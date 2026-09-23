"""Build a small llama-compatible tokenizer cache from a native HF model."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
from pathlib import Path
from typing import Any

from mfq.architectures.hf_config import load_hf_model_config
from mfq.architectures.tensor_schema import (
    canonical_source_tensor_map,
    graph_spec_for_source_names,
    source_runtime_assets,
)
from mfq.formats.assets import (
    ASSET_PREFIX,
    HF_CHAT_TEMPLATE_ASSET,
    HF_GENERATION_CONFIG_ASSET,
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
    TOKENIZER_GGUF_ASSET,
    hf_source_map_asset,
    model_graph_asset,
)
from mfq.formats.io import open_mmap
from mfq.server.protocol.processor_prompt_protocols import (
    DEEPSEEK_V4_CHAT_TEMPLATE,
    DEEPSEEK_V41_CHAT_TEMPLATE,
    processor_prompt_protocol_for_architecture,
    processor_prompt_protocols,
)


class HfTokenizerError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HfTokenizerError(f"cannot read {path.name}") from error
    if not isinstance(value, dict):
        raise HfTokenizerError(f"invalid {path.name}")
    return value


def _safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise HfTokenizerError(f"truncated Safetensors header: {path.name}")
            size = struct.unpack("<Q", raw_size)[0]
            value = json.loads(stream.read(size))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HfTokenizerError(f"cannot read Safetensors header: {path.name}") from error
    if not isinstance(value, dict):
        raise HfTokenizerError(f"invalid Safetensors header: {path.name}")
    return value


def _hf_runtime_tensor_names(root: Path) -> tuple[str, ...]:
    index_path = root / "model.safetensors.index.json"
    expected: dict[str, str] | None = None
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise HfTokenizerError("invalid model.safetensors.index.json")
        expected = dict(weight_map)
        shards = tuple(root / name for name in sorted(set(expected.values())))
    else:
        shards = tuple(sorted(root.glob("*.safetensors")))
    if not shards or any(not path.is_file() for path in shards):
        raise HfTokenizerError("HF checkpoint is missing Safetensors shards")

    tensor_dtypes: dict[str, str] = {}
    for shard in shards:
        for name, entry in _safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            if expected is not None and expected.get(name) != shard.relative_to(root).as_posix():
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
                raise HfTokenizerError(f"invalid Safetensors tensor: {name}")
            if name in tensor_dtypes:
                raise HfTokenizerError(f"duplicate Safetensors tensor: {name}")
            tensor_dtypes[name] = entry["dtype"]
    if expected is not None and set(expected) != set(tensor_dtypes):
        raise HfTokenizerError("Safetensors index references missing tensors")
    # E8M0 records are folded into their MX weight at execution time, but they
    # remain in this physical-source map so the container can pair them by
    # canonical semantics instead of guessing an upstream naming convention.
    return tuple(sorted(tensor_dtypes))


def _token_content(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, int)), None)
    return None


def _tokenizer_pre(model_type: str) -> str:
    if model_type.startswith(("qwen3_5", "qwen3_6", "qwen3_8", "qwen4_exp")):
        return "qwen35"
    if model_type.startswith(("glm5_next", "glm_moe_dsa")):
        return "glm4"
    if model_type.startswith("gemma4"):
        return "gemma4"
    if model_type.startswith("minicpmo"):
        return "qwen2"
    if model_type.startswith("deepseek_v4"):
        return "joyai-llm"
    raise HfTokenizerError(f"unsupported native HF tokenizer family: {model_type}")


def _special_id(
    name: str,
    tokenizer_config: dict[str, Any],
    config: dict[str, Any],
    generation_config: dict[str, Any],
    token_ids: dict[str, int],
) -> int | None:
    content = _token_content(tokenizer_config.get(f"{name}_token"))
    if content is not None and content in token_ids:
        return token_ids[content]
    text_config = config.get("text_config")
    candidates = (
        config.get(f"{name}_token_id"),
        text_config.get(f"{name}_token_id") if isinstance(text_config, dict) else None,
        generation_config.get(f"{name}_token_id"),
    )
    return next((value for item in candidates if (value := _integer(item)) is not None), None)


def _fingerprint_payloads(payloads: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"mfq-hf-tokenizer-gguf-v9\0")
    # Processor-owned fallback templates are generated by MFQ rather than
    # checkpoint files, so their contents must participate in cache identity.
    for protocol in processor_prompt_protocols():
        digest.update(protocol.name.encode("utf-8"))
        digest.update(protocol.bootstrap_chat_template.encode("utf-8"))
    for name, payload in payloads:
        digest.update(name.encode("utf-8"))
        digest.update(payload)
    return digest.hexdigest()[:20]


def _fingerprint(root: Path) -> str:
    payloads = tuple(
        (name, path.read_bytes())
        for name in (
            "config.json",
            "inference/config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
        if (path := root / name).is_file()
    )
    return _fingerprint_payloads(payloads)


def _cache_root(cache_directory: str | Path | None) -> Path:
    root = (
        Path(cache_directory).expanduser().resolve()
        if cache_directory is not None
        else Path(os.environ.get("MFQ_SERVER_TOKENIZER_CACHE_DIR", "~/.cache/mfq/tokenizers"))
        .expanduser()
        .resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_payload(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HfTokenizerError(f"cannot read embedded {name}") from error
    if not isinstance(value, dict):
        raise HfTokenizerError(f"invalid embedded {name}")
    return value


def _has_chat_template(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, list) and bool(value)


def _decode_chat_template(payload: bytes, source: str) -> str:
    try:
        template = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HfTokenizerError(f"cannot read {source}") from error
    if not template.strip():
        raise HfTokenizerError(f"empty {source}")
    return template


def _resolve_chat_template(
    tokenizer_config: dict[str, Any],
    config: dict[str, Any],
    template_asset: bytes | None,
) -> str | list[Any] | None:
    configured = tokenizer_config.get("chat_template")
    if _has_chat_template(configured):
        return configured
    if template_asset is not None:
        return _decode_chat_template(template_asset, "chat_template.jinja")

    # DeepSeek-V4 publishes its message encoder beside the checkpoint instead
    # of a Hugging Face template.  This minimal template initializes the native
    # tokenizer/parser; the managed API renders the released processor format.
    raw_text_config = config.get("text_config", {})
    text_config = raw_text_config if isinstance(raw_text_config, dict) else {}
    raw_architectures = config.get("architectures", [])
    architectures = raw_architectures if isinstance(raw_architectures, list) else []
    identities = (
        text_config.get("model_type"),
        config.get("model_type"),
        *architectures,
    )
    for identity in identities:
        if not isinstance(identity, str) or not identity.strip():
            continue
        protocol = processor_prompt_protocol_for_architecture(identity)
        if protocol is not None:
            return protocol.bootstrap_chat_template
    return None


def _write_tokenizer_gguf(
    *,
    output: Path,
    model_name: str,
    tokenizer: dict[str, Any],
    tokenizer_config: dict[str, Any],
    config: dict[str, Any],
    generation_config: dict[str, Any],
    chat_template_asset: bytes | None = None,
) -> Path:
    if output.is_file() and output.stat().st_size > 0:
        return output

    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise HfTokenizerError("HF config.json has no model_type")
    chat_template = _resolve_chat_template(
        tokenizer_config,
        config,
        chat_template_asset,
    )
    if chat_template is not None:
        tokenizer_config = dict(tokenizer_config)
        tokenizer_config["chat_template"] = chat_template
    model = tokenizer.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise HfTokenizerError("only HF BPE tokenizers are supported by the native runtime")
    raw_vocab = model.get("vocab")
    raw_merges = model.get("merges")
    raw_added = tokenizer.get("added_tokens", [])
    if not isinstance(raw_vocab, dict) or not isinstance(raw_merges, list) or not isinstance(raw_added, list):
        raise HfTokenizerError("invalid HF BPE tokenizer.json")

    try:
        from gguf import GGUFWriter, TokenType
    except ModuleNotFoundError as error:
        raise HfTokenizerError("native HF loading requires the lightweight 'gguf' package") from error

    assigned: dict[int, tuple[str, int]] = {}
    token_ids: dict[str, int] = {}
    for token, raw_id in raw_vocab.items():
        if not isinstance(token, str) or not isinstance(raw_id, int) or raw_id < 0:
            raise HfTokenizerError("invalid HF BPE vocabulary entry")
        assigned[raw_id] = (token, int(TokenType.NORMAL))
        token_ids[token] = raw_id
    for entry in raw_added:
        if not isinstance(entry, dict):
            raise HfTokenizerError("invalid HF added token entry")
        token = entry.get("content")
        raw_id = entry.get("id")
        if not isinstance(token, str) or not isinstance(raw_id, int) or raw_id < 0:
            raise HfTokenizerError("invalid HF added token entry")
        control_alias = (
            token.startswith("<|fim_") or
            token in {"<|repo_name|>", "<|file_sep|>", "</s>"}
        )
        token_type = (
            TokenType.CONTROL
            if entry.get("special") is True or control_alias
            else TokenType.USER_DEFINED
        )
        assigned[raw_id] = (token, int(token_type))
        token_ids[token] = raw_id

    text_config = config.get("text_config")
    configured_vocab = _integer(config.get("vocab_size"))
    if configured_vocab is None and isinstance(text_config, dict):
        configured_vocab = _integer(text_config.get("vocab_size"))
    minimum_vocab = max(assigned, default=-1) + 1
    vocabulary_size = max(minimum_vocab, configured_vocab or 0)
    if vocabulary_size <= 0:
        raise HfTokenizerError("HF tokenizer has an empty vocabulary")
    tokens: list[str] = []
    token_types: list[int] = []
    for token_id in range(vocabulary_size):
        token, token_type = assigned.get(
            token_id, (f"[PAD{token_id}]", int(TokenType.UNUSED))
        )
        tokens.append(token)
        token_types.append(token_type)

    merges: list[str] = []
    for merge in raw_merges:
        if isinstance(merge, str):
            merges.append(merge)
        elif (isinstance(merge, list) and len(merge) == 2 and
              all(isinstance(item, str) for item in merge)):
            merges.append(" ".join(merge))
        else:
            raise HfTokenizerError("invalid HF BPE merge entry")

    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    temporary.unlink(missing_ok=True)
    try:
        writer = GGUFWriter(temporary, "llama")
        try:
            writer.add_name(model_name)
            writer.add_tokenizer_model("gpt2")
            writer.add_tokenizer_pre(_tokenizer_pre(model_type))
            writer.add_token_list(tokens)
            writer.add_token_types(token_types)
            writer.add_token_merges(merges)
            for name, method_name in (
                ("bos", "add_bos_token_id"),
                ("eos", "add_eos_token_id"),
                ("unk", "add_unk_token_id"),
                ("pad", "add_pad_token_id"),
            ):
                token_id = _special_id(
                    name, tokenizer_config, config, generation_config, token_ids
                )
                if token_id is not None:
                    getattr(writer, method_name)(token_id)
            if isinstance(tokenizer_config.get("add_bos_token"), bool):
                writer.add_add_bos_token(tokenizer_config["add_bos_token"])
            if isinstance(tokenizer_config.get("add_eos_token"), bool):
                writer.add_add_eos_token(tokenizer_config["add_eos_token"])
            chat_template = tokenizer_config.get("chat_template")
            if isinstance(chat_template, (str, list)):
                writer.add_chat_template(chat_template)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def ensure_hf_tokenizer_gguf(
    model_directory: str | Path,
    cache_directory: str | Path | None = None,
) -> Path:
    """Return a reusable tokenizer-only GGUF for a supported HF checkpoint."""

    root = Path(model_directory).expanduser().resolve()
    bundled = root / "tokenizer.gguf"
    if bundled.is_file():
        return bundled
    if not root.is_dir():
        raise HfTokenizerError(f"HF model directory does not exist: {root}")

    tokenizer = _read_json(root / "tokenizer.json")
    tokenizer_config = _read_json(root / "tokenizer_config.json")
    try:
        config = load_hf_model_config(root)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise HfTokenizerError("cannot read model configuration") from error
    generation_path = root / "generation_config.json"
    generation_config = _read_json(generation_path) if generation_path.is_file() else {}
    chat_template_path = root / "chat_template.jinja"
    chat_template_asset = (
        chat_template_path.read_bytes() if chat_template_path.is_file() else None
    )
    output = _cache_root(cache_directory) / (
        f"{root.name}-{_fingerprint(root)}.tokenizer.gguf"
    )
    return _write_tokenizer_gguf(
        output=output,
        model_name=root.name,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        config=config,
        generation_config=generation_config,
        chat_template_asset=chat_template_asset,
    )


def ensure_mfq_tokenizer_gguf(
    model_file: str | Path,
    cache_directory: str | Path | None = None,
) -> Path:
    """Build a reusable tokenizer GGUF from assets embedded in an MFQ model."""

    path = Path(model_file).expanduser().resolve()
    if not path.is_file():
        raise HfTokenizerError(f"MFQ model does not exist: {path}")
    with open_mmap(path) as store:
        if TOKENIZER_GGUF_ASSET in store.records:
            payload = store.read_blob(TOKENIZER_GGUF_ASSET)
            fingerprint = hashlib.sha256(payload).hexdigest()[:20]
            output = _cache_root(cache_directory) / (
                f"{path.stem}-{fingerprint}.tokenizer.gguf"
            )
            if output.is_file() and output.stat().st_size > 0:
                return output
            temporary = output.with_name(
                f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
            )
            temporary.unlink(missing_ok=True)
            try:
                temporary.write_bytes(payload)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            return output
        required = (
            MODEL_CONFIG_ASSET,
            HF_TOKENIZER_JSON_ASSET,
            HF_TOKENIZER_CONFIG_ASSET,
        )
        missing = [name for name in required if name not in store.records]
        if missing:
            raise HfTokenizerError(
                "MFQ model has no embedded native tokenizer metadata: "
                + ", ".join(missing)
            )
        payloads = tuple(
            (name, store.read_blob(name))
            for name in (
                MODEL_CONFIG_ASSET,
                HF_TOKENIZER_JSON_ASSET,
                HF_TOKENIZER_CONFIG_ASSET,
                HF_GENERATION_CONFIG_ASSET,
                HF_CHAT_TEMPLATE_ASSET,
            )
            if name in store.records
        )

    values = dict(payloads)
    tokenizer = _json_payload(values[HF_TOKENIZER_JSON_ASSET], "tokenizer.json")
    tokenizer_config = _json_payload(
        values[HF_TOKENIZER_CONFIG_ASSET], "tokenizer_config.json"
    )
    config = _json_payload(values[MODEL_CONFIG_ASSET], "model_config.json")
    generation_config = (
        _json_payload(values[HF_GENERATION_CONFIG_ASSET], "generation_config.json")
        if HF_GENERATION_CONFIG_ASSET in values
        else {}
    )
    model_name = path.stem
    output = _cache_root(cache_directory) / (
        f"{model_name}-{_fingerprint_payloads(payloads)}.tokenizer.gguf"
    )
    return _write_tokenizer_gguf(
        output=output,
        model_name=model_name,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        config=config,
        generation_config=generation_config,
        chat_template_asset=values.get(HF_CHAT_TEMPLATE_ASSET),
    )


def native_hf_asset_environment(
    model_directory: str | Path,
    cache_directory: str | Path | None = None,
) -> dict[str, str]:
    """Expose generated constants through the generic runtime-asset directory."""

    root = Path(model_directory).expanduser().resolve()
    if not root.is_dir():
        return {}
    try:
        config = load_hf_model_config(root)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise HfTokenizerError("cannot read model configuration") from error
    cache_root = (
        Path(cache_directory).expanduser().resolve()
        if cache_directory is not None
        else Path(os.environ.get("MFQ_SERVER_ASSET_CACHE_DIR", "~/.cache/mfq/assets"))
        .expanduser()
        .resolve()
    )
    try:
        source_names = _hf_runtime_tensor_names(root)
        graph = graph_spec_for_source_names(config, source_names)
        if graph is None:
            return {}
        assets = [
            hf_source_map_asset(
                canonical_source_tensor_map(config, source_names)
            ),
            model_graph_asset(graph.as_dict()),
        ]
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise HfTokenizerError(
            "cannot build the native HF canonical source view"
        ) from error
    try:
        assets.extend(source_runtime_assets(root, config))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise HfTokenizerError(
            "cannot build architecture runtime assets from the source checkpoint"
        ) from error
    if not assets:
        return {}

    fingerprint_hash = hashlib.sha256()
    for asset in assets:
        fingerprint_hash.update(asset.name.encode("utf-8"))
        fingerprint_hash.update(asset.data)
    asset_root = cache_root / fingerprint_hash.hexdigest()[:20]
    for asset in assets:
        relative = asset.name.removeprefix(ASSET_PREFIX)
        if not relative or relative == asset.name:
            raise HfTokenizerError("runtime asset is outside the reserved namespace")
        output = asset_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file() and output.read_bytes() == asset.data:
            continue
        temporary = output.with_name(
            f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        try:
            temporary.write_bytes(asset.data)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"MFQ_RUNTIME_ASSET_DIRECTORY": str(asset_root)}


__all__ = [
    "DEEPSEEK_V41_CHAT_TEMPLATE",
    "DEEPSEEK_V4_CHAT_TEMPLATE",
    "HfTokenizerError",
    "ensure_hf_tokenizer_gguf",
    "ensure_mfq_tokenizer_gguf",
    "native_hf_asset_environment",
]
