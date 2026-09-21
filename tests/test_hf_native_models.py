from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFReader

from mfq.architectures.hf_config import load_hf_model_config
from mfq.formats.assets import (
    HF_GENERATION_CONFIG_ASSET,
    HF_SOURCE_MAP_ASSET,
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
    MODEL_GRAPH_ASSET,
)
from mfq.formats.header import FileHeader
from mfq.formats.hf_source import HfSourceTensorStore
from mfq.formats.io import save
from mfq.server.catalog import ModelCatalog
from mfq.server.hf_tokenizer import (
    DEEPSEEK_V4_CHAT_TEMPLATE,
    DEEPSEEK_V41_CHAT_TEMPLATE,
    ensure_hf_tokenizer_gguf,
    ensure_mfq_tokenizer_gguf,
    native_hf_asset_environment,
)
from mfq.server.runtime.native import native_runtime_environment, native_tokenizer_arguments


def _hf_fixture(
    root: Path,
    *,
    model_type: str = "qwen3_5",
    tensor_name: str = "model.language_model.embed_tokens.weight",
) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type, "text_config": {"vocab_size": 6}}),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 3, "eos_token_id": 4, "pad_token_id": 3}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<bos>",
                "chat_template": "{{ messages[0].content }}",
                "add_bos_token": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "ab": 2},
                    "merges": [["a", "b"]],
                },
                "added_tokens": [
                    {
                        "id": 3,
                        "content": "<bos>",
                        "special": True,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                    },
                    {
                        "id": 4,
                        "content": "<eos>",
                        "special": True,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
        separators=(",", ":"),
    ).encode()
    with (root / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.write(b"\x00\x00")


def test_catalog_discovers_native_hf_checkpoint(tmp_path: Path) -> None:
    model = tmp_path / "Qwen-Test"
    _hf_fixture(model)
    catalog = ModelCatalog([tmp_path], cache_seconds=0)
    artifacts = asyncio.run(catalog.list())

    assert len(artifacts.data) == 1
    assert artifacts.data[0].name == "Qwen-Test"
    assert artifacts.data[0].format == "hf"
    assert artifacts.data[0].architecture == "qwen3_5-hf-full-mfq"
    assert artifacts.data[0].tensor_count == 1
    assert artifacts.data[0].dtypes == ["BF16"]
    assert artifacts.data[0].complete
    assert artifacts.data[0].loadable


def test_hf_inference_config_only_fills_missing_text_fields(
    tmp_path: Path,
) -> None:
    model = tmp_path / "DeepSeek-Vision"
    _hf_fixture(model, model_type="deepseek_v4", tensor_name="embed.weight")
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "num_nextn_predict_layers": 3,
            }
        ),
        encoding="utf-8",
    )
    (model / "inference").mkdir()
    (model / "inference" / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 1,
                "n_mtp_layers": 3,
                "dspark_block_size": 5,
                "dspark_target_layer_ids": [40, 41, 42],
            }
        ),
        encoding="utf-8",
    )

    config = load_hf_model_config(model)

    assert config["hidden_size"] == 4096
    assert config["n_mtp_layers"] == 3
    assert config["dspark_block_size"] == 5
    assert config["dspark_target_layer_ids"] == [40, 41, 42]


def test_catalog_tracks_native_hf_routed_expert_bytes(tmp_path: Path) -> None:
    model = tmp_path / "Qwen-MoE"
    _hf_fixture(
        model,
        model_type="qwen4_exp",
        tensor_name=(
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
        ),
    )

    artifact = asyncio.run(
        ModelCatalog([tmp_path], cache_seconds=0).resolve_path(model)
    )
    assert artifact.routed_expert_bytes == 2


@pytest.mark.parametrize("model_type", ["qwen4_exp", "glm5_next"])
def test_catalog_uses_registered_schema_for_native_hf(
    tmp_path: Path,
    model_type: str,
) -> None:
    model = tmp_path / model_type
    _hf_fixture(model)
    (model / "config.json").write_text(
        json.dumps({"model_type": model_type}),
        encoding="utf-8",
    )

    artifact = asyncio.run(ModelCatalog([tmp_path], cache_seconds=0).list()).data[0]

    assert artifact.complete
    assert artifact.format == "hf"
    assert artifact.architecture == f"{model_type}-hf-full-mfq"
    assert artifact.loadable
    assert artifact.error is None


def test_catalog_prefers_same_name_mfq_over_its_hf_source(tmp_path: Path) -> None:
    source = tmp_path / "same-name"
    _hf_fixture(source)
    converted = tmp_path / "same-name.mfq"
    save(
        converted,
        FileHeader(version=2, model_arch="qwen4_exp-test"),
        {"weight": np.ones((1,), dtype=np.float16)},
    )

    artifacts = asyncio.run(ModelCatalog([tmp_path], cache_seconds=0).list())

    assert len(artifacts.data) == 1
    assert artifacts.data[0].name == "same-name"
    assert artifacts.data[0].format == "mfq"
    assert artifacts.data[0].loadable


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen4_exp", "glm5_next"])
def test_native_hf_source_view_comes_from_registered_schema(
    tmp_path: Path,
    model_type: str,
) -> None:
    model = tmp_path / model_type
    _hf_fixture(model)
    (model / "config.json").write_text(
        json.dumps({"model_type": model_type}),
        encoding="utf-8",
    )

    environment = native_hf_asset_environment(model, tmp_path / "assets")
    asset_root = Path(environment["MFQ_RUNTIME_ASSET_DIRECTORY"])
    source_map = json.loads(
        (asset_root / HF_SOURCE_MAP_ASSET.removeprefix("__mfq_asset__/")).read_text()
    )
    graph = json.loads(
        (asset_root / MODEL_GRAPH_ASSET.removeprefix("__mfq_asset__/")).read_text()
    )

    assert source_map["schema"] == "mfq.hf-source-map"
    assert source_map["canonical_to_source"] == {
        "model.token_embedding.weight": (
            "model.language_model.embed_tokens.weight"
        )
    }
    assert graph["canonical_naming"]["namespace"] == "mfq.tensor"


@pytest.mark.parametrize(
    ("model_type", "source_name"),
    [
        ("qwen3_5", "model.language_model.embed_tokens.weight"),
        ("qwen4_exp", "model.language_model.embed_tokens.weight"),
        ("glm5_next", "model.language_model.embed_tokens.weight"),
        ("deepseek_v4", "embed.weight"),
        ("deepseek_v41", "embed.weight"),
        ("minicpmo", "llm.model.embed_tokens.weight"),
        ("glm_moe_dsa", "model.embed_tokens.weight"),
        ("gemma4", "model.language_model.embed_tokens.weight"),
    ],
)
def test_every_registered_architecture_inherits_the_same_hf_source_loader(
    tmp_path: Path,
    model_type: str,
    source_name: str,
) -> None:
    model = tmp_path / model_type
    _hf_fixture(model, model_type=model_type, tensor_name=source_name)

    environment = native_hf_asset_environment(model, tmp_path / "assets")
    asset_root = Path(environment["MFQ_RUNTIME_ASSET_DIRECTORY"])
    source_map = json.loads(
        (asset_root / HF_SOURCE_MAP_ASSET.removeprefix("__mfq_asset__/")).read_text()
    )

    assert source_map["canonical_to_source"] == {
        "model.token_embedding.weight": source_name
    }
    with HfSourceTensorStore(model) as store:
        assert "model.token_embedding.weight" in store
        assert np.asarray(store["model.token_embedding.weight"]).shape == (1,)


def test_native_hf_source_view_is_not_metal_specific(tmp_path: Path) -> None:
    model = tmp_path / "Qwen-Test"
    _hf_fixture(model)

    environment = native_runtime_environment(
        tmp_path / "unused-runtime",
        "cuda",
        {},
        model=model,
    )

    asset_root = Path(environment["MFQ_RUNTIME_ASSET_DIRECTORY"])
    assert (
        asset_root / HF_SOURCE_MAP_ASSET.removeprefix("__mfq_asset__/")
    ).is_file()


def test_hf_tokenizer_cache_is_reusable_and_runtime_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "Qwen-Test"
    cache = tmp_path / "cache"
    _hf_fixture(model)

    tokenizer = ensure_hf_tokenizer_gguf(model, cache)
    assert ensure_hf_tokenizer_gguf(model, cache) == tokenizer
    reader = GGUFReader(tokenizer, "r")
    assert reader.get_field("tokenizer.ggml.model").contents() == "gpt2"
    assert reader.get_field("tokenizer.ggml.pre").contents() == "qwen35"
    assert reader.get_field("tokenizer.ggml.tokens").contents() == [
        "a",
        "b",
        "ab",
        "<bos>",
        "<eos>",
        "[PAD5]",
    ]
    assert reader.get_field("tokenizer.ggml.bos_token_id").contents() == 3
    assert reader.get_field("tokenizer.ggml.eos_token_id").contents() == 4

    monkeypatch.setenv("MFQ_SERVER_TOKENIZER_CACHE_DIR", str(cache))
    arguments = native_tokenizer_arguments(model, "metal")
    assert arguments[0] == "--tokenizer"
    assert Path(arguments[1]).is_file()
    assert native_tokenizer_arguments(model, "cuda") == arguments


@pytest.mark.parametrize(
    "model_type",
    ["deepseek_v4", "deepseek_v4_text", "deepseek_v4_vision"],
)
def test_deepseek_v4_without_hf_template_uses_processor_fallback(
    tmp_path: Path,
    model_type: str,
) -> None:
    model = tmp_path / "DeepSeek-V4-Vision-Test"
    _hf_fixture(model, model_type=model_type, tensor_name="embed.weight")
    tokenizer_config_path = model / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config.pop("chat_template")
    tokenizer_config_path.write_text(json.dumps(tokenizer_config))

    tokenizer = ensure_hf_tokenizer_gguf(model, tmp_path / "cache")
    reader = GGUFReader(tokenizer, "r")

    assert (
        reader.get_field("tokenizer.chat_template").contents()
        == DEEPSEEK_V4_CHAT_TEMPLATE
    )


@pytest.mark.parametrize(
    "model_type",
    ["deepseek_v41", "deepseek_v41_text", "deepseek_v41_vision"],
)
def test_deepseek_v41_without_hf_template_uses_processor_bootstrap(
    tmp_path: Path,
    model_type: str,
) -> None:
    model = tmp_path / "DeepSeek-V41-Vision-Test"
    _hf_fixture(model, model_type=model_type, tensor_name="embed.weight")
    tokenizer_config_path = model / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config.pop("chat_template")
    tokenizer_config_path.write_text(json.dumps(tokenizer_config))

    tokenizer = ensure_hf_tokenizer_gguf(model, tmp_path / "cache")
    reader = GGUFReader(tokenizer, "r")

    assert reader.get_field("tokenizer.chat_template").contents() == (
        DEEPSEEK_V41_CHAT_TEMPLATE
    )


def test_processor_bootstrap_covers_text_variant_config(
    tmp_path: Path,
) -> None:
    model = tmp_path / "DeepSeek-V4-Source-Identity-Test"
    _hf_fixture(model, model_type="deepseek_v4", tensor_name="embed.weight")
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "deepseek_v4_text"
    config["architectures"] = ["DeepseekV4ForCausalLM"]
    config_path.write_text(json.dumps(config))
    tokenizer_config_path = model / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config.pop("chat_template")
    tokenizer_config_path.write_text(json.dumps(tokenizer_config))

    tokenizer = ensure_hf_tokenizer_gguf(model, tmp_path / "cache")
    reader = GGUFReader(tokenizer, "r")

    assert reader.get_field("tokenizer.chat_template").contents() == (
        DEEPSEEK_V4_CHAT_TEMPLATE
    )


def test_native_hf_uses_standalone_chat_template_and_fingerprints_it(
    tmp_path: Path,
) -> None:
    model = tmp_path / "GLM-Test"
    cache = tmp_path / "cache"
    _hf_fixture(model, model_type="glm5_next")
    tokenizer_config_path = model / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config.pop("chat_template")
    tokenizer_config_path.write_text(json.dumps(tokenizer_config))
    template_path = model / "chat_template.jinja"
    template_path.write_text("{{ messages[0].content }}-first")

    first = ensure_hf_tokenizer_gguf(model, cache)
    first_reader = GGUFReader(first, "r")
    assert first_reader.get_field("tokenizer.chat_template").contents() == (
        "{{ messages[0].content }}-first"
    )

    template_path.write_text("{{ messages[0].content }}-second")
    second = ensure_hf_tokenizer_gguf(model, cache)
    second_reader = GGUFReader(second, "r")
    assert second != first
    assert second_reader.get_field("tokenizer.chat_template").contents() == (
        "{{ messages[0].content }}-second"
    )


def test_native_hf_prefers_tokenizer_config_template_over_standalone_file(
    tmp_path: Path,
) -> None:
    model = tmp_path / "Qwen-Test"
    _hf_fixture(model)
    (model / "chat_template.jinja").write_text("standalone")

    tokenizer = ensure_hf_tokenizer_gguf(model, tmp_path / "cache")
    reader = GGUFReader(tokenizer, "r")

    assert reader.get_field("tokenizer.chat_template").contents() == (
        "{{ messages[0].content }}"
    )


def test_mfq_embedded_hf_tokenizer_cache_is_reusable_and_runtime_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "Qwen-Source"
    cache = tmp_path / "cache"
    _hf_fixture(source)
    model = tmp_path / "Qwen-Converted.mfq"
    save(
        model,
        FileHeader(version=2, model_arch="qwen4_exp-hf-mfq-nint-recipe"),
        {
            "weight": np.ones((1,), dtype=np.float16),
            MODEL_CONFIG_ASSET: (source / "config.json").read_bytes(),
            HF_TOKENIZER_JSON_ASSET: (source / "tokenizer.json").read_bytes(),
            HF_TOKENIZER_CONFIG_ASSET: (
                source / "tokenizer_config.json"
            ).read_bytes(),
            HF_GENERATION_CONFIG_ASSET: (
                source / "generation_config.json"
            ).read_bytes(),
        },
    )

    tokenizer = ensure_mfq_tokenizer_gguf(model, cache)
    assert ensure_mfq_tokenizer_gguf(model, cache) == tokenizer
    reader = GGUFReader(tokenizer, "r")
    assert reader.get_field("tokenizer.ggml.pre").contents() == "qwen35"
    assert reader.get_field("tokenizer.ggml.tokens").contents() == [
        "a",
        "b",
        "ab",
        "<bos>",
        "<eos>",
        "[PAD5]",
    ]

    monkeypatch.setenv("MFQ_SERVER_TOKENIZER_CACHE_DIR", str(cache))
    arguments = native_tokenizer_arguments(model, "metal")
    assert arguments == ["--tokenizer", str(tokenizer)]
    assert native_tokenizer_arguments(model, "cuda") == []


def test_minicpmo_native_runtime_materializes_exact_resampler_asset(
    tmp_path: Path,
) -> None:
    model = tmp_path / "MiniCPM-Test"
    _hf_fixture(model)
    (model / "config.json").write_text(
        json.dumps({"model_type": "minicpmo", "vocab_size": 6}),
        encoding="utf-8",
    )
    environment = native_hf_asset_environment(model, tmp_path / "assets")
    asset = (
        Path(environment["MFQ_RUNTIME_ASSET_DIRECTORY"])
        / "minicpmo45-resampler-pos-embed-v1.bf16"
    )
    assert asset.is_file()
    assert asset.read_bytes()[:20] == b"MFQRSPB1" + struct.pack("<III", 70, 70, 4096)


def test_deepseek_v41_materializes_canonical_engram_asset(
    tmp_path: Path,
) -> None:
    model = tmp_path / "DeepSeek-V41-Test"
    _hf_fixture(model)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "bos_token_id": 3,
                "eos_token_id": 4,
                "text_config": {
                    "model_type": "deepseek_v41_text",
                    "vocab_size": 5,
                    "engram_layer_ids": [1],
                    "engram_num_embeddings": [36],
                    "engram_max_ngram_size": 3,
                    "engram_vocab_size": 5,
                    "engram_n_heads": 2,
                    "engram_compressed_vocab_size": 5,
                    "engram_pad_token_id": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    tokenizer = ensure_hf_tokenizer_gguf(model, tmp_path / "tokenizers")
    reader = GGUFReader(tokenizer, "r")
    assert reader.get_field("tokenizer.chat_template").contents() == (
        "{{ messages[0].content }}"
    )

    environment = native_hf_asset_environment(model, tmp_path / "assets")
    asset = (
        Path(environment["MFQ_RUNTIME_ASSET_DIRECTORY"])
        / "deepseek-v41-engram-v1.bin"
    )
    payload = asset.read_bytes()
    assert payload[:28] == struct.pack("<8sIIIII", b"MFQENGR1", 5, 5, 1, 3, 2)
    assert len(payload) == 152
