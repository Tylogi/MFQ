"""Derive an ``mfq.gguf-recipe.v1`` tensor-type file from a GGUF tensor list.

A tensor list is the read-only header dump of a GGUF file (names, types and
shapes, no payload).  Two derivations are supported:

* ``--ftype source`` keeps the listed types.  This registers a published UD
  file whose tensor types are read from its own header.
* ``--ftype IQ4_XS`` (or ``IQ4_NL``) reproduces llama.cpp
  ``llama_tensor_get_type``: ``llama_ftype_get_default_type``,
  ``tensor_allows_quantization``, the per-category branches of
  ``llama_tensor_get_type_impl`` for that file type and
  ``tensor_type_fallback``.  Only the 4-bit IQ4 families are modelled; the
  stateful K-quant/I-quant branches of other file types are rejected instead
  of being approximated.

GGUF payload bytes follow ggml block sizes, 32-byte tensor alignment and the
file's own header/metadata overhead, so the derived whole-file ceiling stays
comparable with the source file it was derived from.

Example::

    python -m mfq.tools.derive_gguf_recipe UD-IQ2_M.header.json \\
        --ftype IQ4_XS --has-imatrix --n-gqa 0 --n-expert 512 \\
        --reference-file-bytes 55876393864 \\
        --repo unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF \\
        --revision 75a5829055f69a0359085ae6bbd0a81b60bc05c6 \\
        --output mfq/calibration/recipes/nemotron_h/IQ4_XS.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

RECIPE_FORMAT = "mfq.gguf-recipe.v1"
TENSOR_ALIGNMENT = 32
MTP_BLOCKS = re.compile(r"^blk\.(\d+)\.")


class TensorRecord(TypedDict):
    """One entry of a GGUF header tensor list."""

    name: str
    type: str
    shape: list[int]


class RecipeDocument(TypedDict):
    """A recipe file plus the provenance of its derivation."""

    format: str
    tensor_types: dict[str, str]
    repo: str
    revision: str
    derivation: dict[str, object]


# ggml type name -> (elements per block, bytes per block).
GGML_TYPE_LAYOUT: dict[str, tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "IQ1_S": (256, 50),
    "IQ1_M": (256, 56),
    "IQ2_XXS": (256, 66),
    "IQ2_XS": (256, 74),
    "IQ2_S": (256, 82),
    "IQ3_XXS": (256, 98),
    "IQ3_S": (256, 110),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
    "Q1_0": (128, 18),
    "Q2_0": (64, 18),
    "TQ1_0": (256, 54),
    "TQ2_0": (256, 66),
    "MXFP4": (32, 17),
}

# ggml enum values that appear in a serialized header but are not spelled out
# in the tensor list.
GGML_TYPE_IDS: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
    41: "Q1_0",
    42: "Q2_0",
}

# llama_ftype_get_default_type for the modelled file types.
FTYPE_DEFAULT_TYPE = {"IQ4_XS": "IQ4_XS", "IQ4_NL": "IQ4_NL"}

# tensor_type_fallback: incompatible row widths demote to a smaller block.
TYPE_FALLBACK = {
    "IQ1_S": "IQ4_NL",
    "IQ1_M": "IQ4_NL",
    "IQ2_XXS": "IQ4_NL",
    "IQ2_XS": "IQ4_NL",
    "IQ2_S": "IQ4_NL",
    "IQ3_XXS": "IQ4_NL",
    "IQ3_S": "IQ4_NL",
    "IQ4_XS": "IQ4_NL",
    "Q2_0": "Q4_0",
    "Q2_K": "Q4_0",
    "Q3_K": "Q4_0",
    "TQ1_0": "Q4_0",
    "TQ2_0": "Q4_0",
    "Q4_K": "Q5_0",
    "Q5_K": "Q5_1",
    "Q6_K": "Q8_0",
}

# tensor_allows_quantization: names llama.cpp never narrows.
_PROTECTED_NAME_PARTS = (
    "ffn_gate_inp.weight",
    "ffn_gate_tid2eid.weight",
    "ssm_conv1d",
    "shortconv.conv.weight",
    "altup",
    "laurel",
    "per_layer_model_proj",
    "indexer.k_proj.weight",
    "indexer.q_proj.weight",
    "time_mix_",
    "attn_rel_b.weight",
    ".position_embd",
    "sam.pos_embd",
    "sam.neck.",
    "sam.net_",
    ".rel_pos",
    ".patch_embd",
    ".patch_merger",
    "a.rvq.codebook",
    "mm.a.code_embd",
)

_CATEGORIES = (
    ("attn_qkv.weight", "attn_qkv"),
    ("attn_kv_b.weight", "attn_kv_b"),
    ("attn_v.weight", "attn_v"),
    ("attn_k.weight", "attn_k"),
    ("attn_q.weight", "attn_q"),
    ("attn_output.weight", "attn_output"),
    ("ffn_up", "ffn_up"),
    ("ffn_gate", "ffn_gate"),
    ("ffn_latent", "ffn_latent"),
    ("attn_gate.weight", "attn_gate"),
    ("ffn_down", "ffn_down"),
)


def type_name(value: str | int) -> str:
    """Normalize a listed ggml type to its header spelling."""

    if isinstance(value, int) or str(value).isdigit():
        index = int(value)
        if index not in GGML_TYPE_IDS:
            raise ValueError(f"unsupported ggml type id: {value}")
        return GGML_TYPE_IDS[index]
    name = str(value).upper()
    if name not in GGML_TYPE_LAYOUT:
        raise ValueError(f"unsupported ggml tensor type: {value}")
    return name


def tensor_elements(shape: Sequence[int]) -> int:
    """Return the element count of one GGUF tensor shape."""

    count = 1
    for size in shape:
        count *= int(size)
    return count


def payload_bytes(shape: Sequence[int], ggml_type: str) -> int:
    """Return the ggml payload bytes of a tensor, without file padding."""

    layout = GGML_TYPE_LAYOUT[ggml_type]
    elements = tensor_elements(shape)
    if elements % layout[0]:
        raise ValueError(
            f"{ggml_type} requires {layout[0]} elements per block, "
            f"tensor has {elements}"
        )
    return elements // layout[0] * layout[1]


def aligned_bytes(nbytes: int, alignment: int = TENSOR_ALIGNMENT) -> int:
    """Return the bytes one tensor occupies when written 32-byte aligned."""

    return (int(nbytes) + alignment - 1) // alignment * alignment


def tensor_category(name: str) -> str:
    """Return the llama.cpp quantization category of a GGUF tensor name."""

    if name == "output.weight":
        return "output"
    if name in {"token_embd.weight", "per_layer_token_embd.weight"}:
        return "token_embd"
    for marker, category in _CATEGORIES:
        if marker in name:
            return category
    return "other"


def allows_quantization(
    name: str,
    shape: Sequence[int],
    *,
    quantize_output: bool = True,
) -> bool:
    """Reproduce ``tensor_allows_quantization`` for a header tensor."""

    if len(shape) < 2:
        return False
    if not name.endswith("weight") or "_norm.weight" in name:
        return False
    if name == "output.weight" and not quantize_output:
        return False
    return not any(part in name for part in _PROTECTED_NAME_PARTS)


def block_index(name: str) -> int | None:
    """Return the ``blk.N`` index of a block tensor, else ``None``."""

    match = MTP_BLOCKS.match(name)
    return None if match is None else int(match.group(1))


def _layer_position(
    name: str,
    counter: int,
    n_layer_all: int,
    n_expert: int,
) -> int:
    """Return the layer a MoE file's tensor belongs to, as llama.cpp parses it."""

    if n_expert <= 1:
        return counter
    index = block_index(name)
    if index is None:
        raise ValueError(f"llama.cpp cannot determine the layer of {name}")
    if not 0 <= index < n_layer_all:
        raise ValueError(
            f"llama.cpp rejects block {index} of {name}: "
            f"expected [0, {n_layer_all})"
        )
    return index


def _fallback_type(row_elements: int, ggml_type: str) -> str:
    """Demote a type whose block does not divide the row width."""

    if row_elements % GGML_TYPE_LAYOUT[ggml_type][0] == 0:
        return ggml_type
    fallback = TYPE_FALLBACK.get(ggml_type, ggml_type)
    if row_elements % GGML_TYPE_LAYOUT[fallback][0] == 0:
        return fallback
    return "F16"


def derive_tensor_type(
    name: str,
    shape: Sequence[int],
    source_type: str,
    *,
    ftype: str,
    has_imatrix: bool,
    n_gqa: int,
    n_expert: int,
    n_layer_all: int,
    counters: dict[str, int],
) -> str:
    """Return the llama.cpp type for one tensor, with stateful counters.

    ``counters`` holds ``i_ffn_down`` and ``i_attention_wv`` in header order,
    exactly as ``llama_tensor_get_type_impl`` walks the tensor list.
    """

    if ftype not in FTYPE_DEFAULT_TYPE:
        raise ValueError(f"unsupported recipe derivation file type: {ftype}")
    if not allows_quantization(name, shape):
        return source_type
    category = tensor_category(name)
    new_type = FTYPE_DEFAULT_TYPE[ftype]
    if category == "output":
        # output_tensor_type is unset, so any non-Q8_0 default becomes Q6_K.
        new_type = "Q6_K"
    elif category == "token_embd":
        pass  # no 4-bit override replaces the default lookup type
    elif category in {"attn_v", "attn_qkv", "attn_kv_b"}:
        if n_gqa >= 4:
            new_type = "Q5_K"
        counters["i_attention_wv"] += 1
    elif category == "attn_output" and n_expert == 8:
        # The 8-expert models bump attention output whatever the 4-bit file is.
        new_type = "Q5_K"
    elif category == "ffn_down":
        position = _layer_position(name, counters["i_ffn_down"], n_layer_all, n_expert)
        if not has_imatrix and position < n_layer_all // 8:
            new_type = "Q5_K"
        counters["i_ffn_down"] += 1
    return _fallback_type(int(shape[0]), new_type)


def derive_tensor_types(
    records: Iterable[TensorRecord],
    *,
    ftype: str,
    has_imatrix: bool,
    n_gqa: int,
    n_expert: int,
    n_layer_all: int,
) -> dict[str, str]:
    """Map one GGUF tensor list onto the derived file type."""

    counters = {"i_ffn_down": 0, "i_attention_wv": 0}
    types: dict[str, str] = {}
    for record in records:
        name = record["name"]
        if name in types:
            raise ValueError(f"duplicate GGUF tensor name: {name}")
        shape = tuple(record["shape"])
        source_type = type_name(record["type"])
        if ftype == "source":
            types[name] = source_type
            continue
        types[name] = derive_tensor_type(
            name,
            shape,
            source_type,
            ftype=ftype,
            has_imatrix=has_imatrix,
            n_gqa=n_gqa,
            n_expert=n_expert,
            n_layer_all=n_layer_all,
            counters=counters,
        )
    return types


def tensor_payload_bytes(
    records: Iterable[TensorRecord],
    types: Mapping[str, str],
) -> int:
    """Return the aligned on-disk bytes of every listed tensor."""

    total = 0
    for record in records:
        name = record["name"]
        total += aligned_bytes(payload_bytes(tuple(record["shape"]), types[name]))
    return total


def main_tensor_elements(
    records: Iterable[TensorRecord],
    *,
    block_count: int,
    mtp_blocks: int,
) -> int:
    """Return the main text element count, excluding the trailing MTP blocks."""

    first_mtp = block_count - mtp_blocks
    total = 0
    for record in records:
        index = block_index(record["name"])
        if mtp_blocks > 0 and index is not None and index >= first_mtp:
            continue
        total += tensor_elements(record["shape"])
    return total


def build_recipe_document(
    records: Sequence[TensorRecord],
    *,
    ftype: str,
    has_imatrix: bool,
    n_gqa: int,
    n_expert: int,
    block_count: int,
    mtp_blocks: int,
    reference_file_bytes: int,
    repo: str,
    revision: str,
    architecture: str = "",
) -> RecipeDocument:
    """Build the recipe document, its payload accounting and its ceiling."""

    if ftype != "source" and ftype not in FTYPE_DEFAULT_TYPE:
        raise ValueError(f"unsupported recipe derivation file type: {ftype}")
    if reference_file_bytes <= 0:
        raise ValueError("reference file bytes must be positive")
    types = derive_tensor_types(
        records,
        ftype=ftype,
        has_imatrix=has_imatrix,
        n_gqa=n_gqa,
        n_expert=n_expert,
        n_layer_all=block_count,
    )
    source_types = {record["name"]: type_name(record["type"]) for record in records}
    # The header/metadata overhead is the reference file's non-tensor bytes; a
    # derived recipe reuses it so both ceilings describe the same container.
    overhead = reference_file_bytes - tensor_payload_bytes(records, source_types)
    if overhead < 0:
        raise ValueError(
            f"reference file is smaller than its own tensor payload: "
            f"{reference_file_bytes} < {tensor_payload_bytes(records, source_types)}"
        )
    payload = tensor_payload_bytes(records, types)
    return {
        "format": RECIPE_FORMAT,
        "tensor_types": types,
        "repo": repo,
        "revision": revision,
        "derivation": {
            "tool": "mfq/tools/derive_gguf_recipe.py",
            "architecture": architecture,
            "ftype": ftype,
            "has_imatrix": has_imatrix,
            "n_gqa": n_gqa,
            "n_expert": n_expert,
            "block_count": block_count,
            "mtp_blocks": mtp_blocks,
            "tensor_count": len(types),
            "elements": sum(tensor_elements(record["shape"]) for record in records),
            "main_text_elements": main_tensor_elements(
                records,
                block_count=block_count,
                mtp_blocks=mtp_blocks,
            ),
            "tensor_payload_bytes": payload,
            "reference_tensor_payload_bytes": tensor_payload_bytes(records, source_types),
            "file_overhead_bytes": overhead,
            "ud_file_bytes": payload + overhead,
        },
    }


def load_tensor_list(path: Path) -> tuple[list[TensorRecord], dict[str, object]]:
    """Read a header dump produced by the read-only GGUF header reader."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("tensors"), list):
        raise ValueError(f"tensor list must contain a tensors array: {path}")
    records: list[TensorRecord] = []
    for entry in document["tensors"]:
        if not isinstance(entry, dict):
            raise ValueError(f"tensor list entry is not an object: {entry!r}")
        records.append(
            TensorRecord(
                name=str(entry["name"]),
                type=str(entry["type"]),
                shape=[int(value) for value in entry["shape"]],
            )
        )
    metadata = document.get("metadata", {})
    return records, dict(metadata) if isinstance(metadata, dict) else {}


def metadata_int(metadata: Mapping[str, object], key: str) -> int:
    """Return one integer metadata value, or zero when it is absent."""

    value = metadata.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tensor_list", type=Path, help="GGUF header dump JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ftype",
        default="source",
        help="source keeps the listed types; IQ4_XS or IQ4_NL derive llama.cpp types",
    )
    parser.add_argument("--has-imatrix", action="store_true")
    parser.add_argument(
        "--n-gqa",
        type=int,
        default=0,
        help="hparams.n_gqa() of block 0, which llama.cpp evaluates for attn_v",
    )
    parser.add_argument("--n-expert", type=int, default=0)
    parser.add_argument("--block-count", type=int, default=0)
    parser.add_argument("--mtp-blocks", type=int, default=0)
    parser.add_argument(
        "--reference-file-bytes",
        type=int,
        required=True,
        help="whole-file size of the file the tensor list was read from",
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, metadata = load_tensor_list(args.tensor_list)
    architecture = str(metadata.get("general.architecture", ""))
    block_count = args.block_count or metadata_int(metadata, f"{architecture}.block_count")
    mtp_blocks = args.mtp_blocks or metadata_int(
        metadata, f"{architecture}.nextn_predict_layers"
    )
    n_expert = args.n_expert or metadata_int(metadata, f"{architecture}.expert_count")
    document = build_recipe_document(
        records,
        ftype=args.ftype,
        has_imatrix=args.has_imatrix,
        n_gqa=args.n_gqa,
        n_expert=n_expert,
        block_count=block_count,
        mtp_blocks=mtp_blocks,
        reference_file_bytes=args.reference_file_bytes,
        repo=args.repo,
        revision=args.revision,
        architecture=architecture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    derivation = document["derivation"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ftype": derivation["ftype"],
                "tensors": derivation["tensor_count"],
                "tensor_payload_bytes": derivation["tensor_payload_bytes"],
                "ud_file_bytes": derivation["ud_file_bytes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
