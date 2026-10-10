from __future__ import annotations

import math
from typing import Any


def yarn_context_limits(config: dict[str, Any], architecture: str = "") -> tuple[int | None, int | None, float | None]:
    text = config.get("text_config", config.get("language_config", config))
    if not isinstance(text, dict):
        return declared_context_size(config), None, None
    native = declared_context_size(text)
    model_type = text.get("model_type") or config.get("model_type") or architecture
    model_type = model_type.replace("-", "_") if isinstance(model_type, str) else ""
    supported = model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_6", "qwen3_8", "qwen4_exp", "qwen4_exp_text"}
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if isinstance(rope, dict) and isinstance(rope.get("full_attention"), dict):
        rope = rope["full_attention"]
    if not isinstance(rope, dict):
        return native, None, None
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type == "yarn":
        original = rope.get("original_max_position_embeddings")
        if type(original) is int and original > 0:
            native = original
    factor = rope.get("factor", 4.0)
    if not supported or rope_type not in {"default", "yarn"} or native is None or type(factor) not in {int, float}:
        return native, None, None
    if not math.isfinite(factor) or factor <= 1 or native * factor > 2**31 - 1:
        return native, None, None
    return native, math.floor(native * factor), float(factor)


def declared_context_size(config: dict[str, Any], tokenizer: dict[str, Any] | None = None) -> int | None:
    keys = ("max_position_embeddings", "max_seq_len", "max_seq_length", "seq_length", "n_positions")
    for source in (config, config.get("text_config"), config.get("language_config")):
        if isinstance(source, dict):
            for key in keys:
                value = source.get(key)
                if type(value) is int and 0 < value <= 2**31 - 1:
                    return value
    value = (tokenizer or {}).get("model_max_length")
    return value if type(value) is int and 0 < value <= 2**31 - 1 else None
