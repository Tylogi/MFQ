"""Dependency-light loading of Hugging Face model configuration metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def text_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the part of a composite config that describes the text model.

    Most families nest the language model under ``text_config``, NVIDIA Omni
    wrappers use ``llm_config``, and a plain text checkpoint is itself the
    text section.  Importers and topology readers share this one rule instead
    of spelling the nesting key per architecture.
    """

    for key in ("text_config", "llm_config"):
        raw = config.get(key)
        if isinstance(raw, Mapping):
            return raw
    return config


def merge_hf_inference_config(
    config: Mapping[str, Any],
    supplemental: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill absent model fields from an upstream inference sidecar.

    Some official checkpoints keep runtime-shape metadata in
    ``inference/config.json`` while retaining the Transformers-facing fields
    in the root config.  Root values remain authoritative.  For composite
    configs, the supplemental model fields belong to ``text_config``.
    """

    merged = dict(config)
    raw_target = merged.get("text_config")
    raw_source = supplemental.get("text_config")
    source = raw_source if isinstance(raw_source, Mapping) else supplemental
    if isinstance(raw_target, Mapping):
        target = dict(raw_target)
        for key, value in source.items():
            target.setdefault(str(key), value)
        merged["text_config"] = target
    else:
        for key, value in source.items():
            merged.setdefault(str(key), value)
    return merged


def load_hf_model_config(root: str | Path) -> dict[str, Any]:
    """Load one HF config plus any standard inference-sidecar additions."""

    directory = Path(root).expanduser().resolve()
    config = _json_object(directory / "config.json")
    supplemental_path = directory / "inference" / "config.json"
    if supplemental_path.is_file():
        config = merge_hf_inference_config(
            config,
            _json_object(supplemental_path),
        )
    return config


__all__ = ["load_hf_model_config", "merge_hf_inference_config", "text_section"]
