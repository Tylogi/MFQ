"""Streaming AlphaQ collection using the existing MFQ quantizer's source plans."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mfq.calibration.alphaq import (
    METHOD,
    STATISTICS_FORMAT,
    AlphaQTensorStatistics,
    alphaq_weight_statistics,
)

_BANK = re.compile(r"^model\.block\.(\d+)\.mlp\.experts\.(gate|up|down|gate_up)\.weight$")
_LEGACY_BANK = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _progress(**value: Any) -> None:
    # A closed pipe or failed optional console must not discard collected work.
    with suppress(OSError):
        print(json.dumps(value), flush=True)


def collect_alphaq(
    model: str | Path, statistics: str | Path, *, device: str = "auto"
) -> tuple[AlphaQTensorStatistics, ...]:
    """Read one expert bank at a time; save and resume completed bank statistics.

    Source tensor discovery, split-expert aggregation and native MX/FP8
    decoding are shared with ``mfq quantize``. Cached identities use file
    metadata, not a second full-model hash pass. No forward model is loaded.
    """
    import torch

    from mfq.tools.quantize_hf_to_mfq import (
        SourceTensorMetadata,
        _MfqGlmExpertRowSource,
        _PackedExpertProjectionSource,
        _plan,
        _raw_source_for_plan,
        _ScaledFp8TensorSlice,
        _SeparateExpertRowSource,
    )

    root, output = Path(model).resolve(), Path(statistics).resolve()
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.device(device).type not in {"cpu", "cuda"}:
        raise ValueError("AlphaQ spectral collection supports cpu and cuda devices")
    checkpoint = None
    try:
        if root.is_dir():
            index = root / "model.safetensors.index.json"
            if index.is_file():
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                files = sorted({root / shard for shard in weight_map.values()})
            else:
                files = sorted(root.glob("*.safetensors"))
            files += [
                p
                for p in (root / "config.json", root / "model.safetensors.index.json")
                if p.is_file()
            ]
            inventory, config = None, None
        else:
            from mfq.quantize.mfq_source import FullPrecisionMfqCheckpoint

            checkpoint = FullPrecisionMfqCheckpoint(root)
            files = list(checkpoint.store.paths)
            config = checkpoint.model_config()
            inventory = {
                name: SourceTensorMetadata(
                    name, f"mfq-{info.source_index:05d}", info.shape, info.dtype, info.nbytes
                )
                for name, info in checkpoint.infos.items()
            }
        identity = {
            "model": str(root),
            "files": [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files],
            "method": METHOD,
        }
        if output.exists():
            saved = json.loads(output.read_text(encoding="utf-8"))
            if saved.get("format") != STATISTICS_FORMAT or saved.get("identity") != identity:
                raise ValueError(
                    "AlphaQ cache belongs to a different source or method; choose another --statistics path"
                )
        else:
            saved = {"format": STATISTICS_FORMAT, "identity": identity, "tensors": {}}
        # The bank-name filter below restricts scope to text-model routed
        # experts, including legacy MFQ files without embedded HF config.
        plans = _plan(root, False, None, "f32", source_inventory=inventory, source_config=config)
        banks = [(item, _BANK.match(item.name) or _LEGACY_BANK.match(item.name)) for item in plans]
        banks = [(item, match) for item, match in banks if match is not None]
        if not banks:
            raise ValueError(
                "no supported routed expert banks found; AlphaQ currently allocates routed MoE tensors"
            )
        results, begin = [], time.monotonic()
        for item, match in banks:
            # Quantizer scheme lookup uses the raw source key for an existing
            # bank, and the canonical derived key for separately stored experts.
            name = (
                item.name
                if item.expert_source_names is not None or (item.transform and item.transform.startswith("expert_"))
                else (item.source_name or item.name)
            )
            shape = tuple(item.expert_shape or item.shape)
            if len(shape) != 3:
                raise ValueError(f"AlphaQ requires an expert bank [E,O,I]: {name} {shape}")
            scaled_mfq_experts = (
                checkpoint is not None
                and item.expert_source_names is not None
                and any(
                    scheme is not None
                    for row in (item.expert_source_quantizations or ())
                    for scheme in row
                )
            )
            raw = saved["tensors"].get(name)
            # Older collectors read separate MFQ experts without their FP8
            # scales. Recompute only those banks, preserving unaffected work.
            if raw is not None and (
                not scaled_mfq_experts or raw.get("mfq_expert_scales_applied") is True
            ):
                value = AlphaQTensorStatistics(
                    raw["name"],
                    raw["layer"],
                    raw["projection"],
                    tuple(raw["shape"]),
                    tuple(raw["alpha"]),
                    tuple(raw["variance"]),
                )
                if (
                    value.shape != shape
                    or value.name != name
                    or value.layer != int(match[1])
                    or value.projection != match[2]
                ):
                    raise ValueError(f"AlphaQ cached bank identity differs: {name}")
            else:
                if item.expert_source_names is not None:
                    source = (
                        _MfqGlmExpertRowSource(
                            checkpoint,
                            shape,
                            item.expert_source_names,
                            item.expert_source_quantizations,
                            item.expert_source_scale_names,
                        )
                        if checkpoint is not None
                        else _SeparateExpertRowSource(
                            root,
                            shape,
                            item.expert_source_names,
                            item.expert_source_shards,
                            item.expert_source_quantizations,
                            item.expert_source_scale_names,
                            item.expert_source_scale_shards,
                            item.expert_source_scale_dtypes,
                        )
                    )
                elif checkpoint is not None:
                    source = checkpoint.tensor_source(name)
                    if item.source_quantization is not None:
                        source = _ScaledFp8TensorSlice(
                            source,
                            checkpoint.tensor_source(item.source_scale_name),
                            item.source_quantization,
                        )
                else:
                    source = _raw_source_for_plan(root, item)
                if item.transform and item.transform.startswith("expert_"):
                    source = _PackedExpertProjectionSource(source, item)
                try:
                    alpha, variance = [], []
                    start, batch = 0, shape[0]
                    while start < shape[0]:
                        count = min(batch, shape[0] - start)
                        try:
                            a, v = _collect_batch(source, start, count, shape, device)
                        except torch.OutOfMemoryError:
                            if count == 1:
                                raise
                            batch = max(1, count // 2)
                            continue
                        alpha.extend(a.tolist())
                        variance.extend(v.tolist())
                        start += count
                    value = AlphaQTensorStatistics(
                        name, int(match[1]), match[2], shape, tuple(alpha), tuple(variance)
                    )
                finally:
                    if hasattr(source, "close"):
                        source.close()
                saved["tensors"][name] = asdict(value)
                if scaled_mfq_experts:
                    saved["tensors"][name]["mfq_expert_scales_applied"] = True
                saved["complete"] = False
                _atomic_json(output, saved)
            results.append(value)
            _progress(
                stage="alphaq_statistics",
                completed=len(results),
                total=len(banks),
                tensor=name,
                elapsed_seconds=time.monotonic() - begin,
            )
        saved["complete"] = True
        _atomic_json(output, saved)
        return tuple(results)
    finally:
        if checkpoint is not None:
            checkpoint.close()


def _collect_batch(source, start, count, shape, device):
    weight = source.read_rows(start * shape[1], (start + count) * shape[1], device=device)
    return alphaq_weight_statistics(weight.reshape(count, shape[1], shape[2]))


__all__ = ["collect_alphaq"]
