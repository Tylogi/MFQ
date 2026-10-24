"""Compare recurrent and chunk GDN with a generated single-layer model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mfq.formats import io
from mfq.formats.assets import MODEL_CONFIG_ASSET, model_config_asset


def fixture(directory: Path) -> Path:
    """Use the native PLE/MoE fixture with a D128, 16/48-head GDN layer."""
    source = ROOT / "tests" / "test_native_nint_ple.py"
    spec = importlib.util.spec_from_file_location("gdn_fixture", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture generator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, original = module._models(directory, context_size=65536)
    header, tensors = io.load(original)
    config = json.loads(bytes(tensors[MODEL_CONFIG_ASSET]))
    config["text_config"]["linear_num_key_heads"] = 16
    config["text_config"]["linear_num_value_heads"] = 48
    asset = model_config_asset(config)
    tensors[asset.name] = asset.data
    rng = np.random.default_rng(20261024)
    root = "model.block.0.linear_attention"
    channels = (2 * 16 + 48) * 128
    for role, shape in {
        "qkv.weight": (channels, 128),
        "gate.weight": (48 * 128, 128),
        "alpha.weight": (48, 128),
        "beta.weight": (48, 128),
        "output.weight": (128, 48 * 128),
        "conv.weight": (channels, 1, 4),
    }.items():
        tensors[f"{root}.{role}"] = rng.normal(0, 0.03, shape).astype(np.float16)
    tensors[f"{root}.dt_bias"] = np.zeros(48, dtype=np.float16)
    tensors[f"{root}.a"] = np.zeros(48, dtype=np.float16)
    model = directory / "gdn.mfq"
    io.save(model, header, tensors)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="native mfq-diagnostics")
    parser.add_argument("--gpu-lock", type=Path, required=True, help="shared GPU flock file")
    parser.add_argument("--repeats", type=int, default=16, help="even count, including four warmups")
    args = parser.parse_args()
    if args.repeats < 8 or args.repeats % 2:
        parser.error("--repeats must be even and at least eight")
    variants = [
        {"name": "recurrent", "environment": {"MFQ_GDN_PREFILL_CHUNK": "0"}},
        {"name": "chunk", "environment": {"MFQ_GDN_PREFILL_CHUNK": "1"}},
    ]
    environment = {
        **os.environ,
        "MFQ_PREFILL_SWEEP_CHUNKS": "65536",
        "MFQ_PREFILL_VARIANTS_JSON": json.dumps(variants),
    }
    with tempfile.TemporaryDirectory(prefix="mfq-gdn-chunk-") as temporary:
        model = fixture(Path(temporary))
        try:
            completed = subprocess.run(
                [
                    "flock", str(args.gpu_lock), str(args.binary.resolve()), "--model", str(model),
                    "--ctx-size", "65536", "--prefill-sweep", "1024,4096,16384,65536",
                    "--prefill-sweep-reps", str(args.repeats),
                ],
                env=environment, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as error:
            print(error.stdout + error.stderr, file=sys.stderr)
            raise
    rows: dict[int, list[list[float]]] = {}
    pattern = r"prefill_chunk_sample_m=(\d+) chunk=\d+ repeat=(\d+) elapsed_ms=([0-9.eE+-]+)"
    for match in re.finditer(pattern, completed.stdout):
        tokens, trial, elapsed = int(match[1]), int(match[2]), float(match[3])
        if trial >= 4:
            rows.setdefault(tokens, [[], []])[trial % 2].append(elapsed)
    if set(rows) != {1024, 4096, 16384, 65536}:
        raise RuntimeError("diagnostics omitted a requested prompt length")
    for tokens, (baseline, candidate) in rows.items():
        expected = (args.repeats - 4) // 2
        if len(baseline) != expected or len(candidate) != expected:
            raise RuntimeError(f"incomplete paired trials for {tokens} tokens")
        before, after = statistics.median(baseline), statistics.median(candidate)
        print(json.dumps({
            "tokens": tokens, "baseline_ms": before, "chunk_ms": after, "speedup": before / after,
            "paired_trials": len(baseline), "baseline_samples_ms": baseline,
            "chunk_samples_ms": candidate, "candidate_forced": True,
        }))


if __name__ == "__main__":
    main()
