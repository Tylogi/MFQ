"""Generate small compressed banks for --mfe-expert-plan-check/--mfe-prepare-bench DIR."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from mfq.formats import io
from mfq.formats.header import FileHeader
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.mx import MxTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from tests.mixed_family_fixtures import make_flat_family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    experts = 768
    for family in ("nint", "nvq", "mxfp4"):
        for down in (False, True):
            rows, width = (384, 96) if down else (96, 384)
            rng = np.random.default_rng(1100 + int(down))
            shape = (experts * rows, width)
            if family == "nint":
                values = rng.normal(0, 0.015, shape).astype(np.float32)
                tensor = quantize(values, NintSpec(4, 24, 6), axis=0)
                del values
            elif family == "nvq":
                tensor = make_flat_family("NVQ2J-L", rows=shape[0], neuron_len=width)
            else:
                tensor = MxTensor(
                    "MXFP4", shape,
                    rng.integers(0, 256, (shape[0], width // 2), dtype=np.uint8),
                    rng.integers(116, 120, (shape[0], width // 32), dtype=np.uint8),
                )
            bank = MfeTensor((experts, rows, width), (MfePool(np.arange(experts, dtype=np.int32), tensor),))
            path = args.directory / f"plan-{family}-{'down' if down else 'up'}.mfq"
            io.save(path, FileHeader(version=2, model_arch="mfe-plan-test"), {"linear.weight": bank})
            print(path)


if __name__ == "__main__":
    main()
