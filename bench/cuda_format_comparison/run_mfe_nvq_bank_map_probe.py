"""Run real-model cached JSC mapping checks or a same-process FFN comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("check", "bench", "capacity-check", "capacity-bench"))
parser.add_argument("--model", type=Path)
args = parser.parse_args()
mode = args.mode
config_path = ROOT / "workspace/local-server/decode-matched-config.json"
config = json.loads(config_path.read_text()) if config_path.exists() else {}
environment = dict(os.environ, **config.get("environment", {}))
environment.update(TEMP=str(ROOT / "workspace/tmp"), TMP=str(ROOT / "workspace/tmp"))
for name in ("MFQ_NVQ2_EXEC", "MFQ_NVQ_EXTENDED_GROUP_EXEC", "MFQ_NVQ1_GROUP_RECORDS",
             "MFQ_NVQ1_INTEGER_DELTA", "MFQ_MFE_SHARED_DENSE_MATH", "MFQ_MFE_NINT_LATE_SCALE",
             "MFQ_MFE_CANONICAL_QUAD", "MFQ_NINT_ROUTE_METADATA", "MFQ_MFE_DEFAULT_MATH",
             "MFQ_MOE_NVQ_DENSE", "MFQ_MFE_NVQ_DENSE_NARROW", "MFQ_MFE_E8_NARROW"):
    environment[name] = "0"
environment.update(MFQ_MFE_GROUP_DOT="1", MFQ_BENCH_NVQ_DENSE="1",
                   MFQ_MFE_GPU_BUNDLE="1", MFQ_MFE_PARALLEL_GU="1",
                   MFQ_MFE_SHARED_FIRST="1", MFQ_MFE_GU_WARPS="8",
                   MFQ_MFE_FORMAT_SPECIALIZE="1", MFQ_MFE_NINT_WHOLE="1")
executable = ROOT / "build/cuda-parity/cuda/mfq-cuda-mfe-nvq-bank-map-probe.exe"
if args.model is None and "command" not in config:
    parser.error("--model is required without workspace/local-server/decode-matched-config.json")
model = args.model.resolve() if args.model is not None else ROOT / config["command"][2]
log = ROOT / f"workspace/local-server/nvq-bank-map-real-mfe-{mode}.log"
log.parent.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
with log.open("wb") as output:
    result = subprocess.run([str(executable), str(model), "--" + mode],
                            cwd=ROOT, env=environment, stdout=output,
                            stderr=subprocess.STDOUT, timeout=120)
print(json.dumps({"mode": mode, "seconds": time.monotonic() - started,
                  "exit_code": result.returncode}))
print("\n".join(log.read_text(errors="replace").splitlines()[-7:]))
raise SystemExit(result.returncode)
