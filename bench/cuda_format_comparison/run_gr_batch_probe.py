"""Check or time actual GR weights at continuous-batch row counts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("down-direct-s16-sweep", "norm-inline-down-sweep", "norm-register-quant-sweep", "norm-fixed-types-sweep", "norm-many-slices-sweep", "down-mma-fused-up-sweep", "up-four-outputs-sweep", "up-half-fused-prepare-sweep", "up-vector-weights-sweep", "norm-slices-sweep", "up-int16-mma-sweep", "norm-fast-rsqrt-sweep", "up-fast-sigmoid-sweep", "down-split-sweep", "norm-stats-sweep", "norm-threads-sweep", "up-direct-stats-sweep", "fused-unsigned-sweep", "up-fused-split-sweep", "up-split-major-sweep", "up-parts-vector-sweep", "up-fused-direct-sweep", "up-half-unroll-sweep", "up-half-register-sweep", "up-direct-s16-sweep", "check", "bench", "up-capacity-sweep", "down-capacity-sweep", "up-thread-sweep", "group-read-sweep", "up-metadata-sweep", "up-unroll-sweep", "norm-cache-sweep", "norm-quant-sweep", "norm-wide-sweep", "norm-group-quant-sweep", "unsigned-dot-sweep", "two-plane-sweep", "three-plane-sweep", "integer16-sweep", "up-integer-sweep", "integer-best-sweep", "up-mix-sweep", "int-pair-sweep", "up-prepare-sweep", "up-uniform-prepare-sweep", "up-groups-sweep", "up-mma-sweep", "up-mma-uniform16-sweep", "up-mma-f16-sweep", "up-mma-f16-m1-sweep", "up-coefficient-sweep", "down-mma-sweep", "combined-affine-sweep", "packed24-sweep", "packed24-pair-sweep", "chain-post-sweep"))
parser.add_argument("--model", type=Path)
parser.add_argument("--role", choices=("attention","mlp"))
parser.add_argument("--rows", type=int, choices=(1,2,3,4,5,6,8,9,16,17,32))
parser.add_argument("--numerical-only", action="store_true")
parser.add_argument("--branch-float", action="store_true")
args = parser.parse_args()
config_path = ROOT / "workspace/local-server/decode-matched-config.json"
config = json.loads(config_path.read_text()) if config_path.exists() else {}
if args.model is None and "command" not in config:
    parser.error("--model is required without the local benchmark config")
model = args.model.resolve() if args.model else ROOT / config["command"][2]
environment = dict(os.environ, **config.get("environment", {}))
environment.update(TEMP=str(ROOT / "workspace/tmp"), TMP=str(ROOT / "workspace/tmp"))
suffix = f"-m{args.rows}" if args.rows else ""
if args.mode=='down-split-sweep' and environment.get('MFQ_GR_DOWN_SPLIT_PROBE_TILE'):
    suffix+='-tile'+environment['MFQ_GR_DOWN_SPLIT_PROBE_TILE']
if args.mode=='up-groups-sweep' and environment.get('MFQ_GR_UP_PROBE_MODE'):
    suffix+='-mode'+environment['MFQ_GR_UP_PROBE_MODE']
    probe_threads=environment.get('MFQ_GR_UP_PROBE_THREADS') or environment.get('MFQ_GR_UP_GROUP_THREADS')
    if probe_threads:
        suffix+='-threads'+probe_threads
    if environment.get('MFQ_GR_UP_PROBE_REFERENCE'):
        suffix+='-ref'+environment['MFQ_GR_UP_PROBE_REFERENCE']
if args.mode=='up-mma-sweep' and environment.get('MFQ_GR_UP_MMA_PROBE_MODE'):
    suffix+='-mode'+environment['MFQ_GR_UP_MMA_PROBE_MODE']
if args.mode=='norm-quant-sweep' and environment.get('MFQ_GR_NORM_QUANT_PROBE_VARIANT'):
    suffix+='-'+environment['MFQ_GR_NORM_QUANT_PROBE_VARIANT']
if args.mode=='norm-group-quant-sweep' and environment.get('MFQ_GR_NORM_GROUP_PROBE_MODE'):
    suffix+='-mode'+environment['MFQ_GR_NORM_GROUP_PROBE_MODE']
if args.mode=='down-mma-sweep' and environment.get('MFQ_GR_DOWN_MMA_SPLITS'):
    suffix+='-splits'+environment['MFQ_GR_DOWN_MMA_SPLITS']
if args.role:
    environment["MFQ_GR_PROBE_ROLE"] = args.role
    suffix += "-" + args.role
if args.numerical_only:
    environment["MFQ_GR_PROBE_NO_TIMING"] = "1"
    suffix += "-check"
if args.branch_float:
    environment["MFQ_GR_CHAIN_BRANCH_FLOAT"] = "1"
    suffix += "-branch-float"
log = ROOT / f"workspace/local-server/gr-batch-real-{args.mode}{suffix}.log"
log.parent.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
with log.open("wb") as output:
    command = [str(ROOT / "build/cuda-parity/cuda/mfq-cuda-gr-batch-probe.exe"), str(model), "--" + args.mode]
    if args.rows:
        command.extend(["--rows", str(args.rows)])
    result = subprocess.run(command, cwd=ROOT, env=environment,
                            stdout=output, stderr=subprocess.STDOUT, timeout=180)
print(json.dumps({"mode": args.mode, "seconds": time.monotonic() - started,
                  "exit_code": result.returncode}))
print("\n".join(log.read_text(errors="replace").splitlines()[-12:]))
raise SystemExit(result.returncode)
