"""Check or time actual GR weights at continuous-batch row counts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("check", "bench"))
parser.add_argument("--model", type=Path)
parser.add_argument("--rows", type=int, choices=(9,16,17,32))
args = parser.parse_args()
config_path = ROOT / "workspace/local-server/decode-matched-config.json"
config = json.loads(config_path.read_text()) if config_path.exists() else {}
if args.model is None and "command" not in config:
    parser.error("--model is required without the local benchmark config")
model = args.model.resolve() if args.model else ROOT / config["command"][2]
environment = dict(os.environ, **config.get("environment", {}))
environment.update(TEMP=str(ROOT / "workspace/tmp"), TMP=str(ROOT / "workspace/tmp"))
log = ROOT / f"workspace/local-server/gr-batch-real-{args.mode}.log"
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
