"""Aggregate native CUDA phase events by the stdio benchmark's window IDs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "workspace/local-server/continuous-throughput"


def fields(line):
    return dict(word.split("=", 1) for word in line.split()[1:] if "=" in word)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    log_path = args.log or RESULTS / f"{args.prefix}-worker.log"
    if not args.log and not log_path.exists():
        log_path = RESULTS / "worker.log"
    windows = {}
    current = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("native_cuda_window "):
            values = fields(line)
            current = (int(values["capture_ns"]), int(values["replay_ns"]))
            windows[current] = []
        elif line.startswith("native_layer_gpu ") and current:
            windows[current].append(fields(line))
    report = []
    for path in sorted(RESULTS.glob(f"{args.prefix}-*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "case" not in result:
            continue
        totals = defaultdict(int)
        replays = 0
        legacy_tiles = False
        for window in result.get("windows", []):
            key = (window["capture_ns"], window["replay_ns"])
            replays += window["replays"]
            for phase in windows.get(key, []):
                legacy_tiles |= "tile" not in phase
                totals[(phase["from"], phase["to"])] += int(phase["total_ns"])
        pairs = [{"from": pair[0], "to": pair[1], "ms_per_replay": value / replays / 1e6}
                 for pair, value in sorted(totals.items(), key=lambda pair: -pair[1])] if replays else []
        record = {"case": result["case"], "errors": result.get("errors", []),
                  "replays": replays, "phase_ms_per_replay": pairs}
        if legacy_tiles and result["case"]["requests"] > 8:
            record["ffn_phase_note"] = "Repeated FFN tile markers currently time the last tile; earlier tiles enter the preceding phase."
        report.append(record)
        print(json.dumps({**record, "phase_ms_per_replay": pairs[:16]}, ensure_ascii=False))
    (RESULTS / f"{args.prefix}-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
