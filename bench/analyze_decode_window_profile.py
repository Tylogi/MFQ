"""Summarize measured CUDA window phases, preserving the window batch size."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def analyze(path, tasks=None):
    windows = []
    current = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("native_cuda_window ", "native_layer_gpu ")):
            continue
        record = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if line.startswith("native_cuda_window "):
            current = {"window": {key: int(value) for key, value in record.items()}, "phases": []}
            windows.append(current)
        elif current is not None:
            current["phases"].append(record)
    windows = [window for window in windows if window["phases"]]
    if not windows:
        raise ValueError("no measured CUDA window phases")
    selected_tasks = max(window["window"]["tasks"] for window in windows) if tasks is None else tasks
    selected = [window for window in windows if window["window"]["tasks"] == selected_tasks]
    if not selected:
        raise ValueError(f"no profiled window with tasks={selected_tasks}")
    totals = defaultdict(lambda: {"total_ns": 0, "events": 0})
    measured_ns = 0
    for window in selected:
        for phase in window["phases"]:
            boundary = f'{phase["from"]} -> {phase["to"]}'
            if phase["from_layer"] != phase["layer"]:
                boundary += " (layer boundary)"
            if phase.get("from_tile") != phase.get("tile"):
                boundary += " (tile boundary)"
            totals[boundary]["total_ns"] += int(phase["total_ns"])
            totals[boundary]["events"] += int(phase["samples"])
            measured_ns += int(phase["total_ns"])
    replays = sum(window["window"]["replays"] for window in selected)
    phases = [{"boundary": boundary, **values,
               "ms_per_replay": values["total_ns"] / replays / 1e6,
               "percent": values["total_ns"] * 100 / measured_ns if measured_ns else 0}
              for boundary, values in totals.items()]
    phases.sort(key=lambda phase: phase["total_ns"], reverse=True)
    return {"tasks": selected_tasks, "profiled_windows": len(selected), "replays": replays,
            "gpu_ms_per_replay": measured_ns / replays / 1e6,
            "wall_ms_per_replay": sum(window["window"]["replay_ns"] for window in selected) / replays / 1e6,
            "phases": phases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.log, args.tasks)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "phases"}))
    for phase in result["phases"]:
        print(f'{phase["ms_per_replay"]:9.3f} ms {phase["percent"]:6.2f}% {phase["boundary"]}')
