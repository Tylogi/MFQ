"""Compare graph timings and native output fixtures from the format runners."""
import argparse
import json
from pathlib import Path

import numpy as np


def timings(path):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")
    rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    return {(row["case"], row["M"]): row for row in rows}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("llama")
    parser.add_argument("--baseline-outputs")
    parser.add_argument("--candidate-outputs")
    parser.add_argument("--output")
    args = parser.parse_args()
    old, new, llama = map(timings, (args.baseline, args.candidate, args.llama))
    result = []
    for key, row in new.items():
        before, reference = old[key], llama[key]
        summary = {"case": key[0], "M": key[1], "baseline_us": before["us"],
                   "candidate_us": row["us"], "llama_us": reference["us"],
                   "speedup": before["us"] / row["us"], "latency_ratio": row["us"] / reference["us"],
                   "effective_GBs": row["effective_GBs"], "llama_GBs": reference["effective_GBs"]}
        if args.baseline_outputs and args.candidate_outputs:
            name = f"{key[0]}.y{key[1]}.f32"
            a = np.fromfile(Path(args.baseline_outputs) / name, dtype=np.float32).astype(np.float64)
            b = np.fromfile(Path(args.candidate_outputs) / name, dtype=np.float32).astype(np.float64)
            if a.shape != b.shape or not np.all(np.isfinite(b)):
                raise ValueError(f"Invalid output {name}")
            summary.update(changed=int(np.count_nonzero(a != b)), elements=a.size,
                           max_abs=float(np.max(np.abs(a - b))),
                           relative_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)))
        result.append(summary)
        print(f"{key[0]:32} M={key[1]:2} {before['us']:6.2f} -> {row['us']:6.2f} us; "
              f"llama {reference['us']:6.2f}; ratio {summary['latency_ratio']:.2f}" +
              (f"; changed {summary['changed']}; relL2 {summary['relative_l2']:.3g}" if "changed" in summary else ""))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
