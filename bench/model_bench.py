"""Model-level benchmark driving the native runtime over the stdio transport.

Runs each model against synthetic prompts of an exact token count and
records prefill throughput, decode throughput, and time to first token.
Works for any backend binary (HIP, CUDA, Metal); GPU selection and runtime
knobs pass through to the child environment.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

PROMPT_TEXT = (
    "The kernel scheduler divides continuous decode batches into tiles so "
    "that recurring experts stay resident between steps. Explain the memory "
    "and compute tradeoffs of this design for quantized inference."
)
METRIC_KEYS = (
    "prompt_tokens",
    "prefill_tokens",
    "completion_tokens",
    "ttft_ms",
    "prefill_ms",
    "prefill_tps",
    "decode_ms",
    "decode_tps",
    "generation_tps",
    "mtp_used",
    "mtp_drafted_tokens",
    "mtp_accepted_tokens",
)


def default_runtime(root):
    for candidate in ("hip", "cuda", "cpp_runtime"):
        path = root / "build" / candidate / "mfq-runtime"
        if path.exists():
            return path
    raise SystemExit("no mfq-runtime found under build/; pass --runtime")


def precision_label(path):
    stem = Path(path).stem.lower()
    if "s4m" in stem or "s4-m" in stem:
        return "S4-M"
    for tag in ("bf16", "f16", "fp16"):
        if tag in stem:
            return "BF16"
    for tag in ("w4", "nint4", "mxfp4"):
        if tag in stem:
            return tag.upper()
    return "native"


class RuntimeProcess:
    def __init__(self, command, env, cwd):
        self.tail = []
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
        )
        threading.Thread(target=self._drain, daemon=True).start()
        greeting = self.read_message(120)
        if not greeting or greeting.get("type") != "ready":
            raise RuntimeError(f"runtime did not greet: {greeting!r}")

    def _drain(self):
        for line in self.proc.stderr:
            self.tail.append(line.rstrip()[:400])
            del self.tail[:-50]

    def read_message(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            line = self.stdout_line(remaining)
            if line is None:
                return None
            if line.strip():
                return json.loads(line)

    def stdout_line(self, timeout):
        result = []

        def reader():
            result.append(self.proc.stdout.readline())

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout)
        return result[0] if result and result[0] else None

    def request(self, name, payload, timeout):
        self.proc.stdin.write(json.dumps({"v": 1, "id": name, **payload}) + "\n")
        self.proc.stdin.flush()
        metrics = {}
        while True:
            message = self.read_message(timeout)
            if message is None:
                return {"status": "timeout", "tail": self.tail[-5:]}
            if message.get("type") == "error":
                return {"status": "error", "detail": str(message)[:400]}
            if message.get("type") == "result":
                metrics = message.get("metrics") or {}
            data = message.get("data") or {}
            if isinstance(data, dict) and "metrics" in data:
                usage = data.get("usage") or {}
                metrics = {
                    **data["metrics"],
                    **{
                        key: usage[key]
                        for key in ("completion_tokens", "prompt_tokens")
                        if key in usage
                    },
                }
            if message.get("type") == "done":
                return {"status": "ok", "metrics": metrics}

    def generate(self, name, prompt_tokens, decode_tokens, mtp, timeout):
        params = {
            "model": "bench",
            "input": {
                "messages": [{"role": "user", "content": PROMPT_TEXT}],
                "preformatted_prompt": PROMPT_TEXT,
                "benchmark_prompt_tokens": prompt_tokens,
            },
            "cache": {"enabled": False},
            "sampling": {
                "temperature": 0.0,
                "max_new_tokens": decode_tokens,
                "seed": 1,
                "enable_mtp": bool(mtp),
            },
            "stream": False,
        }
        return self.request(name, {"op": "generate", "params": params}, timeout)

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"stop": True}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


def run_model(runtime_path, model, args, env):
    command = [str(runtime_path), "--model", str(model), "--transport", "stdio"]
    if args.ctx_size:
        command += ["--ctx-size", str(args.ctx_size)]
    try:
        runtime = RuntimeProcess(command, env, Path.cwd())
    except (RuntimeError, OSError) as error:
        return {"model": Path(model).name, "runs": [], "error": str(error)[:400]}
    runs = []
    for length in args.prefill_lengths:
        for mode in ("decode", "mtp") if args.mtp else ("decode",):
            mtp = mode == "mtp"
            for index in range(args.warmup_runs + args.repetitions):
                name = f"r-{length}-{mode}-{index}"
                result = runtime.generate(name, length, args.decode_tokens, mtp, args.timeout)
                entry = {
                    "prompt_tokens_requested": length,
                    "mode": mode,
                    "warmup": index < args.warmup_runs,
                    "status": result["status"],
                }
                entry.update({key: result.get("metrics", {}).get(key) for key in METRIC_KEYS})
                runs.append(entry)
                if result["status"] != "ok":
                    entry["detail"] = result.get("detail") or " | ".join(result.get("tail", []))
                    break
    runtime.close()
    return {"model": Path(model).name, "runs": runs, "error": None}


def summarize(report):
    rows = []
    for model in report:
        cells = {}
        for run in model["runs"]:
            if run["status"] != "ok" or run["warmup"]:
                continue
            cells.setdefault((run["prompt_tokens_requested"], run["mode"]), []).append(run)
        for (length, mode), samples in sorted(cells.items()):

            def median(key, samples=samples):
                values = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
                return statistics.median(values) if values else None

            rows.append(
                {
                    "model": Path(model["model"]).name,
                    "precision": precision_label(model["model"]),
                    "prompt_tokens": length,
                    "mode": mode,
                    "samples": len(samples),
                    "decode_tps": median("decode_tps"),
                    "prefill_tps": median("prefill_tps"),
                    "ttft_ms": median("ttft_ms"),
                    "completion_tokens": median("completion_tokens"),
                    "mtp_drafted": median("mtp_drafted_tokens"),
                    "mtp_accepted": median("mtp_accepted_tokens"),
                }
            )
    return rows


def print_markdown(rows, args):
    print("| Model | Precision | Mode | Prompt | Decode tok/s | Prefill tok/s | TTFT ms |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:

        def fmt(value, digits=1):
            return "-" if value is None else f"{value:.{digits}f}"

        print(
            f"| {row['model']} | {row['precision']} | {row['mode']} "
            f"| {row['prompt_tokens']} | {fmt(row['decode_tps'])} "
            f"| {fmt(row['prefill_tps'])} | {fmt(row['ttft_ms'], 0)} |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="model files to benchmark")
    parser.add_argument("--runtime", type=Path, help="mfq-runtime binary")
    parser.add_argument(
        "--prefill-lengths",
        default="32,512,2048,8192",
        help="comma-separated synthetic prompt token counts",
    )
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--ctx-size", type=int)
    parser.add_argument("--mtp", action="store_true", help="also measure speculative MTP decode")
    parser.add_argument("--timeout", type=float, default=900, help="per-request timeout in seconds")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="K=V",
        help="environment override for the runtime",
    )
    parser.add_argument("--output", type=Path, help="write raw JSON here")
    args = parser.parse_args()
    args.prefill_lengths = [int(value) for value in args.prefill_lengths.split(",")]

    root = Path(__file__).resolve().parents[1]
    runtime_path = args.runtime or default_runtime(root)
    env = {**os.environ}
    for override in args.env:
        key, sep, value = override.partition("=")
        if not sep:
            raise SystemExit(f"--env expects K=V, got {override!r}")
        env[key] = value

    report = []
    for model in args.models:
        print(f"# benchmarking {model}", file=sys.stderr, flush=True)
        report.append(run_model(runtime_path, model, args, env))

    document = {
        "runtime": runtime_path.name,
        "prefill_lengths": args.prefill_lengths,
        "decode_tokens": args.decode_tokens,
        "repetitions": args.repetitions,
        "warmup_runs": args.warmup_runs,
        "env": dict(item.split("=", 1) for item in args.env),
        "results": report,
        "summary": summarize(report),
    }
    if args.output:
        args.output.write_text(json.dumps(document, indent=2) + "\n")
    print_markdown(document["summary"], args)
    failed = [
        model["model"]
        for model in report
        if not any(run["status"] == "ok" for run in model["runs"])
    ]
    if failed:
        print(f"failed models: {failed}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
