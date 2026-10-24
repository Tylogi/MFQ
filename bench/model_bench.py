"""Model-level benchmark driving the native runtime over the stdio transport.

Runs each model against synthetic prompts of an exact token count and
records prefill throughput, decode throughput, and time to first token.
Works for any backend binary (HIP, CUDA, Metal); GPU selection and runtime
knobs pass through to the child environment.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfq.evaluation.inference import BenchmarkValidationError, validate_benchmark_result

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
        self.lines = queue.Queue()
        self.closed = False
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
            start_new_session=os.name == "posix",
        )
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._drain, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()
        try:
            greeting = self.read_message(120)
            if greeting.get("type") != "ready":
                raise RuntimeError(f"runtime did not greet: {greeting!r}")
        except BaseException:
            self.close(force=True)
            raise

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def _drain(self):
        for line in self.proc.stderr:
            self.tail.append(line.rstrip()[:400])
            del self.tail[:-50]

    def read_message(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("runtime response timed out")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("runtime response timed out") from error
            if line is None:
                raise RuntimeError("runtime closed stdout before completing the request")
            if line.strip():
                try:
                    message = json.loads(line)
                except ValueError as error:
                    raise RuntimeError("runtime returned invalid JSON") from error
                if not isinstance(message, dict):
                    raise RuntimeError("runtime returned a non-object frame")
                return message

    def request(self, name, payload, timeout):
        if self.closed:
            return {"status": "error", "detail": "runtime is closed"}
        deadline = time.monotonic() + timeout
        try:
            self.proc.stdin.write(json.dumps({"v": 1, "id": name, **payload}) + "\n")
            self.proc.stdin.flush()
            result = None
            while True:
                message = self.read_message(deadline - time.monotonic())
                if message.get("id") != name:
                    continue
                if message.get("type") == "error":
                    raise RuntimeError(str(message)[:400])
                data = message.get("data", message)
                if isinstance(data, dict) and "metrics" in data:
                    result = data
                if message.get("type") == "done":
                    if result is None:
                        raise RuntimeError("runtime completed without benchmark metrics")
                    return {"status": "ok", "result": result}
        except (OSError, RuntimeError) as error:
            self.close(force=True)
            return {"status": "timeout" if isinstance(error, TimeoutError) else "error",
                    "detail": str(error), "tail": self.tail[-5:]}

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
        result = self.request(name, {"op": "generate", "params": params}, timeout)
        if result["status"] != "ok":
            return result
        try:
            run = validate_benchmark_result(result["result"], prompt_tokens, decode_tokens, "mtp" if mtp else "decode")
        except BenchmarkValidationError as error:
            self.close(force=True)
            return {"status": "error", "detail": str(error)}
        return {"status": "ok", "metrics": {**run["metrics"],
                "prompt_tokens": run["prompt_tokens"], "completion_tokens": run["completion_tokens"]}}

    def _kill(self):
        with contextlib.suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(self.proc.pid, signal.SIGKILL)
            else:
                self.proc.kill()

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc.poll() is None:
                if force:
                    self._kill()
                else:
                    self.proc.stdin.write(json.dumps({"v": 1, "op": "shutdown"}) + "\n")
                    self.proc.stdin.flush()
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self._kill()
            self.proc.wait()
        finally:
            if os.name == "posix":
                self._kill()
            for thread, stream in ((self.stdout_thread, self.proc.stdout),
                                   (self.stderr_thread, self.proc.stderr)):
                thread.join(timeout=1)
                if not thread.is_alive():
                    stream.close()


def run_model(runtime_path, model, args, env):
    command = [str(runtime_path), "--model", str(model), "--transport", "stdio"]
    if args.ctx_size:
        command += ["--ctx-size", str(args.ctx_size)]
    try:
        runtime = RuntimeProcess(command, env, Path.cwd())
    except (RuntimeError, OSError) as error:
        return {"model": Path(model).name, "runs": [], "error": str(error)[:400]}
    runs = []
    try:
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
                        return {"model": Path(model).name, "runs": runs, "error": entry["detail"]}
    finally:
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
    parser.add_argument(
        "--report-root", type=Path, help="archive results and generate charts after the benchmark"
    )
    parser.add_argument(
        "--report-metadata",
        type=Path,
        help="public model, hardware and build labels for the report",
    )
    args = parser.parse_args()
    try:
        args.prefill_lengths = [int(value) for value in args.prefill_lengths.split(",")]
    except ValueError:
        parser.error("--prefill-lengths requires comma-separated integers")
    if not args.prefill_lengths or any(value < 1 for value in args.prefill_lengths) or len(set(args.prefill_lengths)) != len(args.prefill_lengths):
        parser.error("--prefill-lengths requires distinct positive integers")
    if args.decode_tokens < 2 or args.repetitions < 1 or args.warmup_runs < 0:
        parser.error("decode tokens must be at least 2, repetitions positive, and warmups nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.ctx_size is not None and max(args.prefill_lengths) + args.decode_tokens > args.ctx_size:
        parser.error("prompt and output tokens exceed --ctx-size")

    root = Path(__file__).resolve().parents[1]
    runtime_path = args.runtime or default_runtime(root)
    env = {**os.environ}
    for override in args.env:
        key, sep, value = override.partition("=")
        if not sep:
            raise SystemExit(f"--env expects K=V, got {override!r}")
        env[key] = value

    report_metadata = {}
    if args.report_metadata and not args.report_root:
        parser.error("--report-metadata requires --report-root")
    if args.report_root:
        if len(args.models) != 1:
            parser.error("--report-root supports one model per run")
        from bench import report_workflow

        try:
            report_metadata = (
                report_workflow.report.read_json(args.report_metadata)
                if args.report_metadata
                else {}
            )
            protocol = dict(
                input_tokens=args.prefill_lengths,
                output_tokens=args.decode_tokens,
                repetitions=args.repetitions,
                warmup_runs=args.warmup_runs,
                temperature=0,
                seed=1,
                batch=1,
                prefix_cache=False,
                cooldown_seconds=0,
            )
            if args.ctx_size is not None:
                protocol["context_tokens"] = args.ctx_size
            if env.get("MFQ_DISABLE_MMA_ATTENTION") == "1":
                protocol["mma_attention"] = False
            report_workflow.preflight(report_metadata, protocol)
            declared = report_metadata.get("protocol", {})
            for key, value in protocol.items():
                if key in declared and declared[key] != value:
                    raise ValueError(f"Metadata conflicts with effective {key}")
            report_metadata["protocol"] = {**declared, **protocol}
            digest = hashlib.sha256()
            with runtime_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            runtime_hash = digest.hexdigest()
            if report_metadata.get("runtime_sha256", runtime_hash) != runtime_hash:
                raise ValueError("Metadata conflicts with runtime binary hash")
            report_metadata["runtime_sha256"] = runtime_hash
        except (ValueError, OSError, ImportError) as error:
            parser.error(f"Report preflight failed: {error}")

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
        if model["error"] or any(
            sum(run["status"] == "ok" and not run["warmup"]
                and run["prompt_tokens_requested"] == length and run["mode"] == mode
                for run in model["runs"]) != args.repetitions
            for length in args.prefill_lengths
            for mode in (("decode", "mtp") if args.mtp else ("decode",))
        )
    ]
    if failed:
        print(f"failed models: {failed}", file=sys.stderr)
    if args.report_root:
        try:
            output = report_workflow.archive_report(
                document,
                report_metadata,
                args.report_root,
                modes=("decode", "mtp") if args.mtp else ("decode",),
            )
            print(f"Report ready: {output / 'index.html'}")
        except Exception as error:
            print(
                f"Report failed ({type(error).__name__}); details remain in the private archive.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
