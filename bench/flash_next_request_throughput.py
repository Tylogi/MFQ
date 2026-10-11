"""Measure the native stdio request path without repeated model loads.

All assets, temporary files and results remain inside the repository. This is
an interactive benchmark worker: each input line is a JSON benchmark command.
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from mfq.server.runtime.client import StdioRuntimeClient


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "workspace/local-server/continuous-throughput"


async def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "workspace/local-server/decode-matched-config.json").read_text())
    model = str((ROOT / config["command"][2]).resolve())
    tokenizers = list((ROOT / "workspace/local-server/tokenizer-cache").glob(
        "Qwen3.8-Flash-Next-EWQ-MFQ-S2-L-00001-of-00005-*.tokenizer.gguf"))
    assert len(tokenizers) == 1, tokenizers
    executable = ROOT / "build/cuda-parity/mfq-runtime.exe"
    environment = dict(os.environ, **config["environment"])
    environment["MFQ_TRACE_LAYER_TIMINGS"] = os.environ.get("MFQ_TRACE_LAYER_TIMINGS", "0")
    environment["MFQ_MOE_HOST_CACHE_BYTES"] = "42949672960"
    environment["MFQ_TRACE_NATIVE_CUDA_GRAPH"] = "1"
    command = [str(executable), "--model", model, "--tokenizer", str(tokenizers[0]),
               "--transport", "stdio", "--ctx-size", "4096", "--continuous-batching", "32",
               "--moe-gpu-cache-gb", "10.66", "--prefill-chunk-size", "2048"]
    with (RESULTS / "worker.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=environment,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log)
        client = StdioRuntimeClient(process, control_timeout_seconds=180)
        try:
            status = await client.status()
            print(json.dumps({"ready": True, "pid": process.pid,
                              "free_bytes": status.get("device_free_bytes")}), flush=True)
            reference = json.loads((ROOT / "workspace/local-server/prefill-restored-service-mtp-warm.json")
                                   .read_text(encoding="utf-8"))["content"]
            while line := await asyncio.to_thread(sys.stdin.readline):
                case = json.loads(line)
                if case.get("stop"):
                    break
                requests, tokens = case.get("requests", 1), case.get("tokens", 256)
                assert 1 <= requests <= 32 and 1 <= tokens <= 2048
                started = time.perf_counter()
                before = await client.status()
                log_start = (RESULTS / "worker.log").stat().st_size

                async def generate(row):
                    prompt = "请从一开始连续列出数字，用逗号分隔，至少列出1000个数字。"
                    if case.get("heterogeneous"):
                        prompt = f"本次请求编号{row}。" + "请按要求回答。" * (row % 7) + prompt
                    payload = {"messages": [{"role": "user", "content": prompt}],
                        "max_tokens": tokens, "temperature": 0, "top_k": 1,
                        "enable_mtp": bool(case.get("mtp")), "mtp_max_draft_tokens": 5,
                        "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
                        "stream_options": {"include_usage": True},
                        "mfq_prefix_cache_enabled": False}
                    if case.get("exclusive"):
                        payload["mfq_session_id"] = "throughput-exclusive"
                    first, text, reasoning, usage = None, "", "", None
                    async with client.generate(payload) as events:
                        async for event in events:
                            if event is None:
                                continue
                            if event.get("usage"):
                                usage = event["usage"]
                            for choice in event.get("choices", []):
                                delta = choice.get("delta", {})
                                chunk = delta.get("content", "")
                                reasoning += delta.get("reasoning_content", "")
                                if chunk and first is None:
                                    first = time.perf_counter()
                                text += chunk
                    finished = time.perf_counter()
                    return {"row": row, "text": text, "reasoning": reasoning, "usage": usage,
                            "ttft_s": None if first is None else first - started,
                            "finish_s": finished - started,
                            "decode_s": None if first is None else finished - first}

                try:
                    if case.get("serial"):
                        completed = [await generate(row) for row in range(requests)]
                    else:
                        completed = await asyncio.wait_for(asyncio.gather(
                            *(generate(row) for row in range(requests)),return_exceptions=True), timeout=180)
                    outputs = [output for output in completed if not isinstance(output,Exception)]
                    after = await client.status()
                    seconds = time.perf_counter() - started
                    graph_log = (RESULTS / "worker.log").read_bytes()[log_start:].decode("utf-8", errors="replace")
                    windows = [dict((key, int(value)) for key, value in re.findall(r"(\w+)=(\d+)", line))
                               for line in graph_log.splitlines() if line.startswith("native_cuda_window ")]
                    completion_tokens = sum((out["usage"] or {}).get("completion_tokens", 0)
                                            for out in outputs)
                    errors = sorted({str(output) for output in completed if isinstance(output,Exception)})
                    if completion_tokens != requests * tokens:
                        errors.append("early stop invalidates throughput measurement")
                    if not outputs or not all(out["text"] for out in outputs):
                        errors.append("no ordinary text output")
                    if not case.get("heterogeneous") and not case.get("mtp"):
                        if not all(out["text"][:len(reference)] == reference[:len(out["text"])]
                                   for out in outputs):
                            errors.append("batch output differs from the verified serial response")
                    if reference_tag := case.get("reference_tag"):
                        expected = json.loads((RESULTS / f"{reference_tag}.json").read_text(encoding="utf-8"))["outputs"]
                        if [out["text"] for out in outputs] != [out["text"] for out in expected]:
                            errors.append("heterogeneous batch output differs from its serial oracle")
                    metric_delta = {key: value - before.get(key, 0) for key, value in after.items()
                                    if isinstance(value, (int, float)) and
                                    key.startswith(("continuous_batching_", "moe_pipeline_", "moe_cache_"))}
                    result = {"case": case, "seconds": seconds, "tokens": completion_tokens,
                              "total_tps": completion_tokens / seconds if not errors else None,
                              "errors": errors, "metrics": metric_delta,
                              "status": after, "outputs": outputs, "windows": windows}
                    replay_ns = sum(window.get("replay_ns", 0) for window in windows)
                    decode_tokens = metric_delta.get("continuous_batching_decode_tokens", 0)
                    result["graph_decode_tps"] = decode_tokens * 1e9 / replay_ns if replay_ns and not errors else None
                    if requests == 1 and outputs and outputs[0]["decode_s"] and not errors:
                        result["single_request_decode_tps"] = (completion_tokens-1)/outputs[0]["decode_s"]
                    tag = case.get("tag", f"b{requests}-{'mtp' if case.get('mtp') else 'ordinary'}")
                    (RESULTS / f"{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
                    summary = {key: result[key] for key in ("case", "seconds", "tokens", "total_tps", "graph_decode_tps", "errors")}
                    summary["decode_batches"] = metric_delta.get("continuous_batching_decode_batches")
                    print(json.dumps(summary), flush=True)
                except Exception as error:
                    failed = {"case": case, "error": str(error), "seconds": time.perf_counter()-started}
                    try:
                        failed["status"] = await client.status()
                    except Exception as status_error:
                        failed["status_error"] = str(status_error)
                    tag = case.get("tag", f"b{requests}-failed")
                    (RESULTS / f"{tag}.json").write_text(json.dumps(failed,ensure_ascii=False,indent=2),encoding="utf-8")
                    print(json.dumps({key: failed[key] for key in ("case","error","seconds")}), flush=True)
        finally:
            await client.aclose()
            if process.returncode is None:
                process.terminate()
            await process.wait()


if __name__ == "__main__":
    asyncio.run(main())
