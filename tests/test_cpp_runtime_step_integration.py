"""Real-weight stdio gate. Set MFQ_STEP_TEST_MODEL and MFQ_STEP_TEST_TOKENIZER."""

import asyncio
import base64
import json
import os
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("batch_size", [0, 2])
def test_stdio_step_lifecycle(batch_size: int, tmp_path: Path) -> None:
    model = os.environ.get("MFQ_STEP_TEST_MODEL")
    tokenizer = os.environ.get("MFQ_STEP_TEST_TOKENIZER")
    if not model or not tokenizer:
        pytest.skip("real-weight model and tokenizer not configured")

    async def run() -> None:
        with (tmp_path / "runtime.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                os.environ.get("MFQ_STEP_TEST_RUNTIME", str(ROOT / "build/csrc/mfq-runtime")),
                "--model", model, "--tokenizer", tokenizer,
                "--transport", "stdio", "--ctx-size", "256",
                "--prefill-chunk-size", "8", "--continuous-batching", str(batch_size),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
                env={**os.environ, "MFQ_RUNTIME_DISABLE_PREFIX_CACHE": "1"},
            )
            frames: dict[str, list[dict]] = {}

            async def receive() -> dict:
                line = await asyncio.wait_for(process.stdout.readline(), 60)
                assert line, (tmp_path / "runtime.log").read_text()
                frame = json.loads(line)
                frames.setdefault(frame.get("id", ""), []).append(frame)
                return frame

            async def send(request_id: str, op: str, params: dict | None = None) -> None:
                process.stdin.write(json.dumps({"v": 1, "id": request_id, "op": op, "params": params or {}}).encode() + b"\n")
                await process.stdin.drain()

            async def until(request_id: str, kind: str = "done") -> list[dict]:
                while not any(frame["type"] == kind for frame in frames.get(request_id, [])):
                    frame = await receive()
                    assert frame["type"] != "error", frame
                return frames[request_id]

            def generation(stream: bool, tokens: int = 16) -> dict:
                return {
                    "input": {"messages": [{"role": "user", "content": "Count from one to ten."}]},
                    "sampling": {"max_new_tokens": tokens, "temperature": 0, "top_k": 1, "enable_mtp": False},
                    "stream": stream, "include_usage": True,
                }

            try:
                assert (await receive())["type"] == "ready"
                await send("models", "models")
                metadata = (await until("models", "result"))[0]["data"]["models"][0]
                assert metadata["type"]
                assert metadata["capabilities"]["architecture_family"] != "unknown"
                assert metadata["capabilities"]["source"] == "model-graph+cuda-adapters"
                assert metadata["capabilities"]["features"]["text"]
                if os.environ.get("MFQ_STEP_TEST_MTP") == "1":
                    assert metadata["capabilities"]["features"]["mtp"]
                await send("plain", "generate", generation(False))
                plain = await until("plain")
                result = next(frame["data"] for frame in plain if frame["type"] == "event")
                assert result["usage"]["prompt_tokens"] > 0
                assert result["usage"]["completion_tokens"] > 0
                assert "mtp_available" in result["metrics"]
                if os.environ.get("MFQ_STEP_TEST_MTP") == "1":
                    assert result["metrics"]["mtp_available"]
                if result["metrics"]["mtp_available"]:
                    speculative = generation(False)
                    speculative["sampling"]["enable_mtp"] = True
                    await send("mtp", "generate", speculative)
                    mtp = next(frame["data"] for frame in await until("mtp") if frame["type"] == "event")
                    assert mtp["output"] == result["output"]
                    assert mtp["metrics"]["mtp_used"]
                    assert mtp["metrics"]["mtp_cycles"] > 0
                await send("stream", "generate", generation(True))
                streamed = await until("stream")
                deltas = [frame["data"]["delta"] for frame in streamed if frame.get("data", {}).get("event") == "delta"]
                assert "".join(delta.get("content", "") for delta in deltas) == result["output"]["text"]
                assert "".join(delta.get("reasoning_content", "") for delta in deltas) == result["output"].get("reasoning", "")
                assert sum(frame.get("data", {}).get("event") == "complete" for frame in streamed) == 1
                # Media preparation happens after admission. A request error
                # must release its state without poisoning the entire engine.
                def tensor(dtype: str, shape: list[int], values: list[int]) -> dict:
                    code = "f" if dtype == "float32" else "i"
                    return {"dtype": dtype, "shape": shape, "encoding": "base64",
                            "data": base64.b64encode(struct.pack("<" + code * len(values), *values)).decode()}

                invalid_media = generation(False)
                invalid_media["input"]["preformatted_prompt"] = "Hello"
                invalid_media["media"] = {
                    "pixel_values": tensor("float32", [1, 3, 1, 1], [1, 1, 1]),
                    "vision_types": tensor("int32", [1], [1]),
                    "image_grid_thw": tensor("int32", [1, 3], [1, 1, 1]),
                }
                await send("invalid-media", "generate", invalid_media)
                error = await receive()
                assert error["id"] == "invalid-media" and error["type"] == "error"
                assert error["error"]["status_code"] == 400
                await send("after-invalid", "generate", generation(False))
                recovered = await until("after-invalid")
                assert next(frame["data"]["output"] for frame in recovered if frame["type"] == "event") == result["output"]
                constrained = generation(False, 32)
                constrained["template"] = {"enable_thinking": False}
                constrained["response_format"] = {
                    "type": "json_schema", "json_schema": {"schema": {
                        "type": "object", "properties": {"ok": {"const": True}},
                        "required": ["ok"], "additionalProperties": False,
                    }},
                }
                await send("schema", "generate", constrained)
                schema = await until("schema")
                assert json.loads(next(frame["data"]["output"]["text"] for frame in schema if frame["type"] == "event")) == {"ok": True}
                await send("cancelled", "generate", generation(True, 200))
                await send("cancel", "request.cancel", {"target_id": "cancelled"})
                cancelled = await until("cancelled")
                await until("cancel", "result")
                assert frames["cancel"][0]["data"]["cancelled"]
                assert next(frame["data"]["finish_reason"] for frame in cancelled if frame.get("data", {}).get("event") == "complete") == "cancelled"
                await send("reload-victim", "generate", generation(True, 200))
                await send("reload", "reload", {"context_size": 128})
                victim = await until("reload-victim")
                await until("reload", "result")
                assert next(frame["data"]["finish_reason"] for frame in victim if frame.get("data", {}).get("event") == "complete") == "cancelled"
                assert frames["reload"][0]["data"]["max_context"] == 128
                await send("after", "generate", generation(False))
                after = await until("after")
                assert next(frame["data"]["output"] for frame in after if frame["type"] == "event") == result["output"]
                await send("shutdown", "shutdown")
                await until("shutdown", "result")
                assert await asyncio.wait_for(process.wait(), 30) == 0
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()

    asyncio.run(run())


def test_diagnostic_engine_generation() -> None:
    import subprocess

    model = os.environ.get("MFQ_STEP_TEST_MODEL")
    if not model:
        pytest.skip("real-weight model not configured")
    command = [str(ROOT / "build/csrc/mfq-diagnostics"), "--model", model,
               "--ctx-size", "64", "--ids", "1,2,3", "--gen", "16"]
    outputs = []
    for graph in ("0", "1"):
        result = subprocess.run(command, env={**os.environ, "MFQ_CUDA_GRAPH": graph},
                                capture_output=True, text=True, timeout=60, check=True)
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        assert fields["generation_path"] == "engine_step"
        tokens = fields["generated_ids"].split(",")
        assert len(tokens) == 16 and int(fields["decode_tokens"]) == 15
        assert float(fields["decode_sec"]) > 0
        assert "decode_steady_tok_per_s" not in fields
        outputs.append(tokens)
    assert outputs[0] == outputs[1]
