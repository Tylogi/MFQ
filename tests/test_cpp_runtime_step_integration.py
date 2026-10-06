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
        result = subprocess.run(command, env={**os.environ, "MFQ_RUNTIME_CUDA_GRAPH": graph},
                                capture_output=True, text=True, timeout=60, check=True)
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        assert fields["generation_path"] == "engine_step"
        tokens = fields["generated_ids"].split(",")
        assert len(tokens) == 16 and int(fields["decode_tokens"]) == 15
        assert float(fields["decode_sec"]) > 0
        assert "decode_steady_tok_per_s" not in fields
        outputs.append(tokens)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("transport", ["http", "stdio"])
def test_generation_protocol_through_framing_adapters(transport: str) -> None:
    import socket
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    executable = ROOT / "build/csrc/tests/mfq-runtime-transport-test"
    if not executable.exists():
        pytest.skip("native transport test binary not built")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    async def run() -> None:
        command = [str(executable), "--stdio"] if transport == "stdio" else [
            str(executable), "--http", str(port),
        ]
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE if transport == "stdio" else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        def http(op: str, params: dict) -> list[dict]:
            path = {"session.fork": "sessions/fork", "cache.clear": "cache/clear",
                    "cache.trim": "cache/trim", "realtime.capabilities": "realtime/capabilities"}.get(op, op)
            method = "GET" if op in {"health", "status", "models", "realtime.capabilities"} else "POST"
            if op in {"session.close", "session.cancel"}:
                path = f"sessions/{params['session_id']}" + ("/cancel" if op == "session.cancel" else "")
                method = "DELETE" if op == "session.close" else "POST"
            data = json.dumps(params).encode() if method == "POST" else None
            request = Request(f"http://127.0.0.1:{port}/runtime/{path}", data=data, method=method,
                              headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=2) as response:
                    if response.headers.get_content_type() != "text/event-stream":
                        return [json.load(response)]
                    frames = []
                    for line in response:
                        if not line.startswith(b"data: "):
                            continue
                        if line.strip() == b"data: [DONE]":
                            break
                        frames.append(json.loads(line[6:]))
                    return frames
            except HTTPError as error:
                return [{"error": {**json.load(error)["error"], "status_code": error.code}}]

        async def invoke(id: str, op: str, params: dict | None = None) -> list[dict]:
            if transport == "http":
                return await asyncio.to_thread(http, op, params or {})
            process.stdin.write(json.dumps({"v": 1, "id": id, "op": op, "params": params or {}}).encode() + b"\n")
            await process.stdin.drain()
            frames = []
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), 2)
                assert line
                frame = json.loads(line)
                assert frame["id"] == id
                if frame["type"] == "error":
                    return [{"error": {**frame["error"], "type": frame["error"]["code"]}}]
                if frame["type"] == "done":
                    return frames
                frames.append(frame["data"])
                if frame["type"] == "result":
                    return frames

        params = {
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "sampling": {"max_new_tokens": 2, "seed": 1},
            "stream": False, "include_usage": True,
        }
        try:
            if transport == "stdio":
                assert json.loads(await asyncio.wait_for(process.stdout.readline(), 2))["type"] == "ready"
            else:
                for _ in range(100):
                    try:
                        await invoke("ready", "health")
                        break
                    except URLError:
                        await asyncio.sleep(0.01)
                else:
                    pytest.fail("fake HTTP transport did not start")
            plain, = await invoke("plain", "generate", params)
            assert plain["output"]["text"] == "hello"
            assert plain["output"]["reasoning"] == "thinking"
            assert plain["output"]["tool_calls"][0]["function"]["arguments"] == '{"ok":true}'
            assert plain["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
            params["stream"] = True
            streamed = await invoke("stream", "generate", params)
            assert [frame["event"] for frame in streamed] == ["delta", "delta", "complete", "usage"]
            assert "".join(frame.get("delta", {}).get("content", "") for frame in streamed) == plain["output"]["text"]
            assert "".join(frame.get("delta", {}).get("reasoning_content", "") for frame in streamed) == plain["output"]["reasoning"]
            assert streamed[-1]["usage"] == plain["usage"]
            assert streamed[-2]["finish_reason"] == plain["finish_reason"]
            assert all(frame["request_id"] == streamed[0]["request_id"] for frame in streamed)
            for performance in (plain["metrics"], streamed[-2]["metrics"]):
                assert performance["mtp_available"] and performance["mtp_cycles"] == 2
                assert "test_health_only" not in performance and "test_session_only" not in performance
            params["stream"] = False
            params["input"]["messages"][0]["content"] = "invalid_request"
            failed, = await invoke("failed", "generate", params)
            assert failed["error"]["status_code"] == 400
            assert failed["error"]["type"] == "invalid_request_error"
            status, = await invoke("status", "status")
            assert status["active_requests"] == 0
            assert status["total_requests"] == 3 and status["failed_requests"] == 1
            assert status["test_health_only"] == 99 and status["test_session_only"] == 99
            forked, = await invoke("fork", "session.fork", {"source_session_id": "a", "target_session_id": "b"})
            assert forked["copied_snapshots"] == 7
            closed, = await invoke("close-session", "session.close", {"session_id": "a"})
            assert closed["released_snapshots"] == 7
            for op, invalid in [("session.fork", {"source_session_id": "a", "target_session_id": "a"}),
                                ("session.fork", {"source_session_id": "../bad", "target_session_id": "b"}),
                                ("cache.trim", {"target_bytes": -1}),
                                ("reload", {"context_size": (1 << 64) - 1}),
                                ("reload", {"context_size": 129})]:
                rejected, = await invoke("rejected-control", op, invalid)
                assert rejected["error"]["status_code"] == 400 and rejected["error"]["type"] == "invalid_request_error"
            cleared, = await invoke("clear", "cache.clear")
            assert cleared["released_snapshots"] == 7 and cleared["test_session_only"] == 99
            trimmed, = await invoke("trim", "cache.trim", {"target_bytes": (1 << 64) - 1})
            assert trimmed["released_bytes"] == 7 and trimmed["target_bytes"] == (1 << 64) - 1
            loaded, = await invoke("reload", "reload", {"context_size": 64})
            assert loaded["max_context"] == 64
            health, = await invoke("health", "health")
            assert health["max_context"] == 64 and health["status"] == "ok"
            failed, = await invoke("reload-failed", "reload", {"context_size": 13})
            assert failed["error"]["status_code"] == 500
            health, = await invoke("after-failed-reload", "health")
            assert health["status"] != "loading" and health["max_context"] == 64
        finally:
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 2)

    asyncio.run(run())


def test_stdio_duplex_close_interrupts_pending_control() -> None:
    """The real stdio reader must accept close while a fake device is generating."""
    executable = ROOT / "build/csrc/tests/mfq-runtime-transport-test"
    if not executable.exists():
        pytest.skip("native transport test binary not built")

    async def run() -> None:
        process = await asyncio.create_subprocess_exec(
            str(executable), "--stdio", stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        async def send(id: str, op: str, params: dict | None = None) -> None:
            process.stdin.write(json.dumps({"v": 1, "id": id, "op": op, "params": params or {}}).encode() + b"\n")
            await process.stdin.drain()

        async def until(id: str, kind: str) -> dict:
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), 2)
                assert line
                frame = json.loads(line)
                if frame.get("id") == id and frame["type"] == kind:
                    return frame
                assert frame["type"] != "error", frame

        try:
            assert json.loads(await asyncio.wait_for(process.stdout.readline(), 2))["type"] == "ready"
            await send("channel", "realtime.open")
            await until("channel", "result")
            await send("init", "realtime.send", {"target_id": "channel", "data": json.dumps({"type": "session.init"})})
            await until("init", "result")
            await send("step", "realtime.send", {"target_id": "channel", "data": json.dumps({
                "type": "input.append", "input": {"text": "hello", "max_new_speak_tokens": 1000},
            })})
            await asyncio.sleep(0.03)
            await send("close", "realtime.close", {"target_id": "channel"})
            frames = []
            async def closed() -> None:
                while not any(frame.get("id") == "close" and frame["type"] == "result" for frame in frames):
                    frames.append(json.loads(await process.stdout.readline()))
            await asyncio.wait_for(closed(), 0.2)
            await send("shutdown", "shutdown")
            await asyncio.wait_for(process.wait(), 2)
            assert process.returncode == 0, (await process.stderr.read()).decode()
            remaining = [json.loads(line) for line in (await process.stdout.read()).splitlines()]
            assert sum(frame.get("id") == "step" and frame["type"] == "error" for frame in frames + remaining) == 1
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
    asyncio.run(run())


def test_websocket_duplex_disconnect_interrupts_pending_control() -> None:
    import socket
    import websockets

    executable = ROOT / "build/csrc/tests/mfq-runtime-transport-test"
    if not executable.exists():
        pytest.skip("native transport test binary not built")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    async def run() -> None:
        process = await asyncio.create_subprocess_exec(
            str(executable), "--http", str(port), stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        connection = None
        try:
            for _ in range(100):
                try:
                    connection = await websockets.connect(f"ws://127.0.0.1:{port}/runtime/realtime")
                    break
                except OSError:
                    await asyncio.sleep(0.01)
            assert connection is not None
            await connection.send(json.dumps({"type": "session.init"}))
            created = json.loads(await asyncio.wait_for(connection.recv(), 2))
            assert created["type"] == "session.created", created
            await connection.send(json.dumps({"type": "input.append", "input": {
                "text": "hello", "max_new_speak_tokens": 2,
            }}))
            outputs = [json.loads(await asyncio.wait_for(connection.recv(), 2)) for _ in range(3)]
            assert [frame.get("kind", frame["type"]) for frame in outputs] == ["text", "audio_tokens", "response.step.done"]
            assert outputs[0]["text"] == "hello" and outputs[1]["audio_tokens"] == [4, 5]
            assert outputs[1]["force_flush"]
            assert all(frame["session_id"] == created["session_id"] for frame in outputs)
            await connection.send(json.dumps({"type": "input.append", "input": {
                "text": "hello", "max_new_speak_tokens": 1000,
            }}))
            await asyncio.sleep(0.03)
            await asyncio.wait_for(connection.close(), 0.2)
        finally:
            if connection is not None:
                await connection.close()
            process.terminate()
            await asyncio.wait_for(process.wait(), 2)
    asyncio.run(run())
