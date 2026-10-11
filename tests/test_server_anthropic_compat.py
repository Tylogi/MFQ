import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.api.anthropic_compat import collect_message, parse_messages_request, stream_message
from mfq.server.api.openai_compat import OpenAIRequestError
from mfq.server.protocol.models import TokenUsage
from mfq.server.runtime.backend import BackendDelta, BackendError, BackendToolCallDelta


class Backend:
    def __init__(self, deltas=None):
        self.requests = []
        self.closed = False
        self.deltas = deltas if deltas is not None else [
            BackendDelta(reasoning_delta="plan"), BackendDelta(content_delta="answer"),
            BackendDelta(finish_reason="length"),
            BackendDelta(usage=TokenUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16)),
        ]

    async def stream(self, **request):
        self.requests.append(request)
        try:
            for delta in self.deltas:
                if isinstance(delta, Exception):
                    raise delta
                yield delta
        finally:
            self.closed = True


class Service:
    def __init__(self, backend):
        self.backend = backend

    def resolve_model_alias(self, model):
        return "real-model" if model == "alias" else model

    async def advertised_models(self):
        return {"data": [{"id": "alias"}, {"id": "other"}]}


def body(**updates):
    return {"model": "alias", "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}], **updates}


def events(frames):
    return [json.loads(frame.split("data: ", 1)[1]) for frame in frames]


def test_messages_convert_system_images_tool_results_and_thinking():
    request = parse_messages_request(body(system=[{"type": "text", "text": "system"}],
        messages=[{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "reason", "signature": ""},
            {"type": "text", "text": "check"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"x": 1}},
        ]}, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
            {"type": "text", "text": "continue"},
        ]}], tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "lookup"}, thinking={"type": "disabled"}))
    messages = request.chat.messages
    assert [message["role"] for message in messages] == ["system", "assistant", "tool", "user"]
    assert messages[1]["reasoning_content"] == "reason"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    assert messages[2]["tool_call_id"] == "toolu_1"
    assert messages[3]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert request.chat.sampling.enable_thinking is False
    assert request.chat.tool_choice.function.name == "lookup"
    assert "temperature" not in request.chat.sampling.model_fields_set


@pytest.mark.parametrize("updates", [
    {"max_tokens": None}, {"max_tokens": True}, {"max_tokens": 0}, {"temperature": 2},
    {"messages": [{"role": "system", "content": "x"}]},
    {"messages": [{"role": "user", "content": []}]},
    {"messages": [{"role": "user", "content": [{"type": "tool_use", "id": "x"}]}]},
    {"tools": [{"type": "web_search_20250305", "name": "search"}]},
    {"tool_choice": {"type": "tool", "name": "absent"}},
    {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}},
    {"stop_sequences": [""]}, {"system": [{"type": "image"}]},
    {"thinking": {"type": "enabled", "budget_tokens": 128}},
    {"output_config": {"format": {"type": "json_schema"}}},
])
def test_invalid_or_unsupported_requests_fail_explicitly(updates):
    with pytest.raises(OpenAIRequestError):
        parse_messages_request(body(**updates))


@pytest.mark.parametrize("stream", [False, True])
def test_messages_route_auth_alias_usage_and_stop_reason(stream):
    async def run():
        backend = Backend()
        app = create_app(Service(backend), api_key="test-key")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            denied = await client.post("/v1/messages", json=body(stream=stream))
            assert denied.status_code == 401
            assert denied.json()["type"] == "error"
            assert denied.json()["error"]["type"] == "authentication_error"
            response = await client.post("/v1/messages", json=body(stream=stream),
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"})
            assert response.status_code == 200
            if stream:
                frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                assert frames[0]["type"] == "message_start"
                assert frames[-1]["type"] == "message_stop"
                assert frames[-2]["delta"]["stop_reason"] == "max_tokens"
                assert frames[-2]["usage"] == {"input_tokens": 12, "output_tokens": 4}
                assert "[DONE]" not in response.text
            else:
                result = response.json()
                assert result["type"] == "message"
                assert result["model"] == "alias"
                assert result["stop_reason"] == "max_tokens"
                assert result["usage"] == {"input_tokens": 12, "output_tokens": 4}
                assert [block["type"] for block in result["content"]] == ["thinking", "text"]
            assert backend.requests[0]["model"] == "real-model"
            assert backend.closed
            assert (await client.get("/api/v1/runtime/listener", headers={"x-api-key": "test-key"})).status_code == 401
            oai = await client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
            assert oai.json()["object"] == "list"
            anthropic = await client.get("/v1/models?limit=1", headers={"x-api-key": "test-key"})
            assert anthropic.json()["has_more"] is True
            assert anthropic.json()["last_id"] == "alias"
            following = await client.get("/v1/models?after_id=alias", headers={"x-api-key": "test-key"})
            assert following.json()["data"][0]["id"] == "other"
            invalid = await client.post("/v1/messages", json=body(max_tokens=0), headers={"x-api-key": "test-key"})
            assert invalid.status_code == 400
            assert invalid.json()["error"]["type"] == "invalid_request_error"
    asyncio.run(run())


def test_stream_tools_have_stable_block_indices_and_partial_json():
    async def run():
        backend = Backend([
            BackendDelta(content_delta="checking"),
            BackendDelta(tool_calls=(BackendToolCallDelta(index=0, call_id="toolu_1", name="lookup", arguments_delta='{"x":'),)),
            BackendDelta(tool_calls=(BackendToolCallDelta(index=0, arguments_delta="1}"),)),
            BackendDelta(finish_reason="tool_calls"),
        ])
        request = parse_messages_request(body(stream=True))
        frames = events([frame async for frame in stream_message(backend, request)])
        starts = [frame for frame in frames if frame["type"] == "content_block_start"]
        assert [frame["index"] for frame in starts] == [0, 1]
        assert starts[1]["content_block"]["id"] == "toolu_1"
        fragments = [frame["delta"]["partial_json"] for frame in frames
                     if frame["type"] == "content_block_delta" and frame["delta"]["type"] == "input_json_delta"]
        assert json.loads("".join(fragments)) == {"x": 1}
        assert frames[-2]["delta"]["stop_reason"] == "tool_use"
        assert backend.closed
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_stop_sequences_are_removed_including_across_chunks(stream):
    async def run():
        backend = Backend([BackendDelta(content_delta=text) for text in ["abc<ST", "OP>ignored", "tail"]]
                          + [BackendDelta(finish_reason="stop")])
        request = parse_messages_request(body(stream=stream, stop_sequences=["<STOP>"]))
        if stream:
            frames = events([frame async for frame in stream_message(backend, request)])
            text = "".join(frame["delta"]["text"] for frame in frames
                          if frame["type"] == "content_block_delta" and frame["delta"]["type"] == "text_delta")
            finish = frames[-2]["delta"]
        else:
            result = await collect_message(backend, request)
            text = result["content"][0]["text"]
            finish = result
        assert text == "abc"
        assert finish["stop_reason"] == "stop_sequence"
        assert finish["stop_sequence"] == "<STOP>"
        assert backend.closed
    asyncio.run(run())


def test_stream_error_and_disconnect_never_emit_successful_message_stop():
    async def run():
        request = parse_messages_request(body(stream=True))
        backend = Backend([BackendDelta(content_delta="partial"), BackendError("unavailable", "failed")])
        frames = events([frame async for frame in stream_message(backend, request)])
        assert frames[-1] == {"type": "error", "error": {"type": "api_error", "message": "failed"}}
        assert backend.closed
        backend = Backend()
        async def disconnected():
            return bool(backend.requests)
        frames = events([frame async for frame in stream_message(backend, request, disconnected=disconnected)])
        assert frames[-1]["type"] != "message_stop"
        assert backend.closed
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_stop_cancels_native_generation_and_releases_only_its_temporary_session(stream):
    class Cancellable(Backend):
        cancelled = None
        released = None

        async def cancel_response(self, session_id):
            self.cancelled = session_id
            return True

        async def close_session(self, session_id):
            self.released = session_id
            return True

        async def stream(self, **request):
            self.requests.append(request)
            try:
                yield BackendDelta(content_delta="before STOP after")
                assert self.cancelled == request["session_id"]
                yield BackendDelta(finish_reason="cancelled")
                yield BackendDelta(usage=TokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14))
            finally:
                self.closed = True

    async def run():
        backend = Cancellable()
        request = parse_messages_request(body(stream=stream, stop_sequences=["STOP"]))
        if stream:
            frames = events([frame async for frame in stream_message(backend, request)])
            assert frames[-2]["delta"]["stop_reason"] == "stop_sequence"
            assert frames[-2]["usage"]["output_tokens"] == 4
        else:
            result = await collect_message(backend, request)
            assert result["stop_reason"] == "stop_sequence"
            assert result["usage"]["output_tokens"] == 4
        assert backend.cancelled == backend.released
        assert backend.cancelled is not None
        assert backend.closed
    asyncio.run(run())


def test_ping_and_task_cancellation_close_underlying_generation():
    class Slow(Backend):
        async def stream(self, **request):
            self.requests.append(request)
            try:
                await asyncio.sleep(60)
                yield BackendDelta(content_delta="late")
            finally:
                self.closed = True

    async def run():
        backend = Slow()
        request = parse_messages_request(body(stream=True))
        iterator = stream_message(backend, request, keepalive_seconds=.01)
        assert events([await anext(iterator)])[0]["type"] == "message_start"
        assert events([await anext(iterator)])[0]["type"] == "ping"
        await iterator.aclose()
        assert backend.closed
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_unloaded_models_fail_before_committing_a_stream(tmp_path, stream):
    from mfq.server.runtime.runtime_pool import RuntimePool
    from mfq.server.state.catalog import ModelCatalog

    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime")
        try:
            app = create_app(Service(pool))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                result = await client.post("/v1/messages", json=body(stream=stream))
            assert result.status_code == 404
            assert result.json()["error"]["type"] == "not_found_error"
            assert not pool._instances
        finally:
            await pool.aclose()
    asyncio.run(run())
