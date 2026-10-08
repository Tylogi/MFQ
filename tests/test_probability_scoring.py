import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.runtime.backend import OpenAIChatBackend
from mfq.server.runtime.client import BackendProtocolError, HttpRuntimeClient
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_runtime_stdio_client import _client
from tests.test_server_service import FakeBackend


def result():
    return {"prompt_tokens": 3, "scores": [{"token_ids": [1, 2], "token_logprobs": [-1., -2.], "log_likelihood": -3.}],
        "log_base": "e", "tokenization": "raw-no-special-tokens"}


def test_http_and_stdio_probability_transport():
    async def run():
        payload = {"prompt": "raw prompt:", "continuations": [" answer"], "mode": "continuation"}
        seen = []
        def http(request):
            import json
            assert request.url.path == "/runtime/score"
            assert request.method == "POST"
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=result())
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            backend = OpenAIChatBackend(HttpRuntimeClient("http://runtime", client=client))
            assert await backend.score(payload) == result()
        def stdio(request):
            assert request["op"] == "score"
            seen.append(request["params"])
            return [{"v": 1, "id": request["id"], "type": "result", "data": result()}]
        native = _client(stdio)
        try:
            assert await native.score(payload) == result()
        finally:
            await native.aclose()
        assert seen == [payload, payload]
    asyncio.run(run())


@pytest.mark.parametrize("bad", ["nan", "positive", "total", "tokens", "base", "candidate_count", "next_token"])
def test_probability_responses_fail_closed(bad):
    async def run():
        value = result()
        if bad == "nan":
            value["scores"][0]["token_logprobs"][0] = float("nan")
        elif bad == "positive":
            value["scores"][0]["token_logprobs"][0] = 1
        elif bad == "total":
            value["scores"][0]["log_likelihood"] = -4
        elif bad == "tokens":
            value["scores"][0]["token_ids"] = [1]
        elif bad == "base":
            value["log_base"] = "10"
        elif bad == "candidate_count":
            value["scores"] = []
        class Runtime:
            async def score(self, _payload):
                return value
        backend = OpenAIChatBackend(Runtime())
        with pytest.raises(BackendProtocolError):
            await backend.score({"prompt": "prompt", "continuations": ["suffix"], "mode": "next_token" if bad == "next_token" else "continuation"})
    asyncio.run(run())


def test_probability_api_leases_loaded_model_and_preserves_raw_text(tmp_path):
    async def run():
        class Backend:
            payload = None
            async def probability_available(self):
                return True
            async def score(self, payload):
                self.payload = payload
                return result()
        class Pool:
            active = False
            model = Backend()
            @asynccontextmanager
            async def benchmark_runtime(self, _instance_id):
                assert not self.active
                self.active = True
                try:
                    yield SimpleNamespace(), self.model
                finally:
                    self.active = False
        pool = Pool()
        service = ServerService(SessionStore(tmp_path / "db.sqlite3"), FakeBackend(), tool_handlers=SimpleNamespace(runtime_manager=pool))
        payload = {"instance_id": str(uuid4()), "prompt": "Question:\nAnswer:", "continuations": [" raw answer"], "mode": "continuation"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
            response = await client.post("/api/v1/evaluations/probabilities", json=payload)
            assert response.status_code == 200, response.text
            assert response.json() == result()
            assert pool.model.payload == {key: value for key, value in payload.items() if key != "instance_id"}
            assert not pool.active
            for changes in ({"mode": "generate"}, {"continuations": []}, {"prompt": ""}, {"temperature": 1}):
                assert (await client.post("/api/v1/evaluations/probabilities", json={**payload, **changes})).status_code == 422
    asyncio.run(run())


def test_truthful_adapter_retains_original_token_slicing_and_sum():
    import torch

    from mfq.server.services.probability_benchmark import NativeLogProbs
    values = torch.tensor([0., 0., -9., -8., -7., -2., -3., 0.])
    outputs = NativeLogProbs(values).squeeze(0).log_softmax(-1)
    outputs = outputs[2:-1, :]
    selected = outputs[range(outputs.shape[0]), torch.zeros(5, dtype=torch.int64)]
    assert selected[3:].sum().item() == -5
