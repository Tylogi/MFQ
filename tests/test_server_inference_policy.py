from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.protocol.models import RuntimeInferencePolicy, RuntimeInstanceState, SamplingParams, TokenUsage
from mfq.server.runtime.backend import BackendDelta
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.services.service import ServerService
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore
from tests.test_server_models import IdleBackend, _model


class SamplingBackend(IdleBackend):
    def __init__(self):
        self.sampling = []

    async def stream(self, **options):
        self.sampling.append(options["sampling"])
        yield BackendDelta(content_delta="ok", finish_reason="stop")
        yield BackendDelta(usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    async def runtime_status(self):
        return {"status": "ok", "sampling_defaults": {"enable_mtp": True}}


def test_service_mtp_policy_controls_api_and_chat_without_reloading_and_survives_restart(tmp_path):
    async def run():
        _model(tmp_path / "model.mfq")
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        artifact = await catalog.resolve("model")
        backend = SamplingBackend()
        pool = RuntimePool(catalog, tmp_path / "runtime", automatic_memory_budget=False)
        instance = _Runtime(id=uuid4(), artifact=artifact,
            process=SimpleNamespace(returncode=None), backend=backend, port=0,
            context_size=4096, sampling_defaults=SamplingParams(enable_mtp=True, temperature=0.4),
            state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1))
        pool._instances[instance.id] = instance
        store = SessionStore(tmp_path / "studio.sqlite3")
        service = ServerService(store, pool, runtime_manager=pool, catalog=catalog)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
            response = await client.get("/api/v1/runtime/inference-policy")
            assert response.json() == {"mtp_enabled": True}
            response = await client.put("/api/v1/runtime/inference-policy", json={"mtp_enabled": False})
            assert response.status_code == 200
            assert response.json() == {"mtp_enabled": False}
            body = {"model": "model", "messages": [{"role": "user", "content": "hello"}]}
            for override in ({}, {"enable_mtp": True}, {"enable_mtp": False}):
                response = await client.post("/v1/chat/completions", json={**body, **override})
                assert response.status_code == 200, response.text
                assert backend.sampling[-1].enable_mtp is False
                assert backend.sampling[-1].temperature == 0.4
            async for _ in pool.stream(model="model", messages=body["messages"],
                    sampling=SamplingParams(enable_mtp=True), session_id=uuid4()):
                pass
            assert backend.sampling[-1].enable_mtp is False
            response = await client.get(f"/api/v1/runtime/status?instance_id={instance.id}")
            assert response.json()["mtp_service_enabled"] is False
            restored = RuntimePool(catalog, tmp_path / "runtime", automatic_memory_budget=False)
            ServerService(store, restored, runtime_manager=restored, catalog=catalog)
            assert restored.inference_policy.mtp_enabled is False
            assert (await restored.runtime_status())["mtp_service_enabled"] is False
            response = await client.put("/api/v1/runtime/inference-policy", json={"mtp_enabled": True})
            assert response.status_code == 200
            for override, expected in (({}, True), ({"enable_mtp": False}, False), ({"enable_mtp": True}, True)):
                response = await client.post("/v1/chat/completions", json={**body, **override})
                assert response.status_code == 200, response.text
                assert backend.sampling[-1].enable_mtp is expected
            for value in ("false", 0, None):
                response = await client.put("/api/v1/runtime/inference-policy", json={"mtp_enabled": value})
                assert response.status_code == 422
            assert pool.inference_policy.mtp_enabled is True
        assert pool._instances[instance.id] is instance
        assert instance.backend is backend
        assert instance.state == RuntimeInstanceState.READY
        assert store.runtime_inference_policy() == {"mtp_enabled": True}

    asyncio.run(run())


def test_failed_policy_save_does_not_change_the_live_setting(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", automatic_memory_budget=False)
        def fail(_policy):
            raise OSError("disk full")
        pool.store = SimpleNamespace(save_runtime_inference_policy=fail)
        with pytest.raises(OSError, match="disk full"):
            await pool.configure_inference_policy(RuntimeInferencePolicy(mtp_enabled=False))
        assert pool.inference_policy.mtp_enabled is True

    asyncio.run(run())


def test_service_mtp_policy_also_controls_fallback_requests(tmp_path):
    async def run():
        backend = SamplingBackend()
        pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", fallback=backend, automatic_memory_budget=False)
        await pool.configure_inference_policy(RuntimeInferencePolicy(mtp_enabled=False))
        async for _ in pool.stream(model="external", messages=({"role": "user", "content": "hello"},),
                sampling=SamplingParams(enable_mtp=True)):
            pass
        assert backend.sampling[-1].enable_mtp is False

    asyncio.run(run())
