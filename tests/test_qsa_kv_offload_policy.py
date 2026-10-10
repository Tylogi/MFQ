import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import numpy as np
import pytest

from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.server.api import create_app
from mfq.server.protocol.models import (
    ErrorDetail,
    ModelLoadRequest,
    QsaKvOffloadPolicy,
    RuntimeContextPolicy,
    RuntimeInstanceState,
)
from mfq.server.runtime.client import BackendError
from mfq.server.runtime.qsa_kv_policy import qsa_index_requirement
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.services.jobs import JobExecutionError
from mfq.server.state.catalog import ModelCatalog
from tests.test_server_models import _TestJobContext


def artifact(tmp_path, kind="qwen4_exp_text"):
    path = tmp_path / "arbitrary-name.mfq"
    text = {"model_type": kind, "max_position_embeddings": 262144, "num_hidden_layers": 48,
        "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)],
        "indexer_head_dim": 128, "indexer_compress_ratio": 4, "mtp_num_hidden_layers": 1,
        "num_key_value_heads": 2, "head_dim": 256}
    save(path, FileHeader(version=2, model_arch="qwen4_exp-hf-mfq-custom"),
        {"predictor.block.0.stub.weight": np.ones((1,), np.float16),
         MODEL_CONFIG_ASSET: json.dumps({"model_type": "qwen4_exp", "text_config": text}).encode()})
    return asyncio.run(ModelCatalog([tmp_path]).resolve("arbitrary-name"))


def test_requirement_includes_predictor_and_actual_index_storage(tmp_path):
    model = artifact(tmp_path)
    assert qsa_index_requirement(model, 262144) == 13 * (128 * (9 * 2 + 65536 * 4) + 2)


def test_yarn_extended_streaming_context_uses_the_enabled_model_limit(tmp_path, monkeypatch):
    model = artifact(tmp_path)
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'native', backend='metal')
    policy = QsaKvOffloadPolicy(enabled=True, budget_bytes=128 << 10, target_context=678920)
    with pytest.raises(JobExecutionError, match='exceeds the model capacity'):
        asyncio.run(pool._validate_qsa_kv_policy(model, policy))
    pool.context_policy = RuntimeContextPolicy(model_overrides={model.resource.name: 678920},
        model_yarn_enabled={model.resource.name: True})
    expected = 13 * (128 * (9 * 2 + (678920 // 4) * 4) + 2)
    assert asyncio.run(pool._validate_qsa_kv_policy(model, policy)) == expected
    instance = SimpleNamespace(id=uuid4(), artifact=model, qsa_kv_offload_supported=True, context_size=678920)
    pool._instances[instance.id] = instance
    pool._load_requests[model.resource.name] = ModelLoadRequest(model=model.resource.name, context_size=678920)
    info = asyncio.run(pool.qsa_kv_offload_info(instance.id))
    assert info['target_context'] == 678920 and info['required_index_bytes'] == expected
    monkeypatch.setattr('mfq.server.runtime.runtime_pool.native_tokenizer_arguments', lambda *_: [])
    argv, environment = pool._launch_configuration(model, ModelLoadRequest(model=model.resource.name,
        qsa_kv_offload=policy), port=0)
    assert argv[argv.index('--ctx-size') + 1] == '678920'
    assert '--yarn-context-extension' in argv
    assert environment['MFQ_QSA_KV_BUDGET_BYTES'] == str(128 << 10)
    with pytest.raises(JobExecutionError, match='exceeds the model capacity'):
        asyncio.run(pool._validate_qsa_kv_policy(model, policy.model_copy(update={'target_context': 1048577})))


def test_small_budget_does_not_reject_target_indexer_spill(tmp_path):
    model = artifact(tmp_path)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="metal")
    policy = QsaKvOffloadPolicy(enabled=True, budget_bytes=128 << 10, target_context=262144)
    assert asyncio.run(pool._validate_qsa_kv_policy(model, policy)) > policy.budget_bytes


def test_impossible_microblock_buffer_is_rejected_before_reloading(tmp_path):
    model = artifact(tmp_path)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="metal")
    with pytest.raises(JobExecutionError, match="microblock I/O"):
        asyncio.run(pool._validate_qsa_kv_policy(model, QsaKvOffloadPolicy(enabled=True, budget_bytes=1)))


def test_non_qsa_config_cannot_bypass_capability_with_its_name(tmp_path):
    model = artifact(tmp_path, "qwen3_5_text")
    pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="metal")
    with pytest.raises(JobExecutionError, match="QSA only"):
        asyncio.run(pool._validate_qsa_kv_policy(model, QsaKvOffloadPolicy(enabled=True)))


def test_cuda_cannot_enable_metal_only_feature(tmp_path):
    pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="cuda")
    with pytest.raises(JobExecutionError, match="Metal QSA"):
        asyncio.run(pool._validate_qsa_kv_policy(artifact(tmp_path), QsaKvOffloadPolicy(enabled=True)))


def test_launch_off_is_not_overridden_by_inherited_environment(tmp_path, monkeypatch):
    model = artifact(tmp_path)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.native_tokenizer_arguments", lambda *_: [])
    pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="metal",
        runtime_environment={"MFQ_QSA_KV_BUDGET_BYTES": "123", "MFQ_QSA_KV_TARGET_CONTEXT": "456"})
    _, environment = pool._launch_configuration(model, ModelLoadRequest(model=model.resource.name), port=0)
    assert "MFQ_QSA_KV_BUDGET_BYTES" not in environment
    enabled = ModelLoadRequest(model=model.resource.name, context_size=32768,
        qsa_kv_offload={"enabled": True, "budget_bytes": 128 << 10, "target_context": 262144})
    _, environment = pool._launch_configuration(model, enabled, port=0)
    assert environment["MFQ_QSA_KV_BUDGET_BYTES"] == str(128 << 10)
    assert "MFQ_QSA_KV_TARGET_CONTEXT" not in environment


@pytest.mark.parametrize("supported,active", [(False, 0), (True, 1)])
def test_unsupported_or_busy_model_is_not_unloaded(tmp_path, supported, active):
    model = artifact(tmp_path)
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="metal")
        instance = SimpleNamespace(id=uuid4(), artifact=model, qsa_kv_offload_supported=supported,
            context_size=32768, state=RuntimeInstanceState.READY, active_requests=active, queued_requests=0,
            control_leases=0, pinned=False, idle_ttl_seconds=None)
        pool._instances[instance.id] = instance
        pool._load_requests[model.resource.name] = ModelLoadRequest(model=model.resource.name)
        pool._retire_instance = AsyncMock()
        with pytest.raises(JobExecutionError):
            await pool.configure_qsa_kv_offload(_TestJobContext(), {"instance_id": str(instance.id), "enabled": True,
                "budget_bytes": 128 << 10})
        pool._retire_instance.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize("fail", [False, True])
def test_settings_reload_only_selected_model_and_recover_on_failure(tmp_path, fail):
    model = artifact(tmp_path)
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="metal")
        instance = SimpleNamespace(id=uuid4(), artifact=model, qsa_kv_offload_supported=True,
            context_size=32768, state=RuntimeInstanceState.READY, active_requests=0, queued_requests=0,
            control_leases=0, pinned=True, idle_ttl_seconds=300)
        other = SimpleNamespace(id=uuid4(), artifact=SimpleNamespace(resource=SimpleNamespace(name="other")),
            state=RuntimeInstanceState.READY)
        pool._instances = {instance.id: instance, other.id: other}
        previous = ModelLoadRequest(model=model.resource.name, context_size=32768)
        pool._load_requests[model.resource.name] = previous
        async def retire(item):
            assert item is instance
            pool._instances.pop(item.id, None)
        pool._retire_instance = AsyncMock(side_effect=retire)
        pool.load = AsyncMock(side_effect=[RuntimeError("load failed"), {"instance_id": "restored"}] if fail
            else [{"instance_id": "updated"}])
        payload = {"instance_id": str(instance.id), "enabled": True, "budget_bytes": 128 << 10}
        if fail:
            with pytest.raises(RuntimeError, match="load failed"):
                await pool.configure_qsa_kv_offload(_TestJobContext(), payload)
            assert pool._load_requests[model.resource.name] == previous
            assert pool.load.await_args_list[1].args[1] == previous.model_dump(mode="json")
        else:
            result = await pool.configure_qsa_kv_offload(_TestJobContext(), payload)
            assert result["qsa_kv_offload"]["budget_bytes"] == 128 << 10
        requested = pool.load.await_args_list[0].args[1]
        assert requested["context_size"] == 32768
        assert requested["qsa_kv_offload"]["target_context"] == 32768
        assert requested["pin"] is True and requested["idle_ttl_seconds"] == 300
        assert pool._instances[other.id] is other and other.state == RuntimeInstanceState.READY
        assert pool._memory_configuration_job is None
    asyncio.run(run())


@pytest.mark.parametrize("error,status,code", [
    (BackendError("runtime_instance_not_found", "missing", status_code=404), 404, "runtime_instance_not_found"),
    (JobExecutionError(ErrorDetail(code="qsa_kv_configuration_invalid", message="bad target")), 400, "qsa_kv_configuration_invalid"),
])
def test_preflight_routes_return_structured_errors(error, status, code):
    async def run():
        manager = SimpleNamespace(qsa_kv_offload_info=AsyncMock(side_effect=error))
        service = SimpleNamespace(runtime_manager=manager, create_job=AsyncMock())
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            instance_id = str(uuid4())
            responses = [await client.get(f"/api/v1/runtime/qsa-kv/{instance_id}"),
                await client.put("/api/v1/runtime/qsa-kv", json={"instance_id": instance_id,
                    "enabled": True, "budget_bytes": 128 << 10})]
        for response in responses:
            assert response.status_code == status
            assert response.json()["error"]["code"] == code
        service.create_job.assert_not_called()
    asyncio.run(run())


def test_preflight_does_not_create_jobs_for_non_qsa_but_accepts_fractional_budget():
    async def run():
        manager = SimpleNamespace(qsa_kv_offload_info=AsyncMock(return_value={"supported": False}))
        service = SimpleNamespace(runtime_manager=manager, create_job=AsyncMock(return_value=SimpleNamespace(id=uuid4())))
        app = create_app(service)
        payload = {"instance_id": str(uuid4()), "enabled": True, "budget_bytes": 128 << 10}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put("/api/v1/runtime/qsa-kv", json=payload)
            assert response.status_code == 400
            service.create_job.assert_not_called()
            manager.qsa_kv_offload_info.return_value = {"supported": True}
            response = await client.put("/api/v1/runtime/qsa-kv", json=payload)
            assert response.status_code == 202
            assert service.create_job.await_args.args[0].payload == payload
    asyncio.run(run())


def test_offload_configuration_cannot_set_context():
    async def run():
        manager = SimpleNamespace(qsa_kv_offload_info=AsyncMock(return_value={"supported": True}))
        service = SimpleNamespace(runtime_manager=manager, create_job=AsyncMock())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
            response = await client.put("/api/v1/runtime/qsa-kv", json={"instance_id": str(uuid4()),
                "enabled": True, "budget_bytes": 128 << 10, "target_context": 65536})
        assert response.status_code == 422
        service.create_job.assert_not_called()
    asyncio.run(run())


def test_load_normalizes_legacy_offload_context_to_model_context(tmp_path):
    model = artifact(tmp_path)
    async def run():
        pool = RuntimePool(ModelCatalog([tmp_path]), tmp_path / "native", backend="metal")
        pool._validate_qsa_kv_policy = AsyncMock(side_effect=RuntimeError("validation boundary"))
        with pytest.raises(RuntimeError, match="validation boundary"):
            await pool.load(_TestJobContext(), {"model": model.resource.name, "context_size": 32768,
                "qsa_kv_offload": {"enabled": True, "target_context": 262144}})
        assert pool._validate_qsa_kv_policy.await_args.args[1].target_context == 32768
    asyncio.run(run())


def test_info_follows_live_context_not_saved_offload_target(tmp_path):
    model = artifact(tmp_path)
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="metal")
        instance = SimpleNamespace(id=uuid4(), artifact=model, context_size=32768, qsa_kv_offload_supported=True)
        pool._instances[instance.id] = instance
        pool._load_requests[model.resource.name] = ModelLoadRequest(model=model.resource.name, context_size=32768,
            qsa_kv_offload={"enabled": True, "target_context": 262144})
        info = await pool.qsa_kv_offload_info(instance.id)
        assert info["target_context"] == info["policy"]["target_context"] == 32768
        assert info["required_index_bytes"] == qsa_index_requirement(model, 32768)
    asyncio.run(run())


@pytest.mark.parametrize("size", [8192, 65536])
def test_regular_context_reload_remains_available_with_offload_enabled(tmp_path, size):
    model = artifact(tmp_path)
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / "native", backend="metal")
        backend = SimpleNamespace(reload_runtime=AsyncMock(return_value={"max_context": size}))
        instance = _Runtime(id=uuid4(), artifact=model, backend=backend, process=None, port=0,
            context_size=32768, context_capacity=262144, state=RuntimeInstanceState.READY)
        pool._instances[instance.id] = instance
        pool._load_requests[model.resource.name] = ModelLoadRequest(model=model.resource.name, context_size=32768,
            qsa_kv_offload={"enabled": True, "budget_bytes": 128 << 10, "target_context": 262144})
        await pool.reload_runtime(size, instance.id)
        backend.reload_runtime.assert_awaited_once_with(size)
        request = pool._load_requests[model.resource.name]
        assert instance.context_size == request.context_size == request.qsa_kv_offload.target_context == size
        assert request.qsa_kv_offload.enabled and request.qsa_kv_offload.budget_bytes == 128 << 10
    asyncio.run(run())
