import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from mfq.server.protocol.models import (
    ModelLoadRequest,
    RuntimeContextPolicy,
    RuntimeInstanceState,
    RuntimeReloadRequest,
)
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.services.jobs import JobExecutionError
from mfq.server.services.service import ServerService
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore
from tests.test_qsa_kv_offload_policy import artifact
from tests.test_server_models import _TestJobContext


def fixture(tmp_path):
    model = artifact(tmp_path)
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', backend='metal', automatic_memory_budget=False)
    service = ServerService(SessionStore(tmp_path / 'server.sqlite3'), pool, runtime_manager=pool)
    instance = SimpleNamespace(id=uuid4(), artifact=model, context_size=32768, state=RuntimeInstanceState.READY,
        qsa_kv_offload_supported=True, active_requests=0, queued_requests=0, control_leases=0, pinned=True, idle_ttl_seconds=300)
    other = SimpleNamespace(id=uuid4(), artifact=SimpleNamespace(resource=SimpleNamespace(name='other')), state=RuntimeInstanceState.READY)
    pool._instances = {instance.id: instance, other.id: other}
    previous = ModelLoadRequest(model=model.resource.name, context_size=32768)
    pool._load_requests[model.resource.name] = previous
    pool._retire_instance = AsyncMock(side_effect=lambda item: pool._instances.pop(item.id, None))
    return model, pool, service, instance, other, previous


@pytest.mark.parametrize('requested,target,warning', [(524288, 524288, None), (2000000, 1048576, 'yarn_context_size_exceeded')])
def test_one_save_reloads_once_with_yarn_and_streaming_then_persists_together(tmp_path, requested, target, warning):
    model, pool, service, instance, other, _ = fixture(tmp_path)

    async def run():
        async def load(context, payload):
            assert service.store.runtime_context_policy() == {}
            assert pool.context_policy.model_overrides == {} and pool.context_policy.model_yarn_enabled == {}
            assert payload['context_size'] == target and payload['yarn_enabled'] is True
            assert payload['qsa_kv_offload'] == {'enabled': True, 'budget_bytes': 128 << 10, 'target_context': target}
            assert payload['pin'] and payload['idle_ttl_seconds'] == 300
            await pool.configure_context_policy(RuntimeContextPolicy(max_context_size=65536))
            return {'instance_id': str(uuid4()), 'model': model.resource.name, 'context_size': target}

        pool.load = AsyncMock(side_effect=load)
        result = await pool.configure_context(_TestJobContext(), {'instance_id': str(instance.id), 'context_size': requested,
            'yarn_enabled': True, 'qsa_kv_offload': {'enabled': True, 'budget_bytes': 128 << 10}})
        assert result['max_context'] == target and result['warning'] == warning
        pool.load.assert_awaited_once()
        pool._retire_instance.assert_awaited_once_with(instance)
        saved = service.store.runtime_context_policy()
        assert saved['max_context_size'] == 65536
        assert saved['model_overrides'] == {model.resource.name: target}
        assert saved['model_yarn_enabled'] == {model.resource.name: True}
        assert saved['model_qsa_kv_offload'] == {model.resource.name: {'enabled': True, 'budget_bytes': 128 << 10}}
        restored = RuntimeContextPolicy.model_validate(saved)
        assert restored.model_qsa_kv_offload[model.resource.name].enabled
        assert pool._instances[other.id] is other
        pool._instances.clear()
        await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize('budget,supported', [(1, True), (128 << 10, False)])
def test_invalid_candidate_does_not_unload_or_save_any_setting(tmp_path, budget, supported):
    _, pool, service, instance, _, _ = fixture(tmp_path)

    async def run():
        instance.qsa_kv_offload_supported = supported
        pool.load = AsyncMock()
        with pytest.raises(JobExecutionError):
            await pool.configure_context(_TestJobContext(), {'instance_id': str(instance.id), 'context_size': 524288,
                'yarn_enabled': True, 'qsa_kv_offload': {'enabled': True, 'budget_bytes': budget}})
        pool.load.assert_not_awaited()
        pool._retire_instance.assert_not_awaited()
        assert service.store.runtime_context_policy() == {}
        assert instance.state == RuntimeInstanceState.READY
        pool._instances.clear()
        await service.aclose()

    asyncio.run(run())


def test_failed_reload_recovers_previous_model_without_publishing_candidate_settings(tmp_path):
    model, pool, service, instance, other, previous = fixture(tmp_path)

    async def run():
        pool.load = AsyncMock(side_effect=[RuntimeError('load failed'), {'instance_id': str(uuid4()), 'context_size': 32768}])
        with pytest.raises(RuntimeError, match='load failed'):
            await pool.configure_context(_TestJobContext(), {'instance_id': str(instance.id), 'context_size': 524288,
                'yarn_enabled': True, 'qsa_kv_offload': {'enabled': True, 'budget_bytes': 128 << 10}})
        assert pool.load.await_count == 2
        assert pool.load.await_args_list[1].args[1] == previous.model_dump(mode='json')
        assert pool._load_requests[model.resource.name] == previous
        assert service.store.runtime_context_policy() == {} and pool.context_policy.model_yarn_enabled == {}
        assert pool._instances[other.id] is other
        pool._instances.clear()
        await service.aclose()

    asyncio.run(run())


def test_saved_streaming_policy_is_reused_on_future_model_loads(tmp_path):
    model = artifact(tmp_path)
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', backend='metal')
    pool.context_policy = RuntimeContextPolicy(model_overrides={model.resource.name: 524288},
        model_yarn_enabled={model.resource.name: True},
        model_qsa_kv_offload={model.resource.name: {'enabled': True, 'budget_bytes': 128 << 10}})
    pool.catalog.resolve = AsyncMock(return_value=model)
    pool._validate_qsa_kv_policy = AsyncMock(side_effect=RuntimeError('validation boundary'))

    async def run():
        with pytest.raises(RuntimeError, match='validation boundary'):
            await pool.load(_TestJobContext(), {'model': model.resource.name})
        policy = pool._validate_qsa_kv_policy.await_args.args[1]
        assert policy.enabled and policy.budget_bytes == 128 << 10 and policy.target_context == 524288

    asyncio.run(run())


def test_combined_settings_do_not_accept_a_second_context_setting_inside_streaming_policy():
    with pytest.raises(ValueError):
        RuntimeReloadRequest.model_validate({'qsa_kv_offload': {'enabled': True, 'target_context': 4096}})
