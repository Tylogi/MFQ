import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mfq.server.protocol.models import ModelArtifactResource, ModelLoadRequest, RuntimeInstanceState, RuntimeMemoryPolicy
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.services.jobs import JobExecutionError
from mfq.server.state.catalog import DiscoveredModel, ModelCatalog
from mfq.server.state.storage import SessionStore
from tests.test_server_models import _TestJobContext

GIB = 1 << 30

def artifact(tmp_path, name, total, experts=0):
    return DiscoveredModel(path=tmp_path / f'{name}.mfq', routed_expert_bytes=experts * GIB,
        resource=ModelArtifactResource(id='a' * 32, name=name, architecture='test', total_bytes=total * GIB,
            shard_count=1, tensor_count=1, record_count=1, complete=True, loadable=True,
            modified_at=datetime.now(timezone.utc)))

def test_weight_and_prefix_limits_are_shared_across_models(tmp_path):
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=112 * GIB)
    moe, dense = artifact(tmp_path, 'moe', 80, 70), artifact(tmp_path, 'dense', 20)
    policy = RuntimeMemoryPolicy(model_limit_bytes=64 * GIB, prefix_limit_bytes=8 * GIB)
    weights, prefixes = pool._plan_memory_policy(policy, [moe, dense])
    assert weights == {'moe': 44 * GIB, 'dense': 20 * GIB}
    assert prefixes == {'moe': 4 * GIB, 'dense': 4 * GIB}
    request = pool._apply_automatic_expert_residency(moe, ModelLoadRequest(model='moe'), memory_ceiling=weights['moe'])
    assert request.moe_gpu_cache_gb is not None and 0 < request.moe_gpu_cache_gb < 70
    assert pool._estimated_load_bytes(moe, request) <= weights['moe']

def test_tight_multi_expert_budgets_keep_each_cache_usable(tmp_path):
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=112 * GIB)
    large, small = artifact(tmp_path, 'large', 90, 80), artifact(tmp_path, 'small', 30, 20)
    weights, _ = pool._plan_memory_policy(RuntimeMemoryPolicy(model_limit_bytes=31 * GIB), [large, small])
    assert sum(weights.values()) <= 31 * GIB
    for item in (large, small):
        request = pool._apply_automatic_expert_residency(item, ModelLoadRequest(model=item.resource.name), memory_ceiling=weights[item.resource.name])
        assert request.moe_gpu_cache_gb is not None and request.moe_gpu_cache_gb > 0
        assert pool._estimated_load_bytes(item, request) <= weights[item.resource.name]

def test_impossible_dense_budget_is_rejected_before_unloading(tmp_path):
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=112 * GIB)
    with pytest.raises(JobExecutionError):
        pool._plan_memory_policy(RuntimeMemoryPolicy(model_limit_bytes=20 * GIB),
            [artifact(tmp_path, 'moe', 80, 70), artifact(tmp_path, 'dense', 20)])
    with pytest.raises(JobExecutionError):
        pool._plan_memory_policy(RuntimeMemoryPolicy(prefix_limit_bytes=128 * GIB), [])

def test_policy_reloads_dense_before_experts_preserving_each_context_and_pin(tmp_path, monkeypatch):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=112 * GIB)
        pool.store = SessionStore(tmp_path / 'settings.sqlite3')
        moe, dense = artifact(tmp_path, 'moe', 80, 70), artifact(tmp_path, 'dense', 20)
        instances = [SimpleNamespace(id=uuid4(), artifact=item, state=RuntimeInstanceState.READY,
            active_requests=0, queued_requests=0, control_leases=0, context_size=size, pinned=pin, idle_ttl_seconds=ttl)
            for item, size, pin, ttl in ((moe, 32768, True, None), (dense, 8192, False, 300))]
        pool._instances = {item.id: item for item in instances}
        pool._load_requests = {item.artifact.resource.name: ModelLoadRequest(model=item.artifact.resource.name) for item in instances}
        loaded = []
        async def retire(item):
            pool._instances.pop(item.id, None)
        async def load(context, payload):
            loaded.append(payload)
            await context.progress(0.5)
        monkeypatch.setattr(pool, '_retire_instance', retire)
        monkeypatch.setattr(pool, 'load', load)
        context = _TestJobContext()
        context.job_id, context.cancel_requested = uuid4(), False
        result = await pool.configure_memory_policy(context, {'model_limit_bytes': 64 * GIB, 'prefix_limit_bytes': 8 * GIB})
        assert [item['model'] for item in loaded] == ['dense', 'moe']
        assert [item['context_size'] for item in loaded] == [8192, 32768]
        assert loaded[0]['idle_ttl_seconds'] == 300 and loaded[1]['pin'] is True
        assert result['models_replanned'] == 2 and pool._memory_configuration_job is None
        assert pool.store.runtime_memory_policy()['model_limit_bytes'] == 64 * GIB
    asyncio.run(run())

def test_failed_replan_restores_previous_policy_and_models(tmp_path, monkeypatch):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=112 * GIB)
        dense = artifact(tmp_path, 'dense', 20)
        instance = SimpleNamespace(id=uuid4(), artifact=dense, state=RuntimeInstanceState.READY,
            active_requests=0, queued_requests=0, control_leases=0, context_size=16384, pinned=False, idle_ttl_seconds=None)
        pool._instances = {instance.id: instance}
        pool._load_requests = {'dense': ModelLoadRequest(model='dense')}
        restored = []
        async def retire(item): pool._instances.pop(item.id, None)
        async def load(context, payload):
            if pool.memory_policy.model_limit_bytes is not None: raise RuntimeError('load failed')
            restored.append(payload)
        monkeypatch.setattr(pool, '_retire_instance', retire)
        monkeypatch.setattr(pool, 'load', load)
        context = _TestJobContext()
        context.job_id, context.cancel_requested = uuid4(), False
        with pytest.raises(RuntimeError): await pool.configure_memory_policy(context, {'model_limit_bytes': 64 * GIB})
        assert pool.memory_policy.model_limit_bytes is None and pool._memory_configuration_job is None
        assert restored[0]['context_size'] == 16384
    asyncio.run(run())

def test_prefix_directory_is_real_persisted_and_rejects_relative_paths(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        pool.store = SessionStore(tmp_path / 'settings.sqlite3')
        assert pool.memory_policy_status()['actual_prefix_directory'] == str(tmp_path / 'prefix-cache')
        context = _TestJobContext()
        context.job_id, context.cancel_requested = uuid4(), False
        directory = tmp_path / 'manual-cache'
        await pool.configure_memory_policy(context, {'prefix_directory': str(directory)})
        assert directory.is_dir()
        assert pool.memory_policy_status()['actual_prefix_directory'] == str(directory)
        assert pool.store.runtime_memory_policy()['prefix_directory'] == str(directory)
        with pytest.raises(JobExecutionError):
            await pool.configure_memory_policy(context, {'prefix_directory': 'relative/cache'})
        assert pool.memory_policy_status()['actual_prefix_directory'] == str(directory)
        await pool.configure_memory_policy(context, {'prefix_directory': None})
        assert pool.memory_policy_status()['actual_prefix_directory'] == str(tmp_path / 'prefix-cache')
    asyncio.run(run())
