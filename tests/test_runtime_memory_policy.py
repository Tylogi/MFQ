import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mfq.server.protocol.models import ModelArtifactResource, ModelLoadRequest, RuntimeInstanceState, RuntimeMemoryPolicy, RuntimeMemoryResources
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.services.jobs import JobExecutionError
from mfq.server.state.catalog import DiscoveredModel, ModelCatalog
from mfq.server.state.storage import SessionStore
from tests.test_server_models import _TestJobContext

GIB = 1 << 30

class CacheBackend:
    def __init__(self, hot, limit, pressure=0):
        self.hot, self.limit, self.pressure = hot, limit, pressure
        self.calls = []
        self.disk = None

    async def set_prefix_cache_budget(self, target, disk_target_bytes=None):
        self.calls.append(target)
        self.limit = target
        self.hot = min(self.hot, target)
        self.pressure = 0
        if disk_target_bytes is not None:
            self.disk = disk_target_bytes
        return {'status': 'ok'}

    async def runtime_status(self):
        result = {'prefix_cache_dynamic_budget': 1, 'prefix_cache_hot_bytes': self.hot,
            'prefix_cache_max_bytes': self.limit, 'prefix_cache_hot_pressure_bytes': self.pressure}
        if self.disk is not None:
            result['prefix_cache_disk_max_bytes'] = self.disk
        return result


def cached_instance(tmp_path, name, hot, limit, pressure=0, active=0):
    backend = CacheBackend(hot, limit, pressure)
    return SimpleNamespace(id=uuid4(), artifact=artifact(tmp_path, name, 1), state=RuntimeInstanceState.BUSY if active else RuntimeInstanceState.READY,
        prefix_dynamic_budget=True, prefix_pressure_bytes=pressure, prefix_pending_writes=0,
        active_requests=active, queued_requests=0, control_leases=0, read_control_leases=0, started_at=datetime.now(timezone.utc), last_used_at=None,
        memory=RuntimeMemoryResources(prefix_cache_bytes=hot, prefix_cache_limit_bytes=limit), backend=backend, process=None)


def test_auto_prefix_budget_does_not_multiply_with_model_count(tmp_path):
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
    _, hot = pool._plan_memory_policy(RuntimeMemoryPolicy(), [artifact(tmp_path, 'a', 1), artifact(tmp_path, 'b', 1)])
    assert sum(hot.values()) == 2 * GIB


def test_metal_launch_passes_cache_limits_to_native_environment(tmp_path, monkeypatch):
    monkeypatch.setattr('mfq.server.runtime.runtime_pool.native_tokenizer_arguments', lambda *_: [])
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', backend='metal')
    pool.memory_policy = RuntimeMemoryPolicy(prefix_disk_limit_bytes=512 << 20)
    _, environment = pool._launch_configuration(artifact(tmp_path, 'a', 1),
        ModelLoadRequest(model='a', prefix_cache_hot_bytes=0, prefix_cache_block_tokens=256), port=0)
    assert environment['MFQ_SERVER_PREFIX_CACHE_HOT_BYTES'] == '0'
    assert environment['MFQ_SERVER_PREFIX_CACHE_DISK_BYTES'] == str(512 << 20)
    assert environment['MFQ_SERVER_PREFIX_CACHE_BLOCK_TOKENS'] == '256'
    assert environment['MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES'] == '0'


def test_hot_capacity_borrows_by_demand_and_trims_lru_without_reloads(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        a = cached_instance(tmp_path, 'a', 3 * GIB, 4 * GIB)
        b = cached_instance(tmp_path, 'b', GIB, 4 * GIB, pressure=6 * GIB)
        pool.memory_policy = RuntimeMemoryPolicy(prefix_limit_bytes=8 * GIB)
        pool._instances = {a.id: a, b.id: b}
        await pool._rebalance_prefix_cache_budget(b.id)
        assert a.backend.limit == 2 * GIB and b.backend.limit == 6 * GIB
        assert a.backend.hot == 2 * GIB
        assert sum(item.backend.limit for item in (a, b)) == 8 * GIB
        assert a.control_leases == b.control_leases == 0
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_busy_cache_keeps_its_capacity_until_request_finishes(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        a = cached_instance(tmp_path, 'a', GIB, 2 * GIB, active=1)
        b = cached_instance(tmp_path, 'b', 0, 0, pressure=GIB)
        pool._instances = {a.id: a, b.id: b}
        await pool._rebalance_prefix_cache_budget(b.id)
        assert a.backend.calls == [] and b.backend.limit == 0
        a.active_requests = 0
        a.state = RuntimeInstanceState.READY
        await pool._rebalance_prefix_cache_budget(b.id)
        assert b.backend.limit == GIB and a.backend.limit == GIB
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_loading_workers_reserve_shared_hot_capacity(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        assert pool._reserve_prefix_capacity_locked('loading', set()) == 2 * GIB
        pool._loading_model_names.add('loading')
        assert pool._reserve_prefix_capacity_locked('second', set()) == 0
        ready = cached_instance(tmp_path, 'ready', 0, 0, pressure=GIB)
        pool._instances = {ready.id: ready}
        await pool._rebalance_prefix_cache_budget(ready.id)
        assert ready.backend.limit == 0 and ready.backend.calls == []
        pool._loading_model_names.clear()
        await pool._rebalance_prefix_cache_budget(ready.id)
        assert ready.backend.limit == 2 * GIB
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_reloaded_model_cannot_reuse_an_outdated_capacity(tmp_path):
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
    current = cached_instance(tmp_path, 'current', GIB, 2 * GIB)
    pool._instances = {current.id: current}
    pool._policy_prefix_limits = {'current': GIB, 'old': 2 * GIB}
    assert pool._reserve_prefix_capacity_locked('old', set()) == 0


def test_disk_quota_changes_propagate_without_weight_reloads(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        instance = cached_instance(tmp_path, 'a', 0, 2 * GIB)
        instance.prefix_disk_capacity = 100 * GIB
        pool._instances = {instance.id: instance}
        pool.memory_policy = RuntimeMemoryPolicy(prefix_disk_limit_bytes=0)
        await pool._rebalance_prefix_cache_budget()
        assert instance.backend.disk == 0 and instance.prefix_disk_capacity == 0
        pool.memory_policy = RuntimeMemoryPolicy(prefix_disk_limit_bytes=GIB)
        await pool._rebalance_prefix_cache_budget()
        assert instance.backend.disk == GIB and instance.backend.limit == 2 * GIB
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('readers', [0, 3])
def test_prefix_only_setting_changes_do_not_retire_models(tmp_path, monkeypatch, readers):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        instance = cached_instance(tmp_path, 'a', GIB, 2 * GIB)
        instance.control_leases = instance.read_control_leases = readers
        instance.context_size, instance.pinned, instance.idle_ttl_seconds = 8192, False, None
        pool._instances = {instance.id: instance}
        pool._load_requests = {'a': ModelLoadRequest(model='a')}
        async def retire(_): raise AssertionError('prefix-only update reloaded weights')
        monkeypatch.setattr(pool, '_retire_instance', retire)
        context = _TestJobContext()
        context.job_id, context.cancel_requested = uuid4(), False
        result = await pool.configure_memory_policy(context, {'prefix_limit_bytes': 512 << 20})
        assert result['models_replanned'] == 0 and instance.backend.limit == 512 << 20
        assert instance.context_size == 8192 and pool._memory_configuration_job is None
        assert instance.control_leases == instance.read_control_leases == readers
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('blocked', ['active', 'queued', 'exclusive', 'reload_readers'])
def test_memory_configuration_still_rejects_busy_workers_and_reload_with_readers(tmp_path, blocked):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        instance = cached_instance(tmp_path, 'a', GIB, 2 * GIB)
        instance.context_size, instance.pinned, instance.idle_ttl_seconds = 8192, False, None
        instance.control_leases = instance.read_control_leases = 3
        payload = {'prefix_limit_bytes': 512 << 20, 'prefix_directory': str(tmp_path / 'prefix-cache')}
        if blocked == 'active':
            instance.active_requests = 1
        elif blocked == 'queued':
            instance.queued_requests = 1
        elif blocked == 'exclusive':
            instance.control_leases += 1
        else:
            payload['prefix_directory'] = str(tmp_path / 'another-prefix-cache')
        pool._instances = {instance.id: instance}
        pool._load_requests = {'a': ModelLoadRequest(model='a')}
        context = _TestJobContext()
        context.job_id, context.cancel_requested = uuid4(), False
        with pytest.raises(JobExecutionError) as failure:
            await pool.configure_memory_policy(context, payload)
        assert failure.value.detail.code == 'runtime_busy'
        assert not instance.backend.calls and instance.backend.limit == 2 * GIB
        assert pool._memory_configuration_job is None
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())

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
