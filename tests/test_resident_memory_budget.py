import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mfq.server.protocol.models import ModelLoadRequest, RuntimeInstanceState, RuntimeMemoryPolicy, RuntimeMemoryResources
from mfq.server.runtime.client import BackendError
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.state.catalog import ModelCatalog
from tests.test_runtime_memory_policy import GIB, artifact
from tests.test_server_models import _TestJobContext


class BudgetBackend:
    def __init__(self, dense, experts, kv, prefix, limit, events):
        self.dense, self.experts, self.kv, self.prefix, self.limit = dense, experts, kv, prefix, limit
        self.events = events
        self.fail = False

    @property
    def used(self):
        return self.dense + self.experts + self.kv + self.prefix

    async def trim_runtime_cache(self, target):
        self.events.append(('prefix', target))
        self.prefix = min(self.prefix, target)
        return {'status': 'ok'}

    async def set_resident_memory_budget(self, target):
        self.events.append(('budget', target))
        if self.fail:
            raise RuntimeError('injected native failure')
        room = target - min(256 << 20, target // 20)
        self.prefix -= min(self.prefix, max(0, self.used - room))
        self.experts -= min(self.experts, max(0, self.used - room))
        if self.used > room:
            raise RuntimeError('live KV does not fit')
        self.limit = target
        return {'status': 'ok'}

    async def runtime_status(self):
        return {'resident_memory_dynamic_budget': 1, 'resident_memory_budget_bytes': self.limit,
            'resident_memory_used_bytes': self.used, 'resident_reclaimable_weight_bytes': self.experts,
            'resident_weight_bytes': self.dense + self.experts, 'kv_cache_bytes': self.kv,
            'prefix_cache_hot_bytes': self.prefix}


def instance(tmp_path, name, *, dense=GIB, experts=3 * GIB, kv=GIB, prefix=GIB, limit=8 * GIB, events=None):
    backend = BudgetBackend(dense, experts, kv, prefix, limit, events if events is not None else [])
    return SimpleNamespace(id=uuid4(), artifact=artifact(tmp_path, name, 8, 6), backend=backend, process=None,
        state=RuntimeInstanceState.READY, active_requests=0, queued_requests=0, control_leases=0,
        resident_dynamic_budget=True, resident_budget_limit=limit, resident_memory_used=backend.used, prefix_pending_writes=0,
        reclaimable_weight_bytes=experts, prefix_dynamic_budget=True, prefix_pressure_bytes=0,
        memory=RuntimeMemoryResources(prefix_cache_bytes=prefix), started_at=datetime.now(timezone.utc), last_used_at=None,
        context_size=4096, pinned=True, idle_ttl_seconds=None)


def test_total_budget_reclaims_all_prefixes_before_lru_experts(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=20 * GIB)
        pool.memory_policy = RuntimeMemoryPolicy(total_limit_bytes=8 * GIB)
        events = []
        old, new = instance(tmp_path, 'old', events=events), instance(tmp_path, 'new', events=events)
        old.started_at -= timedelta(days=1)
        pool._instances = {old.id: old, new.id: new}
        await pool._rebalance_resident_budget(new.id)
        assert [event[0] for event in events[:2]] == ['prefix', 'prefix']
        assert old.backend.prefix == new.backend.prefix == 0
        assert old.backend.experts < new.backend.experts
        assert sum(item.backend.limit for item in (old, new)) == 8 * GIB
        assert old.backend.kv == new.backend.kv == GIB
        assert old.control_leases == new.control_leases == 0
        assert pool._claim_over_budget_instances_for_unload_locked(memory_ceiling=1) == []
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_busy_worker_capacity_is_reserved_and_not_mutated(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=12 * GIB)
        busy, idle = instance(tmp_path, 'busy', limit=8 * GIB), instance(tmp_path, 'idle')
        busy.state, busy.active_requests = RuntimeInstanceState.BUSY, 1
        pool._instances = {busy.id: busy, idle.id: idle}
        await pool._rebalance_resident_budget(idle.id)
        assert busy.backend.events == []
        assert idle.backend.limit == 4 * GIB
        assert busy.backend.limit + idle.backend.limit == 12 * GIB
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_irreducible_budget_failure_does_not_modify_workers(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=20 * GIB)
        pool.memory_policy = RuntimeMemoryPolicy(total_limit_bytes=3 * GIB)
        a, b = instance(tmp_path, 'a'), instance(tmp_path, 'b')
        pool._instances = {a.id: a, b.id: b}
        with pytest.raises(BackendError) as error:
            await pool._rebalance_resident_budget()
        assert error.value.status_code == 413
        assert a.backend.events == b.backend.events == []
        assert a.control_leases == b.control_leases == 0
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('readers', [0, 3])
def test_budget_configuration_preserves_process_and_live_context(tmp_path, readers):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=20 * GIB)
        a = instance(tmp_path, 'a', dense=GIB, experts=6 * GIB, prefix=0)
        a.control_leases = a.read_control_leases = readers
        a.artifact = artifact(tmp_path, 'a', 8, 7)
        pool._instances = {a.id: a}
        pool._load_requests = {'a': ModelLoadRequest(model='a')}
        context = _TestJobContext()
        context.job_id, context.cancel_requested = uuid4(), False
        result = await pool.configure_memory_policy(context, {'total_limit_bytes': 6 * GIB})
        assert result['models_replanned'] == 0
        assert pool._instances[a.id] is a and a.context_size == 4096 and a.backend.kv == GIB
        assert a.backend.used < 6 * GIB
        assert a.control_leases == a.read_control_leases == readers
        assert pool.memory_policy_status()['effective_total_limit_bytes'] == 6 * GIB
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_native_failure_restores_previously_applied_caps(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=20 * GIB)
        pool.memory_policy = RuntimeMemoryPolicy(total_limit_bytes=10 * GIB)
        a, b = instance(tmp_path, 'a'), instance(tmp_path, 'b')
        b.backend.fail = True
        a.started_at -= timedelta(days=1)
        pool._instances = {a.id: a, b.id: b}
        with pytest.raises(RuntimeError):
            await pool._rebalance_resident_budget(b.id)
        assert a.backend.limit == a.resident_budget_limit == 8 * GIB
        assert a.control_leases == b.control_leases == 0
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('activation_state', [RuntimeInstanceState.READY, RuntimeInstanceState.LOADING])
def test_unsupported_runtime_cannot_claim_manual_total_enforcement(tmp_path, activation_state):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=20 * GIB)
        a = instance(tmp_path, 'a')
        a.resident_dynamic_budget = False
        a.state = activation_state
        a.control_leases = int(activation_state == RuntimeInstanceState.LOADING)
        pool._instances = {a.id: a}
        pool.memory_policy = RuntimeMemoryPolicy(total_limit_bytes=10 * GIB)
        with pytest.raises(BackendError) as error:
            await pool._rebalance_resident_budget(activating=a if a.state == RuntimeInstanceState.LOADING else None)
        assert error.value.status_code == 501 and a.backend.events == []
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_load_reservation_is_shared_and_passed_to_native(tmp_path, monkeypatch):
    async def run():
        monkeypatch.setattr('mfq.server.runtime.runtime_pool.native_tokenizer_arguments', lambda *_: [])
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=8 * GIB)
        await pool._rebalance_resident_budget(additional_bytes=6 * GIB, prospective_model='a')
        _, environment = pool._launch_configuration(artifact(tmp_path, 'a', 5), ModelLoadRequest(model='a'), port=0)
        assert environment['MFQ_SERVER_RESIDENT_BUDGET_BYTES'] == str(8 * GIB)
        with pytest.raises(BackendError):
            await pool._rebalance_resident_budget(additional_bytes=3 * GIB, prospective_model='b')
        assert pool._resident_load_limits == {'a': 8 * GIB}
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('readers', [0, 2])
@pytest.mark.parametrize('activation_state', [RuntimeInstanceState.READY, RuntimeInstanceState.LOADING])
def test_activation_rebalances_its_own_load_lease_without_releasing_it(tmp_path, readers, activation_state):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=8 * GIB)
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        reserve = 256 << 20
        old = instance(tmp_path, 'old', dense=GIB + (16 << 20), experts=0, kv=0, prefix=0, limit=GIB + reserve)
        new = instance(tmp_path, 'new', dense=GIB, experts=0, kv=0, prefix=0, limit=7 * GIB - reserve)
        new.control_leases = 1 + readers
        new.read_control_leases = readers
        new.state = activation_state
        pool._instances = {old.id: old, new.id: new}
        with pytest.raises(BackendError):
            await pool._rebalance_resident_budget(new.id)
        assert old.backend.events == new.backend.events == []
        await pool._rebalance_resident_budget(new.id, activating=new)
        assert old.backend.limit + new.backend.limit == 8 * GIB
        assert old.backend.limit > GIB + reserve
        assert new.backend.limit < 7 * GIB - reserve
        assert old.backend.used == GIB + (16 << 20) and new.backend.used == GIB
        assert old.control_leases == 0 and new.control_leases == 1 + readers
        assert new.read_control_leases == readers
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('blocked', ['additional_control', 'active_request', 'queued_request'])
def test_activation_does_not_rebalance_a_worker_with_another_lease_or_request(tmp_path, blocked):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=8 * GIB)
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        new = instance(tmp_path, 'new', dense=GIB, experts=0, kv=0, prefix=0, limit=4 * GIB)
        new.control_leases = 1
        if blocked == 'additional_control':
            new.control_leases += 1
        elif blocked == 'active_request':
            new.active_requests = 1
        else:
            new.queued_requests = 1
        pool._instances = {new.id: new}
        leases = new.control_leases
        assert await pool._rebalance_resident_budget(new.id, activating=new) is None
        assert not new.backend.events and new.control_leases == leases
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('readers', [1, 3])
@pytest.mark.parametrize('exclusive', [False, True])
def test_new_load_can_reclaim_idle_capacity_from_read_only_monitoring(tmp_path, readers, exclusive):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', max_runtime_memory_bytes=8 * GIB)
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        old = instance(tmp_path, 'old', dense=GIB, experts=0, kv=0, prefix=0, limit=8 * GIB)
        old.read_control_leases = readers
        old.control_leases = readers + int(exclusive)
        pool._instances = {old.id: old}
        if exclusive:
            with pytest.raises(BackendError):
                await pool._rebalance_resident_budget(additional_bytes=GIB, prospective_model='new')
            assert not old.backend.events and not pool._resident_load_limits
        else:
            incoming = await pool._rebalance_resident_budget(additional_bytes=GIB, prospective_model='new')
            assert old.backend.limit + incoming == 8 * GIB
            assert incoming > GIB and old.backend.used == GIB
            assert pool._resident_load_limits == {'new': incoming}
        assert old.control_leases == readers + int(exclusive) and old.read_control_leases == readers
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())
