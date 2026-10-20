import asyncio
from datetime import timedelta

from mfq.server.protocol.models import RuntimeMemoryPolicy
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.state.catalog import ModelCatalog
from tests.test_runtime_memory_policy import CacheBackend, cached_instance, GIB


class RamBackend(CacheBackend):
    def __init__(self):
        super().__init__(0, GIB)
        self.ram, self.ram_limit = GIB, GIB

    async def set_prefix_cache_budget(self, target, disk_target_bytes=None, ram_target_bytes=None):
        result = await super().set_prefix_cache_budget(target, disk_target_bytes)
        if ram_target_bytes is not None:
            self.ram_limit = ram_target_bytes
            self.ram = min(self.ram, self.ram_limit)
        return result

    async def runtime_status(self):
        return {**await super().runtime_status(), 'prefix_cache_ram_bytes': self.ram,
            'prefix_cache_ram_max_bytes': self.ram_limit, 'prefix_cache_ram_blocks': int(self.ram > 0)}


def test_ram_prefixes_yield_to_pressure_and_model_loads(tmp_path, monkeypatch):
    monkeypatch.setattr('mfq.server.runtime.runtime_pool.memory_topology', lambda: {'memory_architecture': 'discrete'})
    available = [4 * GIB]
    monkeypatch.setattr('mfq.server.runtime.runtime_pool.host_memory_capacity', lambda: (64 * GIB, available[0]))

    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', backend='cuda')
        pool.memory_policy = RuntimeMemoryPolicy(prefix_ram_limit_bytes=2 * GIB)
        items = [cached_instance(tmp_path, name, 0, GIB) for name in ('a', 'b')]
        items[0].last_used_at = items[1].started_at - timedelta(seconds=1)
        items[1].last_used_at = items[1].started_at
        for item in items:
            item.backend = RamBackend()
            item.memory = pool._memory_resources(await item.backend.runtime_status())
        pool._instances = {item.id: item for item in items}
        await pool._rebalance_prefix_cache_budget()
        assert sum(item.backend.ram_limit for item in items) == 2 * GIB
        available[0] = 256 << 20
        await pool._rebalance_prefix_cache_budget()
        assert sum(item.backend.ram_limit for item in items) == (2 * GIB) - (256 << 20)
        assert items[0].backend.ram_limit == (768 << 20)
        assert items[1].backend.ram_limit == GIB  # Most recently used model keeps its blocks.
        await pool._rebalance_prefix_cache_budget(reclaim_ram=True)
        assert all(item.backend.ram == item.backend.ram_limit == 0 for item in items)
        available[0] = 4 * GIB
        await pool._rebalance_prefix_cache_budget()
        assert sum(item.backend.ram_limit for item in items) == 2 * GIB
        assert items[1].backend.ram_limit == 2 * GIB
        assert all(item.control_leases == 0 for item in items)
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_ram_prefixes_are_not_counted_in_vram():
    memory = RuntimePool._memory_resources({'prefix_cache_hot_bytes': 100,
        'kv_cache_bytes': 200, 'prefix_cache_ram_bytes': 300, 'prefix_cache_ram_max_bytes': 400,
        'prefix_cache_ram_blocks': 3})
    assert memory.kv_bytes == 300
    assert memory.prefix_cache_bytes == 100
    assert memory.prefix_cache_ram_bytes == 300
    assert memory.prefix_cache_ram_limit_bytes == 400
    assert memory.prefix_cache_ram_blocks == 3
    assert RuntimePool._memory_resources({}, memory) == memory
