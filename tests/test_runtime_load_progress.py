import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mfq.server.protocol.models import RuntimeInstanceState
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.state.catalog import ModelCatalog
from tests.test_server_models import IdleBackend, _model


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_load_progress_uses_worker_records_and_never_regresses(tmp_path, transport):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport=transport)
        reader = asyncio.StreamReader()
        reader.feed_data(b"ordinary log\n"
                         b"mfq_load_progress completed=2 total=10\n"
                         b"mfq_load_progress completed=1 total=10\n"
                         b"mfq_load_progress completed=11 total=10\n"
                         b"mfq_load_progress completed=0 total=0\n"
                         b"mfq_load_progress completed=5 total=10\n"
                         b"mfq_load_progress stage=finalizing\n"
                         b"mfq_load_progress completed=10 total=10\n"
                         b"mfq_load_progress stage=compiling completed=0 total=12\n"
                         b"mfq_load_progress stage=compiling completed=6 total=12\n"
                         b"mfq_load_progress stage=compiling completed=5 total=12\n"
                         b"mfq_load_progress stage=compiling completed=13 total=12\n"
                         b"mfq_load_progress stage=compiling completed=0 total=0\n"
                         b"mfq_load_progress stage=compiling completed=12 total=12\n"
                         b"mfq_load_progress stage=warming completed=0 total=13\n"
                         b"mfq_load_progress stage=warming completed=6 total=13\n"
                         b"mfq_load_progress stage=warming completed=5 total=13\n"
                         b"mfq_load_progress stage=warming completed=14 total=13\n"
                         b"mfq_load_progress stage=warming completed=0 total=0\n"
                         b"mfq_load_progress stage=warming completed=13 total=13\n")
        reader.feed_eof()
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path),
                            process=SimpleNamespace(stdout=reader, stderr=reader),
                            backend=IdleBackend(), port=0, context_size=4096)
        values = []
        logs = []

        async def progress(value, **options):
            values.append((value, options.get("data")))

        async def log(message, **options):
            logs.append(message)

        context = SimpleNamespace(progress=progress, log=log)
        await pool._pump_output(instance, context)
        assert [value for value, _ in values] == pytest.approx([0.20, 0.47, 0.94, 0.945, 0.95, 0.95 + 0.04 * 6 / 13, 0.99])
        assert values[0][1] == {"phase": "weights", "completed": 2, "total": 10}
        assert values[2][1] == {"phase": "finalizing"}
        assert values[4][1] == {"phase": "compiling", "completed": 12, "total": 12}
        assert values[-1][1] == {"phase": "warming", "completed": 13, "total": 13}
        assert "ordinary log" in logs
        instance.state = RuntimeInstanceState.READY
        reader = asyncio.StreamReader()
        reader.feed_data(b"mfq_load_progress stage=finalizing\n")
        reader.feed_eof()
        instance.process = SimpleNamespace(stdout=reader, stderr=reader)
        await pool._pump_output(instance, context)
        assert len(values) == 7

    asyncio.run(run())
