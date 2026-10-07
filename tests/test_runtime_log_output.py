from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mfq.server.protocol.models import JobEventType, RuntimeInstanceState
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.services.jobs import JobContext
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import InvalidJobStateError, JobNotFoundError, SessionStore
from tests.test_server_models import IdleBackend, _model


@pytest.mark.parametrize("transport", ["http", "stdio"])
@pytest.mark.parametrize("state", [RuntimeInstanceState.LOADING, RuntimeInstanceState.READY])
def test_oversized_runtime_log_lines_are_truncated_without_stopping_pipe_drain(tmp_path, transport, state):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport=transport, automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        stream = asyncio.StreamReader()
        first = "long-line:" + "x" * 100_000
        stream.feed_data((first + "\nnext-row\n").encode())
        stream.feed_eof()
        process = SimpleNamespace(stdout=stream if transport == "http" else None,
            stderr=stream if transport == "stdio" else None, returncode=None)
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=process, port=0, context_size=512, state=state, request_slots=asyncio.Semaphore(1))
        messages = []

        async def log(message, **options):
            messages.append(message)

        context = SimpleNamespace(log=log)
        await asyncio.wait_for(pool._pump_output(instance, context), 1)
        records = pool.store.list_runtime_logs()
        assert [item.message for item in records] == [first[:4096], "next-row"]
        assert all(item.instance_id == instance.id for item in records)
        assert all(item.fields["model"] == instance.artifact.resource.name for item in records)
        assert all(item.fields["source"] == f"runtime.{'stdout' if transport == 'http' else 'stderr'}" for item in records)
        assert messages == ([first[:4096], "next-row"] if state == RuntimeInstanceState.LOADING else [])
        await pool.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("newline", [True, False])
def test_runtime_output_preserves_utf8_prefix_and_final_unterminated_line(tmp_path, newline):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport="http", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        stream = asyncio.StreamReader()
        message = "汉字🙂" * 20_000
        payload = (message + ("\n" if newline else "")).encode()
        process = SimpleNamespace(stdout=stream, stderr=None, returncode=None)
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=process, port=0, context_size=512, state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1))
        pumping = asyncio.create_task(pool._pump_output(instance, SimpleNamespace()))
        for offset in range(0, len(payload), 16381):
            stream.feed_data(payload[offset:offset + 16381])
            await asyncio.sleep(0)
        stream.feed_eof()
        await asyncio.wait_for(pumping, 1)
        assert [item.message for item in pool.store.list_runtime_logs()] == [message[:4096]]
        await pool.aclose()

    asyncio.run(run())


def test_real_runtime_pipe_drains_a_line_larger_than_the_production_frame_limit(tmp_path):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport="http", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        process = await asyncio.create_subprocess_exec(sys.executable, "-c",
            "import sys; out=sys.stdout.buffer; out.write(b'log-start:'); chunk=b'x'*(1<<20); "
            "[out.write(chunk) for _ in range(96)]; out.write(b'\\nafter-wide-log\\n'); out.flush()",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=64 * 1024 * 1024)
        try:
            instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
                process=process, port=0, context_size=512, state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1))
            await asyncio.wait_for(pool._pump_output(instance, SimpleNamespace()), 10)
            assert await asyncio.wait_for(process.wait(), 3) == 0
            assert [item.message for item in pool.store.list_runtime_logs()] == ["log-start:" + "x" * (4096 - 10), "after-wide-log"]
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                await asyncio.wait_for(process.wait(), 3)
            await pool.aclose()

    asyncio.run(run())


def test_runtime_log_fragments_preserve_loading_progress_and_ignore_blank_lines(tmp_path):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport="http", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        stream = asyncio.StreamReader()
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=SimpleNamespace(stdout=stream, stderr=None), port=0, context_size=512,
            state=RuntimeInstanceState.LOADING, request_slots=asyncio.Semaphore(1))
        messages, progress = [], []

        async def log(message, **options):
            messages.append(message)

        async def report(value, **kwargs):
            progress.append((value, kwargs))

        pumping = asyncio.create_task(pool._pump_output(instance, SimpleNamespace(log=log, progress=report)))
        for part in (b"\n\n" + b"x" * 100_000, b"\nmfq_load_prog", b"ress completed=1 total=2\n",
                b"mfq_load_progress stage=finalizing\n\n"):
            stream.feed_data(part)
            await asyncio.sleep(0)
        stream.feed_eof()
        await asyncio.wait_for(pumping, 1)
        assert [item[0] for item in progress] == pytest.approx([.47, .94])
        assert progress[0][1]["data"] == {"phase": "weights", "completed": 1, "total": 2}
        assert progress[1][1]["data"] == {"phase": "finalizing"}
        assert messages == ["x" * 4096, "mfq_load_progress completed=1 total=2", "mfq_load_progress stage=finalizing"]
        assert len(pool.store.list_runtime_logs()) == 3
        await pool.aclose()

    asyncio.run(run())


def test_incomplete_runtime_log_line_remains_cancellable(tmp_path):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport="http", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        stream = asyncio.StreamReader()
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=SimpleNamespace(stdout=stream, stderr=None), port=0, context_size=512,
            state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1))
        pumping = asyncio.create_task(pool._pump_output(instance, SimpleNamespace()))
        stream.feed_data(b"x" * 100_000)
        await asyncio.sleep(0)
        pumping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pumping, 1)
        assert pool.store.list_runtime_logs() == []
        await pool.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("diagnostic,level", [
    ("mfq-decode-metal: missing MFQ record: model.token_embedding.weight", "error"),
    ("error: missing MFQ record: model.token_embedding.weight", "error"),
    ("Warning: Metal memory residency unavailable: API unavailable", "warning"),
    ("warning: prefix restore error; continuing without cache", "warning"),
    ("common_chat_templates_init: warning: vocab does not have a role token", "warning"),
    ("Loading warning-model tensors", "info"),
    ("mfq-decode-metal: warning: invalid model configuration", "error"),
])
def test_native_diagnostics_keep_the_same_severity_in_runtime_and_load_job_logs(tmp_path, diagnostic, level):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport="stdio", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        job = pool.store.create_job("model.load", {"model": "model"})
        pool.store.claim_job(job.id)
        stream = asyncio.StreamReader()
        stream.feed_data(("Loading native model...\n" + diagnostic + "\n").encode())
        stream.feed_eof()
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=SimpleNamespace(stdout=None, stderr=stream), port=0, context_size=512,
            state=RuntimeInstanceState.LOADING, request_slots=asyncio.Semaphore(1))
        await pool._pump_output(instance, JobContext(pool.store, job.id, asyncio.Event()))
        records = pool.store.list_runtime_logs()
        events = [item for item in pool.store.list_job_events(job.id) if item.type == JobEventType.LOG]
        assert [(item.message, item.level.value) for item in records] == [("Loading native model...", "info"), (diagnostic, level)]
        assert [(item.message, item.level.value) for item in events] == [(item.message, item.level.value) for item in records]
        await pool.aclose()

    asyncio.run(run())


def test_stopping_an_exited_runtime_drains_pending_output_before_cancelling_reader(tmp_path):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport="http", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        stream = asyncio.StreamReader()
        stream.feed_data(("Preparing tensor\n" * 200 + "mfq-decode-metal: final-diagnostic\n").encode())
        stream.feed_eof()
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=SimpleNamespace(stdout=stream, stderr=None, returncode=2), port=0, context_size=512,
            state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1))
        instance.output_task = asyncio.create_task(pool._pump_output(instance, SimpleNamespace()))
        await asyncio.sleep(0)
        await asyncio.wait_for(pool._stop_process(instance), 3)
        records = pool.store.list_runtime_logs(limit=1000)
        assert len(records) == 201
        assert records[-1].message == "mfq-decode-metal: final-diagnostic"
        assert records[-1].level.value == "error"
        await pool.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["retiring", "finished", "deleted"])
def test_load_failure_cleanup_keeps_logs_after_progress_or_job_lifetime_ends(tmp_path, phase):
    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport="stdio", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        stream = asyncio.StreamReader()
        lines = ["mfq_load_progress completed=1 total=2", "mfq-decode-metal: final-diagnostic"]
        stream.feed_data(("\n".join(lines) + "\n").encode())
        stream.feed_eof()
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=SimpleNamespace(stdout=None, stderr=stream, returncode=2), port=0, context_size=512,
            state=RuntimeInstanceState.UNLOADING if phase == "retiring" else RuntimeInstanceState.LOADING,
            log_to_load_job=True, request_slots=asyncio.Semaphore(1))
        messages = []

        async def log(message, **options):
            if phase == "deleted":
                raise JobNotFoundError("load job was deleted")
            messages.append(message)

        async def progress(*args, **kwargs):
            assert phase != "retiring"
            raise JobNotFoundError("load job was deleted") if phase == "deleted" else InvalidJobStateError("load job ended")

        await pool._pump_output(instance, SimpleNamespace(log=log, progress=progress))
        assert messages == ([] if phase == "deleted" else lines)
        assert [item.message for item in pool.store.list_runtime_logs()] == lines
        await pool.aclose()

    asyncio.run(run())


def test_runtime_output_drain_has_a_deadline_and_reports_incomplete_output(tmp_path, monkeypatch):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool._RUNTIME_OUTPUT_DRAIN_TIMEOUT_SECONDS", .01)

    async def run():
        path = tmp_path / "model.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", transport="http", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "server.sqlite3")
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=SimpleNamespace(stdout=asyncio.StreamReader(), stderr=None, returncode=0), port=0,
            context_size=512, state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1))
        instance.output_task = asyncio.create_task(pool._pump_output(instance, SimpleNamespace()))
        await asyncio.wait_for(pool._stop_process(instance), 1)
        assert instance.output_task.cancelled()
        records = pool.store.list_runtime_logs()
        assert len(records) == 1 and records[0].level.value == "warning"
        assert "output may be incomplete" in records[0].message
        assert records[0].fields == {"source": "runtime.log-drain", "model": instance.artifact.resource.name}
        await pool.aclose()

    asyncio.run(run())
