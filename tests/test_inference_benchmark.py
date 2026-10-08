import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mfq.server.protocol.models import RuntimeInstanceState, CreateJobRequest
from mfq.server.services.inference_benchmark import InferenceBenchmarkPayload, validate_benchmark_run, aggregate_benchmark_runs, run_inference_benchmark
from mfq.server.services.jobs import JobExecutionError, JobManager
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore


def native_run(mode="decode", prompt=32, output=8, drafted=0, accepted=0):
    return {"metrics": {"prefill_tokens": prompt, "prefill_ms": 10, "decode_ms": 20, "ttft_ms": 12,
        "mtp_used": mode == "mtp", "mtp_drafted_tokens": drafted, "mtp_accepted_tokens": accepted},
        "usage": {"prompt_tokens": prompt, "completion_tokens": output}, "finish_reason": "length"}


def request(**updates):
    return InferenceBenchmarkPayload(instance_id=uuid4(), prompt="natural text", prompt_tokens=32, output_tokens=8, **updates)


def test_aggregation_uses_token_weighted_timings_and_draft_weighted_acceptance():
    req = request()
    a = validate_benchmark_run(native_run("mtp", drafted=2, accepted=1), req, "mtp")
    b = validate_benchmark_run(native_run("mtp", output=4, drafted=10, accepted=9), req, "mtp")
    b["metrics"]["decode_ms"] = 40
    result = aggregate_benchmark_runs([a, b])
    assert result["mtp_decode_tps"] == pytest.approx(10 * 1000 / 60)
    assert result["mtp_acceptance_rate"] == pytest.approx(10 / 12)
    assert result["mtp_prefill_tps"] == 3200


@pytest.mark.parametrize("mutation,mode", [
    (lambda x: x["metrics"].update(prefill_tokens=0), "decode"),
    (lambda x: x["metrics"].update(decode_ms=float("nan")), "decode"),
    (lambda x: x["metrics"].update(mtp_used=True), "decode"),
    (lambda x: x["metrics"].update(mtp_accepted_tokens=9), "decode"),
    (lambda x: x["usage"].update(completion_tokens=0), "decode"),
    (lambda x: None, "mtp"),
])
def test_invalid_or_mislabeled_runs_are_rejected(mutation, mode):
    result = native_run()
    mutation(result)
    with pytest.raises(JobExecutionError):
        validate_benchmark_run(result, request(), mode)


def test_job_warms_each_mode_excludes_warmups_and_preserves_explicit_mtp(tmp_path):
    class Backend:
        def __init__(self): self.payloads = []
        async def benchmark(self, payload):
            self.payloads.append(payload)
            return native_run("mtp" if payload["enable_mtp"] else "decode", drafted=4 if payload["enable_mtp"] else 0, accepted=3 if payload["enable_mtp"] else 0)
    class Pool:
        executable = tmp_path / "runtime"
        backend = "metal"
        leased = False
        engine = Backend()
        @asynccontextmanager
        async def benchmark_runtime(self, instance_id):
            self.leased = True
            try:
                yield SimpleNamespace(id=instance_id, context_size=128, mtp_available=True,
                    artifact=SimpleNamespace(resource=SimpleNamespace(name="model", id="asset"))), self.engine
            finally: self.leased = False
    async def run():
        store = SessionStore(tmp_path / "jobs.sqlite3")
        pool = Pool()
        manager = JobManager(store, {"benchmark.inference": lambda context, payload: run_inference_benchmark(context, payload, pool)})
        req = request(repetitions=2, warmup_runs=1)
        job = await manager.submit(CreateJobRequest(kind="benchmark.inference", payload=req.model_dump(mode="json")))
        for _ in range(100):
            await asyncio.sleep(.01)
            finished = store.get_job(job.id)
            if finished.status.value in {"succeeded", "failed"}: break
        assert finished.status.value == "succeeded", finished.error
        assert not pool.leased
        assert len(pool.engine.payloads) == 6
        assert all(p["mfq_prefix_cache_enabled"] is False and p["mfq_benchmark_prompt_tokens"] == 32 for p in pool.engine.payloads)
        assert [p["enable_mtp"] for p in pool.engine.payloads] == [False, True, True, False, False, True]
        assert len(finished.result["runs"]) == 4
        assert len(finished.result["warmup_runs"]) == 2
        assert finished.result["mtp_acceptance_rate"] == .75
        assert store.list_evaluations()[0].kind == "inference_benchmark"
        await manager.close()
    asyncio.run(run())


def test_runtime_benchmark_lease_blocks_busy_models_and_releases_on_cancel(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime")
        instance = SimpleNamespace(id=uuid4(), state=RuntimeInstanceState.READY, active_requests=0,
            queued_requests=0, control_leases=0, backend=SimpleNamespace(benchmark=lambda _: None), last_used_at=None)
        pool._instances[instance.id] = instance
        with pytest.raises(asyncio.CancelledError):
            async with pool.benchmark_runtime(instance.id):
                assert instance.state == RuntimeInstanceState.BUSY
                assert instance.control_leases == instance.active_requests == 1
                raise asyncio.CancelledError()
        assert instance.state == RuntimeInstanceState.READY
        assert instance.control_leases == instance.active_requests == 0
        instance.queued_requests = 1
        with pytest.raises(Exception, match="idle runtime"):
            async with pool.benchmark_runtime(instance.id): pass
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_benchmark_waits_for_short_status_leases_but_cancellation_leaves_no_reservation(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime")
        instance = SimpleNamespace(id=uuid4(), state=RuntimeInstanceState.READY, active_requests=0,
            queued_requests=0, control_leases=1, backend=SimpleNamespace(benchmark=lambda _: None), last_used_at=None)
        pool._instances[instance.id] = instance
        async def finish_status():
            await asyncio.sleep(.03)
            instance.control_leases -= 1
        status_task = asyncio.create_task(finish_status())
        async with pool.benchmark_runtime(instance.id):
            assert instance.active_requests == instance.control_leases == 1
        await status_task
        assert instance.state == RuntimeInstanceState.READY and instance.control_leases == 0
        instance.control_leases = 1
        async def acquire():
            async with pool.benchmark_runtime(instance.id): raise AssertionError("must not acquire")
        pending = asyncio.create_task(acquire())
        await asyncio.sleep(.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError): await pending
        assert instance.control_leases == 1 and instance.active_requests == 0 and instance.state == RuntimeInstanceState.READY
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("lengths", [[], [0], [1048577], [32, 32], list(range(1, 10))])
def test_invalid_input_matrix_is_rejected(lengths):
    with pytest.raises(ValueError):
        request(prompt_token_lengths=lengths)


def test_multi_length_job_warms_and_records_each_length_without_blending_results(tmp_path):
    class Backend:
        def __init__(self): self.payloads = []
        async def benchmark(self, payload):
            self.payloads.append(payload)
            return native_run("mtp" if payload["enable_mtp"] else "decode", prompt=payload["mfq_benchmark_prompt_tokens"], drafted=2 if payload["enable_mtp"] else 0, accepted=1 if payload["enable_mtp"] else 0)
    class Pool:
        executable = tmp_path / "runtime"
        backend = "metal"
        engine = Backend()
        @asynccontextmanager
        async def benchmark_runtime(self, instance_id):
            yield SimpleNamespace(id=instance_id, context_size=128, mtp_available=True,
                artifact=SimpleNamespace(resource=SimpleNamespace(name="model", id="asset"))), self.engine
    async def run():
        store = SessionStore(tmp_path / "jobs.sqlite3")
        pool = Pool()
        from mfq.server.services.jobs import JobContext
        job = store.create_job("benchmark.inference", {})
        store.claim_job(job.id)
        context = JobContext(store, job.id, asyncio.Event())
        result = await run_inference_benchmark(context, request(prompt_token_lengths=[64, 32], repetitions=1, warmup_runs=1).model_dump(mode="json"), pool)
        assert [row["prompt_tokens"] for row in result["series"]] == [32, 64]
        assert [row["decode_prefill_tps"] for row in result["series"]] == [3200, 6400]
        assert len(result["warmup_runs"]) == len(result["runs"]) == 4
        assert len(pool.engine.payloads) == 8
        evaluation = store.list_evaluations()[0]
        another = await run_inference_benchmark(context, request(prompt_token_lengths=[32, 64], repetitions=1, warmup_runs=1).model_dump(mode="json"), pool)
        assert store.list_evaluations()[0].comparison_key == evaluation.comparison_key
        assert another["series"] == result["series"]
        with pytest.raises(JobExecutionError, match="exceed"):
            await run_inference_benchmark(context, request(prompt_token_lengths=[32, 128]).model_dump(mode="json"), pool)
    asyncio.run(run())
