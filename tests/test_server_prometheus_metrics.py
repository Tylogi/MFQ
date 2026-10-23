"""Prometheus exposition for runtime and request telemetry."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.api.auth import ApiKeyManager
from mfq.server.protocol.models import (
    CreateApiKeyRequest,
    ModelArtifactResource,
    ResponsePerformance,
    RuntimeInstanceState,
    SamplingParams,
    TokenUsage,
)
from mfq.server.runtime.backend import BackendDelta, BackendError
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.services.metrics import (
    PROMETHEUS_CONTENT_TYPE,
    Histogram,
    RequestMetrics,
    RuntimeMetricsSnapshot,
    render_prometheus,
)
from mfq.server.services.service import ServerService
from mfq.server.state.catalog import DiscoveredModel, ModelCatalog
from mfq.server.state.storage import SessionStore

_SAMPLE_LINE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})? (?P<value>[^ ]+)$'
)
_LABEL = re.compile(r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:[^"\\]|\\.)*)"')
_SUFFIXES = ("_bucket", "_sum", "_count")


def _unescape_label_value(value: str) -> str:
    """Decode the text format escapes so tests compare real label values."""

    decoded: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            escaped = value[index + 1]
            decoded.append({"n": "\n", "\\": "\\", '"': '"'}.get(escaped, "\\" + escaped))
            index += 2
            continue
        decoded.append(char)
        index += 1
    return "".join(decoded)


def parse_exposition(text: str) -> dict[str, dict[str, Any]]:
    """Parse a scrape and assert the text format rules the exporter promises.

    Returns one entry per metric family with its declared type, samples, and
    parsed labels. HELP/TYPE pairs, sample syntax, and cumulative histogram
    buckets are validated here so every test that scrapes also checks them.
    """

    families: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in text.splitlines():
        assert line == line.strip() and line, f"blank, padded, or empty line: {line!r}"
        if line.startswith("#"):
            parts = line.split(" ", 3)
            assert len(parts) == 4 and parts[0] == "#" and parts[1] in {"HELP", "TYPE"}, line
            kind, name, body = parts[1], parts[2], parts[3]
            family = families.setdefault(name, {"help": None, "type": None, "samples": []})
            assert family[kind.lower()] is None, f"duplicate {kind} for {name}"
            family[kind.lower()] = body
            current = name
            continue
        matched = _SAMPLE_LINE.match(line)
        assert matched is not None, f"invalid sample line: {line!r}"
        assert current is not None, "sample before any HELP/TYPE"
        name = matched.group("name")
        base = next((name[: -len(suffix)] for suffix in _SUFFIXES if name.endswith(suffix)), name)
        if base != current:
            assert base in families, f"sample {name} has no HELP/TYPE"
            current = base
        labels = {
            label.group("name"): _unescape_label_value(label.group("value"))
            for label in _LABEL.finditer(matched.group("labels") or "")
        }
        value = float(matched.group("value"))
        assert value == value and abs(value) != float("inf"), line
        families[base]["samples"].append(
            {"name": name, "labels": labels, "value": value, "text": matched.group("value")}
        )

    for name, family in families.items():
        assert family["help"] is not None and family["type"] is not None, f"{name} lacks HELP or TYPE"
        assert family["type"] in {"counter", "gauge", "histogram"}, family["type"]
        assert family["samples"], f"{name} was declared without samples"
        if family["type"] != "histogram":
            assert all(sample["name"] == name for sample in family["samples"]), name
            continue
        _check_histogram(name, family["samples"])
    return families


def _check_histogram(name: str, samples: list[dict[str, Any]]) -> None:
    """Buckets must be monotonic and cumulative, ending at +Inf."""

    groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for sample in samples:
        group = tuple(sorted((key, value) for key, value in sample["labels"].items() if key != "le"))
        groups.setdefault(group, []).append(sample)
    for group in groups.values():
        buckets = [sample for sample in group if sample["name"] == f"{name}_bucket"]
        assert buckets, f"{name} has no buckets"
        assert buckets[-1]["labels"].get("le") == "+Inf", f"{name} has no +Inf bucket"
        previous = -1.0
        for bucket in buckets:
            assert bucket["value"] >= previous, f"{name} buckets are not cumulative"
            previous = bucket["value"]
        assert {sample["name"] for sample in group} == {f"{name}_bucket", f"{name}_sum", f"{name}_count"}
        count = next(sample for sample in group if sample["name"] == f"{name}_count")
        assert buckets[-1]["value"] == count["value"], f"{name} +Inf bucket disagrees with _count"


def family_samples(families: dict[str, dict[str, Any]], name: str, **labels: str) -> list[dict[str, Any]]:
    """Return the samples named ``name``; histogram suffixes share one family."""

    base = next((name[: -len(suffix)] for suffix in _SUFFIXES if name.endswith(suffix)), name)
    family = families.get(base)
    if family is None:
        return []
    return [
        sample
        for sample in family["samples"]
        if sample["name"] == name
        and all(sample["labels"].get(key) == value for key, value in labels.items())
    ]


def sample_value(families: dict[str, dict[str, Any]], name: str, **labels: str) -> float:
    samples = family_samples(families, name, **labels)
    assert len(samples) == 1, f"{name}{labels} matched {len(samples)} samples"
    return samples[0]["value"]


def _artifact(name: str) -> DiscoveredModel:
    return DiscoveredModel(
        resource=ModelArtifactResource(
            id="0" * 32,
            name=name,
            architecture="qwen35",
            format="mfq",
            shard_count=1,
            total_bytes=1,
            tensor_count=1,
            record_count=1,
            complete=True,
            loadable=True,
            modified_at=datetime.now(timezone.utc),
        ),
        path=Path("model.mfq"),
    )


class _FakeProcess:
    returncode: int | None = None
    # The resident-memory probe is skipped for pid 0, keeping tests off ``ps``.
    pid = 0


class _FakeBackend:
    def __init__(self, *, mtp: bool = False, failure: str | None = None) -> None:
        self.mtp = mtp
        self.failure = failure
        self.status_calls = 0

    async def stream(self, **_kwargs: Any):
        if self.failure is not None:
            raise BackendError(self.failure, "backend failed")
        await asyncio.sleep(0)
        yield BackendDelta(
            backend_request_id="request-1",
            content_delta="a",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            performance=ResponsePerformance(
                prefill_tokens=11,
                ttft_ms=40.0,
                prefill_ms=20.0,
                prefill_tps=550.0,
                decode_ms=30.0,
                decode_tps=100.0,
                generation_ms=50.0,
                generation_tps=300.0,
                mtp_drafted_tokens=8 if self.mtp else 0,
                mtp_accepted_tokens=6 if self.mtp else 0,
                sampling=SamplingParams(),
            ),
        )

    async def runtime_status(self) -> dict[str, Any]:
        self.status_calls += 1
        return {"prefix_cache_hot_bytes": 1024, "prefix_cache_max_bytes": 4096}

    async def aclose(self) -> None:
        return None


def _instance(
    *,
    model: str = "model",
    capacity: int = 1,
    backend: _FakeBackend | None = None,
) -> _Runtime:
    return _Runtime(
        id=uuid4(),
        artifact=_artifact(model),
        process=_FakeProcess(),
        backend=backend or _FakeBackend(),
        port=43123,
        context_size=4096,
        state=RuntimeInstanceState.READY,
        request_slots=asyncio.Semaphore(capacity),
        request_capacity=capacity,
        batch_capacity=capacity,
    )


def _pool(monkeypatch: pytest.MonkeyPatch, *, queue_timeout: float = 0.0, max_queued: int = 8) -> RuntimePool:
    pool = RuntimePool(
        ModelCatalog([]),
        Path("mfq-runtime"),
        backend="cuda",
        max_requests_per_instance=4,
        max_queued_requests_per_instance=max_queued,
        queue_timeout_seconds=queue_timeout,
    )

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pool, "_rebalance_prefix_cache_budget", noop)
    monkeypatch.setattr(pool, "_rebalance_resident_budget", noop)
    return pool


def _route(pool: RuntimePool, instance: _Runtime) -> None:
    pool._instances[instance.id] = instance

    async def resolved(_model: str, *, session_id: Any = None) -> _Runtime:
        return instance

    pool._select = resolved  # type: ignore[method-assign]


async def _drain(pool: RuntimePool, model: str = "model") -> None:
    async for _delta in pool.stream(model=model, messages=[], sampling=SamplingParams()):  # type: ignore[arg-type]
        pass


def _scrape(instances: list[RuntimeMetricsSnapshot]) -> dict[str, dict[str, Any]]:
    return parse_exposition(render_prometheus(instances))


def test_exposition_declares_each_family_with_monotonic_buckets() -> None:
    metrics = RequestMetrics()
    metrics.record_request(
        "ok",
        ResponsePerformance(
            prefill_tokens=1,
            ttft_ms=1500.0,
            prefill_ms=1000.0,
            prefill_tps=1.0,
            decode_ms=10.0,
            decode_tps=3.0,
            generation_ms=1010.0,
            generation_tps=3.0,
            mtp_drafted_tokens=4,
            mtp_accepted_tokens=2,
            sampling=SamplingParams(),
        ),
        TokenUsage(prompt_tokens=5, completion_tokens=6, total_tokens=11),
    )
    metrics.observe_queue_wait(0.02)
    families = _scrape([RuntimeMetricsSnapshot(
        instance="instance-a",
        model="model-a",
        request_capacity=4,
        batch_capacity=2,
        active_requests=1,
        queued_requests=0,
        rejected_requests=3,
        resident_bytes=2048,
        kv_bytes=1024,
        kv_limit_bytes=4096,
        requests=metrics,
    )])

    labels = {"model": "model-a", "instance": "instance-a"}
    assert sample_value(families, "mfq_runtime_batch_capacity", **labels) == 2
    assert sample_value(families, "mfq_runtime_request_capacity", **labels) == 4
    assert sample_value(families, "mfq_runtime_active_requests", **labels) == 1
    assert sample_value(families, "mfq_runtime_rejected_requests_total", **labels) == 3
    assert sample_value(families, "mfq_runtime_resident_bytes", **labels) == 2048
    assert sample_value(families, "mfq_runtime_kv_bytes", **labels) == 1024
    assert sample_value(families, "mfq_runtime_kv_usage_ratio", **labels) == 0.25
    assert sample_value(families, "mfq_runtime_mtp_acceptance_ratio", **labels) == 0.5
    assert sample_value(families, "mfq_requests_total", result="ok", **labels) == 1
    assert sample_value(families, "mfq_request_prompt_tokens_total", **labels) == 5
    assert sample_value(families, "mfq_request_completion_tokens_total", **labels) == 6
    assert sample_value(families, "mfq_request_ttft_seconds_count", **labels) == 1
    assert sample_value(families, "mfq_request_prefill_tokens_per_second_count", **labels) == 1
    assert sample_value(families, "mfq_request_decode_tokens_per_second_count", **labels) == 1
    assert sample_value(families, "mfq_runtime_queue_wait_seconds_count", **labels) == 1
    # Every declared request result is exported from the first scrape so rates
    # work without waiting for each outcome to happen once.
    assert {sample["labels"]["result"] for sample in families["mfq_requests_total"]["samples"]} == {
        "ok", "error", "cancelled", "timeout",
    }

def test_labels_are_escaped_and_series_follow_each_runtime() -> None:
    first = _instance(model='qwen "quoted"\nline', capacity=2)
    second = _instance(model="other-model", capacity=4)
    families = _scrape([
        RuntimePool._metrics_snapshot_of(first),
        RuntimePool._metrics_snapshot_of(second),
    ])

    models = {
        (sample["labels"]["model"], sample["labels"]["instance"])
        for sample in families["mfq_runtime_request_capacity"]["samples"]
    }
    assert models == {(first.artifact.resource.name, str(first.id)), (second.artifact.resource.name, str(second.id))}
    assert sample_value(
        families, "mfq_runtime_request_capacity",
        model=first.artifact.resource.name, instance=str(first.id),
    ) == 2
    assert sample_value(
        families, "mfq_runtime_request_capacity",
        model="other-model", instance=str(second.id),
    ) == 4


def test_mtp_ratio_is_absent_when_the_predictor_never_ran() -> None:
    instance = _instance()
    families = _scrape([RuntimePool._metrics_snapshot_of(instance)])
    assert "mfq_runtime_mtp_acceptance_ratio" not in families
    instance.request_metrics.record_request("ok", ResponsePerformance(
        prefill_tokens=1, ttft_ms=1.0, prefill_ms=1.0, prefill_tps=1.0,
        decode_ms=1.0, decode_tps=1.0, generation_ms=2.0, generation_tps=2.0,
        sampling=SamplingParams(),
    ))
    families = _scrape([RuntimePool._metrics_snapshot_of(instance)])
    assert "mfq_runtime_mtp_acceptance_ratio" not in families


def test_scrape_counts_requests_and_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        pool = _pool(monkeypatch)
        instance = _instance(backend=_FakeBackend(mtp=True))
        _route(pool, instance)

        before = _scrape(await pool.metrics_snapshot())
        labels = {"model": "model", "instance": str(instance.id)}
        assert sample_value(before, "mfq_requests_total", result="ok", **labels) == 0

        await _drain(pool)
        after = _scrape(await pool.metrics_snapshot())
        assert sample_value(after, "mfq_requests_total", result="ok", **labels) == 1
        assert sample_value(after, "mfq_request_prompt_tokens_total", **labels) == 11
        assert sample_value(after, "mfq_request_completion_tokens_total", **labels) == 4
        assert sample_value(after, "mfq_request_ttft_seconds_sum", **labels) == pytest.approx(0.04)
        assert sample_value(after, "mfq_request_decode_tokens_per_second_sum", **labels) == 100.0
        assert sample_value(after, "mfq_runtime_mtp_acceptance_ratio", **labels) == 0.75
        assert sample_value(after, "mfq_runtime_kv_usage_ratio", **labels) == 0.25
        assert sample_value(after, "mfq_runtime_active_requests", **labels) == 0

    asyncio.run(run())


def test_failed_requests_change_the_result_label(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        pool = _pool(monkeypatch)
        instance = _instance(backend=_FakeBackend(failure="backend_oom"))
        _route(pool, instance)
        with pytest.raises(BackendError):
            await _drain(pool)
        families = _scrape(await pool.metrics_snapshot())
        labels = {"model": "model", "instance": str(instance.id)}
        assert sample_value(families, "mfq_requests_total", result="error", **labels) == 1
        assert sample_value(families, "mfq_requests_total", result="timeout", **labels) == 0

    asyncio.run(run())


def test_queue_timeout_and_rejection_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        pool = _pool(monkeypatch, queue_timeout=0.05, max_queued=0)
        gate = asyncio.Event()

        class SlowBackend(_FakeBackend):
            async def stream(self, **_kwargs: Any):
                await gate.wait()
                async for delta in super().stream():
                    yield delta

        instance = _instance(backend=SlowBackend())
        _route(pool, instance)
        first = asyncio.create_task(_drain(pool))
        await asyncio.sleep(0.02)
        with pytest.raises(BackendError) as failure:
            await _drain(pool)
        assert failure.value.code == "runtime_queue_full"
        families = _scrape(await pool.metrics_snapshot())
        labels = {"model": "model", "instance": str(instance.id)}
        assert sample_value(families, "mfq_runtime_rejected_requests_total", **labels) == 1
        assert sample_value(families, "mfq_runtime_queued_requests", **labels) == 0
        gate.set()
        await first

        timing_out = _pool(monkeypatch, queue_timeout=0.05)
        slow_gate = asyncio.Event()

        class GatedBackend(_FakeBackend):
            async def stream(self, **_kwargs: Any):
                await slow_gate.wait()
                async for delta in super().stream():
                    yield delta

        second = _instance(backend=GatedBackend())
        _route(timing_out, second)
        holder = asyncio.create_task(_drain(timing_out))
        await asyncio.sleep(0.02)
        with pytest.raises(BackendError) as timeout:
            await _drain(timing_out)
        assert timeout.value.code == "runtime_queue_timeout"
        families = _scrape(await timing_out.metrics_snapshot())
        labels = {"model": "model", "instance": str(second.id)}
        assert sample_value(families, "mfq_requests_total", result="timeout", **labels) == 1
        assert sample_value(families, "mfq_runtime_queue_wait_seconds_count", **labels) == 1
        slow_gate.set()
        await holder

    asyncio.run(run())


def test_cancelled_stream_is_counted_as_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        pool = _pool(monkeypatch)
        instance = _instance()
        _route(pool, instance)
        stream = pool.stream(model="model", messages=[], sampling=SamplingParams())  # type: ignore[arg-type]
        await anext(stream)
        await stream.aclose()
        families = _scrape(await pool.metrics_snapshot())
        labels = {"model": "model", "instance": str(instance.id)}
        assert sample_value(families, "mfq_requests_total", result="cancelled", **labels) == 1
        assert sample_value(families, "mfq_runtime_active_requests", **labels) == 0

    asyncio.run(run())


def test_restarted_runtime_gets_a_new_instance_label(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        pool = _pool(monkeypatch)
        first = _instance()
        _route(pool, first)
        await _drain(pool)
        pool._instances.clear()
        second = _instance()
        _route(pool, second)
        await _drain(pool)

        families = _scrape(await pool.metrics_snapshot())
        samples = families["mfq_requests_total"]["samples"]
        # Only the live instance is exported; its counters start from its own load.
        assert {sample["labels"]["instance"] for sample in samples} == {str(second.id)}
        assert sample_value(
            families, "mfq_requests_total", model="model", instance=str(second.id), result="ok",
        ) == 1

    asyncio.run(run())


def test_scrape_reuses_cached_usage_and_never_probes_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        pool = _pool(monkeypatch)
        backend = _FakeBackend()
        instance = _instance(backend=backend)
        _route(pool, instance)

        def fail_nvml() -> list[dict[str, Any]]:
            raise AssertionError("a scrape must not initialize NVML")

        monkeypatch.setattr("mfq.server.runtime.runtime_pool.gpu_memory_status", fail_nvml)
        await pool.metrics_snapshot()
        await pool.metrics_snapshot()
        assert backend.status_calls == 1

    asyncio.run(run())


class _NullBackend:
    async def stream(self, **_kwargs: Any):
        yield BackendDelta(content_delta="a")


def _app(
    tmp_path: Path,
    instances: list[_Runtime],
    *,
    api_key: str = "",
    api_keys: ApiKeyManager | None = None,
) -> Any:
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda")
    for instance in instances:
        pool._instances[instance.id] = instance
    service = ServerService(SessionStore(tmp_path / "mfq.server.sqlite3"), _NullBackend(),
        runtime_manager=pool)
    return create_app(service, api_key=api_key, api_keys=api_keys)


def test_metrics_endpoint_serves_the_prometheus_text_format(tmp_path: Path) -> None:
    async def run() -> None:
        instance = _instance()
        app = _app(tmp_path, [instance])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert response.headers["content-type"] == PROMETHEUS_CONTENT_TYPE
            families = parse_exposition(response.text)
            assert str(instance.id) in response.text
            assert sample_value(
                families, "mfq_runtime_request_capacity",
                model="model", instance=str(instance.id),
            ) == 1

    asyncio.run(run())


def test_metrics_endpoint_without_runtimes_is_empty_and_valid(tmp_path: Path) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(tmp_path, [])), base_url="http://test",
        ) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert response.text == ""
            assert response.headers["content-type"] == PROMETHEUS_CONTENT_TYPE

    asyncio.run(run())


def test_metrics_endpoint_requires_the_configured_credential(tmp_path: Path) -> None:
    async def run() -> None:
        app = _app(tmp_path, [_instance()], api_key="example-metrics-key")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            denied = await client.get("/metrics")
            assert denied.status_code == 401
            assert "mfq_runtime_request_capacity" not in denied.text
            allowed = await client.get("/metrics", headers={"Authorization": "Bearer example-metrics-key"})
            assert allowed.status_code == 200
            assert "mfq_runtime_request_capacity" in allowed.text

    asyncio.run(run())


def test_metrics_scope_matches_the_runtime_read_routes(tmp_path: Path) -> None:
    async def run() -> None:
        manager = ApiKeyManager(SessionStore(tmp_path / "keys.sqlite3"), "root-secret")
        viewer = manager.create(CreateApiKeyRequest(name="viewer", role="viewer"))
        jobs = manager.create(CreateApiKeyRequest(name="jobs", scopes=["jobs"]))
        app = _app(tmp_path, [_instance()], api_keys=manager)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/metrics")).status_code == 401
            allowed = await client.get("/metrics", headers={"Authorization": f"Bearer {viewer.token}"})
            assert allowed.status_code == 200
            assert allowed.headers["content-type"] == PROMETHEUS_CONTENT_TYPE
            denied = await client.get("/metrics", headers={"Authorization": f"Bearer {jobs.token}"})
            assert denied.status_code == 403

    asyncio.run(run())


def test_histogram_rejects_unsorted_buckets() -> None:
    with pytest.raises(ValueError):
        Histogram((0.5, 0.1))
