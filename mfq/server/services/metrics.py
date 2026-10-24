"""Prometheus text exposition for managed runtime and request telemetry.

The server serves ``GET /metrics`` in the Prometheus text format 0.0.4 without
depending on ``prometheus_client``. The runtime pool records request outcomes,
latencies, throughputs and queue waits while it serves, so a scrape only reads
that in-process state plus the runtime snapshot the pool already caches. A
scrape never initializes NVML or reads SQLite, and it reuses the cached snapshot
instead of polling a worker for every scrape.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# The exposition format version is part of the content type; Prometheus does not
# negotiate it, but clients and proxies expect this exact string.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4"

# Upper bounds in seconds. Queue waits are bounded by --queue-timeout, and TTFT
# spans a cold prompt up to a long prefill, so both cover sub-millisecond queue
# hits through multi-minute prefill without unbounded cardinality.
QUEUE_WAIT_SECONDS_BUCKETS = (
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)
TTFT_SECONDS_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
)

# Upper bounds in tokens per second. Prefill is measured over short prompts too,
# where a single token can inflate the rate, so its scale starts low and reaches
# the batched prefill range. Decode stays below the sampler-bound ceiling.
PREFILL_TOKENS_PER_SECOND_BUCKETS = (
    1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0, 50000.0,
)
DECODE_TOKENS_PER_SECOND_BUCKETS = (
    1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0, 240.0, 320.0, 480.0, 640.0, 1000.0,
)

# Every request ends in exactly one of these buckets; ``timeout`` is separate
# from ``error`` so retryable deadline failures stay visible next to rejections.
REQUEST_RESULTS = ("ok", "error", "cancelled", "timeout")


def _is_measurement(value: Any) -> bool:
    """True for finite non-negative numbers, excluding booleans."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


class Histogram:
    """Cumulative observations against fixed upper bounds.

    Observations are counted into one bucket and rendered cumulatively, so
    recording is O(1) and a scrape only adds up the bounded bucket list.
    """

    def __init__(self, buckets: Sequence[float]) -> None:
        if len(buckets) < 1 or any(
            later <= earlier for earlier, later in zip(buckets, buckets[1:], strict=False)
        ):
            raise ValueError("histogram buckets must be strictly increasing")
        self._buckets = tuple(float(bound) for bound in buckets)
        self._counts = [0] * len(self._buckets)
        self._sum = 0.0
        self._count = 0

    def observe(self, value: Any) -> None:
        if not _is_measurement(value):
            return
        measured = float(value)
        self._sum += measured
        self._count += 1
        for index, upper_bound in enumerate(self._buckets):
            if measured <= upper_bound:
                self._counts[index] += 1
                return

    @property
    def count(self) -> int:
        return self._count

    @property
    def total(self) -> float:
        """Sum of all observations; named for its metric suffix, not the builtin."""

        return self._sum

    def cumulative(self) -> Iterator[tuple[float, int]]:
        observed = 0
        for upper_bound, count in zip(self._buckets, self._counts, strict=True):
            observed += count
            yield upper_bound, observed


class RequestMetrics:
    """Counters and histograms recorded for one loaded runtime instance.

    A reload or restart creates a new runtime instance and therefore a new
    object, which is why the ``instance`` label changes and the counters may
    start over; a counter never decreases under a stable set of labels.
    """

    def __init__(self) -> None:
        self.results: dict[str, int] = {result: 0 for result in REQUEST_RESULTS}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.ttft = Histogram(TTFT_SECONDS_BUCKETS)
        self.prefill_rate = Histogram(PREFILL_TOKENS_PER_SECOND_BUCKETS)
        self.decode_rate = Histogram(DECODE_TOKENS_PER_SECOND_BUCKETS)
        self.queue_wait = Histogram(QUEUE_WAIT_SECONDS_BUCKETS)
        self.mtp_drafted_tokens = 0
        self.mtp_accepted_tokens = 0

    def observe_queue_wait(self, seconds: Any) -> None:
        self.queue_wait.observe(seconds)

    def record_request(self, result: str, performance: Any = None, usage: Any = None) -> None:
        """Account for one finished request, including the failed outcomes.

        Telemetry must never fail a served request, so unmeasured values are
        skipped instead of raising.
        """

        self.results[result if result in self.results else "error"] += 1
        if usage is not None:
            self._add_tokens(usage.prompt_tokens, usage.completion_tokens)
        if performance is not None:
            self.ttft.observe(_milliseconds_to_seconds(performance.ttft_ms))
            self.prefill_rate.observe(performance.prefill_tps)
            self.decode_rate.observe(performance.decode_tps)
            self._add_mtp_tokens(performance.mtp_drafted_tokens, performance.mtp_accepted_tokens)

    def _add_tokens(self, prompt_tokens: Any, completion_tokens: Any) -> None:
        if _is_measurement(prompt_tokens):
            self.prompt_tokens += int(prompt_tokens)
        if _is_measurement(completion_tokens):
            self.completion_tokens += int(completion_tokens)

    def _add_mtp_tokens(self, drafted: Any, accepted: Any) -> None:
        if _is_measurement(drafted):
            self.mtp_drafted_tokens += int(drafted)
        if _is_measurement(accepted):
            self.mtp_accepted_tokens += int(accepted)

    @property
    def mtp_acceptance_ratio(self) -> float | None:
        """Accepted draft tokens per drafted token, or None when MTP never ran."""

        return self.mtp_accepted_tokens / self.mtp_drafted_tokens if self.mtp_drafted_tokens else None


def _milliseconds_to_seconds(value: Any) -> Any:
    return value / 1000.0 if _is_measurement(value) else None


@dataclass(frozen=True)
class RuntimeMetricsSnapshot:
    """One serving runtime as reported by the runtime pool for a single scrape."""

    instance: str
    model: str
    request_capacity: int
    batch_capacity: int
    active_requests: int
    queued_requests: int
    rejected_requests: int
    resident_bytes: int | None
    kv_bytes: int | None
    kv_limit_bytes: int | None
    requests: RequestMetrics


class _ExpositionWriter:
    """Minimal prometheus_client-free serializer for the text format 0.0.4."""

    def __init__(self) -> None:
        self._blocks: list[list[str]] = []
        self._block: list[str] | None = None

    def family(self, name: str, kind: str, help_text: str) -> None:
        if self._block is not None:
            self._blocks.append(self._block)
        self._block = [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]

    def sample(self, name: str, labels: Mapping[str, str], value: Any) -> None:
        if self._block is None:
            raise RuntimeError("a metric family must be opened before its samples")
        self._block.append(f"{name}{_format_labels(labels)} {_format_value(value)}")

    def render(self) -> str:
        blocks = [*self._blocks, *([self._block] if self._block is not None else [])]
        # A family without samples (for example memory the worker omits, or MTP
        # on a model without a predictor) is left out instead of emitted empty.
        return "".join(line + "\n" for block in blocks if len(block) > 2 for line in block)


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{name}="{_escape_label_value(value)}"' for name, value in labels.items())
    return "{" + rendered + "}"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _runtime_labels(item: RuntimeMetricsSnapshot) -> dict[str, str]:
    return {"model": item.model, "instance": item.instance}


def render_prometheus(instances: Sequence[RuntimeMetricsSnapshot]) -> str:
    """Render one scrape for every serving runtime; no runtime renders nothing."""

    writer = _ExpositionWriter()
    _write_capacity(writer, instances)
    _write_queue(writer, instances)
    _write_memory(writer, instances)
    _write_mtp(writer, instances)
    _write_requests(writer, instances)
    return writer.render()


def _write_capacity(writer: _ExpositionWriter, instances: Sequence[RuntimeMetricsSnapshot]) -> None:
    writer.family("mfq_runtime_batch_capacity", "gauge",
        "Rows the loaded model batches together, as reported by the worker.")
    for item in instances:
        writer.sample("mfq_runtime_batch_capacity", _runtime_labels(item), item.batch_capacity)
    writer.family("mfq_runtime_request_capacity", "gauge",
        "Concurrent request slots the server admits for the runtime instance.")
    for item in instances:
        writer.sample("mfq_runtime_request_capacity", _runtime_labels(item), item.request_capacity)
    writer.family("mfq_runtime_active_requests", "gauge",
        "Requests holding a runtime slot, including their queue wait.")
    for item in instances:
        writer.sample("mfq_runtime_active_requests", _runtime_labels(item), item.active_requests)
    writer.family("mfq_runtime_queued_requests", "gauge",
        "Requests waiting for a runtime slot.")
    for item in instances:
        writer.sample("mfq_runtime_queued_requests", _runtime_labels(item), item.queued_requests)
    writer.family("mfq_runtime_rejected_requests_total", "counter",
        "Requests refused because the runtime queue was full.")
    for item in instances:
        writer.sample("mfq_runtime_rejected_requests_total", _runtime_labels(item), item.rejected_requests)


def _write_queue(writer: _ExpositionWriter, instances: Sequence[RuntimeMetricsSnapshot]) -> None:
    name = "mfq_runtime_queue_wait_seconds"
    writer.family(name, "histogram", "Time a request waited for a runtime slot.")
    for item in instances:
        labels = _runtime_labels(item)
        _write_histogram(writer, name, labels, item.requests.queue_wait)


def _write_memory(writer: _ExpositionWriter, instances: Sequence[RuntimeMetricsSnapshot]) -> None:
    writer.family("mfq_runtime_resident_bytes", "gauge",
        "Resident memory of the runtime process, in bytes.")
    for item in instances:
        if item.resident_bytes is not None:
            writer.sample("mfq_runtime_resident_bytes", _runtime_labels(item), item.resident_bytes)
    writer.family("mfq_runtime_kv_bytes", "gauge",
        "KV cache bytes resident in the runtime, in bytes.")
    for item in instances:
        if item.kv_bytes is not None:
            writer.sample("mfq_runtime_kv_bytes", _runtime_labels(item), item.kv_bytes)
    writer.family("mfq_runtime_kv_usage_ratio", "gauge",
        "Resident KV cache bytes over the configured KV cache budget.")
    for item in instances:
        if item.kv_bytes is not None and item.kv_limit_bytes:
            writer.sample("mfq_runtime_kv_usage_ratio", _runtime_labels(item),
                item.kv_bytes / item.kv_limit_bytes)


def _write_mtp(writer: _ExpositionWriter, instances: Sequence[RuntimeMetricsSnapshot]) -> None:
    writer.family("mfq_runtime_mtp_acceptance_ratio", "gauge",
        "Accepted MTP draft tokens over drafted tokens since the runtime loaded.")
    for item in instances:
        acceptance = item.requests.mtp_acceptance_ratio
        if acceptance is not None:
            writer.sample("mfq_runtime_mtp_acceptance_ratio", _runtime_labels(item), acceptance)


def _write_requests(writer: _ExpositionWriter, instances: Sequence[RuntimeMetricsSnapshot]) -> None:
    writer.family("mfq_requests_total", "counter",
        "Requests served by result: ok, error, cancelled, or timeout.")
    for item in instances:
        for result in REQUEST_RESULTS:
            writer.sample("mfq_requests_total", {**_runtime_labels(item), "result": result},
                item.requests.results[result])
    writer.family("mfq_request_prompt_tokens_total", "counter",
        "Prompt tokens accepted from served requests.")
    for item in instances:
        writer.sample("mfq_request_prompt_tokens_total", _runtime_labels(item), item.requests.prompt_tokens)
    writer.family("mfq_request_completion_tokens_total", "counter",
        "Completion tokens generated for served requests.")
    for item in instances:
        writer.sample("mfq_request_completion_tokens_total", _runtime_labels(item), item.requests.completion_tokens)
    writer.family("mfq_request_ttft_seconds", "histogram",
        "Time to first generated token, measured by the worker.")
    for item in instances:
        _write_histogram(writer, "mfq_request_ttft_seconds", _runtime_labels(item), item.requests.ttft)
    writer.family("mfq_request_prefill_tokens_per_second", "histogram",
        "Prompt prefill throughput reported by the worker.")
    for item in instances:
        _write_histogram(writer, "mfq_request_prefill_tokens_per_second", _runtime_labels(item),
            item.requests.prefill_rate)
    writer.family("mfq_request_decode_tokens_per_second", "histogram",
        "Decode throughput reported by the worker.")
    for item in instances:
        _write_histogram(writer, "mfq_request_decode_tokens_per_second", _runtime_labels(item),
            item.requests.decode_rate)


def _write_histogram(writer: _ExpositionWriter, name: str, labels: Mapping[str, str], histogram: Histogram) -> None:
    for upper_bound, observed in histogram.cumulative():
        writer.sample(f"{name}_bucket", {**labels, "le": _format_value(upper_bound)}, observed)
    writer.sample(f"{name}_bucket", {**labels, "le": "+Inf"}, histogram.count)
    writer.sample(f"{name}_sum", labels, histogram.total)
    writer.sample(f"{name}_count", labels, histogram.count)
