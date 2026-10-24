# Prometheus metrics

`GET /metrics` exposes runtime and request telemetry in the Prometheus text
exposition format 0.0.4. The endpoint is compatible with a plain `scrape_config`
and needs no extra Python or C++ dependency.

```console
$ curl -s http://127.0.0.1:8090/metrics | head -4
# HELP mfq_runtime_batch_capacity Rows the loaded model batches together, as reported by the worker.
# TYPE mfq_runtime_batch_capacity gauge
mfq_runtime_batch_capacity{model="qwen3-8b",instance="4a0f0a1e-…"} 8
```

The response uses `Content-Type: text/plain; version=0.0.4`. `/metrics` is not
part of the OpenAPI contract.

## Authentication

`/metrics` follows the `/api/` rules rather than the `/health` rules: when a root
credential is configured (`MFQ_SERVER_API_KEY` or `--api-key-env`), a scrape must
send the same bearer credential as any other API call and needs the `inference`
scope that `GET /api/v1/runtime/*` requires:

```yaml
scrape_configs:
  - job_name: mfq
    authorization:
      credentials_file: /etc/mfq/metrics-token
    static_configs:
      - targets: ["127.0.0.1:8090"]
```

Metrics carry model names and traffic volume, so they stay behind the same
credential as the rest of the control plane. Without a configured credential the
endpoint is open, exactly like `/api/`. `/health` stays unauthenticated because it
returns only liveness, not telemetry.

## Labels

Every series carries:

- `model` — the artifact name the runtime loaded.
- `instance` — the runtime instance id (a UUID). Loading or restarting a model
  creates a new instance id, so counters restart under a new label value instead
  of decreasing under an old one. Removing a runtime removes its series.

Request series add `result` (`ok`, `error`, `cancelled`, `timeout`), and histogram
series add `le`.

## Runtime metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `mfq_runtime_batch_capacity` | gauge | Rows the worker can batch together (its reported `batch_capacity`). |
| `mfq_runtime_request_capacity` | gauge | Concurrent request slots the server admits for this runtime. |
| `mfq_runtime_active_requests` | gauge | Requests holding a slot, including their queue wait. |
| `mfq_runtime_queued_requests` | gauge | Requests waiting for a slot. |
| `mfq_runtime_rejected_requests_total` | counter | Requests refused with `429 runtime_queue_full`. |
| `mfq_runtime_queue_wait_seconds` | histogram | Time a request waited for a slot before the worker saw it. |
| `mfq_runtime_resident_bytes` | gauge | Resident memory of the runtime process. |
| `mfq_runtime_kv_bytes` | gauge | Resident KV (prefix-cache) bytes reported by the worker. |
| `mfq_runtime_kv_usage_ratio` | gauge | `mfq_runtime_kv_bytes` over the configured KV budget; omitted when the worker reports no budget. |
| `mfq_runtime_mtp_acceptance_ratio` | gauge | Accepted MTP draft tokens over drafted tokens since the runtime loaded; omitted when MTP never ran. |

`request_capacity` is the requested admission limit and `batch_capacity` is what
the worker actually batches; they differ when a model has no per-request batch
state, which is why both are exported. See
[concurrency.md](concurrency.md#capacity-negotiation).

## Request metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `mfq_requests_total` | counter | Requests served, by `result`. |
| `mfq_request_prompt_tokens_total` | counter | Prompt tokens of those requests. |
| `mfq_request_completion_tokens_total` | counter | Completion tokens of those requests. |
| `mfq_request_ttft_seconds` | histogram | Time to first token as measured by the worker. |
| `mfq_request_prefill_tokens_per_second` | histogram | Prefill throughput per request. |
| `mfq_request_decode_tokens_per_second` | histogram | Decode throughput per request. |

All four `result` values are exported from the first scrape (as `0`) so `rate()`
works without waiting for each outcome to occur. `ok` means the stream completed,
`cancelled` covers client disconnects and worker-side cancellation, `timeout`
covers retryable deadline failures (`runtime_queue_timeout`,
`runtime_maintenance_timeout`, `backend_timeout`), and `error` covers every other
failure. Admission refusals are counted once, in
`mfq_runtime_rejected_requests_total`, not in `mfq_requests_total`.

Throughput is exported as histograms rather than summaries: summaries cannot be
aggregated across runtime instances or server processes, while histogram buckets
can be summed and used with `histogram_quantile()`. Server-side queueing is kept
out of `mfq_request_ttft_seconds`; it has its own histogram.

## Histogram buckets

Buckets are fixed, in base units (seconds or tokens per second), and end at
`+Inf`:

| Histogram | Upper bounds |
| --- | --- |
| `mfq_runtime_queue_wait_seconds` | 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60 |
| `mfq_request_ttft_seconds` | 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120 |
| `mfq_request_prefill_tokens_per_second` | 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000 |
| `mfq_request_decode_tokens_per_second` | 1, 2, 5, 10, 20, 40, 60, 80, 120, 160, 240, 320, 480, 640, 1000 |

Queue waits start at 1 ms because an uncontended slot is granted in microseconds,
and stop at the `--queue-timeout` default ceiling. TTFT spans fast short-prompt
prefill through minute-long prompts. Prefill throughput is measured over short
prompts as well, where one token can look very fast, so its buckets reach the
batched-prefill range. Decode throughput buckets follow the range seen on the
supported backends, from tiny models to batched Flash-Next decoding.

## Scrape cost and freshness

A scrape reads in-process state only:

- Request counters and histograms are recorded by the runtime pool while it
  serves, not recomputed from storage.
- Runtime gauges come from the cached instance snapshot that
  `GET /api/v1/runtime/instances` also uses. That snapshot refreshes at most once
  per runtime-pool interval (2 s by default), so its cached hardware and storage
  readings are reused instead of taken again for the scrape.
- A scrape never initializes NVML, reads SQLite, or issues more than one worker
  status call per refresh interval.

Gauges can therefore be up to one interval stale; counters and histograms update
as requests finish.

## Known limits

- Only runtimes managed by this server process are reported. Remote cluster nodes
  (`/api/v1/cluster/*`) keep their own telemetry; a node's `/metrics` is scraped
  directly if needed.
- Histograms aggregate the requests of one server process. Sum bucket counts
  across replicas before applying `histogram_quantile()`.
- MTP acceptance is a cumulative ratio since the runtime loaded, reset by the next
  load, not a moving window.
