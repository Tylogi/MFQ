# Multi-request concurrency

MFQ serves several requests at once wherever the loaded model provides
per-request state. The control plane discovers that capacity from the worker
instead of assuming it, so one flag works across CUDA, HIP, and Metal without
turning a hardware or checkpoint limit into a failed load.

## Capacity negotiation

1. `mfq serve` asks each worker for a bounded capacity
   (`--max-requests-per-runtime`, default `8` on CUDA, HIP and Metal).
   The CLI accepts up to `32`; per-model overrides must also
   fit the configured server ceiling.
2. The worker loads the model and decides whether it can batch at all. A model
   without per-request batch state keeps the single-request path, logs
   `continuous_batching unavailable model=… requested_capacity=…`, and reports
   capacity `1`.
3. The worker publishes `batch_capacity` in its `health` and `status` payloads.
   The server adopts that value after the load handshake and shrinks its
   admission limit when the model is narrower than the requested capacity.

Shrinking never disturbs work in flight: it only happens while the runtime has
no active or queued requests. The effective value is visible in
`GET /api/v1/runtime/instances` (`request_capacity`, `batch_capacity`) and in
`GET /api/v1/runtime/status`.

### Supported batching paths

| Backend | Architecture | Concurrent rows | Notes |
| --- | --- | --- | --- |
| CUDA | Qwen3.5 family | up to the requested capacity | Paged per-row KV and recurrent state |
| CUDA | Qwen Flash-Next / Qwen4-style | up to 32 | Padded to the configured width; GPU sample fast path when every row is greedy |
| CUDA | Other families | 1 | Loads normally with a warning |
| Metal | Qwen Flash-Next | up to 32 | Packed prefill and per-request cache; batch MTP when the predictor supports it |
| Metal | Other families | 1 | Loads normally with a warning |
| HIP | Qwen3.5 family | up to the requested capacity | Implemented and compiled; runtime reports effective capacity |
| HIP | Other families | 1 | Loads normally with a warning |

Requests that cannot join a batch still execute correctly. They wait in the
native scheduler while a batch is running and then run on the single-request
path; the capacity report tells the server not to count them as parallel work.

## Admission and queueing

Each runtime instance admits `request_capacity` concurrent requests. Waiting
requests are bounded by `--max-queued-requests-per-runtime` (default
`max(32, 4 × capacity)`).

- When `active + queued` reaches `capacity + queue limit`, the server rejects the
  request with `429 runtime_queue_full` and a `Retry-After` header.
- `--queue-timeout SECONDS` bounds the wait for a slot. The default `0` waits
  indefinitely. On expiry the request fails with `503 runtime_queue_timeout`,
  which is retryable; the queue slot is released immediately.
- Queue wait is tracked per runtime (`queue_wait_ms_avg`, `queue_wait_ms_max`,
  `queue_wait_samples`, `rejected_requests`). Each response also carries
  `server_queue_wait_ms` inside its performance block, separate from `ttft_ms`,
  so client-observed time to first token splits into server queueing and native
  prefill.

## Per-model settings

`GET /api/v1/runtime/concurrency` returns `defaults` and `limits` for
`max_requests`, `max_queued_requests`, and `queue_timeout_seconds`. The server's
CLI values are both the defaults and the ceilings. A server timeout of `0`
means unlimited waiting, represented by a `null` timeout ceiling. With a finite
server timeout, a per-model timeout must be positive and no larger than that
ceiling; `0` would bypass it and is rejected.

`POST /api/v1/models/load` accepts these optional fields:

```json
{
  "model": "model",
  "max_requests": 16,
  "max_queued_requests": 24,
  "queue_timeout_seconds": 30
}
```

Omitted or null fields inherit server defaults independently. An override
above a ceiling returns `422 runtime_concurrency_limit` with the permitted
limit before creating a load job. `max_requests` ranges from `1` to `32`;
`max_queued_requests` can be `0` to reject waiting requests.

Each `RuntimeInstanceResource` includes a `concurrency` object containing
`requested`, `limits`, and `effective`. For example, a requested width of `16`
can have an effective width of `8`, or `1` for a model without batching. The
existing `request_capacity` and `batch_capacity` fields remain available.

### Live queue updates and explicit reloads

`PUT /api/v1/runtime/instances/{instance_id}/concurrency` accepts any nonempty
subset of the three settings. Queue settings take effect immediately without
restarting the worker. Existing waiters adopt a changed timeout, measured from
their original admission time. Reducing the queue limit keeps existing queued
requests and applies the new ceiling to new arrivals.

The response contains `concurrency` and `reload_required`. If `max_requests`
differs from the current requested width, `reload_required` is `true`; the
current requested and effective widths remain unchanged. Queue edits in that
same request still apply immediately.

To change the width, explicitly submit
`POST /api/v1/runtime/concurrency/reload` with `instance_id` and the desired
settings. It returns a `202` job. Reloading requires an idle model; a busy
runtime fails the job with retryable `runtime_busy`. The reload preserves the
other load options, including context, devices, cache settings, pinning, and
idle TTL. If loading fails, the server attempts to restore the previous load
request. It never implicitly reloads in response to a queue update.

MFQ Studio provides concurrency controls in the model load policy and the
runtime overview. The overview displays requested/effective widths, queue
length, average/maximum queue wait, and rejection counts. Rejections include
full queues and expired waits. The settings page stores defaults for new loads
in the existing browser preferences; empty fields inherit server defaults.
Load-memory estimates and the KV planner scale cache requirements by requested
rows while keeping resident weights fixed. Estimates use the requested width
until the loaded worker reports its actual capacity.

## Scheduling behavior in the runtime

The shared engine (`cpp_runtime/engine/`) owns request lifecycle, queueing, and
batch membership for every backend:

- Pending prefills run in bounded chunks (`--prefill-chunk-size`). Eligible
  decode rows wait for their peers so requests with different prompt lengths
  form a batch instead of alternating single-row decodes.
- A request paused by output backpressure does not hold up other rows.
- Cancelling a request releases only that request's state.
- Capacity failures during prefill or decode-growth reject the affected request
  without poisoning the backend.

Batched execution currently covers ordinary token decoding. On CUDA, requests
that use MTP, an explicit session ID, multimodal input, or a restorable prefix
run exclusively; the Metal Flash-Next runtime keeps MTP and prefix reuse inside
the batch. See [cuda-continuous-batching.md](cuda-continuous-batching.md) for
the CUDA graph, KV-growth, and failure-isolation details.

## Serving overhead

Long-lived servers poll runtime telemetry continuously. Two paths were on the
per-request hot path and now cache:

- NVML readings are cached for one second, so `/runtime/status` polling no
  longer initializes and shuts down NVML on every call.
- The remote-cluster node list is refreshed at most once per health TTL instead
  of on every request; explicit API calls still force a refresh.

MFQ Studio sends one coalesced runtime refresh and reuses the server's cached
model catalog for its five-second poll. Only explicit user actions (register,
load, delete) request a full catalog rescan.

## Measuring concurrency

```shell
uv run mfq serve --model /models/model.mfq --max-requests-per-runtime 8 --queue-timeout 120
curl -s http://127.0.0.1:8090/api/v1/runtime/instances
```

The instance payload reports `request_capacity`, `batch_capacity`,
`queued_requests`, `queue_wait_ms_avg`, `queue_wait_ms_max`, and
`rejected_requests`. Throughput claims should use more than one request in
flight; a single request cannot exercise the batching path even when the
capacity is greater than one.
