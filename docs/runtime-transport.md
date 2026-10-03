# Native runtime transport

`mfq serve` owns the public API. Each managed C++ inference runtime only exposes
its private runtime control surface through one selected communication module:

```text
Python Server -> RuntimeClient -> stdio or private HTTP -> C++ Runtime
                                              Transport -> Scheduler -> Engine
```

The transport only speaks the private protocol; scheduling owns request
lifecycle and dispatches work to the backend inference engine.

Finite generation requests (text, prepared multimodal input, MTP and continuous
batches) all enter `Scheduler::submit` and advance through `Engine::step` and
one `RequestExecutor`. They share preparation, session transactions, output,
cancellation and terminal cleanup. Batching groups pending numerical operations;
each row uses the same prefill/decode sequence as single-request execution.
Internal batching yields never enter transport outboxes. Requests requiring
MTP, session reuse or multimodal preparation currently execute individually
through this same lifecycle. Duplex migration is deferred and retains its
existing control API.

Internal execution steps distinguish waiting, progress and completion; these
states are separate from public request events. Waiting does not count as an
advanced request. Grid-Vision and MiniCPM image/audio preparation yield between
encoder layers and preparation stages, so cancellation can release their
suspended tensors before text prefill. CUDA measures each preparation quantum
separately, excluding time spent between steps.

The generation flow owns serial model cleanup, and each batch row owns its
physical slot cleanup. The executor destroys the suspended flow, observes any
cleanup failure, and only then publishes a terminal event. Cleanup errors mark
the engine unhealthy, including errors encountered during explicit shutdown.

Select it with:

```bash
mfq serve --transport stdio   # default
mfq serve --transport http
```

CUDA runtime tuning uses `MFQ_RUNTIME_*` environment variables.

## stdio framing

stdin and stdout carry UTF-8 NDJSON protocol version 1. stderr carries all
runtime logs. Every request has a string `id`, an `op`, and an object `params`:

```json
{"v":1,"id":"1","op":"health","params":{}}
{"v":1,"id":"2","op":"generate","params":{"model":"...","input":{"messages":[{"role":"user","content":"hi"}]},"sampling":{"max_new_tokens":128},"stream":true,"include_usage":true}}
```

Responses use the same ID:

```json
{"v":1,"type":"ready"}
{"v":1,"id":"1","type":"result","data":{"status":"ok"}}
{"v":1,"id":"2","type":"event","data":{"event":"delta","request_id":"run-...","delta":{"content":"hello"}}}
{"v":1,"id":"2","type":"done"}
{"v":1,"id":"2","type":"error","error":{"code":"...","message":"...","status_code":500,"retryable":false}}
```

Supported operations are `generate`, `health`, `status`, `models`,
`session.fork`, `session.close`, `session.cancel`, `request.cancel`, `reload`,
`cache.clear`, `cache.trim`, `realtime.capabilities`, `realtime.open`,
`realtime.send`, `realtime.close`, and `shutdown`.

Writes are serialized and response frames are multiplexed by ID. EOF cancels
unfinished generation and closes realtime state. Large multimodal tensors stay
out of band in the existing controlled temporary files.

## Private HTTP

HTTP mode exposes the same runtime concepts under `/runtime/*`, including
`/runtime/generate`, `/runtime/health`, `/runtime/status`, `/runtime/models`,
cache/session controls, and `/runtime/realtime`. It does not expose public
`/v1/*` or OpenAI-compatible routes; Python performs that translation.
