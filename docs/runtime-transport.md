# Native runtime transport

`mfq serve` owns the public API. Each managed C++ inference runtime only exposes
its private runtime control surface through one selected communication module:

```text
Python Server -> RuntimeClient -> stdio or private HTTP -> C++ Runtime
                                              Transport -> Scheduler -> InferenceEngine
```

The transport only speaks the private protocol; scheduling owns request
lifecycle and dispatches work to the backend inference engine.

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
