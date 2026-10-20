# MFQ C++ runtime

The native runtime is organized by responsibility. Shared models are grouped
by model family:

- `core/` — canonical model graph, request/value contracts, policies, and
  generated tables; temporary old-artifact name adapters live in
  `core/compat/`;
- `models/<family>/` — each family owns `config.h` and `causal_lm.h`; the latter
  defines its concrete `CausalLm`, forward graph, topology, and position rules.
  MiniCPM also defines `TtsCausalLm`. `models/common/` contains `CausalModelBase`,
  causal traversal, attention, MLP, MoE, recurrence, and shared layer compositions;
- `transport/` — private stdio/HTTP protocol adapters;
- `apps/` and `server/` — native public server entry and initial HTTP/lifecycle support;
- `scheduler/` — backend-neutral request dispatch and lifecycle boundary;
- `engine/` — the sole cross-backend `Engine` interface plus reusable
  Engine instance lifecycle, shared runtime configuration, generation,
  continuous-batching, cache, tokenizer, output, and MTP components; CUDA tensors and device execution stay in `backends/cuda/`;
- `components/` — focused integrated components (`ggml`, `tokenizer`, `http`,
  and `json`);
- `backends/cuda/` — native Engine bindings, model execution adapters,
  operators, applications, build definition, and tests; `core/causal_model.h`
  supplies native resources and operations to the shared causal model.
  Native Transformer, predictor and graph resources belong to `core/`, lower-level
  operators to `ops/`, and device setup, loading plus session codecs/cache storage
  to `storage/`.
  `models/` contains family bindings and implementations; `engine/` contains
  Engine, generation, batching and MTP execution bindings;
- `backends/metal/` — Metal/MLX storage, runtime utilities, operators, model
  implementations, kernels, applications, tests, benchmarks, and diagnostics;
- `tests/` — backend-independent native tests;

A native `Runtime` composes a protocol `Transport`, a backend-neutral
`Scheduler`, and one concrete `Engine`. `Engine` is the only compute-backend
polymorphic boundary. Python still owns the product's runtime process lifecycle
and full public API. The separate `mfq-server` target starts the native server migration.

The mandatory ownership and canonicalization rules are defined in the
repository [development rules](../CONTRIBUTING.md). In particular, reusable
state machines, cache lifecycle, sampling, dispatch, metrics, multimodal
pipelines, and MTP orchestration must never live in a model-architecture
directory.

CUDA text and prepared grid-Vision requests share one
restore/prefill/output/snapshot lifecycle. Media embeddings, positions, cache
identity materialization, numerical verification, CUDA Graph execution, and
physical batch operations stay CUDA-specific. Generation, MTP draft/accept/commit
rules, and the continuous-batching request state machine are shared. Qwen3.5
and the other seven model families now keep their forward definitions in
`models/<family>/causal_lm.h`, including sparse/recurrent attention, MoE,
predictors, and MiniCPM audio/vision/TTS traversal. CUDA family `ops.h/.cpp`
files bind these definitions to native tensors, kernels, weights, and caches.
Qwen4/GLM5 decoder order, DeepSeek-V4.1 target/DSpark mega-layers, and
DeepSeek/GLM5 output collapse/norm order are shared. Local and parallel FFNs
use shared gated-MLP composition; CUDA owns shard placement, reduction, and
stream scheduling. MiniCPM ordinary TTS and text evaluation use the Engine's
synchronous token loop, with TTS sampling order in its shared model.
HC input expansion choices, Gemma FFN finishing, position/mask rules, and
MiniCPM/Grid-Vision multimodal composition are shared. Model definition,
output-weight tying and layer traversal live in `CausalModelBase`; Gemma also
owns its layer geometry and parameter-role assembly. CUDA's `CausalResources`
supplies native bindings, and `storage/model_loader.cpp` handles device loading
and placement. `EngineInstance<Backend>` directly implements `Engine` and owns
request execution, text controls and ordered reload/shutdown. CUDA's construction
entry returns `std::unique_ptr<Engine>`; model metadata and capabilities use
`EngineInfo`. There is no public `CudaEngine` facade or PImpl. Native batch operations
feed `ContinuousBatch<QwenBatchOperations>` directly, without a batch executor wrapper. Session/prefix settings and batch budgets use shared config;
CUDA load settings stay in `storage/load_options.cpp`. Ordinary, speculative
and batch generation share CUDA sampling buffer/count initialization. Model
users include their actual family headers directly; there is no include-only
model registry or CUDA MXFP4 blob forwarding header.
Metal integration and duplex control remain pending;
multi-device and non-Qwen3.5 real-weight validation are also incomplete.
Qwen3.8-27B uses the `qwen3_5` backbone;
Qwen3.8-Flash-Next uses `qwen4_exp`.

`CMakeLists.txt` is the single entry point. Runtime executable targets are
`mfq-runtime`, `mfq-decode-metal`, and `mfq-perplexity`; the established Metal
build output directory remains unchanged. Integrated upstream-derived
code retains its original licensing as documented in the repository `NOTICE`
and `LICENSES/` directory.

## Native server bootstrap

Stage 1 infrastructure is implemented and accepted on Linux: seven scenarios
with 951 checks, plus 34 native regression tests. Windows and macOS acceptance
remain pending. Stage 2 currently provides the local model catalog; in-process
model loading and inference are the next step. Studio and release entry points
still use the existing service.

Build the initial server without GPU backends or the private inference transport:

SQLite development headers and libraries must be installed. On Debian/Ubuntu:

```sh
sudo apt install libsqlite3-dev
```

```sh
cmake -S cpp_runtime -B build/native-server \
  -DMFQ_BUILD_SERVER=ON -DMFQ_BUILD_CUDA_RUNTIME=OFF \
  -DMFQ_BUILD_HIP_RUNTIME=OFF -DMFQ_BUILD_METAL_RUNTIME=OFF \
  -DMFQ_BUILD_RUNTIME_COMMUNICATION=OFF -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-server --target mfq-server-test -j
ctest --test-dir build/native-server -L native-server --output-on-failure
build/native-server/server/mfq-server --data-dir .mfq --port 8090
```

The server and its C++ integration test need no Python interpreter or test
framework. CTest launches the native test executable, which starts real server
processes and checks HTTP, lifecycle, authentication and database migrations.
Version metadata comes from `mfq/_version.py`, read as text at configure time.
Use `ctest --test-dir build/native-server -L stage1 --output-on-failure` for the
infrastructure acceptance suite alone; model catalog checks have the `stage2`
label.

The initial routes are `/health` and `/health/identity`. Optional `--web-root PATH`
serves an existing frontend build, including browser navigation fallback; unknown
API paths remain errors. Without this flag, static serving is disabled. Port `0`
selects an available port; the startup JSON line reports the actual listener port.
`--host` accepts numeric IPv4/IPv6 addresses and defaults to `127.0.0.1`.

Authentication uses `MFQ_SERVER_API_KEY`, or a variable selected with
`--api-key-env NAME`. A saved `auth.json` policy takes precedence over the
environment. Non-loopback listeners require an enabled policy, which cannot be
disabled while listening on the network. Authentication routes are:

- `GET/PUT /api/v1/auth/policy`;
- `GET/POST /api/v1/auth/keys`;
- `POST /api/v1/auth/keys/{id}/revoke` and `/rotate`.

Root credentials and administrator keys manage authentication. Issued keys use
scopes (`inference`, `models`, `jobs`, `admin`) or roles (`viewer`, `operator`,
`administrator`), with optional expiry. Tokens are returned only on creation or
rotation; persistence stores SHA-256 hashes. The request gate accepts Bearer
credentials, Anthropic `X-Api-Key`, and WebSocket `access_token` credentials
before upgrade. Studio CORS preflights and instance checks are also enforced;
the realtime WebSocket API itself remains pending.

The data directory selects the same instance lock and workspace identity as the
Python service. It cannot be served by both implementations at once.
`--exit-on-stdin-close` exits when the parent closes its
input pipe; `--studio-instance-id UUID` requires that lifecycle flag. SIGINT and
SIGTERM also stop the server. Lock files remain on disk and unlock on process exit.
Shutdown finishes active database handlers before closing SQLite and skips HTTP
connections that are still queued. Current-version databases with missing
authentication columns fail startup instead of advertising a healthy service.
Version 19 databases skip schema initialization and version rewrites. This
native rewrite retains version 19; upgrading older databases preserves existing
user data rather than introducing a new schema version.

SQLite uses `server.sqlite3` in the data directory, falling back to an existing
`mfq-server.sqlite3`; `--db PATH` selects an explicit file. The shared schema is
embedded at build time from `mfq/server/state/schema.sql`. Versions 1–18 migrate
transactionally to version 19, preserving existing rows. Migration failures roll
back; corrupt databases, unknown versions and nonempty databases without schema
metadata are rejected.

The local model catalog defaults to `DATA_DIR/models`. Repeated `--model-dir PATH`
arguments select other roots; when omitted, `MFQ_SERVER_MODEL_DIRS` accepts an
OS-separated path list. Registration persists the existing `.mfq-files.json`
format in the first root. The catalog serves:

- `GET /api/v1/models`, with `?refresh=true` to rescan;
- `GET /api/v1/models/directories`, with `directory_id` or `path` to browse;
- `POST /api/v1/models/directories/register`, with exactly one directory source;
- `GET /api/v1/models/{id}/directory`.

Discovery reuses the native MFQ and Safetensors readers without reading full
weight payloads. It reports file sizes, tensor counts, dtypes, declared context
capacity and incomplete artifacts. Model listings omit host paths; directory
APIs require the `models` scope or an appropriate role. Directory IDs change on
restart. Duplicate names within one format fail; MFQ takes precedence when an
HF source and converted MFQ share a name. Failed registrations preserve the
previous index and snapshot.

Snapshots refresh synchronously after five seconds or on request. Each root
scan is limited to 100,000 entries and 32 nested directory levels; registration
allows at most 10,000 model paths and a 4 MiB index, enforced before saving.
Slow volumes can delay catalog requests. The `stage2` CTest suite covers two
scenarios with 112 checks.
Models currently report `loadable: false` because in-process execution is still
pending. Opening a server-side desktop file manager is also pending and is
reported as unavailable in directory responses.

Other management and inference APIs, and the Studio/release switch are pending.
This target is not yet a replacement for `mfq serve`. Validation currently covers
Linux; Windows and macOS remain unverified.
