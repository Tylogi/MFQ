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

Build the initial server without GPU backends or the private inference transport:

```sh
cmake -S cpp_runtime -B build/native-server \
  -DMFQ_BUILD_SERVER=ON -DMFQ_BUILD_CUDA_RUNTIME=OFF \
  -DMFQ_BUILD_HIP_RUNTIME=OFF -DMFQ_BUILD_METAL_RUNTIME=OFF \
  -DMFQ_BUILD_RUNTIME_COMMUNICATION=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-server --target mfq-server -j
ctest --test-dir build/native-server -R '^mfq-server-test$' --output-on-failure
build/native-server/server/mfq-server --data-dir .mfq --port 8090
```

The native process needs no Python. The process test uses the Python standard
library and runs when CMake finds an interpreter. Version metadata comes from
`mfq/_version.py`, read as text at configure time.

The initial routes are `/health` and `/health/identity`. Optional `--web-root PATH`
serves an existing frontend build, including browser navigation fallback; unknown
API paths remain errors. Without this flag, static serving is disabled. Only
`127.0.0.1` and `::1` may bind until authentication is migrated. Port `0` selects
an available port; the startup JSON line reports the actual listener port.

The data directory selects the same instance lock and workspace identity as the
Python service. It cannot be served by both implementations at once. Existing
databases are untouched. `--exit-on-stdin-close` exits when the parent closes its
input pipe; `--studio-instance-id UUID` requires that lifecycle flag. SIGINT and
SIGTERM also stop the server. Lock files remain on disk and unlock on process exit.

Authentication, SQLite, management and inference APIs, and the Studio/release
switch are pending. This target is not yet a replacement for `mfq serve`.
