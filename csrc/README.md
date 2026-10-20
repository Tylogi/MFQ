# MFQ C++ runtime

The native runtime is organized by responsibility. Shared models are grouped
by model family. The tree below is the **current transitional layout**; the
target layout removes the `core/` aggregation, moves each family to
`engine/model/<family>/`, and adds `formats/`, `model_source/` and
`components/io/`. That end state is defined in
[docs/refactor.md](../docs/refactor.md) and sequenced in
[docs/plan.md](../docs/plan.md); nothing here claims the new execution chain
has landed.

- `core/` — canonical model graph, request/value contracts, policies, and
  generated tables; temporary old-artifact name adapters live in
  `core/compat/`. This is a transitional aggregation target (`mfq-core`), not a
  target module: its responsibilities are reassigned in
  [docs/refactor.md](../docs/refactor.md) and removed by K1–K4 in
  [docs/plan.md](../docs/plan.md);
- `models/<family>/` — each family owns `config.h` and `causal_lm.h`; the latter
  defines its concrete `CausalLm`, forward graph, topology, and position rules.
  MiniCPM also defines `TtsCausalLm`. `models/common/` contains `CausalModelBase`,
  causal traversal, attention, MLP, MoE, recurrence, and shared layer compositions;
- `transport/` — private stdio/HTTP protocol adapters;
- `scheduler/` — backend-neutral logical request lifecycle, admission,
  cancellation decisions and bounded output queues;
- `engine/` — the sole cross-backend `Engine` interface plus reusable
  Engine instance lifecycle, shared runtime configuration, generation,
  continuous-batching, cache, tokenizer, output, and MTP components; CUDA tensors and device execution stay in `backends/cuda/`;
- `components/` — focused integrated components (`ggml`, `tokenizer`, `http`,
  and `json`);
- `backends/cuda/` — native Engine bindings, model execution adapters,
  operators, applications, build definition, and tests. `native/` implements
  Tensor/context primitives in the `mfq-cuda-native` target. `ops/` contains
  device operators, including RoPE; operators consume loaded weights.
  `storage/` owns weight/source primitives, KV storage, paged payload I/O and
  cache storage; it does not select model families.
  `models/common/` supplies common native loading/session mechanics, Transformer
  blocks, predictor interfaces and grid-Vision; `models/<family>/` binds each
  shared family graph to CUDA operations and native state. The common
  `causal_model_ops.h` and `attention_ops.h` names distinguish CUDA bindings
  from shared model headers. `engine/` is the composition root: it owns family
  selection, template bindings, optional runtime components, generation,
  batching, MTP execution and decode Graph bindings;
- `backends/metal/` — Metal/MLX storage, runtime utilities, operators, model
  implementations, kernels, applications, tests, benchmarks, and diagnostics;
- `tests/` — backend-independent native tests;

A native `Runtime` composes a protocol `Transport`, a backend-neutral
`Scheduler`, and one concrete `Engine`. `Engine` is the only compute-backend
polymorphic boundary. Python owns runtime process lifecycle and the public
server API.

The mandatory ownership and canonicalization rules are defined in the
repository [runtime architecture](../docs/architecture.md) (current state) and
repository [development rules](../CONTRIBUTING.md); the target architecture is
[docs/refactor.md](../docs/refactor.md) and known defects are
[docs/bugs.md](../docs/bugs.md). In particular, reusable
state machines, cache lifecycle, sampling, dispatch, metrics, multimodal
pipelines, and MTP orchestration must never live in a model-architecture
directory.

Scheduler calls the shared Engine's bounded `step()` and publishes its typed
results after execution returns. Engine owns physical request state, model
composition, tokenization, sampling semantics, MTP commit rules and session/cache
lifecycle. Backend operations own numerical execution and native resources;
they do not receive output/cancellation callbacks or transport objects. Shared
model and Engine code must not depend on concrete backend headers.

CUDA's public header is `backends/cuda/include/cuda_engine.h`, containing
load options and `load_cuda_engine()`. Tensor/context headers stay in `native/`,
operator headers beside their sources in `ops/`, model plans in `models/`,
and execution configuration in `engine/`. `kernels/` owns the shared CUDA
sources and headers. CMake builds all native kernels; `mfq/kernels/cuda/_ext.py`
builds the subset exposed by the Python bindings, from these same sources.
The Python package keeps wrappers and `mfq_cuda.cpp` only. Python extension
builds require this source checkout, including when using an editable install.
Python keeps its existing fast-math flags; native fast-math remains limited to
the attention kernels. Implementation include paths are PRIVATE, and old paths
have no forwarding headers.

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
owns its layer geometry and parameter-role assembly. CUDA's `CausalResources` supplies native bindings and model-specific resource
geometry. Cross-family composition is confined to `engine/`; model loading and
optional components are wired by `model_bindings.cpp` and `model_components.cpp`.
`EngineInstance<Backend>` directly implements `Engine` and owns request
execution, text controls and ordered reload/shutdown. CUDA's construction entry
returns `std::unique_ptr<Engine>`; model metadata and capabilities use
`EngineInfo`. There is no public `CudaEngine` facade or PImpl. Reusable CUDA batch
execution lives in `engine/batch_execution.*` and feeds shared `ContinuousBatch`
with complete prefill/decode quanta and ordered token results. This includes
sampling state, metadata, Graph planning/storage and execution metrics. Family
native state stays in `models/<family>/`; scoped Qwen35 bindings own KV/recurrent
packing and global cache-position restoration, not a shared StateAdapter protocol.
Paged allocation, chunks and page tables live in `storage/state/batch_paged.h`,
which accepts native geometry rather than a model. Model Ops associate native
request/storage types; only Engine composition enables supported batch execution.
Production batch bindings currently include Qwen35 and a separate
`Qwen4BatchExecution` for Qwen4/Flash-Next. These bindings do not establish
shared Executor ownership or real-weight validation for every family. A synthetic
dense CUDA model with independent request histories tests runner reuse.
Concrete model, batch and request-state types are selected only at Engine composition;
generation accepts an owned sequence, never a concrete batching pointer.
Concurrent admission and physical batching are distinct.
Session/prefix settings and batch budgets use shared config;
CUDA load settings stay in `engine/load_options.cpp`. Ordinary, speculative
and batch generation share CUDA sampling buffer/count initialization. Model
users include their actual family headers directly; there is no include-only
model registry or CUDA MXFP4 blob forwarding header.
Metal integration and duplex control remain pending
([docs/bugs.md](../docs/bugs.md) B26 and B28);
multi-device and non-Qwen3.5 real-weight validation are also incomplete.
Qwen3.8-27B uses the `qwen3_5` backbone;
Qwen3.8-Flash-Next uses `qwen4_exp`.

`CMakeLists.txt` is the single entry point. CUDA executables are `mfq-runtime`,
`mfq-diagnostics` and `mfq-eval`; Metal executables are `mfq-decode-metal` and
`mfq-perplexity`. CLI implementations compile into their executables, not the
CUDA backend library. `mfq build` requests only the current backend's inference
target (`mfq-runtime` on CUDA); the CUDA tools above are built explicitly with
`--target`, see [optional native tools](../docs/cli/build.md#optional-native-tools). `mfq-cuda-runtime` exports the Engine construction API
and has no transport dependency. With `MFQ_BUILD_RUNTIME_COMMUNICATION=OFF`,
the backend, prefix cache, tokenizer, diagnostics and evaluation still build;
communication modules and `mfq-runtime` are omitted. `MFQ_BUILD_CUDA_RUNTIME`
defaults to ON when a CUDA compiler is detected on a non-Apple platform and HIP
is not selected; otherwise it defaults to OFF. Enabling it on Apple is a
configure error. Setting it to OFF omits the CUDA backend entirely:
the CUDA language, toolkit lookup, device tests and CUDA executables are not
configured, so the common engine and its CPU tests still configure, build and run
without invoking a CUDA compiler or linking a CUDA library. The
optional serving application `mfq-runtime-torch` requires communication to be
enabled.

The established Metal build output directory remains unchanged. Integrated upstream-derived
code retains its original licensing as documented in the repository `NOTICE`
and `LICENSES/` directory.
