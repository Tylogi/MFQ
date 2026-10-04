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
  `storage/` owns weight loading, KV storage and session codecs/cache storage.
  `models/common/` supplies common native model bindings, Transformer blocks,
  predictor interfaces and grid-Vision; `models/<family>/` binds each shared
  family graph. The common `causal_model_ops.h` and `attention_ops.h` names
  distinguish CUDA bindings from shared model headers.
  `engine/` contains Engine, generation, batching, MTP execution and decode
  Graph bindings;
- `backends/metal/` — Metal/MLX storage, runtime utilities, operators, model
  implementations, kernels, applications, tests, benchmarks, and diagnostics;
- `tests/` — backend-independent native tests;

A native `Runtime` composes a protocol `Transport`, a backend-neutral
`Scheduler`, and one concrete `Engine`. `Engine` is the only compute-backend
polymorphic boundary. Python owns runtime process lifecycle and the public
server API.

The mandatory ownership and canonicalization rules are defined in the
workspace [runtime architecture](../../docs/architecture.md) and repository
[development rules](../CONTRIBUTING.md). In particular, reusable
state machines, cache lifecycle, sampling, dispatch, metrics, multimodal
pipelines, and MTP orchestration must never live in a model-architecture
directory.

Scheduler calls the shared Engine's bounded `step()` and publishes its typed
results after execution returns. Engine owns physical request state, model
composition, tokenization, sampling semantics, MTP commit rules and session/cache
lifecycle. Backend operations own numerical execution and native resources;
they do not receive output/cancellation callbacks or transport objects. Shared
model and Engine code must not depend on concrete backend headers.

CUDA's public header is `backends/cuda/include/mfq/cuda/engine.h`, containing
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

`CMakeLists.txt` is the single entry point. CUDA executables are `mfq-runtime`,
`mfq-diagnostics` and `mfq-eval`; Metal executables are `mfq-decode-metal` and
`mfq-perplexity`. CLI implementations compile into their executables, not the
CUDA backend library. `mfq-cuda-runtime` exports the Engine construction API
and has no transport dependency. With `MFQ_BUILD_RUNTIME_COMMUNICATION=OFF`,
the backend, prefix cache, tokenizer, diagnostics and evaluation still build;
communication modules and `mfq-runtime` are omitted. The optional serving
application `mfq-runtime-torch` requires communication to be enabled.

The established Metal build output directory remains unchanged. Integrated upstream-derived
code retains its original licensing as documented in the repository `NOTICE`
and `LICENSES/` directory.
