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
- `scheduler/` — backend-neutral request dispatch and lifecycle boundary;
- `engine/` — the sole cross-backend `Engine` interface plus reusable
  generation, continuous-batching, cache, tokenizer, output, and MTP
  components; CUDA tensors and device execution stay in `backends/cuda/`;
- `components/` — focused integrated components (`ggml`, `tokenizer`, `http`,
  and `json`);
- `backends/cuda/` — the concrete `CudaEngine`, model execution adapters,
  operators, applications, build definition, and tests; `core/causal_model.h`
  supplies native resources and operations to the shared causal model.
  Native Transformer, predictor and graph resources belong to `core/`, lower-level
  operators to `ops/`, and loading plus session codecs/cache storage to `storage/`.
  `models/` contains family bindings and implementations; `engine/` contains
  request, generation, batching, MTP and runtime configuration adapters;
- `backends/metal/` — Metal/MLX storage, runtime utilities, operators, model
  implementations, kernels, applications, tests, benchmarks, and diagnostics;
- `tests/` — backend-independent native tests;

A native `Runtime` composes a protocol `Transport`, a backend-neutral
`Scheduler`, and one concrete `Engine`. `Engine` is the only compute-backend
polymorphic boundary. Python owns runtime process lifecycle and the public
server API.

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
and placement. Metal integration and duplex control remain pending;
multi-device and non-Qwen3.5 real-weight validation are also incomplete.
Qwen3.8-27B uses the `qwen3_5` backbone;
Qwen3.8-Flash-Next uses `qwen4_exp`.

`CMakeLists.txt` is the single entry point. Runtime executable targets are
`mfq-runtime`, `mfq-decode-metal`, and `mfq-perplexity`; the established Metal
build output directory remains unchanged. Integrated upstream-derived
code retains its original licensing as documented in the repository `NOTICE`
and `LICENSES/` directory.
