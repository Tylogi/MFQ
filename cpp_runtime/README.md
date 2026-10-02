# MFQ C++ runtime

The native runtime is organized by responsibility. Shared models are grouped
by model family:

- `core/` — canonical model graph, request/value contracts, policies, and
  generated tables; temporary old-artifact name adapters live in
  `core/compat/`;
- `models/<family>/` — shared family configuration and forward composition;
  `models/common/` contains the shared `CausalLm` class, causal traversal, attention,
  gated MLP, and layer compositions;
- `transport/` — private stdio/HTTP protocol adapters;
- `scheduler/` — backend-neutral request dispatch and lifecycle boundary;
- `engine/` — the sole cross-backend `Engine` interface plus reusable
  generation, continuous-batching, cache, tokenizer, output, and MTP
  components; CUDA tensors and device execution stay in `backends/cuda/`;
- `components/` — focused integrated components (`ggml`, `tokenizer`, `http`,
  and `json`);
- `backends/cuda/` — the concrete `CudaEngine`, model execution adapters,
  operators, applications, build definition, and tests; `models/causal_ops.h`
  supplies native resources and operations to the shared causal model;
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
inner attention, dense FFN, and predictor ordering also use shared model code;
other families still require migration of their inner forward definitions. Qwen3.8-27B uses the `qwen3_5` backbone;
Qwen3.8-Flash-Next uses `qwen4_exp`.

`CMakeLists.txt` is the single entry point. Runtime executable targets are
`mfq-runtime`, `mfq-decode-metal`, and `mfq-perplexity`; the established Metal
build output directory remains unchanged. Integrated upstream-derived
code retains its original licensing as documented in the repository `NOTICE`
and `LICENSES/` directory.
