# MFQ C++ runtime

The native runtime is organized by responsibility rather than checkpoint
family or source origin:

- `core/` — canonical model graph, backend-neutral policies, interfaces, and
  generated tables; temporary old-artifact name adapters live in
  `core/compat/`;
- `models/` — shared model configuration and architecture geometry; public headers in `models/include/`;
- `transport/` — private stdio/HTTP protocol adapters;
- `scheduler/` — backend-neutral request dispatch and lifecycle boundary;
- `engine/` — backend-neutral tokenizer, output processing, MTP policies, and
  an opt-in whole-model text request flow; CUDA tensors and device execution
  stay in `backends/cuda/`;
- `components/` — focused integrated components (`ggml`, `tokenizer`, `http`,
  and `json`);
- `backends/cuda/` — the CUDA runtime, scheduler-facing inference engine,
  model adapters, operators, applications, build definition, and tests;
- `backends/metal/` — Metal/MLX storage, runtime utilities, operators, model
  implementations, kernels, applications, tests, benchmarks, and diagnostics;
- `tests/` — backend-independent native tests;

A native `Runtime` composes a protocol `Transport`, a backend-neutral
`Scheduler`, and one backend `InferenceEngine`. Python owns runtime process
lifecycle and the public server API.

The mandatory ownership and canonicalization rules are defined in the
repository [development rules](../CONTRIBUTING.md). In particular, reusable
state machines, cache lifecycle, sampling, dispatch, metrics, multimodal
pipelines, and MTP orchestration must never live in a model-architecture
directory.

Set `MFQ_RUNTIME_QWEN38_TEXT_FLOW=1` to try the opt-in CUDA whole-model
`generate` flow for text on supported non-MiniCPM backbones and prepared
Qwen3.5 grid-Vision input (currently one image, no video). Text and vision
share restore/prefill/output/snapshot handling; media embeddings, positions,
cache identity, and MTP drafting/verification stay CUDA-specific. Without the
flag, or for continuous batching and diagnostic/reprefill modes, the legacy
generator remains available and is the default. The opt-in target-only path
does not yet use CUDA Graph. Qwen3.8-27B uses the `qwen3_5` backbone;
Qwen3.8-Flash-Next uses `qwen4_exp`.

`CMakeLists.txt` is the single entry point. Runtime executable targets are
`mfq-runtime`, `mfq-decode-metal`, and `mfq-perplexity`; the established Metal
build output directory remains unchanged. Integrated upstream-derived
code retains its original licensing as documented in the repository `NOTICE`
and `LICENSES/` directory.
