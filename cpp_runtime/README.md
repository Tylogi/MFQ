# MFQ C++ runtime

The native runtime is organized by responsibility rather than checkpoint
family or source origin:

- `core/` — canonical model graph, backend-neutral policies, interfaces, and
  generated tables; temporary old-artifact name adapters live in
  `core/compat/`;
- `transport/` — private stdio/HTTP protocol adapters;
- `scheduler/` — backend-neutral request dispatch and lifecycle boundary;
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

`CMakeLists.txt` is the single entry point. Runtime executable targets are
`mfq-runtime`, `mfq-decode-metal`, and `mfq-perplexity`; the established Metal
build output directory remains unchanged. Integrated upstream-derived
code retains its original licensing as documented in the repository `NOTICE`
and `LICENSES/` directory.
