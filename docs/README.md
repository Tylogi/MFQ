# MFQ documentation

MFQ combines high-quality, fine-grained mixed-precision quantization, optimized
CUDA/Metal C++ inference, and local serving. Start with the workflow that
matches what you want to do.

## Get started

- [Build the C++ runtime](cli/build.md)
- [Run MFQ Server and Studio](cli/serve.md)
- [Create a self-contained release](packaging.md)
- [Check model and backend support](runtime/support.md)

## Quantize and calibrate

- [Quantize HF, GGUF, or full-precision MFQ](cli/quantize.md)
- [Collect reusable calibration artifacts](cli/calibrate.md)
- [Solve an Expert-Wise precision budget](quantization/ew-joint-solver.md)
- [AlphaQ calibration-free allocation](quantization/alphaq.md)
- [Embed runtime sampling profiles](runtime/sampling-profiles.md)
- [Utility commands](cli/utilities.md)

## Models and runtimes

- [Runtime support matrix](runtime/support.md)
- [Python/C++ runtime transport](runtime/transport.md)
- [CUDA KV memory policies](runtime/cuda-kv-memory.md)
- [Flash-Next continuous batching](runtime/cuda-continuous-batching.md)
- [MiniCPM-o 4.5 multimodal runtime](runtime/minicpmo45.md)
- [HIP runtime validation](runtime/hip-validation.md)
- [HIP model benchmarks](runtime/hip-model-benchmarks.md)
- [C++ runtime layout](../csrc/README.md)

## APIs

MFQ provides OpenAI-compatible `/v1/models` and `/v1/chat/completions`,
Anthropic-compatible `/v1/messages`, and an HTTP control API under `/api/v1`.
Token and job updates stream over SSE; audio and realtime sessions use
WebSocket. Function tools are executed by the client and their results returned
through the inference API.

- [HTTP API](api/http.md)
- [WebSocket and realtime API](api/websocket.md)

## Develop MFQ

- [Architecture and contribution rules](../CONTRIBUTING.md)
- [Current project architecture](architecture.md)
- [Native runtime refactor design](refactor.md)
- [Remaining refactor work and validation](plan.md)
- [Open defects and validation gaps](bugs.md)
- [Repository README](../README.md)

Runtime support is checkpoint- and backend-dependent. Use the support matrix
as the starting point, then validate the exact model artifact on its target
machine before deployment.
