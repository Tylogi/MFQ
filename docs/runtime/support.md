# Runtime support

Model and feature support for CUDA, Metal, and ROCm (HIP).

## Status terms

- **Native C++** means the backend worker loads the model and performs prefill,
  decode, sampling, and serving without a Python model runtime.
- **Reference** means an explicit MFQ runtime exists for correctness and
  integration work, but the managed server does not select it automatically.
- **Optional** means a component is enabled only when the model graph declares
  it, its weights are present, the backend has an adapter, and the user has not
  disabled it.

## Model and backend matrix

| Architecture family | CUDA C++ | Metal C++ | HIP C++ | Current scope |
| --- | --- | --- | --- | --- |
| Qwen3.5–3.8 | Native C++ | Native C++ | Native C++ (dense) | Hybrid full/GDN attention; graph-declared MTP when predictor weights are present |
| Qwen Flash-Next / Qwen4-style | Native C++ | Native C++ | Partial | Routed MoE, PLE, sparse attention, and graph-declared MTP; bounded expert storage on Metal |
| DeepSeek V4 Flash Series | Native C++ | Native C++ | — | Compression, indexer, sparse attention, mHC, and routed MoE; Vision and DSpark availability depends on the artifact and backend |
| DeepSeek V4.1 Flash | Native C++ | Native C++ | — | Engram, routed MoE, mHC, and graph-declared DSpark; native Metal Vision and raw-HF loading |
| MiniCPM-o 4.5 | Native C++ | Native C++ | — | Text, image, video, audio input, audio output, and duplex serving |
| GLM5–5.3 | Native C++ | Reference | — | DSA/KDA and Flash-Next families; Metal production C++ worker integration remains open |
| Gemma 4 | Native C++ | Reference | — | Mixed full/sliding attention and MoE; Metal production C++ worker integration remains open |
| Nemotron-H | Native C++ (text) | — | — | Mamba-2, NoPE attention and latent relu² MoE with a dense expert bank; graph-declared MTP; RADIO vision is not implemented |

The HIP column reflects MI300X validation of the early HIP backend: dense
Qwen3.5–3.8 models (BF16 and quantized `S4` tiers) run end to end, Qwen
Flash-Next loads but generation is blocked by a known logits-dtype defect, and
the other families are not yet ported. See
[hip-validation.md](hip-validation.md).

The top-level architecture family is not used to force optional features on or
off. Vision and MTP are resolved from the graph, declared weight roots, backend
adapters, and per-load settings. A missing optional component does not prevent
the backbone from loading.

## Conversion and storage

- The quantizer accepts recognized Hugging Face Safetensors, GGUF, and
  full-precision MFQ sources and writes canonical tensor names into a
  self-contained `.mfq` container.
- A backend-neutral model-source contract exposes direct MFQ and native HF
  Safetensors loading to CUDA and Metal. CUDA directly consumes native MXFP8
  and lossless `FP8-128SQ/SQ8` views of block-scaled HF `F8_E4M3` weights.
  The native Metal path is production-validated for DeepSeek V4/V4.1; other
  architecture/backend combinations still require exact-artifact validation.
- Model configuration, model graph, tokenizer data, chat templates,
  special-token metadata, and sampling profiles can be embedded as runtime
  assets. Numbered MFQ shards are loaded directly by both Python tooling and
  C++ workers.
- Canonical MFE supports mixed-family HF/GGUF streaming conversion. Legacy
  tensor names are normalized at the compatibility boundary; current
  quantization and runtime code use canonical semantic names internally.

## Shared runtime capabilities

- Descriptor-driven NINTv2 and MFE execution keeps heterogeneous per-neuron,
  per-projection, and per-expert precision inside unified logical containers.
- The server provides OpenAI-compatible chat and completion APIs, streaming
  responses, model-pool lifecycle management, and per-model capability
  reporting.
- Prefix reuse and tiered RAM/SSD cache storage are shared serving features.
  Continuous batching is active on supported native paths, and newly loaded
  workers report the capacity their model can batch so the server sizes
  admission from it. See [multi-request concurrency](concurrency.md).
- Metal supports bounded resident expert caches and SSD-streamed expert access
  for supported large-MoE runtimes.
- One backend-wide MTP engine owns sampling, verification, cache snapshots,
  adaptive depth, and accounting. Model modules provide only predictor math
  and model-specific state adapters.

## Backend notes

### CUDA

The default CUDA worker is a native C++/CUDA runtime with no Python, PyTorch,
ATen, or LibTorch dependency during inference. It supports one-GPU execution
and explicit tensor/expert-parallel device groups; multi-rank deployments still
require physical validation. Packed NINTv2, VQ-family, dense,
routed-MoE, attention, recurrent-state, and
sampling kernels are selected through the shared model graph and backend plan.

### Apple silicon and Metal

The native Metal worker uses C++ model graphs with MLX/Metal operations. Packed
GEMV and small-M paths consume MFQ storage directly; large-M paths can
temporarily dequantize for matrix multiplication without keeping a persistent
FP16 weight copy. DeepSeek V4/V4.1 and Qwen Flash-Next runtimes also support
bounded expert residency, while DeepSeek V4.1 can stream Engram storage from
the internal SSD.

### AMD GPUs and ROCm (HIP)

The HIP worker compiles the CUDA model graph and kernels through ROCm/HIP on
Linux. Dense Qwen3.5–3.8 models (BF16 and quantized `S4` tiers) are validated
end to end on MI300X, together with the torch JIT quantization extension.
Known gaps: multi-tile MFMA attention fails above 32 query tokens, so set
`MFQ_DISABLE_MMA_ATTENTION=1` to route prefill through the flash FMA path;
quantized GEMM uses a scalar dot-product fallback ahead of native MFMA ports;
and several families (DeepSeek V4, GLM DSA sparse, MiniCPM-o 4.5) remain
stubbed. See the [HIP validation report](hip-validation.md).

MiniCPM-o setup and modality details are in the
[MiniCPM-o 4.5 runtime guide](minicpmo45.md).

Build requirements and backend selection: [`mfq build`](../cli/build.md).
Serving and model-pool configuration: [`mfq serve`](../cli/serve.md).
