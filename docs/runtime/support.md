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
| GLM5–5.3 | Native C++ | Reference | — | DSA/KDA and Flash-Next families; GLM-5.3-Flash CUDA image/video Vision with optional MTP; Metal production C++ worker integration remains open |
| Gemma 4 | Native C++ | Reference | — | Mixed full/sliding attention and MoE; Metal production C++ worker integration remains open |

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
  Continuous Batching is active on supported native paths and is being expanded
  across the remaining model graphs.
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

GLM-5.3-Flash Vision uses the graph component `rms_swiglu_vit`, input contract
`grid_vision.v1`, and `sequential` text positions. `mfq serve` selects
`Glm5NextVisionProcessor` and sends its `grid_vision.v1` image/video payload through
the native worker's stdio transport. Vision and MTP default on when declared
and loaded. The tower accepts native BF16/F16/F32 or NINT projection weights;
`MFQ_CUDA_VISION_RESIDENT=1` keeps vision weights on the GPU instead of staging
each encoder stage from RAM.

The numerical and stdio regression tests cover BF16/NINT8, image/video prompt
order, sequential positions, cached embeddings, resident/staged weights, and
MTP present/absent. Build the runtime and tests in Release with `sm_120`:

```sh
cmake -S cpp_runtime -B ~/work/build-vision -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DMFQ_CUDA_ARCHITECTURES=120 -DMFQ_BUILD_CUDA_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build ~/work/build-vision -j 8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
ctest --test-dir ~/work/build-vision --output-on-failure
PYTHONPATH=. MFQ_CUDA_RESOURCE_TEST_RUNTIME=~/work/build-vision/mfq-runtime python -m pytest -q tests/test_cuda_glm5_vision_native.py tests/test_cuda_qwen4_vision_native.py
```

For native Windows, run from an MSVC 2022 x64 developer PowerShell with CUDA
13.0 and the project's Python environment (no WSL path translation):

```powershell
cmake -S cpp_runtime -B build-vision-windows -G "Visual Studio 17 2022" -A x64 -T "cuda=$env:CUDA_PATH" -DMFQ_BUILD_CUDA_RUNTIME=ON -DMFQ_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=ON
cmake --build build-vision-windows --config Release --parallel 8
ctest --test-dir build-vision-windows -C Release --output-on-failure
$env:PYTHONPATH = (Get-Location).Path
$env:MFQ_CUDA_RESOURCE_TEST_RUNTIME = (Resolve-Path build-vision-windows/Release/mfq-runtime.exe).Path
$env:MFQ_CUDA_VISION_NATIVE_TEST = (Resolve-Path build-vision-windows/cuda/Release/mfq-rms-swiglu-vision-test.exe).Path
python -m pytest -q tests/test_cuda_glm5_vision_native.py tests/test_cuda_qwen4_vision_native.py
```

On Linux the bridge is discovered at `cuda/mfq-rms-swiglu-vision-test` in the
build directory; set `MFQ_CUDA_VISION_NATIVE_TEST` if it is elsewhere. The new
GPU tests skip when the bridge reports no CUDA device. Compilation and CPU
checks alone do not verify CUDA numerics or the native Windows build.

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
