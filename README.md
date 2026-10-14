<div align="center">

# TyloQuant MFQ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/figures/tylogi-ai-lab-lockup-dark.svg">
  <img src="./docs/figures/tylogi-ai-lab-lockup-light.svg" alt="Tylogi AI Lab" width="420">
</picture>

### Next-generation quantization & inference infrastructure

**Every Bit. Maximum Fidelity.**

MFQ combines neural-network-aware SQ and VQ formats designed to approach the
practical rate-distortion frontier, high-quality, fine-grained mixed-precision
calibration and quantization, and efficient CUDA, ROCm (HIP), and Metal
inference. The goal is
to deliver the best model quality possible on existing hardware within its
memory, storage, and latency limits.

<p>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0 license">
  <img src="https://img.shields.io/badge/runtime-C%2B%2B-black" alt="C++ runtime">
  <img src="https://img.shields.io/badge/backends-CUDA%20%7C%20ROCm%20%7C%20Metal-6b57ff" alt="CUDA, ROCm and Metal">
</p>

<p>
  <strong>English</strong> · <a href="./README.zh-CN.md">中文</a>
</p>

<p>
  <a href="#deepseek-v41-raw-hf">DeepSeek V4.1</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#features">Features</a> ·
  <a href="#models">Models</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="./docs/README.md">Documentation</a>
</p>

</div>

<a id="deepseek-v41-raw-hf"></a>

## DeepSeek V4.1 raw-HF on one Mac Studio

> **Run a 476 GB DeepSeek V4.1 Flash checkpoint directly from Hugging Face
> Safetensors on a single 512 GB Apple M3 Ultra—without converting it to an
> MFQ container first.** The tuned Metal path keeps the backbone and MoE
> experts resident in unified memory while offloading only Engram storage to
> the Mac's internal SSD.

| Model / hardware | Placement | Model load | Prefill, ~0.5K / ~2K / ~16K tokens | TG, MTP off | TG, high-acceptance MTP |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash raw-HF / Mac Studio M3 Ultra, 512 GB | Full resident; Engram only on internal SSD | **38.6 s** | **390.2 / 458.6 / 471.1 tok/s** | **18.29 tok/s** | **33.46 tok/s** (**+83.0%**) |

MFQ covers calibration, precision allocation, quantization, and inference.
Create self-contained `.mfq` models with tensor-, expert-, and projection-level
mixed precision, then deploy them with MFQ Studio or the inference API.

<p align="center">
  <img src="./docs/figures/tyloquant-mfq-webui-english.png" alt="MFQ Studio running a local model" width="900">
</p>

## Quick start

### Use a prebuilt package

The simplest Apple silicon path is a prebuilt **MFQ Studio** package. Prebuilt
packages are published through the project
[Releases](https://github.com/Tylogi/TyloQuant/releases) when available and
bundle the desktop console, MFQ Server, and C++ runtime. Open Studio,
register a model directory, and load any supported `.mfq` model.

Prebuilt C++ workers can also be supplied directly to the server:

```bash
uv run mfq serve \
  --running-executable /path/to/mfq-decode-metal \
  --model /models/model.mfq
```

### Build from source

MFQ requires Git, [uv](https://docs.astral.sh/uv/), CMake 3.26+, and a
C++ toolchain. Use CUDA 12+ on NVIDIA systems, ROCm 6+ on AMD systems, or
Metal on Apple silicon.

```bash
git clone https://github.com/Tylogi/TyloQuant.git MFQ
cd MFQ

# NVIDIA / CUDA
uv sync
uv run mfq build --backend cuda

# AMD / ROCm (HIP)
uv sync
uv run mfq build --backend hip

# Apple silicon / Metal
uv sync --extra metal
uv run mfq build --backend metal
```

Start an empty local server and open <http://127.0.0.1:8090/>:

```bash
uv run mfq serve
```

Detailed requirements and custom CMake options are in the
[`mfq build` guide](./docs/cli/build.md).

## Quantize and run a model

Use the MFQ **`S4-M` mixed-precision preset**. It keeps sensitive tensors at
higher precision and leaves Vision and MTP components at source precision by
default.

```bash
# Quantization dependencies
uv sync --extra train

# Hugging Face safetensors -> self-contained MFQ
uv run mfq quantize \
  /models/Qwen3.8-27B \
  /models/Qwen3.8-27B-MFQ-S4-M.mfq \
  --preset S4-M \
  --backend auto
```

Load and serve the result:

```bash
uv sync                                # add --extra metal on Apple silicon
uv run mfq build --backend auto
uv run mfq serve \
  --model /models/Qwen3.8-27B-MFQ-S4-M.mfq \
  --context-size 32768
```

MFQ can also quantize full-precision MFQ and GGUF sources, consume per-tensor or
Expert-Wise precision maps and activation importance matrices, and write
sharded output. See
[`mfq quantize`](./docs/cli/quantize.md) and
[`mfq calibrate`](./docs/cli/calibrate.md).

## Features

### Quantization

- **Fine-Grained Quantization & Precision Virtualization.** Assign precision
  to individual experts, projections, chunks, or neurons within unified MFQ
  formats, retaining higher precision where the model needs it most.

- **High-Quality Calibrated Models & Rapid Architecture Support.** Download
  ready-to-run mixed-precision models, with quantization and inference support
  for new architectures added regularly.

### Inference

- **Production Inference & Serving.** Continuous Batching, multi-model loading
  with LRU management, tiered RAM/SSD Prefix and KV caches, and SSD-streamed
  experts let you serve multiple models and run MoE models beyond memory capacity.

- **High-Performance CUDA, ROCm & Metal Backends.** Optimized packed-weight,
  attention, cache, and I/O kernels for CUDA and Metal, with early ROCm/HIP
  support for dense Qwen3.5–3.8 models on MI300X. Optimized CPU backends are planned.

| Model | Model size | Expert budget | Precision | Hardware | Prefill | Decode |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| DeepSeek-V4-Flash-0731 | ~160 GiB | 85.3 GiB | Official Native QAT Precision | Apple M5 Max, 128 GB | **312.4 tok/s** | **18.6 tok/s** |

MFQ keeps heterogeneous MoE execution efficient at both supported precision
granularities:

| Model | Precision granularity | Average BPW | Prefill | Decode |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | One precision per expert | **5.80** | **1.13K tok/s** | **25.4 tok/s** |
| Qwen3.8-Flash-Next | Independent precision for each expert's Gate/Up/Down | **5.79** | **1.12K tok/s** | **24.4 tok/s** |

*Warm end-to-end throughput on Apple M5 Max (40-core GPU, 128 GB).*

## Models

Published `.mfq` models are available from
[Hugging Face](https://huggingface.co/Tylogi) and
[ModelScope](https://www.modelscope.cn/profile/Tylogi).

| Model | Published precision families | Download |
| --- | --- | --- |
| DeepSeek-V4-Flash-0731 | Expert-Wise `V2` tiers | [Hugging Face](https://huggingface.co/Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ) · [ModelScope](https://www.modelscope.cn/models/Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ) |
| Qwen3.8-27B | `V1`–`V4`, `S4`–`S6` tiers | [ModelScope](https://www.modelscope.cn/models/Tylogi/Qwen3.8-27B-MFQ) |
| Qwen3.6-27B | `V2`–`V3`, `S2`–`S6` tiers | [Hugging Face](https://huggingface.co/Tylogi/Qwen3.6-27B-MFQ) |
| MiniCPM-o 4.5 | `S4`–`S8` multimodal tiers | [ModelScope](https://www.modelscope.cn/models/Tylogi/MiniCPM-o-4_5-MFQ) |

Supported architecture groups:

| Architecture group | Backends |
| --- | --- |
| Qwen3.5–3.8 | CUDA C++, Metal C++, HIP C++ (dense models) |
| Qwen Flash-Next / Qwen4-style | CUDA C++, Metal C++ |
| DeepSeek-V4-Flash Series | CUDA C++, Metal C++ |
| MiniCPM-o 4.5 | CUDA C++, Metal C++ |
| GLM5–5.3 | CUDA C++; Metal reference runtime |
| Gemma 4 | CUDA C++; Metal reference runtime |

See the [runtime support matrix](./docs/runtime-support.md) for details.

## Tools and APIs

| Surface | Purpose |
| --- | --- |
| **MFQ Studio** | Local model catalog, load lifecycle, inference playground, server state, and resource controls |
| `mfq build` | Detect the platform and compile the optimized CUDA or Metal C++ worker |
| `mfq quantize` | Apply uniform or mixed precision to HF, GGUF, or full-precision MFQ sources and write packed `.mfq` models |
| `mfq calibrate` / `mfq solve-ew` | Collect activation and loss-sensitivity data, score and validate packed candidates, and allocate precision to exact serialized-byte budgets |
| `mfq serve` | Run MFQ Server, Studio/Web UI, model workers, caches, and persistence |
| `mfq inspect` / `mfq optimize-layout` | Inspect containers and repack tensors into backend-optimized layouts |

MFQ exposes several interfaces for applications and tooling:

- **HTTP control API** under `/api/v1` for models, sessions, responses, jobs,
  datasets, evaluations, caches, and runtime state;
- **Server-Sent Events** for token and job streaming;
- **WebSocket realtime APIs** for audio and full-duplex sessions;
- **OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints** from
  managed C++ workers and runtime adapters;
- **Function-tool calling** through the inference API; clients execute tools and return results.

Start with the [HTTP API](./docs/api/http.md),
[WebSocket API](./docs/api/websocket.md), or
[`mfq serve` reference](./docs/cli/serve.md).

## Benchmarks

### DeepSeek-V4-Flash-0731: model size vs. distribution fidelity

<p align="center">
  <img src="./docs/figures/deepseek-v4-flash-mfq-vs-ud-kld.svg" alt="DeepSeek-V4-Flash MFQ versus matched-size baseline Mean KLD" width="900">
</p>

The evaluation uses the official 0731 weights and a fixed WikiText-2 protocol
covering 573 chunks and 146,115 scored tokens at `ctx=512`.

| Released tier | Size | Mean KLD ↓ | Same-top ↑ |
| --- | ---: | ---: | ---: |
| `EW-V2-S` | 77.519 GiB | `0.313576` | `82.2913%` |
| `EW-V2-M` | 88.007 GiB | `0.244488` | `84.5300%` |
| `EW-V2-L` | 98.007 GiB | `0.201444` | `86.0753%` |

Against the nearest-size Unsloth Dynamic baselines used in this evaluation,
MFQ lowers Mean KLD by **34.24–51.42%**.

### Qwen3.5-9B: matched precision tiers

<p align="center">
  <img src="./docs/figures/qwen35-9b-mfq-vs-ud-size-kld.svg" alt="Qwen3.5-9B MFQ versus matched-size baseline Mean KLD" width="900">
</p>

All plotted tiers use the same BF16 teacher and the complete 145-chunk,
148,335-scored-token evaluation. MFQ records lower raw Mean KLD at every
matched precision point shown.

## Documentation

- [Documentation index](./docs/README.md)
- [Build the C++ runtime](./docs/cli/build.md)
- [Quantize a model](./docs/cli/quantize.md)
- [Run MFQ Server](./docs/cli/serve.md)
- [Runtime support](./docs/runtime-support.md)
- [MiniCPM-o 4.5 multimodal runtime](./docs/minicpmo45.md)
- [Contributing and architecture rules](./CONTRIBUTING.md)

## Acknowledgements

MFQ is deeply inspired by and has learned from
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[oMLX](https://github.com/jundot/omlx),
[MLX](https://github.com/ml-explore/mlx),
[PyTorch](https://github.com/pytorch/pytorch),
[Transformers](https://github.com/huggingface/transformers), and
[Unsloth](https://github.com/unslothai/unsloth).

MFQ is licensed under the [Apache License 2.0](./LICENSE).
