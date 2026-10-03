# Native CUDA runtime validation

MFQ's production CUDA inference executable is `mfq-runtime`. It owns tensor
storage, CUDA streams and events, CUDA Graphs, and generic graph operations
without Python, PyTorch, ATen, or LibTorch. The packed MFQ CUDA kernels are the
same kernels used by the optional reference executable.

`mfq-runtime-torch` is an opt-in migration target. It exists only for A/B
validation and is excluded from normal builds and packages.

## Operator regression targets

`mfq-native-tensor-cuda-test` covers padded, sliced, transposed and broadcast
matrix products (including zero contraction and CUDA Graph capture), small-K
selection with stable ties, and tiled grouped-query attention. The attention
path bounds score storage to 128 query rows and shares the K/V head storage.

`mfq-packed-prefill-test` checks FP8-SQ CPU decoding and packed slicing,
compares MXFP4-SQ Tensor Core execution with the direct packed kernel, and
checks native TopK/graph behavior while printing timings for retained shapes.
`mfq-mxfp4-sq-test` consumes independently decoded wire fixtures, including
packed column/row slicing and CPU decoding. Generate them and check SQ linear
CPU execution and shard composition with:

```shell
python bench/cuda_mxfp4_sq_fixtures.py build/sq-fixtures
build/cuda-native/mfq-mxfp4-sq-test build/sq-fixtures
build/cuda-native/mfq-sq-linear-test build/sq-fixtures
```

SQ tensor-parallel partitions preserve the native scale blocks: MXFP4-SQ input
partitions align to 32 columns; FP8-SQ partitions align to the format's row and
column scale block boundaries. The single-GPU shard test exercises both output
gather and input reduction; it does not certify inter-device transport.

The CUDA workflow is manual and requires a dedicated `mfq-cuda` runner and the
`MFQ_CI_GPU_UUID` repository variable. It executes GPU numerical and graph tests;
source-contract tests alone do not establish CUDA correctness. The macOS
workflow also runs Qwen3.5 and MiniCPM-o text-prefill chunking tests.

MXFP4-SQ uses Tensor Core tiles by default for `M >= 32` and `M*N >= 131072`,
where the retained RTX 3090 Ti comparisons showed gains; smaller projections
keep direct packed execution. `MFQ_FORCE_SQ_TENSOR_CORE=1` tests that kernel on
smaller shapes. `MFQ_DISABLE_SQ_TENSOR_CORE=1`, `MFQ_DISABLE_NATIVE_STRIDED_BATCH_MATMUL=1`, and
`MFQ_DISABLE_NATIVE_TILED_SDPA=1` retain comparison paths for their corresponding
optimizations. These operator checks do not establish model-level token parity
or replace full-model quality evaluation.

## Build and dependency checks

Configure the production runtime on both Linux and Windows:

```shell
cmake -S cpp_runtime -B build/cuda-native \
  -DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=OFF \
  -DBUILD_TESTING=ON
cmake --build build/cuda-native --config Release -j
ctest --test-dir build/cuda-native -C Release --output-on-failure
```

The resulting executable must not import or link Python, Torch, ATen, c10, or
LibTorch. Check the final dependency table with `ldd` on Linux and
`dumpbin /DEPENDENTS` on Windows. CUDA runtime, CUDA driver, cuBLAS, the host C++
runtime, and optional NCCL are expected.

For an A/B build, configure a separate tree with LibTorch discoverable and
`-DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=ON`. This adds `mfq-runtime-torch`; it does
not alter `mfq-runtime`.

Execution policy environment variables are parsed when a
`CudaExecutionContext` is constructed. Each Engine keeps its own typed
`CudaExecutionConfig`; changing the process environment afterwards does not
reconfigure an existing Engine. Diagnostic A/B paths must override the target
Engine's config directly or construct a fresh Engine.

Run the model-backed scheduler and execution-isolation gates on a machine with
enough device memory. The second command loads and generates with two complete
engines concurrently and compares each result with its serial oracle:

```shell
mfq-diagnostics --model model.mfq --check-continuous-batching
mfq-diagnostics --model model.mfq --check-engine-isolation
```

The continuous-batching gate covers stable slot retirement, paged-KV reuse and
release, CUDA Graph capture/replay, callback cancellation, and prefix reuse.

## Tensor and expert parallel execution

The native runtime accepts either a rank count or an ordered CUDA device list.
Split weights stay attached to that device order:

```shell
mfq-runtime --model model.mfq --tensor-parallel 4
mfq-runtime --model moe.mfq \
  --expert-parallel 0,1,2,3 --expert-split 1,1,1,1
mfq-runtime --model moe.mfq \
  --tensor-parallel 0,1,2,3 --tensor-split 1,1,1,1 \
  --expert-parallel 0,1,2,3 --expert-split 1,1,2,4
```

Combined tensor and expert parallel execution currently uses one NCCL rank
group, so both ordered device lists must match. Their split weights may differ.
For compatibility with older launches, routed experts inherit the tensor
parallel device group when `--expert-parallel` is omitted.

Four- and eight-rank changes must pass `mfq-tensor-parallel-test` before a CUDA
run. A real multi-GPU gate must then cover dense tensor shards, routed expert
ownership, NCCL reduction, eager execution, and CUDA Graph replay. Eight-rank
code is not considered device-validated until that physical gate runs.

## Required A/B matrix

Run every row with fixed model files, prompts, seeds, cache limits, and sampling
parameters. Start each executable in a fresh process.

| Area | Required coverage |
| --- | --- |
| Architectures | dense causal LM, GQA, multimodal MiniCPM-o, and routed MoE |
| Formats | dense BF16/F16, NINT, NVQ/NPQ/NEPQ, MXFP8, and MXFP4 where the architecture permits them |
| Shapes | single-token decode, short and long prefill, odd sizes, batched inputs, and GQA head broadcasting |
| State | empty cache, reused prefix cache, context rollover, session reset, and interrupted generation |
| Sampling | greedy, temperature, top-k, top-p, min-p, repetition/presence penalties, and fixed random seed |
| Execution | eager, CUDA Graph capture/replay, one and multiple CUDA streams, one GPU, and tensor/expert parallel when NCCL is present |
| Media | text, image, video, audio, half duplex, and full duplex |

Capture raw logits from both executables at prefill and every decode step. The
packed kernel bodies are unchanged and keep their existing numerical contract.
Generic reductions and cuBLAS may use a different reduction tree from ATen.
Accept results only when logits stay within the model's established tolerance
and greedy tokens match. Explain any token divergence before merging.

Also compare peak device memory, host memory, model-load time, prefill speed,
and decode speed. The native runtime is rejected if it introduces a material
regression or silently falls back to host computation.

## Packaging gate

Create clean Windows and Linux packages from the native build. Install each in
an environment that has no `torch` package and no LibTorch files, then verify:

1. empty-server startup and model discovery;
2. loading and unloading representative dense and MoE models;
3. OpenAI-compatible streaming and non-streaming generation;
4. multimodal processing through MFQd;
5. process shutdown without leaked workers; and
6. package size and dependency inventory.

The legacy MiniCPM-o diagnostic `.pt` filenames use MFQ's `MFQTNSR1` tensor
envelope in the native executable. Python pickle files created by `torch.save`
are intentionally handled only by `mfq-runtime-torch`; production HTTP media
requests do not cross this file boundary.

## CUDA backend modularization (2026-10-03)

Kernel declarations now live in domain headers for norm, activation, attention,
linear attention, MoE, quant, cache, and sampling. Callers include the domains
that they use. Each Engine load owns its MoE cache; target and predictor weights
retain that cache through aliasing source handles, including copied projection
operations. Cache registration and lifecycle APIs take cache resources directly.

Qwen full/linear attention, GLM DSA indexer/MLP choices, MiniCPM text/TTS norm
semantics, and common dense/MoE parameter assembly live in shared models.
CUDA loaders retain quantization layout, projection fusion, important-neuron
materialization, CPU placement, and TP/EP choices.

The build boundaries use the existing targets:

| Target | Responsibility |
| --- | --- |
| `mfq-models`, `mfq-engine` | Backend-neutral model definitions/configuration and Engine execution |
| `mfq-cuda-ops` | CUDA numerical operations and execution resources; no Engine/model-family/transport dependency |
| `mfq-cuda-loading` | Weight materialization, cache loading and model-load bindings |
| `mfq-cuda-runtime` | Concrete Engine/model instantiation and native execution plans |

Native and optional Torch executables use the same loading and instantiation
source lists. Only runtime executables link transport. The generated Ninja
order dependencies for `mfq-cuda-ops` contain only shared model-graph and
model-source targets, with no Engine, model-family, scheduler, or transport.

Incremental build probes used the existing Release/Ninja/CUDA 13.2 build. After
a completed build, touch each source and run `cmake --build build/cpp_runtime`.
Each probe compiled exactly one translation unit before and after the boundary
change; no unrelated CUDA/model compilation occurred.

| Touched source | Before (seconds) | After (seconds) | Recompiled target |
| --- | ---: | ---: | --- |
| `backends/cuda/ops/format.cpp` | 0.96 | 0.99 | `mfq-cuda-ops` |
| `engine/src/generation_policy.cpp` | 0.56 | 0.57 | `mfq-engine` |
| `models/qwen35/config.cpp` | 2.11 | 2.21 | `mfq-models` |

These are single warm-build observations including required relinks, not a
performance benchmark. There is no measured benefit supporting additional
library targets, so none were added. Splitting targets would not remove the
concrete instantiations' actual template-header dependencies.

Validation: full native build and all 52 CTest tests passed on the local RTX
4090. The offset regression compares embedding and KV writers against CPU
assignment with nonzero aligned/unaligned offsets, dispatch thresholds, FP16,
BF16 and FP32, independent K/V/cache misalignment, per-batch positions, ring
wraparound and untouched storage guards. The cache regression covers independent
instances, execution reset, retained projections and final destruction. Shared
assembly tests cover full/linear attention, dense/split/fused MoE, GLM indexers,
MiniCPM norm variants and malformed shapes/partial expert representations.

`pytest -q tests/test_cpp_runtime_*` passed 203 tests with 11 skips. Full Python
execution with `--continue-on-collection-errors` had 1012 passes, 146 skips,
17 failures and 56 collection errors. Its limitations remain separate from
this refactor: Torch/MLX/SciPy are missing,
and the health-metrics and prefill-timing source contracts already fail at the
starting revision `96be1553`. Real model numerical/performance A/B, multi-GPU
communication and the LibTorch build remain subject to the gates above; these
modularization checks do not substitute for them.
