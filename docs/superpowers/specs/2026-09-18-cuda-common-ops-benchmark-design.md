# CUDA Common Operators Benchmark and Incremental Optimization

## Goal

Create a model-free native CUDA microbenchmark for the reusable kernels in
`mfq/kernels/cuda`, then optimize those kernels one small stage at a time. Each
stage must establish a baseline before changing a kernel and must report
correctness and performance separately.

This work does not benchmark the HTTP service, load model weights, or change
model architecture and dispatch policy.

## Ownership boundary

The common-operator runner exercises reusable CUDA entry points from
`mfq-cuda-native-kernels`, such as residual addition, fused residual/RMSNorm,
activation, RoPE, KV-cache writes, and embedding. It is distinct from:

- `cuda_paged_kv_attention_bench.cu`, which owns paged-attention state and
  workspace construction;
- generic Tensor API benchmarking in `mfq_native_tensor_ops.cu`;
- model and server end-to-end benchmarks.

The runner may use native Tensor objects to allocate and describe inputs, but
the measured subject is the public CUDA-kernel entry point used by the native
runtime.

## Runner design

Add one executable, `mfq-cuda-common-ops-bench`, backed initially by one source
file: `bench/cuda_common_ops_bench.cu`. The first positional argument selects
the operation. Operation-specific code owns input construction, validation,
and shape parameters. Shared code owns device selection, warmup, CUDA Event
timing, sample statistics, and JSON output.

Example commands:

```text
mfq-cuda-common-ops-bench acc --rows 1 --width 4096 --dtype f16
mfq-cuda-common-ops-bench acc-rms-norm --rows 128 --width 4096 --dtype f16
```

The JSON record contains a schema version, GPU identity, operation name,
parameters, correctness result, raw samples, median, and p95 latency. The
runner returns 77 when no CUDA device is present. It is built with
`BUILD_TESTING=ON` but performance thresholds are not registered with CTest.

The initial implementation remains in one file. Shared utilities are extracted
only after a second benchmark executable or enough operation cases create real
reuse. Operation cases may be split into separate translation units after the
runner grows beyond a manageable size; the executable name and JSON schema
remain stable.

## Incremental stages

### Stage 1: measurement foundation

- Add the CMake target and the common-operator runner.
- Support only `acc` and `acc-rms-norm`.
- Cover FP16, BF16 where the entry point supports it, and FP32.
- Include decode-sized rows, prefill-sized rows, and non-aligned widths.
- Validate output outside the timed region against a host or existing CUDA
  reference without widening existing tolerances.
- Capture baseline JSON on the local RTX 4060 Laptop GPU.

No production kernel changes occur in this stage.

### Stage 2: first kernel optimization

- Change only `acc.cu`.
- In the generic fused residual/RMSNorm kernel, consume the already-rounded
  `sum_out` value in the normalization pass instead of rereading `a` and `b`
  and recomputing the residual sum.
- Preserve the existing FP16/FP32 result contract and leave the dedicated BF16
  path unchanged.
- Add focused correctness coverage for aligned and tail widths.
- Re-run the exact Stage 1 matrix and compare median and p95 results.

The change is accepted only if correctness is unchanged and target shapes show
a repeatable improvement without a material regression elsewhere. Otherwise it
is reverted or guarded by a shape-specific dispatch.

### Later stages

Add and optimize one family at a time:

1. activation;
2. norm;
3. RoPE;
4. KV-cache writes;
5. embedding.

Each family first gains benchmark cases, then receives one isolated kernel
change. Paged attention, GDN, sampling, quantized matmul, and generic Tensor
`topk` remain separate scopes.

## Correctness and performance protocol

Correctness covers normal, zero-length where the API permits it, and boundary
widths around vector and warp sizes. FP16/BF16/FP32 use their established
contracts. Vectorized paths must retain scalar-tail coverage.

Performance measurement uses the current CUDA stream, pre-created timing
events, warmup iterations, multiple timed samples, and several iterations per
sample. Input creation, host copies, and reference calculation stay outside the
timed interval. Results report raw samples plus median and p95 rather than only
the best or average value.

Before/after measurements use the same executable arguments, GPU, build type,
and process conditions. Baseline and optimized JSON files are retained as
review artifacts; the production runtime does not retain a permanent legacy
dispatch solely for benchmarking.

## Out of scope

- Running or modifying the model service.
- Loading or converting model checkpoints.
- Optimizing more than one operator family in a stage.
- Introducing NVBench, Google Benchmark, PyTorch, or another benchmark
  dependency into the native runner.
- Adding hardware-dependent performance assertions to CI.
