# Native HIP runtime validation

Status of the HIP backend (`cpp_runtime/backends/hip`, `mfq/kernels/hip`) on
ROCm 7.14 / Instinct MI300X (gfx942). The backend is generated from the CUDA
sources (torch hipify for the imported kernel trees, hipify-perl for host
translation units) with the NVIDIA tensor-core tile paths replaced by direct
FMA/dp4a implementations; see commit history for the exact transformations.

## Validated

- **Build**: `mfq build --backend hip` exits zero; `mfq-runtime`,
  `mfq-diagnostics`, and `mfq-eval` compile and link for gfx942.
- **GPU quantization**: the offline quantize kernels build through
  `mfq.quantize.hip._ext` and pass their GPU-vs-CPU parity suites (nint/nvq/
  npq0/nepq0 assign+pack, search packing, second order, alphaq: 231 tests).
- **Quantized end-to-end generation**: Qwen3.5-0.8B quantized with the S4-M
  preset (backbone NINT4/NINT6, norms and MTP at source precision) loads and
  generates coherent text with greedy sampling. First tokens match the BF16
  baseline exactly and outputs diverge only later, the expected signature of
  correct 4-bit quantization. Decode runs at ~104 tok/s with MTP enabled on
  one MI300X.
- **C++ operator tests** (CUDA test sources compiled directly as HIP
  translation units through `tests/cuda_compat`): context, activation,
  nint-rows, packed-prefill (MXFP4 tile path vs direct packed execution),
  and paged-kv pass on device.

## Failing (known gaps)

- `mfq-hip-native-tensor-test` aborts with a GPU memory fault inside a ROCr
  copy buffer kernel during the native tensor exercise (strided/batched
  matmul, graph capture, SDPA). Not yet isolated.
- `mfq-hip-nint-q8-test` reports the NINT q8 fast path differing from the
  generic reference path. Suspect: the direct FMA replacement of the NINT
  prefill wmma tiles. Needs a minimized repro against the CPU decoder.

## Not yet exercised

- MoE grouped-GEMM kernels at runtime (build only).
- DeepSeek DSV4 compression/HC, GLM DSA/MLA, MiniCPM paths: stubbed with
  runtime errors until ported.
- Continuous batching beyond single requests, multimodal, duplex.

## Environment notes

- The torch wheel's bundled ROCm lacks the rocBLAS Tensile library data;
  copy the host ROCm's `lib/rocblas/library` into the image's
  `_rocm_sdk_devel/lib/rocblas/` or point `ROCBLAS_TENSILE_LIBPATH` at it,
  otherwise hipBLAS aborts at the first GEMM.
- `mfq/kernels/hip/nvq_matmul.hip` takes ~25 minutes at `-O3` for gfx942.
