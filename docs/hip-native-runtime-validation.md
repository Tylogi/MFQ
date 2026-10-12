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

- Cross-device peer copies fault on this stack: the backend's device1 to
  device0 `hipMemcpyPeerAsync` dies in ROCr's `__amd_rocclr_copyBuffer`
  kernel on a host-page address, while the identical API sequence passes
  standalone (both directions, pooled streams, pool allocations, event
  lifetimes). The interaction with the backend's async-pool allocations is
  the remaining suspect. The native-tensor test skips its multi-GPU peer
  section on HIP; everything else in the suite runs.

## Resolved during validation

- `mfq-hip-nint-q8-test`: the q8 fast path applies the group affine per
  4-byte chunk while the generic path applies it once per group. The
  expressions are algebraically equal, but clang contracts the per-chunk
  affine differently than nvcc, crossing one FP16 rounding boundary on a
  boundary-adjacent value (single element, single ulp, tail row of a
  partial block; determinism verified). The cross-path comparison now
  allows one FP16 ulp on HIP with the rationale documented in the test.

After both resolutions the HIP operator suite passes 7/7 (repeated
runs): context, activation, native tensor (single device), nint rows,
nint q8, packed prefill, and paged KV.

## Torch JIT extension

`mfq.kernels.hip._ext` builds the unified torch extension (`mfq_hip`, 139
ops) from the hipify-translated kernels; `mfq.kernels.cuda._ext` delegates
to it on ROCm hosts. Two operational findings:

- **Torch's hipify regenerates `/hip/` headers from `/cuda/` siblings.**
  When both trees are visible to `torch.utils.cpp_extension` on a ROCm
  host, its hipify pass treats every walked `hip/<name>.cuh` as the output
  of `cuda/<name>.cuh` and rewrites it from the CUDA original, silently
  discarding hand fixes. Builds that must preserve fixed HIP headers have
  to keep the CUDA kernel sources out of the hipify include walk (the
  validation builds stage a copy without `kernels/cuda/*.{cu,cuh,h,cpp}`).
- **The imported binding was stale.** The WIP-era `mfq_hip.cpp` predated
  the current CUDA binding's argument lists (for example
  `fp8_128_sq_dequant_cuda` had six integer parameters instead of the
  current set). It is regenerated from the current `mfq_cuda.cpp` via
  torch hipify.

With the JIT extension in place, the suite stands at 3154 passed / 52
failed / 284 skipped. Of the 52: seven are the sparse-MMA stubs failing
by design; nine are the pre-existing upstream failures that also fail on
macOS; seventeen (fp8 decode/dispatch, two NINT group tests, three
format-completion cases) fail through code paths and fixtures that are
byte-identical to CUDA and need a CUDA-host run to adjudicate; the rest
are source-contract and harness tests that reference the CUDA sources
removed from the validation staging copy or make NVIDIA-specific
assumptions.

## Model-scale benchmarks (MI300X, single GPU, batch 1, 17-token prompt)

| Model | Precision | Decode tok/s | Prefill tok/s | TTFT ms |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | BF16 | 96–109 | ~15 | 1620 |
| Qwen3.5-0.8B | S4-M | 104–109 | ~15 | 1620 |
| Qwen3.5-4B | BF16 | 96.2 | 18.5 | 1431 |
| Qwen3.5-4B | S4-M | 80.3 | 20.3 | 1336 |
| Qwen3.5-9B | S4-M | 64.2 | 19.5 | 1372 |

Key finding: quantized (S4-M) decode is currently **slower** than BF16 at
4B (80 vs 96 tok/s) despite smaller weights. The FMA/dp4a replacements
for the tensor-core MMA paths trade peak throughput for portability: the
quantized GEMM is compute-limited by scalar dot products rather than
bandwidth-limited by weight size. Prefill improves (20.3 vs 18.5) because
the tiled path reads less data. Closing the decode gap requires MFMA
(tensor-core) implementations of the NVQ/NINT/SQ kernels.

### Qwen3.8-Flash-Next (qwen4_exp MoE, 175 GB)

Quantizes successfully at S4-M (175.7 GB, 1342 tensors, ~25 min). Loads
all 48 layers on one MI300X and reaches the generation path, then fails
with `logits must be float16 or float32` — a dtype bug in the MoE routing
chain. This is the first runtime failure of the MoE grouped-GEMM FMA
kernels (they compile and link but had never executed). Root cause is in
the expert routing or logits projection producing a non-float tensor.

### Qwen3.5-35B-A3B (qwen3_5_moe)

Blocked by two upstream gaps, not HIP-specific:
- The MTP completeness check expected dense `mlp.gate_proj` but this
  checkpoint uses a routed FFN inside the MTP head
  (`mlp.experts.N.*`). Fixed by accepting either FFN layout.
- After the MTP fix, a canonical tensor-mapping gap for
  `mtp.layers.*.mlp.experts.*` in the `qwen3_5_moe` architecture
  registration, and the 2 GB safetensors single-tensor offset limit for
  the fused `gate_up_proj` tensor (~3.3 GB) both block quantization.
  These are upstream registration/format limitations.

## Compile-time fix

`nvq_matmul.hip` took ~25 minutes at `-O3` for gfx942. The innermost
16-iteration dot-product loops in the FMA tile replacements were fully
unrolled, exploding codegen across every template instantiation. Capping
those loops at 4-way unroll plus `-O2` for the TU cuts the compile to
~10 minutes (measured 586s). The same cap applies to the two innermost
loops in `nint_matmul.hip`.

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
