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

The additional `mfq-hip-wave-test` checks width-8/16/32 shuffle semantics and
independent 32-lane ballots on multidimensional blocks. Run it on an AMD host
with `ctest --test-dir build/cpp_runtime -R mfq-hip-wave-test --output-on-failure`.
CPU shim and source-isolation tests in `tests/test_hip_portability.py` do not
replace this device check.

## Torch JIT extension

`mfq.kernels.hip._ext` builds the unified torch extension (`mfq_hip`, 139
ops) from the hipify-translated kernels; `mfq.kernels.cuda._ext` delegates
to it on ROCm hosts. Two operational findings:

- **Torch's hipify regenerates `/hip/` headers from `/cuda/` siblings.**
  When both trees are visible to `torch.utils.cpp_extension` on a ROCm
  host, its hipify pass treats every walked `hip/<name>.cuh` as the output
  of `cuda/<name>.cuh` and rewrites it from the CUDA original, silently
  discarding hand fixes. Builds that must preserve fixed HIP headers have
  to keep the CUDA kernel sources out of the hipify include walk. The loader
  now stages HIP kernels and their required headers in a content-addressed
  directory under the torch extension cache, without CUDA sibling trees.
  Unchanged inputs reuse that directory; changed inputs select a new one.
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

## Prefill fix: real attention MMA kernel

The earlier prefill numbers (13-20 tok/s) were an artifact of
`mfq_attention_mma256_cuda` being a throwing stub: any prompt long
enough to trigger the tiled attention path failed silently, and the
"prefill" time was mostly model loading. The kernel's hipified source
(`attention_mma.hip` -> `mfq_fattn_mma_f16.cuh`) already contains
ggml's native CDNA MFMA/WMMA paths — it was stubbed only because the
ggml-hip headers had unresolved identifiers at import time. Fixing the
remaining SIMD shims (`__vsubss4`, `__vcmpne4`, `__trap`) and the
`CUBLAS_TF32_TENSOR_OP_MATH` enum, then removing the stub, gives:

- Short prompts (~12 tokens): **391 tok/s prefill** (was 13 with the
  stub), 83 tok/s decode, 525ms TTFT.
- Longer prompts (~35+ tokens): `mfq_attention_mma256 launch failed:
  unspecified launch failure`. The initial explanation was a multi-tile
  launch limit; the isolated checks below correct that explanation.

### Attention launch correction (2026-10-13)

Direct calls to `mfq_attention_mma256_cuda` failed at both 32 and 33
query tokens on gfx942. Short model prompts used another attention path,
so their success did not validate the MMA kernel. With `AMD_LOG_LEVEL=3`,
the 33-token call failed at launch with a block of `(32, 4, 1)` and
33,792 bytes of dynamic shared memory.

The native build did not define `GGML_USE_HIP` or the CDNA architecture
macros. The device kernel therefore used the default 32-thread launch
bound and omitted the MFMA body. Meanwhile, the host treated HIP's
`major=9, minor=4` as NVIDIA compute capability 940 and selected a
128-thread NVIDIA configuration. The host and device configurations
did not match.

The attention compatibility header now enables ggml's HIP architecture
selection. The shared HIP shuffle shim preserves explicit widths. Host
configuration uses the AMD architecture parsed from `gcnArchName`, and
launches use
`properties.warpSize` (64 on MI300X). The 256-dimensional prefill path
uses at least 16 columns, as required by the CDNA kernel. The ggml HIP
vendor shim maps the CUDA TF32 math-mode name to `HIPBLAS_DEFAULT_MATH`
so the header also compiles in a clean native build.

The `mfq-hip-attention-mma-test` compares GPU output with a CPU causal
softmax reference using rounded FP16 keys and values. All 32 numerical
cases passed on MI300X: GQA ratios 4 and 8, two KV heads, batches 1 and 2,
and token counts from 1 to 257 including the 32/33, 63/64/65, 127/128/129, and
256/257 boundaries. Maximum relative error was 0.000316 and maximum
absolute error was 0.000527. An additional test checks that unsupported
512-dimensional CDNA tiles fail before launch rather than returning an
uncomputed output.

An isolated diagnostic executable linked with the corrected attention
object also completed Qwen3.5-4B uniform NINT4 prefill at 33, 129, 512, and 2048
tokens with `MFQ_DISABLE_MMA_ATTENTION` unset. The running server and its
build artifacts were not replaced. These checks cover the 256-dimensional
prefill path; other attention families and GPU architectures need
separate validation. The FMA override remains available.


### Clean-build verification (2026-10-14)

The feature branch was updated to upstream master `0cda1e0f`. A clean
native HIP build exposed two additional port gaps: the missing TF32
math-mode mapping described above and missing `gdn_weights`/`qsa_weights`
accessors in the HIP Qwen4 block loader. The accessors load the existing
combined-QKV GDN and QSA weight structures required by the shared loader.

After these fixes, the full native HIP build and all 41 registered CTest
cases passed on MI300X. The attention test includes 32 numerical cases
and the unsupported-tile rejection check. The newly built diagnostics
executable completed uniform NINT4 and S4-M Qwen3.5-4B prefill at 33, 129, 256,
512, and 2048 tokens with `MFQ_DISABLE_MMA_ATTENTION` unset. Both models
also generated 16 tokens from a 129-token input. At that input length,
MMA and FMA logits selected the same top token, with absolute KL
divergence below 0.0000022. These are functional checks, not a language
quality or performance benchmark.

The artifact named `qwen35-4b-bf16.mfq` was later found to contain 249
NINT tensors. The two checks above used uniform NINT4 weights, despite
that filename. A separate conversion of the original HF checkpoint
contains 690 BF16 tensors and 48 F32 tensors, with no NINT tensors. Use
the verified conversion for unquantized comparisons.

On macOS, all 32 backend-neutral CTest cases and 134 focused Python
cases passed. The full Python suite reported 2599 passed, 608 skipped,
and 42 failed. Each of the 42 failures was reproduced against unchanged
upstream `0cda1e0f`; no new failing cases were observed.

### Packed NINT decode optimization

The HIP decode kernel quantizes activations to signed bytes, reads four
packed weight codes, computes integer dots and activation sums, then
applies each group's scale and minimum before the FP32 reduction. NINT8
weights use the same signed dot after a byte bias, with the matching
activation-sum correction. This is the CUDA packed-integer algorithm.

The old scalar `__dp4a` expression was already lowered to
`v_dot4c_i32_i8_e32` by Clang 23 on gfx942, but a standalone assembly
probe showed ten extra vector instructions to repack its operands. The
explicit `__builtin_amdgcn_sdot4` uses the original packed registers.
Saturation is disabled to retain CUDA's wrapping accumulator. The
native branch is limited to gfx908, gfx90a and gfx942; other targets and
host translation units retain the scalar fallback. The first two targets
were compile-checked; GPU execution was checked on gfx942.

Single-token projections now select the existing single-row kernel by
default. This removes the generic kernel's eight-row and routed branches,
uses checked packed-word reads, and caches activations in shared memory
for narrow-output contractions. It preserves the original FP32 reduction
order. `MFQ_NINT_SINGLE_ROW=0` selects the generic path.
`MFQ_NINT_GROUP_DOT=1` opts into integer accumulation across a complete
group, which changes the FP32 reduction order and remains separate.

The full native build and all 41 CTest cases passed after this change.
The wave test now checks 524,288 signed dot-product and accumulator-wrap
cases. The NINT tests passed 1,176 single-row comparisons, 2,592 q8 cases,
54 q8 capture/replay cases, and 111 ordinary plus 333 changing-graph
activation checks. Default single-row output is compared byte for byte
with the generic path, including mixed bit widths, odd group sizes,
unpadded tails, offset storage and shared-activation cases.

On macOS, 32 focused portability and source-contract checks passed. One
additional CUDA source-contract assertion fails on unchanged `37f3de1f`
as well; the isolated baseline reproduction fails at the same assertion.

Alternating before/after runs against `37f3de1f`, on the same MI300X,
used batch 1, a 129-token input, 256 timed decode tokens and HIP graphs.
Median S4-M throughput increased from 182.46 to 219.92 tok/s for 0.8B,
65.77 to 123.35 tok/s for 4B, and 51.17 to 88.84 tok/s for 9B. The 4B
comparison used five runs per version; the other models used three.
Shared-host timing varied: the 4B ranges were 61.57-70.51 and
102.60-123.94 tok/s. Generated token sequences matched across versions
for all three models and for three additional varied 4B prompts.

Individual 4B projection checks used 200 repetitions, including activation
quantization. The GDN QKV projection improved 4.24x, the FFN down
projection 2.35x, and the vocabulary projection 2.80x. The relative,
mean-absolute and maximum-absolute output errors matched the baseline.
Prefill was unchanged in repeat medians: 10,631.4 to 10,628.2 tok/s at
2K, and 10,051.9 to 10,055.7 tok/s at 8K. Each repeat median summarizes
three trials of seven measurements; timing outliers were retained.

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
- Continuous batching beyond single requests, multimodal, duplex. The Qwen3.5
  HIP batcher compiles and the managed server accepts an explicit
  `--max-requests-per-runtime` for it, but end-to-end multi-request execution
  has not been certified on AMD hardware.

## Environment notes

- The torch wheel's bundled ROCm lacks the rocBLAS Tensile library data;
  copy the host ROCm's `lib/rocblas/library` into the image's
  `_rocm_sdk_devel/lib/rocblas/` or point `ROCBLAS_TENSILE_LIBPATH` at it,
  otherwise hipBLAS aborts at the first GEMM.
- `mfq/kernels/hip/nvq_matmul.hip` takes ~25 minutes at `-O3` for gfx942.
