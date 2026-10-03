# Flash-Next Metal decode diagnostics

The gated-hyper-connection kernels accept independent FP16, BF16, and FP32
storage types. Grouped RMSNorm computes the norm and centered affine in FP32
and returns the activation type. Down and injection projections use their
respective MLX-promoted types; the low-rank activation keeps the down type,
while up projection and collapse use the promoted down/up type. Projection
weights are not expanded to FP32 or rounded to a common 16-bit type.

The uniform BF16 decode regression remains bit-exact. Mixed-storage tests
cover all 27 activation/down/up combinations against the unfused MLX graph,
with a `2e-5 * (1 + abs(reference))` elementwise tolerance. FP32 reduction
order can change, so this is not a promise of identical greedy text on every
prompt, nor a substitute for a model-quality evaluation.

Single-token forward submits the first six layers individually, then every
third layer with `async_eval`, overlapping GPU execution with construction of
subsequent layers while amortizing commit overhead. This policy is adapted
from oMLX revision `87460f4d50de79aef9b67e99e215c31f0a89b445`; see `NOTICE`.
Prefill keeps its existing
dispatch policy. Component profiling disables these asynchronous submissions
for the instrumented steps so component boundaries remain meaningful.

## Controls

- `MFQ_METAL_QWEN_GATED_HC_FAST=0`: disable gated-HC fusion for a reference run.
- `MFQ_METAL_QWEN_DECODE_ASYNC=0`: disable layer-wise decode submission.
- `MFQ_METAL_QWEN_DECODE_ASYNC_EVERY=1`: submit every layer (default: 3 after
  the initial six); useful for comparing submission granularity.
- `MFQ_METAL_PROFILE_COMPONENTS=1`: enable synchronized component diagnostics.
- `MFQ_METAL_PROFILE_SKIP_STEPS=32` and `MFQ_METAL_PROFILE_STEPS=3`: instrument
  three ordinary (non-MTP) decode steps after 32 uninstrumented steps.

Component costs include synchronization and host dispatch; they are not pure
GPU kernel times. Do not use instrumented request throughput as a benchmark.

## Validation and timing

```sh
ctest --test-dir build/cpp_runtime/metal \
  -R '^mfq-metal-qwen-gated-hc-test$' --output-on-failure
build/cpp_runtime/metal/mfq-metal-qwen-gated-hc-test --benchmark-mixed
```

The mixed-storage microbenchmark warms both paths, then alternates reference /
fused execution order within one process, reporting every pair. Its resident
synthetic weights and synchronization boundaries do not represent whole-model
throughput.

For end-to-end comparisons preserve a control binary, use the same artifact,
prompt, sampler, context and residency policy, disable component profiling,
and exclude cold requests. Record all runs, power state and thermal telemetry;
reverse process order and allow cooling before drawing a performance
conclusion. A nominal OS thermal state alone does not establish equal GPU
clocks. If clock/temperature conditions cannot be matched, label the numbers
as development diagnostics rather than a controlled speedup claim.

Residual writeback also combines broadcast multiplication and residual
addition, preserving the multiplication's promoted rounding before the add.
Its tests require exact equality across 27 type combinations for both
single-token batches and prefill. The MoE path queries the shared weight
implementation's fused-reduction capability: unsupported representations
retain per-expert outputs until the existing reduce/shared-gate epilogue,
avoiding an extra standalone reduction without model-level format checks.

## Packed-format decoding

NINTv2 specializes the GS24 and GS28 group loops without fixing q: each row
still selects its own q width. The group dot product applies scale/min after
the integer-weight accumulation, reusing the activation sum for the minimum.

NVQ execution packing is internal and does not change the MFQ wire format.
Wide JSC indices use layout 3: one or two indices plus the complete sign mask
in a three- or four-byte record per eight weights. NVQ1-S uses layout 4: three
nine-bit indices, a four-bit state, and a bank/delta bit in one 32-bit record
per GS24 group. Both Decode and grouped Prefill consume these layouts; the
original index/auxiliary streams are not retained in packed device storage.

The MoE regression fixtures vary indices, signs, states, and codebook entries
and compare with an independently constructed dense reference. They cover
M=1/31/32/49/1025, K=640 tails, wide JSC profiles, NVQ1-S, and row-adaptive
NINT GS24/28. Zero-index fixtures alone cannot validate execution packing.

Adaptive-row regressions cover every q width from 1 through 8 at GS24/28.
Shared-expert two-stage tests additionally use GS23/28 with K65 and an
intermediate width of 41, comparing against independently constructed dense
shared weights. Each VQ cohort also receives its own single-token case.
