# HIP NINT GEMV optimization (2026-10-21)

The baseline is commit `2557bd1eab0a39830b8d42fd1d62d1454c11880e`.
Both versions were built with the same Release toolchain and run on one
MI300X (`gfx942`). The raw samples, binary and source hashes, model dtype
counts, token hashes, and validation results are in
[`hip-gemv-bench-20261021.json`](../../bench/hip-gemv-bench-20261021.json).

## Changes

- Fix fused row IDs for unsorted mixed-bit tensors. The loader sorted the
  payload into bit-width segments but appended row IDs in source order.
- Preserve fused payloads and row mappings when projections are concatenated.
  Grouped FFN and input projections now retain their decode path.
- Let single-row sigmoid/SiLU-gated projections use fused weight decoding.
  Gate evaluation and FP16 rounding remain in activation quantization.
- Compute each activation group's integer sum once. Specialized GS24/GS48
  preparation handles four groups per block without block barriers.
- Specialize fused decode for GS24/GS48. Keep packed words in registers and
  use aligned 64/128-bit weight and activation loads, with alignment fallbacks.
- Add whole-group q8 decode for aligned GS48, M=1, at least 16 output rows,
  and at least 32 groups. It applies the affine after twelve integer dots
  and reuses activation sums. `MFQ_NINT_GROUP_DOT=0` keeps the old q8 path.

Slot-fused calls rise from 57 to 145 of 169 NINT calls per 4B decode step.
The remaining 24 use aligned-q8 dispatch. These are dispatch counts from
filtered profiles; event attribution is not a direct HBM bandwidth counter.
NINT6 is supported by the fused kernel. It was not inherently excluded by
its precision.

## Decode throughput

Tokens/s, batch 1, 256 generated tokens, median of three measured requests
after one warmup per prompt length. Both versions set
`MFQ_DISABLE_MMA_ATTENTION=1`. Runtime paths were explicit. The BF16 control
was checked to contain 690 BF16 tensors, 48 F32 tensors, and no NINT tensors.

| Model | Prompt tokens | Before | After | Gain |
| --- | ---: | ---: | ---: | ---: |
| 0.8B S4-M | 32 | 273.8 | 311.4 | +13.7% |
| 0.8B S4-M | 2048 | 254.2 | 287.5 | +13.1% |
| 0.8B S4-M | 8192 | 144.7 | 154.6 | +6.9% |
| 4B S4-M | 32 | 143.2 | 192.6 | +34.5% |
| 4B S4-M | 2048 | 136.3 | 179.3 | +31.5% |
| 4B S4-M | 8192 | 88.4 | 104.8 | +18.6% |
| 4B BF16 | 32 | 123.3 | 123.5 | +0.1% |
| 4B BF16 | 2048 | 118.2 | 118.4 | +0.1% |
| 4B BF16 | 8192 | 80.5 | 80.5 | +0.1% |

These changes target decode. M>1 packed kernels and large-M dequantization/GEMM
remain on their existing paths. Long-context attention remains a separate
bottleneck.

## Operator throughput

Microseconds per activation-quantization + GEMV operation, median of seven
samples. GPU events surround repeated graph replays, so this measurement
avoids a host event pair around every operator. The benchmark rotates a
nominal 256 MiB working set for large matrices; tiny control matrices rotate
four copies. These timings do not establish an HBM bandwidth ceiling.

| Operator | Before us | After us | Speedup |
| --- | ---: | ---: | ---: |
| gate_up | 25.085 | 16.876 | 1.49x |
| gated_output | 18.655 | 8.896 | 2.10x |
| down_nint6 | 29.708 | 13.508 | 2.20x |
| q8_projection | 18.034 | 6.806 | 2.65x |
| q8_control | 12.896 | 5.029 | 2.56x |

The driver also records hot-weight samples. Its nominal byte estimate selects
the rotation count; it is not measured memory traffic. ISA inspection confirms
wide loads and native integer dot instructions, without scratch spills in the
GS24 NINT4/NINT6 and grouped-q8 kernels.

## Confirmation and prefill

A separate p2048 pass uses two warmups and five measured requests per model.
Absolute throughput varied between campaigns, including the BF16 control, so
the confirmation is kept separate from the primary matrix.

| Model | Decode before | Decode after | Gain | Prefill before | Prefill after |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen35-08b-s4m.mfq | 232.7 | 264.0 | +13.4% | 13596 | 10289 |
| qwen35-4b-s4m.mfq | 120.7 | 163.7 | +35.7% | 7178 | 6938 |
| qwen35-4b-fp.mfq | 108.0 | 108.1 | +0.1% | 7221 | 7195 |

All prefill samples are retained in the JSON. The original matrix has noisier
short-prefill timings, including BF16; no prefill speedup is claimed.

A prefill-only 0.8B p2048 check alternates baseline/optimized/optimized/baseline,
with three warmups, seven measurements, and two generated tokens per request.
Its run medians are 12,814 / 15,522 / 16,327 / 16,599 prefill tokens/s. The
baseline itself drifts by 29.5% across that sequence; the last adjacent pair
differs by 1.6%. This does not reproduce the large slowdown seen in the noisy
confirmation samples. All runs are included under `prefill_abba` in the JSON.

## Correctness and validation

- Fresh baseline build: all 46 native tests pass.
- Optimized build: all 46 native tests pass, including 864 fused/grouped/gated
  comparisons and 144 grouped-q8 comparisons with changing-input graph replay.
- Regression fixtures include unsorted bit widths 2..8, unequal grouped output
  sizes, GS4/24/28/32/48/64, short and long group counts, partial output blocks,
  misaligned weight storage, FP16/FP32 inputs and gates, rounding boundaries,
  zero groups, and padded tails.
- Quantized activations, activation scales, and cached integer sums match their
  references exactly. Output checks retain the existing FP16 reduction bounds;
  regrouping q8 sums can change floating-point rounding.
- Six generation comparisons (0.8B/4B, prompt lengths 32/129/2048, 65 generated
  tokens each) match the baseline token-for-token.
- All 37 backend-neutral native tests and 67 focused Python tests pass.
- Full Python suite: 3,258 passed, 770 skipped, 26 failed. The exact same 26
  failures reproduce on untouched `2557bd1e`; the JSON lists their test IDs.

The fused representation retains the canonical tensors for other batch sizes
and prefill. Preserving fused copies for grouped projections increases GPU
weight memory compared with the baseline. `MFQ_NINT_FUSED_DECODE=0` disables
load-time fused layouts.

## Reproduce

Build each revision in its own directory with the same toolchain and flags:

```sh
cmake -S cpp_runtime -B build/hip -DCMAKE_BUILD_TYPE=Release \
  -DMFQ_BUILD_HIP_RUNTIME=ON -DMFQ_BUILD_CUDA_RUNTIME=OFF \
  -DMFQ_BUILD_METAL_RUNTIME=OFF -DMFQ_BUILD_RUNTIME_COMMUNICATION=ON
cmake --build build/hip -j 16
ctest --test-dir build/hip --output-on-failure
ROCR_VISIBLE_DEVICES=0 build/hip/hip/mfq-hip-nint-bench \
  --working-mib 256 --samples 7 --target-ms 30
python bench/model_bench.py qwen35-08b-s4m.mfq qwen35-4b-s4m.mfq qwen35-4b-fp.mfq \
  --runtime build/hip/mfq-runtime --prefill-lengths 32,2048,8192 \
  --decode-tokens 256 --repetitions 3 --warmup-runs 1 \
  --env MFQ_DISABLE_MMA_ATTENTION=1 --output results.json
```

The microbenchmark driver must also be added to the baseline checkout; it
calls the same existing projection APIs in both builds.
