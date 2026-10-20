# HIP model benchmarks

Model-level throughput for the HIP runtime on a single MI300X (gfx942),
comparing true full-precision (BF16) MFQ conversions with their MFQ-quantized
(S4-M) builds across the dense Qwen3.5 size ladder and Qwen3.8-27B, measuring
speculative MTP decode, and referencing an independent llama.cpp run on the
same GPU. Raw results:
[`bench/hip-model-bench-20261018-bf16.json`](../../bench/hip-model-bench-20261018-bf16.json),
[`bench/hip-model-bench-20261018-quantized.json`](../../bench/hip-model-bench-20261018-quantized.json),
and [`bench/hip-model-bench-20261018-mtp.json`](../../bench/hip-model-bench-20261018-mtp.json),
measured after the kernel improvements below; the 2026-10-17 files next
to them record the pre-improvement baseline, and
[`bench/hip-llama-bench-20261017.txt`](../../bench/hip-llama-bench-20261017.txt)
holds the llama.cpp reference.

## Method

`bench/model_bench.py` drives `mfq-runtime --transport stdio` with the
native benchmark protocol: synthetic prompts sized to an exact token
count, prefix cache disabled per request, greedy sampling, and 256
decoded tokens per run. Each cell is the median of 3 runs after 1
warmup run, batch 1, Release build. All runs set
`MFQ_DISABLE_MMA_ATTENTION=1`, which the HIP backend still requires for
prompts longer than 32 tokens (see the
[HIP validation report](hip-validation.md)).

```sh
python bench/model_bench.py model-fp.mfq model-s4m.mfq \
  --runtime build/cpp_runtime/mfq-runtime \
  --prefill-lengths 32,512,2048,8192 --decode-tokens 256 \
  --env MFQ_DISABLE_MMA_ATTENTION=1 --output results.json
```

Full-precision references are `mfq quantize <hf-dir> model-fp.mfq
--full-precision --tokenizer <tokenizer.gguf>`. The `--tokenizer` asset
is required: without it the file loads no tokenizer and the runtime
exits with "model runtime requires model config and tokenizer GGUF".

## Reference-file correction

The earlier "BF16" reference files (`qwen35-0.8b.mfq`,
`qwen35-4b-bf16.mfq`) were default-recipe NINT conversions, not
full-precision files: the 4B file stores 2.21 GiB NINT and 0.85 GiB
BF16. Earlier tables that compared those files against S4-M measured
NINT against NINT. This document compares only true full-precision
files against S4-M. Weight sizes below are the stored tensor bytes.

## Decode and prefill (2026-10-18, fused-kernel build, 2048-token cohort)

Decode, tokens per second. The llama.cpp columns are `tg256` from the
same GPU (see the llama.cpp section for its build and conversion);
llama-bench runs generation at its default context, so treat them as
the nearby-KV reference rather than an exact-cohort match.

| Model | MFQ BF16 | llama F16 | MFQ S4-M | llama Q4_K_M | S4-M vs BF16 | S4-M vs llama Q4_K_M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 165.1 | 406.8 | 254.5 | 414.4 | 1.54x | 0.61x |
| Qwen3.5-2B | 172.4 | 290.7 | 212.9 | 375.6 | 1.24x | 0.57x |
| Qwen3.5-4B | 116.2 | 180.3 | 135.6 | 242.6 | 1.17x | 0.56x |
| Qwen3.5-9B | 71.2 | 122.9 | 101.1 | 193.8 | 1.42x | 0.52x |
| Qwen3.5-27B | 22.0 | 40.7 | 42.2 | 77.1 | 1.92x | 0.55x |
| Qwen3.8-27B | 21.9 | 40.6 | 42.2 | 76.9 | 1.93x | 0.55x |

Prefill at the same cohort, tokens per second (`pp2048` for llama.cpp):

| Model | MFQ BF16 | llama F16 | MFQ S4-M | llama Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 16866 | 19732 | 16512 | 27772 |
| Qwen3.5-2B | 15110 | 15495 | 14659 | 20636 |
| Qwen3.5-4B | 8635 | 8427 | 8462 | 11762 |
| Qwen3.5-9B | 7129 | 6582 | 6823 | 8235 |
| Qwen3.5-27B | 2768 | 2523 | 2517 | 2763 |
| Qwen3.8-27B | 2762 | 2519 | 2440 | 2761 |

S4-M decode wins over BF16 at every size, with the gain growing from
1.17x (4B) to 1.9x (both 27B models); by cohort the speedup spans
1.17-1.96x at 32 tokens and 1.11-1.69x at 8192 tokens. Effective S4-M
weight streaming reaches 0.74 TB/s on 27B. Against llama.cpp, MFQ BF16
prefill matches or beats llama F16 from 4B upward while MFQ S4-M
prefill trails llama Q4_K_M by 1.1-1.7x (the M > 8 dequant fallback);
decode reaches 0.52-0.61x of llama Q4_K_M, which is the remaining
kernel gap quantified in the diagnosis below.

## Diagnosis: where quantized speed is lost

An op-level profile (`mfq-diagnostics --profile --gen 65`, GPU event
timings, 4B models) plus kernel-source analysis and A/B runs locate the
gap:

- **The packed NINT GEMV kernel is the bottleneck, and its cost scales
  with weight bytes at a fixed rate.** Normalizing end-to-end decode by
  stored bytes, the uniform-NINT and S4-M files both stream at about
  0.41 TB/s on 4B and 0.63 TB/s on 27B, independent of bit width. On
  the same GPU the MFQ BF16 path sustains 1.14 TB/s and llama.cpp
  sustains 1.2-1.25 TB/s at 27B. Every other part of the stack is
  competitive; the fused decode kernel converts weight bytes to
  answers at roughly half the demonstrated streaming rate.
- **The scalar `__dp4a` fallback is no longer active.** A probe kernel
  compiled with the current toolchain shows `__gfx942__` defined and
  `__has_builtin(__builtin_amdgcn_sdot4)` true, emitting `v_dot4`. The
  `mfq_hip_compat.h` guard takes the hardware path in current builds;
  the earlier "no hardware dot" finding predates the newer compiler.
- **The per-chunk float epilogue is worth 3-4%.** The
  integer-accumulate path (`nint_group_partial`, previously opt-in via
  `MFQ_NINT_GROUP_DOT=1`) measured +3.4% decode on 4B S4-M (121.2 to
  125.3 tok/s) and +4.4% on 27B (36.4 to 38.0) before further kernel
  work; see the improvement section below.
- **Prefill above 8 rows dequantizes whole matrices.** `nint_matmul`
  (cpp_runtime/backends/hip/ops/nint.cpp) uses the fused workspace GEMV
  only for M <= 8; larger cohorts dequantize the full weight to BF16
  and call a dense GEMM. In the profiled window the dequant kernel
  alone cost 11.3 ms against 6.5 ms for the GEMM it feeds.

What remains is the kernel's per-byte instruction and access pattern:
one output row per 32-lane subwarp, 4-8 byte weight loads per chunk
followed by shift/mask unpacking, and three scattered per-group scale
loads (subgroup scale, minimum, activation scale) that are separate
from the weight stream. llama.cpp's Q4_K_M reaches 1.2 TB/s with an
inline block layout: scales stored next to the weights they scale and
32-byte aligned blocks, so one wide load serves both.

## Kernel improvements (2026-10-17)

Two changes landed in the single-row decode path:

1. Integer group accumulation (`nint_group_partial`) is now the default
   (`MFQ_NINT_GROUP_DOT=0` opts out), and its weight loads use a sliding
   dword window: adjacent chunks reuse the previously loaded words. The
   unpack arithmetic is unchanged, so the grouped path keeps its
   tolerance of the plain path (`mfq-hip-nint-q8-test`, identical
   bounds across repeated runs).
2. A fused slot layout built once per weight at load time: rows are
   grouped into uniform-bit segments and each group becomes one
   4-byte-aligned slot [weight bits | pad | u16 scale/minimum], so the
   templated-per-bits kernel folds all extraction shifts and byte-perm
   selectors to constants and reads scales from the same stream
   (`MFQ_NINT_FUSED_DECODE=0` opts out; aligned-q8 tensors keep their
   dedicated kernel). The fused path is bit-exact against the grouped
   path (exact_half in the q8 test) and costs about one second of extra
   load time plus roughly 30% more VRAM for NINT weights.

| Model | Plain tok/s | Sliding window | + Fused slots | Total gain |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B S4-M | 120.3 | 129.5 | 136.2 | +13.2% |
| Qwen3.5-9B S4-M | 92.9 | 98.3 | 101.7 | +9.5% |
| Qwen3.5-27B S4-M | 36.2 | 40.4 | 42.5 | +17.4% |

Measurements are medians of 3 runs at the 2048-token cohort; prefill is
unchanged. Effective weight streaming rises from 0.63 to 0.74 TB/s on
27B — still short of the 1.3-1.5 TB/s needed to match llama.cpp Q4_K_M,
because one activation row per warp still pays per-chunk activation
loads and a segment launch per projection.

Fix directions, in expected impact order: 16-byte single loads per
group slot where the slot permits; vectorized activation loads; one
kernel launch for all segments; more than one activation row per wave
to amortize weight traffic once MTP batches arrive; extend the fused
path beyond M <= 8 or cache dequantized hot tensors for prefill-sized
cohorts.

## Speculative MTP decode (2026-10-18, 2048-token cohort)

`bench/model_bench.py --mtp` repeats the decode matrix with
`enable_mtp` and records drafted and accepted tokens. Raw results:
[`bench/hip-model-bench-20261018-mtp.json`](../../bench/hip-model-bench-20261018-mtp.json).

| Model | Precision | Plain tok/s | MTP tok/s | Ratio | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | BF16 | 165.1 | 80.3 | 0.49x | 41% |
| Qwen3.5-0.8B | S4-M | 254.5 | 123.5 | 0.49x | 20% |
| Qwen3.5-2B | BF16 | 172.4 | 78.8 | 0.46x | 28% |
| Qwen3.5-2B | S4-M | 212.9 | 113.7 | 0.53x | 40% |
| Qwen3.5-4B | BF16 | 116.2 | 87.0 | 0.75x | 92% |
| Qwen3.5-4B | S4-M | 135.6 | 110.3 | 0.81x | 90% |
| Qwen3.5-9B | BF16 | 71.2 | 61.6 | 0.86x | 100% |
| Qwen3.5-9B | S4-M | 101.1 | 94.7 | 0.94x | 100% |
| Qwen3.5-27B | BF16 | 22.0 | 15.5 | 0.70x | 54% |
| Qwen3.5-27B | S4-M | 42.2 | 29.4 | 0.70x | 59% |
| Qwen3.8-27B | BF16 | 21.9 | 20.2 | 0.92x | 90% |
| Qwen3.8-27B | S4-M | 42.2 | 38.9 | 0.92x | 92% |

MTP is a net loss at every size on the current build; even the previous
break-even cases (9B and Qwen3.8-27B S4-M) now sit at 0.92-0.94x. The
cause is measurable: plain single-row decode uses the fused kernel,
while MTP verification runs multi-row batches that the fused kernel
does not cover yet and falls back to the slower generic path. Extending
the fused layout to M in [2,8] is therefore both the MTP fix and the
next decode win. Small models additionally draft rarely (0.06-0.15
drafted tokens per output token) yet still lose half their throughput,
so per-step MTP overhead remains real; the MTP stall and workspace
optimizations that landed for CUDA have no HIP equivalent yet.

## llama.cpp reference on the same GPU

An independent reference was measured with llama.cpp (`f74e2fa`, HIP
backend, gfx942, same MI300X host, `llama-bench -p 32,2048 -n 256 -r 3
-ngl 99`). Models were converted from the same HF snapshots
(`convert_hf_to_gguf.py --outtype f16`) and quantized to Q4_K_M with
`llama-quantize`. Numbers are means of 3 repeats; llama-bench `tg` runs
at its default context, so compare against the MFQ 32-token cohort for
the closest KV size.

| Model | llama F16 tg | MFQ BF16 | llama Q4_K_M tg | MFQ S4-M |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 406.8 | 173.3 | 414.4 | 273.5 |
| Qwen3.5-2B | 290.7 | 181.2 | 375.6 | 226.7 |
| Qwen3.5-4B | 180.3 | 121.9 | 242.6 | 142.4 |
| Qwen3.5-9B | 122.9 | 73.2 | 193.8 | 105.2 |
| Qwen3.5-27B | 40.7 | 22.3 | 77.1 | 43.7 |
| Qwen3.8-27B | 40.6 | 22.3 | 76.9 | 43.6 |

Prefill is competitive: at the 2048 cohort MFQ BF16 reaches 2762-16866
tok/s against llama F16 pp2048 of 2523-19731 tok/s (within 10% at
every size). Decode still trails: llama.cpp F16 decodes 1.6-1.9x faster
than the MFQ BF16 path, and llama.cpp Q4_K_M decodes 1.5-1.8x faster
than MFQ S4-M after the kernel improvements (down from 1.6-2.1x).
llama.cpp also converts more of its quantization into speed (1.9x
Q4_K_M over F16 at 9B/27B versus 1.4-1.9x for S4-M over BF16), which
independently confirms the bandwidth-efficiency diagnosis above: the
remaining gap is in the quantized GEMV kernels, not in the format.

Quality (KL divergence to the BF16 reference) is measured separately by
`mfq-eval`; this document covers throughput only. The Flash-Next MoE
build is excluded because generation still fails with the known logits
dtype bug, and Qwen3.5-35B-A3B remains blocked on canonical tensor
mapping.
