# Qwen3.8-Flash-Next: Metal runtime comparison

This comparison collects 17 runtime/checkpoint configurations from
MFQ-Publicity revision `f2f9b0fa00005740097b87bd2612a2b92d21bbe8`.
All reported results use an Apple M5 Max with a 40-core GPU and 128 GB memory.
The Strata, llama.cpp and oMLX protocols identify the 4K Python-code input as the
same input used for the MFQ measurements.

<p align="center">
  <img src="../figures/qwen38-flash-next-metal-runtime-comparison.svg" alt="Qwen3.8-Flash-Next Metal comparison: prompt processing and generation with MTP off and on, grouped by MFQ, Strata, llama.cpp and oMLX" width="1100">
</p>

The rows retain each runtime's tested quantization tiers. Their weight sizes,
quality and MTP draft configurations differ, so the figure compares complete
configurations rather than isolating runtime overhead at matched precision.
Each bar starts at zero, and both generation columns use the same scale.
The [data snapshot](../figures/data/qwen38-flash-next-metal-runtime-comparison.json)
preserves every published value, including the MTP-on prefill measurements
that are available for Strata, llama.cpp and oMLX.

## Comparable weight sizes

Shading and bold labels link MFQ V3-XXS with GSQ-RCO IQ3_S in both GGUF
runtimes. The [quality data snapshot](../figures/data/qwen38-flash-next-quality.json)
reports 81.952 GB for V3-XXS and 83.618 GB for IQ3_S, excluding the vision
encoder and MTP head, with 1 GB = 10⁹ bytes. These are similar weight sizes;
they do not imply identical weights, quality or runtime memory use.

With MTP enabled, MFQ V3-XXS generates at 105.4 tok/s, compared with 99.6 tok/s
for Strata and 59.3 tok/s for llama.cpp using GSQ-RCO IQ3_S. With MTP off,
Strata is slightly faster: 72.8 tok/s versus MFQ's 70.8 tok/s; llama.cpp
reports 45.4 tok/s. The MTP comparison includes each runtime's draft setup,
as described below.

## Measurement conditions

The baseline workload uses 4,096 input tokens, 128 generated tokens, greedy
sampling, batch size 1 and a 16,384-token context limit, without an added chat
template. llama.cpp's and oMLX's raw input token IDs match the Strata fixture,
with no extra BOS token. All three baseline runtimes keep model weights
resident and read PLE data from SSD.

For Strata and llama.cpp, warmup, loading and compilation are excluded, prefix
reuse is zero, and each measured run follows 60 seconds idle. Prompt processing
uses the engine's prompt time. Generation measures the 127 token intervals
between the first and last emitted tokens.

oMLX also excludes warmup, loading and compilation, with zero prefix reuse.
The oQ2.5e/oQ3.5e runs cool for 60 seconds and oQ4e for 20 seconds with
full-speed fans, retaining the official 0.5-second GPU keep-warm tick.
Its prefill timing includes
first-token overhead: 4,096 divided by the wall-clock time to the first
producer-generated token. Generation uses 127 intervals between the first
and last producer-generated token timestamps. These timings come from the
native streaming API, without HTTP transport or profiling.

| Runtime | Version | Measured runs per mode | MTP setup |
| --- | --- | --- | --- |
| MFQ | Not specified in the source table | Not specified | Not specified in the source table |
| Strata | `v0.1.44.1` / `296b59d2` | Two; total tokens divided by total measured time | Q2_0 draft experts, default CJK vocabulary shortlist, `spec=2`: one draft token per round |
| llama.cpp | `f74e2fa6a31cc33ab4ee4ae516641cfe62df3924` | One | Self-contained Unsloth Q8_0 MTP head, one draft token per round |
| oMLX | `0.7.1.dev2`; MLX `0.32.4`; mlx-lm `0.31.4.dev132+g94cdcae13` | One | Each checkpoint's included draft head; default adaptive MTP, maximum draft depth 3; controller can reduce or pause drafting |

The MFQ table reports a single prompt-processing rate per tier without
identifying its MTP mode. The figure uses that published value for MFQ and
MTP-off prefill for Strata, llama.cpp and oMLX; it does not fill in missing MFQ
prefill measurements. The source does not specify MFQ's repeat count, context
limit or detailed timing protocol, and gives no uncertainty estimates for
the comparison.

## MFQ

The residency column covers weights; KV cache and runtime allocations are
additional. All throughput values below are in tokens per second.

| Tier | Resident weights (GiB) | Published prefill | Generation: MTP off | Generation: MTP on |
| --- | ---: | ---: | ---: | ---: |
| V1-M | 43.36 | 2,538.5 | 71.6 | 101.2 |
| S2-L | 47.41 | 2,538.9 | 71.0 | 97.9 |
| V3-XXS | 50.33 | 2,527.6 | 70.8 | 105.4 |
| S3-L | 57.95 | 2,565.6 | 68.1 | 98.2 |
| V4-XS | 61.48 | 2,540.1 | 67.7 | 96.7 |
| S4-L | 78.25 | 2,481.7 | 63.6 | 93.6 |

## Strata

| Checkpoint | Prefill: MTP off | Generation: MTP off | Prefill: MTP on | Generation: MTP on | Draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSQ-RCO Q2_0 | 1,796.3 | 84.4 | 1,698.0 | 122.8 | 95.4% |
| GSQ-RCO IQ3_XXS | 1,487.5 | 76.2 | 1,401.4 | 105.7 | 85.5% |
| GSQ-RCO IQ3_S | 1,469.7 | 72.8 | 1,362.1 | 99.6 | 82.9% |
| Unsloth UD-Q4_K_XL | 1,363.7 | 58.4 | 1,242.0 | 85.9 | 89.6% |

Strata uses its persistent native engine over stdio, with prefix reuse and
suffix drafting disabled, 2,048-token prefill chunks and default Q8_0
hyper-connection reads. UD-Q4_K_XL uses the official Metal repack; the source
reports that all 1,223 repacked tensors match the original GGUF payloads by
SHA-256. For all four configurations, both ordinary and MTP runs produce the
same 128 output token IDs.

GSQ-RCO IQ2_XS cannot run on this Strata release because six expert tensors
use unsupported IQ1_M. It is absent from the figure rather than assigned
zero throughput.

## llama.cpp

| Checkpoint | Prefill: MTP off | Generation: MTP off | Prefill: MTP on | Generation: MTP on | Draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSQ-RCO Q2_0 | 1,259.9 | 49.0 | 1,171.9 | 66.9 | 95.4% |
| GSQ-RCO IQ3_XXS | 1,208.5 | 46.5 | 1,112.6 | 62.9 | 96.9% |
| GSQ-RCO IQ3_S | 1,192.8 | 45.4 | 1,108.4 | 59.3 | 86.8% |
| Unsloth UD-Q4_K_XL | 1,191.7 | 42.5 | 1,103.2 | 57.7 | 88.1% |

llama.cpp uses a Release build with Metal and Accelerate, the native
`llama-server` streaming `/completion` endpoint and `cache_prompt=false`.
Its original GGUF shards are loaded with PLE on CPU for lazy SSD access,
FP16 KV storage, `-b 4096 -ub 2048`, and one server slot. MTP uses
`--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-min 1` with
`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`.

The ordinary and MTP output token IDs match for the three GSQ-RCO
configurations. UD-Q4_K_XL differs starting at output token 80. Strata and
llama.cpp use different draft weights and head configurations, so their
MTP results include those differences.

## oMLX

| Checkpoint | Prefill: MTP off | Generation: MTP off | Prefill: adaptive MTP | Generation: adaptive MTP | Draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jundot oQ2.5e | 2,630.2 | 51.6 | 2,405.2 | 94.3 | 88.0% |
| Jundot oQ3.5e | 2,647.4 | 61.5 | 2,575.4 | 111.0 | 92.6% |
| Jundot oQ4e | 2,843.4 | 65.0 | 2,766.7 | 108.3 | 88.4% |

oMLX uses the complete official macOS 26/27 DMG, including its native NAX
kernels, and the `VLMBatchedEngine` streaming API with a 4,096-token prefill
step and speed priority. The allocator cache uses the physical-memory limit,
matching official server startup. Experts remain resident and PLE uses SSD-backed mmap. KV
quantization, TurboQuant and ANE prefill are disabled. Normal warmup generates
16 tokens; adaptive-MTP warmup generates 128 tokens to exercise verification
depths. Ordinary and adaptive-MTP output text matches within each tier.

MTP uses `mtp_enabled=True`, `mtp_fixed_depth=None` and
`mtp_adaptive_max_depth=None`, retaining the model's default maximum draft
depth of 3. These adaptive results include a different drafting policy from
the fixed-single-draft Strata and llama.cpp runs. They do not establish a
comparison at matched model size or quality with MFQ V3-XXS or GSQ-RCO IQ3_S.

The source pins oQ2.5e to revision
`ebfb10474baa3c1c78d38453ce0d5228febd32e9` and oQ3.5e to
`e8216051d1a75ec4a523ed8c6b0ed4a01667bcda`. oQ4e uses a previously downloaded
checkpoint with 106,294,664,646 bytes of weight shards; its repository revision
is not reported. All oMLX runs enable High Power Mode. Recorded prefill GPU
frequency samples were 1,620 MHz, with P13 dominant in request-contained
performance-state samples.

## Sources and regeneration

The source tables are pinned to the same MFQ-Publicity commit:

- MFQ Apple table: `Qwen3.8-Flash-Next-EW-MFQ/README.md`.
- Strata: `Qwen3.8-Flash-Next-EW-MFQ/benchmarks/strata-metal-4k.md`.
- llama.cpp: `Qwen3.8-Flash-Next-EW-MFQ/benchmarks/llamacpp-metal-4k.md`.
- oMLX: `Qwen3.8-Flash-Next-EW-MFQ/benchmarks/omlx-metal-4k.md`.

The snapshot records source-file SHA-256 values and the input checksum.
The original input excerpt is credited to the Apache-2.0
[oMLX Python benchmark corpus](https://github.com/jundot/omlx/blob/v0.6.3/omlx/admin/bench_corpora/code_python.txt);
the prompt itself is not copied here.

Regenerate the English and Chinese SVGs and the English PNG from the snapshot:

```bash
uv run --no-project --no-config --with matplotlib python docs/figures/generate-metal-runtime-comparison.py
```
