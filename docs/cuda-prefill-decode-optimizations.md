# CUDA compact decode and long-prompt prefill

This change integrates the CUDA operator and runtime work for compact
NVQ/NINT expert execution, MFE, GR, QSA, GDN, RAM/VRAM residency, and
long-prompt prefill. It preserves the on-disk MFQ format. The runtime
does not enable expanded E8/D4 execution records by default.

## Operators

- NVQ decode reuses whole compressed groups, signs, paired indices, and
  routed activation reads. NVQ1 has a dedicated compressed decode path.
- NINT decode includes whole-group affine reduction, ordered Q8 inputs,
  route grouping, and reusable row metadata/workspace.
- MFE uses compact specialized gate/up and down kernels with shared
  activation preparation. Generic device code is shared between the
  original and compact translation units.
- GR separates expansion and contraction dispatch, uses register tiling
  for packed prefill projection, and fuses projection/mix stages while
  retaining the established floating-point accumulation order.
- QSA fuses prefill preparation and selected attention stages; GDN has
  column-parallel prefill and next-token q/k software pipelining.

The accepted GR prefill defaults are matmul mode 17 and fused mix mode 3.
Mixed-format NVQ prefill cohort dispatch is enabled for route tile 128.
The GDN pipeline is enabled for D128, four-column tiles, and single
sequences with 48 or 64 value heads. Other shapes retain their existing
dispatch. Experimental narrow-G2 NVQ, expanded layouts, and GR mix-K64
paths remain opt-in.

## Long-prompt execution

Layer-major prefill batches expert FFNs across up to 8192 tokens while
preserving the 1024-row execution used by router/shared projections.
Attention/GR compute chunks and expert FFN batches are independent.
Whole-layer packed transfers can overlap gate/up compute with down
transfer, and registered RAM and reusable GPU workspace avoid repeated
allocation. The runtime bounds pools by its existing memory budget and
preserves ownership across prefill/decode transitions.

The long-prompt benchmark configuration enables
`MFQ_MOE_PREFILL_LAYER`, `MFQ_MOE_PREFILL_LAYER_MAJOR`,
`MFQ_MOE_PREFILL_LAYER_OVERLAP`, `MFQ_MOE_PREFILL_LAYER_PHASED`, and
`MFQ_GDN_PREFILL_FUSED`. Layer asynchronous transfer is disabled for the
accepted measurement. These scheduling switches remain configurable.

## Measurements

Measurements below use S2-L on an RTX 3090 Ti, without MTP, with 8192
input tokens formed by repeating IDs 1 through 1024. Model loading is
excluded. They measure actual generation-path prefill, with 22.5 GiB
VRAM and a 32 GiB process working-set limit.

The accepted mixed-format cohort/GDN A/B run used a 1024-token compute
chunk and an 8192-token expert batch. Median prefill improved from
1392 to 1422 tokens/s, or 2.15%, with identical full-vocabulary logits
and three-token continuations. This is the gain from those two final
changes, rather than from the complete optimization series.

The subsequent compute-chunk comparison used the same default operators
in one process, with three generation-path measurements per chunk:

| Compute chunk | Median prefill time | Tokens/s | Full-vocabulary relative RMS |
| --- | ---: | ---: | ---: |
| 1024 | 5.54570 s | 1477 | 0 |
| 8192 | 5.13676 s | 1595 | 0.073656 |

The 8192 compute chunk fails the 1e-4 numerical threshold and is not an
accepted default. Chunk comparisons retain the requested operators when
reporting speed after a numerical rejection. Performance differences
between independent runs should not be attributed to a code change.

## Validation tools

The CUDA native tests cover compressed payload identity, format bounds,
FP64 operator references, changing graph inputs, recurrent state, and
cache exchange. Additional checks in `mfq-moe-ffn-pipeline-test` cover
canonical prefill, mixed-format cohort defaults, pool reuse, memory
limits, registered RAM, and prefill/decode transitions. Released weights
are not required for the synthetic fixtures.

`bench/cuda_format_comparison` contains reusable expert/operator benchmark
tools and a model manifest template. Raw logs, model weights, generated
device snapshots, and local experiment directories are excluded from the
source changes.
