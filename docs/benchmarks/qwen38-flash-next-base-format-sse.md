# Qwen3.8-Flash-Next: base-format reconstruction

The expert-allocation figure in the README's **Why MFQ → Preserve quality with
fine-grained precision** section includes the following base-format
reconstruction comparison. The values below are transcribed from
the table supplied for the figure; no new measurements were run. The
[data snapshot](../figures/data/qwen38-flash-next-base-format-sse.json)
preserves all seven pairs and the source image's SHA-256.

The comparison covers **layer 0, all 512 experts and each expert's Gate, Up
and Down projections: 1,536 matrices**. Each SSE column is an equally weighted
mean of the per-matrix reconstruction SSE values. Lower SSE is better.

| MFQ format | Actual bpw | GGUF format | Actual bpw | MFQ mean SSE | GGUF mean SSE | SSE reduction |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| NVQ2J | 2.0535 | IQ2_XXS | 2.0625 | 25.6356 | 34.2542 | 25.16% |
| NVQ3J | 3.0498 | IQ3_XXS | 3.0625 | 7.52099 | 9.46618 | 20.55% |
| NVQ3J-L | 3.5535 | IQ3_S | 3.4375 | 4.39932 | 5.54164 | 20.61% |
| NINT4 GS24/k6 | 4.5285 | Q4_K | 4.5000 | 1.35714 | 1.51948 | 10.68% |
| NINT4 GS24/k6 | 4.5285 | IQ4_NL | 4.5000 | 1.35714 | 1.68599 | 19.50% |
| NINT5 GS28/k7 | 5.5489 | Q5_K | 5.5000 | 0.355366 | 0.380979 | 6.72% |
| NINT6 GS24/k7 | 6.6184 | Q6_K | 6.5625 | 0.089583 | 0.0933589 | 4.04% |

For each row, the chart plots `100 × (1 − MFQ mean SSE / GGUF mean SSE)`.
The reported reductions agree with this calculation to the displayed precision.
The actual bpw values differ within each pair and are shown alongside the
formats. The chart preserves the listed pairs and their order, including
both NINT4 comparisons.

This panel measures reconstruction on the listed layer-0 matrices. The README's
opening quality figure separately reports whole-model KLD and Top1 agreement on WikiText-2,
using the [model-quality data](../figures/data/qwen38-flash-next-quality.json).
Their weight sizes exclude the vision encoder and MTP head.

Regenerate the English and Chinese allocation/format comparison SVGs:

```bash
node docs/figures/generate-expert-allocation-comparison.mjs
```
