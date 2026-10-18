# Calibration-free NatScale

`mfq quantize --alphaq-allocator natscale` allocates routed experts under the
same main-text file ceiling as [AlphaQ](alphaq.md). Nonexperts follow the UD
recipe. `--ud-tier` selects both a pinned recipe and its exact `ud_file_bytes`
ceiling; it is mutually exclusive with `--target-bpw`. AlphaQ remains the
default allocator. NatScale requires recipe dense mode and rejects joint mode.

This implements [NatScale](https://arxiv.org/abs/2610.20353) with these explicit
changes from the paper's native-calibration protocol:

- Statistics use checkpoint weights and the supplied imatrix; no calibration
  texts, model execution, forward or backward passes are run. The imatrix
  itself contains observations collected by its producer.
- Output error uses a separate-error surrogate, omitting the paper's joint
  nonlinear expert replay and its cross terms.
- Exposure and errors use one aggregate, rather than six native folds.
- Sensitivity defaults to **uniform**, the paper's “w/o gradient weights”
  variant. Imatrix energy is an experimental proxy, not a gradient estimate.
- Format errors transfer from sampled experts/layers to the nearest sampled
  layer. `analytic` is an explicit bit-decay ablation.
- Shared pools use conservative all-pools reservation, charging only selected
  pools in the returned scheme. The LP bound relaxes pool activation costs.

## Statistics and objective

Exposure is `f_i = counts_i / router_tokens_layer`. Counts must agree across
projections and sum to `router_tokens * topk`. Zero-exposure experts have zero
cost. The imatrix loader already divides `in_sum2` by counts: its `values`
are input second moments `s[i,p,j]`, not sums to divide again. Gate/up use
hidden-input moments; down uses its own entry.

During the same weight pass as the AlphaQ spectra, collection saves
`column_energy[i,p,j] = sum_o W[i,p,o,j]^2` in checksummed `.npy` sidecars
under `column-energies/`, referenced from `statistics.json`. These arrays are
imatrix independent. Complete banks survive interruption. Older banks
lacking energies are recollected; complete new banks remain reusable.
Allocation computes `P[i,p] = sum_j s[i,p,j] * column_energy[i,p,j]`.

Measured relative error is `rho = sum(s*(W_c-W)^2) / sum(s*W^2)`. By default,
six evenly spaced MoE layers and three evenly spaced exposed experts per
layer/projection are measured for **every** selected profile. Numerators and
denominators are pooled across experts. Other layers use the nearest sampled
layer, choosing the earlier layer on ties. A zero denominator yields zero
relative error. Measurement runs the normal MFQ payload writer and decodes
the actual serialized result. NINT is explicitly imatrix weighted, including
NINT8. NVQ-JSC uses the same deterministic full-bank training rows, options,
tables and table cache as final export. Each measured sample checkpoints.

For all `Q^3` gate/up/down configurations (`12^3=1728` by default):

```text
M_i(c) = (P[i,down] / hidden_size) * sum_p rho[layer,p,c_p]
E_i(c) = f_i * sqrt(M_i(c))
A_i(c) = (1/3) * sum_p [(median(alpha)/alpha[i,p]) * Var(W[i,p]) * 2^(-2*b(c_p))]
d_i(c) = w_i * A_i(c) * E_i(c)
```

Spectra, population variance and bit coordinates are AlphaQ's: nominal bits
for NINT and effective payload BPW for NVQ, excluding the expert descriptor.
Actual payload costs include descriptors, padding and shared pools.
`--natscale-error analytic` sets `rho(c)=2^(-2*b(c))`; it does not measure
quality. The default `measured` requires a matching complete error cache and
never silently falls back. Source identity, imatrix checksum, profiles,
sample settings, fitting options and codec code identify that cache.

`--natscale-sensitivity uniform` gives `w=1`, `beta=0`, realized contrast zero;
tau does not change this variant. `imatrix-energy` sets `g=P[i,down]/hidden_size`.
`gradient:PATH` reads external moments. Positive nonconstant scores use
`q=softmax(2*beta*log(g))` and bracketed monotone log-space root finding for
`KL(q||u)+KL(u||q)=tau`. `--natscale-tau` defaults to one nat. Empty or constant
positive support is rejected for nonuniform sources. Zero scores keep zero
weight. Weights are divided by their common maximum before cost normalization
to avoid overflow; allocation comparisons are unchanged. The unweighted
Weibull MLE on `sqrt(g)` has `beta=k/4`, satisfying exactly the one-nat equation.

Complete cost rows are shifted by their minima and divided by
`sum_i max_c d_i(c)`. Nonfinite or zero global scale is rejected. The solver
removes dominated points, constructs lower convex hulls, and consumes edges
in decreasing reduction per byte with stable ties. It rounds the fractional
boundary down, then scans later affordable edges whose preceding endpoint
is selected. The report contains normalized/raw objectives and LP gaps, beta,
realized Jeffreys contrast, tau, per-projection profile counts, per-layer BPW,
exposure statistics and reserved/used pool costs. The certificate bounds
optimization error for this proxy; integer optimality and model quality are
not claimed.

## Commands and reuse

Use the same checkout, source, imatrix, profiles and cache for all commands:

```bash
mfq calibrate natscale-errors --model MODEL --imatrix IMATRIX --cache CACHE \
  --backend cuda --device cuda:0
mfq calibrate natscale-statistics --model MODEL --imatrix IMATRIX --cache CACHE \
  --backend cuda --device cuda:0
mfq quantize MODEL glm-IQ1_S.mfq --imatrix IMATRIX --alphaq-cache CACHE \
  --ud-tier IQ1_S --alphaq-allocator natscale --backend cuda --device cuda:0
mfq quantize MODEL glm-IQ4_XS.mfq --imatrix IMATRIX --alphaq-cache CACHE \
  --ud-tier IQ4_XS --alphaq-allocator natscale --backend cuda --device cuda:0
```

The statistics command makes one complete routed-weight pass. Measurement
reads sampled expert weights and full-bank training rows. Both resume completed
work. Quantize reuses them, trains remaining selected tables, fits selected
weights and calls the normal exporter. `--devices` explicitly assigns multiple
CUDA devices to quantize's statistics and final training/fitting; measurement
accepts one device. No model-quality result follows from completing export.

Sampling options are `--sample-layers`/`--sample-experts` on measurement and
matching `--natscale-sample-layers`/`--natscale-sample-experts` on quantize.
`--profiles` on measurement must match quantize's `--alphaq-profiles`.
`--natscale-error-cache PATH` changes the default `CACHE/natscale-errors.json`.
Allocation checkpoints include every NatScale setting and input digest.
`CACHE/allocations/<key>/allocation.json` records the objective/certificate;
`export-report.json` records actual main-text, vision, MTP and total bytes.
`--dry-run` reads config, index, safetensors headers and the imatrix, and prints
fixed bytes, expert capacity, auxiliary bytes and parameter count. It neither
reads weight data nor requires a measured-error cache.

## GLM storage and components

GLM-5.3-Flash identification requires `glm5_next_text`, hidden size 4096, 45
main layers, 288 routed experts, top-8 and vocabulary 154880. Main float
parameter count is 313,326,811,966. Pinned recipes under
`mfq/calibration/recipes/glm5_next/` have ceilings 93,087,245,088 (IQ1_S),
97,579,344,672 (IQ1_M), and 156,822,111,200 (IQ4_XS) bytes. The Qwen registry
retains its nine recipes and nearest-BPW selection.

Import-boundary aliases cover KDA, MLA, DSA indexers, mHC, dense/shared/routed
FFNs, roots and appended predictor block 45. Absorbed `embed_q`/`unembed_out`
head banks use `attn_k_b`/`attn_v_b` types. Integer runtime hash metadata is
preserved and has no GGUF float policy. GLM per-tensor recipe types are
authoritative; Qwen fused-pair anchors do not apply to GLM. FP8 dense weights
decode E4M3 with 128-square `weight_scale_inv` multipliers through the expert
source decoder. In AlphaQ **joint** mode, scaled FP8 dense matrices are
explicitly excluded from allocation and stored as decoded BF16 fixed payloads;
a native block-scaled FP8 dense candidate is not implemented there.

When present, vision defaults to `--alphaq-vision native` (source BF16).
`NINT8` quantizes rank-two vision matrices and preserves other shapes;
`exclude` omits vision. MTP defaults to `--alphaq-mtp quantize`: mapped UD
per-tensor formats are fitted **without imatrix weighting**, since the supplied
Unsloth imatrix stops at block 44. `exclude` omits it. Existing `--text-only`
and `--exclude-mtp` also omit these components. The written graph declares
the components included in the output.

Vision and MTP payloads are reported outside the main-text ceiling. Shared
assets, the full header and the full tensor table remain charged to main
text. Before publication the writer checks
`actual_total - actual_vision_payload - actual_mtp_payload <= cap`, equivalently
`actual_total <= cap + actual_vision_payload + actual_mtp_payload`.

## External gradient schema

`--natscale-sensitivity gradient:gradients.json` accepts:

```json
{
  "format": "mfq.natscale-gradients.v1",
  "source": {"model": "/absolute/model", "files": [["config.json", 123, 456]]},
  "experts": [
    {"layer": 3, "expert": 0, "second_moment": 0.002},
    {"layer": 3, "expert": 1, "second_moment": 0.005}
  ]
}
```

Expand this abbreviated example to every main routed expert exactly once.
IDs are nonnegative integers. `second_moment` is a finite nonnegative
conditional gradient second moment at the down output divided by output
width, as in the paper. `source` must equal
`mfq.calibration.alphaq_cache.source_identity(Path(MODEL))`, including every
file path/size/mtime entry. Duplicate/missing/extra experts and source
mismatches fail. The gradient checksum enters allocation identity. Supplying
gradients is an externally calibrated sensitivity variant, distinct from the
uniform default.
