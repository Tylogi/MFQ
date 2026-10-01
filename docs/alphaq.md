# AlphaQ: calibration-free expert precision allocation

MFQ implements the **official code-default AlphaQ objective** from
[Superone77/AlphaQ](https://github.com/Superone77/AlphaQ)
([paper](https://arxiv.org/abs/2606.04980)). It estimates importance using weights
alone. No calibration texts, chat templates, activations, gradients, candidate
reconstruction measurements, imatrix or learned sensitivity coefficient are
required. Budget and available quantization formats remain caller choices.

## Fixed method

For each routed expert projection matrix `W`:

1. Partition `W` into disjoint complete 128 x 128 blocks. Pool their squared
   singular values. Incomplete edge blocks are omitted from the spectrum;
   population variance still uses **all weights**. If either dimension is less
   than 128, use the complete matrix's singular values.
2. Sort the pooled eigenvalues `lambda` ascending. Set
   `k = min(n - 1, max(10, floor(0.1 * n)))` and compute the Hill estimate
   `alpha = 1 + k / sum(log(lambda_tail / lambda_threshold))`, with the official
   `1e-12` numerical floors.
3. Compute importance `I = median(alpha) / alpha * Var(W)` (gamma fixed to 1).
4. Minimize `sum(I[e,p] * 2**(-2*b[e,p,c]))` subject to the supplied size budget.
   NINT uses its nominal primary-code width `q` for `b`. Other candidate
   families use the table's **effective payload BPW**. These are scoring
   coordinates; actual storage costs are accounted independently.

A single positive global scale normalizes importance for numerical stability.
There is no per-layer or per-projection normalization. This preserves relative
layer importance. Constant matrices receive zero importance; a one-value
spectrum receives alpha=1 since a Hill tail is undefined.

MFQ uses **all** complete blocks deterministically. This differs from the
official implementation's optional random 256-block cap, avoids a sampling
seed, and matches the uncapped FARMS variant used in our experiments. Batched
FP32 SVD is used on CPU/CUDA; no approximate/randomized SVD is used. GPU batch
sizes decrease only on an actual out-of-memory error.

Packed `gate_up` banks remain a single projection if that is the model's native
quantizer granularity. Separate Gate, Up and Down banks are independently
allocated. This initial command does not invent split-tensor recipes, allocate
dense/attention tensors, or change the weight-fitting algorithm.

## Command and reuse

```shell
mfq calibrate alphaq --model model-hf --target-bpw 3.0 \
  --statistics alphaq-stats.json --output alphaq-3bpw.json

# Same source statistics, another global budget; no repeated SVD.
mfq calibrate alphaq --model model-hf --target-bpw 4.0 \
  --statistics alphaq-stats.json --output alphaq-4bpw.json

mfq quantize model-hf quantized.mfq --scheme alphaq-3bpw.json
```

Inputs can be supported HF Safetensors checkpoints or full-precision MFQ files.
Collection reuses the quantizer's source planner/readers, including native
scaled sources; no inference model is instantiated. Statistics are atomically
saved after each bank and reused by name, shape, method and source file
path/size/mtime identity. This is a local restart check, not a cryptographic
model-revision guarantee. Keep immutable checkpoints for reproducibility.

The default scheme sidecars are `<stem>.alphaq-statistics.json` and
`<stem>.report.json`. `--statistics` and `--report` override these paths.
Optional console-output failures do not interrupt computation. A failed
statistics/scheme write still raises an error.

## Candidate and size accounting

The built-in candidate set uses the existing standard specifications:

| Profile | q | group size | subgroup metadata bits |
|---|---:|---:|---:|
| NINT2 | 2 | 16 | 5 |
| NINT3 | 3 | 24 | 5 |
| NINT4 | 4 | 24 | 6 |
| NINT5 | 5 | 28 | 7 |
| NINT6 | 6 | 24 | 7 |
| NINT8 | 8 | 48 | 7 |

`--target-bpw` bounds **routed expert payload + expert IDs**, divided by routed
weight count. Built-in costs include padded groups, FP16 neuron scale/minimum,
both subgroup metadata streams, the NINTv2 q/k selectors, and byte rounding.
Fixed MFE/pool/file headers and non-routed weights are excluded. Per-expert
rounding can be slightly larger than the final coalesced-pool payload. This is
not a whole-file BPW or a promise that the output file fits this exact byte cap.

For custom/native/NVQ candidates, use the existing
[`mfq.ew-candidates.v1` and `mfq.ew-budget.v1`](ew-joint-solver.md) formats:

```shell
mfq calibrate alphaq --model model-hf \
  --statistics alphaq-stats.json \
  --candidates candidates.json --budget budget.json \
  --solver exact --output alphaq-custom.json
```

Candidate `distortion` can be set to 0; AlphaQ replaces both distortion fields
analytically. Candidate precision descriptors, artifact paths, exact variable
costs, pool charges and tensor charges are preserved. Relative codebook paths
are rebased to the scheme location. No codebooks are trained by this command;
standard MFQ weight quantization happens afterwards. The candidate table must
cover exactly the collected routed tensors, with matching dimensions and
source/quantizer scheme names. Use source-appropriate candidates for native
low-precision checkpoints; adding formats above their source precision does
not restore missing precision.

## Solver

The default `hull` solver solves the single-upper-budget continuous relaxation
using each expert's lower convex rate/loss hull. It rounds the fractional
boundary down, then spends remaining budget on feasible hull upgrades. It
reports the achieved objective, continuous lower bound, and absolute/relative
gap. The output is feasible; **integer optimality is not claimed**. Its work is
roughly `O(N K log(N K))`, with N expert projections and K candidates.

The fast solver rejects nonzero shared-pool activation charges and joint
lower/layer/projection/shape constraints. `--solver exact` explicitly selects
MFQ's existing HiGHS MILP for those cases; it can take substantially longer.
No constraints or fixed charges are silently dropped.

## Python API

```python
from mfq.calibration import (
    collect_alphaq, alphaq_nint_candidates, allocate_alphaq,
    EwBudget, RateBounds, save_scheme,
)

stats = collect_alphaq("model-hf", "alphaq-stats.json", device="cuda:0")
candidates = alphaq_nint_candidates(stats)
budget = EwBudget("AlphaQ", candidates.routed_weight_count, 0,
                  RateBounds(max_bpw=3.0), {}, {}, (), {})
result = allocate_alphaq(stats, candidates, budget)
save_scheme("alphaq.json", result.scheme)
```

`alphaq_weight_statistics` also accepts a floating `[E,O,I]` torch tensor.
`AlphaQTensorStatistics` and `alphaq_importance` expose the reusable statistics
and EW importance table. The latter must be combined with `alphaq_candidates`
when calling `solve_ew_budget` directly; empirical MSE is a different objective.

## Validation status

Tests compare batched spectra against scalar FP64 SVD, CUDA against CPU,
NINTv2 byte costs against serialization, the structured relaxation against
SciPy LP/brute force, and exact allocation with fixed pool charges. CLI tests
check restart/reuse and actual quantizer plan consumption for packed and
separate experts. These validate the implementation, not model quality.

All produced schemes/reports explicitly say `not evaluated; surrogate only`.
Quantization quality requires complete WT2 forward KLD and Top1 agreement
against the reference under the same evaluation contract (normally ctx512).
