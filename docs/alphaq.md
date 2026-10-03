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
FP32 SVD is used on CPU and for small matrices. When the installed cuSOLVER
supports its generic batched symmetric API, CUDA computes complete 128-square
Gram spectra using full FP32 products with TF32 disabled. Low-rank or nearly
flat spectral tails use the SVD reference path to avoid Gram roundoff. No
approximate/randomized spectrum or block subsampling is used. GPU batch sizes
decrease only on an actual out-of-memory error; the batched solver's integer
indexing limit is also respected. `alphaq_weight_statistics(..., backend="svd")`
selects the reference implementation explicitly.

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

Separate FP8 experts apply their tensor or block scale multipliers for both HF
and MFQ inputs, in statistics collection and subsequent quantization. Older
cached MFQ expert statistics without the scale-decoding marker are recomputed
per affected bank; all unaffected cached banks remain reusable.

The default scheme sidecars are `<stem>.alphaq-statistics.json` and
`<stem>.report.json`. `--statistics` and `--report` override these paths.
Optional console-output failures do not interrupt computation. A failed
statistics/scheme write still raises an error.

## Candidate and size accounting

The built-in candidate set uses the existing standard specifications:

| Profile | q | group size | subgroup metadata bits |
|---|---:|---:|---:|
| NINT4 | 4 | 24 | 6 |
| NINT5 | 5 | 28 | 7 |
| NINT6 | 6 | 24 | 7 |
| NINT8 | 8 | 48 | 7 |

The default also includes **NVQ1-S, NVQ1-L, NVQ2J, NVQ2J-L, NVQ2J-XL,
NVQ3J, NVQ3J-512 and NVQ3J-L**: twelve choices in total. NINT2/3 are absent
from the built-in CLI candidate set. Repeat `--profile` to select a subset.
NVQ uses the canonical format group sizes and stream layouts, including
NVQ2J-XL's aligned 64-bit groups. JSC bank counts are explicit: four for
NVQ2J variants and two for NVQ3J variants.

`--target-bpw` bounds routed expert streams, expert IDs, MFE headers, pool
headers and embedded codebooks, divided by routed weight count. Built-in
costs include padded groups, neuron anchors, metadata/selectors and byte
rounding. Shared tables are charged once per selected tensor/profile pool.
Per-expert rounding and separate NINT pool charges may slightly exceed final
coalesced storage. Outer file metadata and non-routed weights are excluded.
For a whole-model budget, use `--budget` with the full model weight count
and all non-routed/container costs in `model_fixed_storage_bits`, then verify
the final serialized file size.

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

For shared pools, the fast solver reserves the cost of all enabled pools
before allocating, then charges only pools actually selected. The report
includes reserved, used and unused pool bits. Its lower bound uses the
relaxation with pool costs removed, so the reported gap remains valid.
This conservative reservation can leave budget unused or reject a very small
budget that an exact activation solve could fit. `--solver exact` selects
the existing HiGHS MILP for those cases and for joint lower/layer/projection/
shape constraints; it can take substantially longer. No fixed costs are
silently dropped. Allocation itself does not fit weights or train codebooks.

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

Dense tensor selections in a calibration scheme accept either HF source names
or canonical MFQ names. NVQ descriptors retain their family, frozen codebook
artifact (relative to the scheme file), and quantizer options through normal
HF conversion. Supplying both aliases for the same tensor is rejected.
Native dense storage uses the existing explicit BF16/F16/F32 tensor override;
it does not introduce a synthetic quantized precision. `calibrate alphaq`
retains its routed-weight budget. The normal `quantize --target-bpw` command
uses the complete-file budget described below.

`mfq.calibration.alphaq_joint` provides the analytical joint allocation API.
`dense_choices` accounts for complete canonical payloads, and `allocate_joint`
accepts a full-file byte cap, an explicit fixed-file cost, expert exposure and
optional router multipliers. Its result includes the quantized scheme and
separate native dtype overrides; both must be consumed by conversion. The
caller remains responsible for PLE, imatrix eligibility and exact container
accounting. No minimum dense precision is imposed. The full retained experiment
matches all 73,728 expert and 768 dense choices and all 9,984 dense candidate
byte costs. This is allocator equivalence, not a joint model-quality result.

The joint API marks affine NINT selections with `imatrix_weighted: true`,
including NINT8. Conversion requires the actual matching imatrix and preserves
this policy in dense and expert pool writers. Legacy recipe NINT8 retains its
unweighted fitting behavior. Weighted and unweighted NINT pools remain separate
when both occur in a scheme; storage estimation includes those separate pools.

Tests compare batched spectra against scalar FP64 SVD, CUDA against CPU,
NINTv2 byte costs against serialization, the structured relaxation against
SciPy LP/brute force, and exact allocation with fixed pool charges. CLI tests
check restart/reuse and actual quantizer plan consumption for packed and
separate experts. These validate the implementation, not model quality.

All produced schemes/reports explicitly say `not evaluated; surrogate only`.
Quantization quality requires complete WT2 forward KLD and Top1 agreement
against the reference under the same evaluation contract (normally ctx512).

## Verified joint workflow: AlphaQ with native router frequency

The joint workflow shares one whole-file budget between routed experts and
eligible non-PLE dense matrices. To use a native routing-count cache collected
separately from the imatrix, select both options explicitly:

```bash
mfq quantize model-hf output.mfq --imatrix calibration.gguf \
  --target-bpw 3.5658365879106404 --backend cuda \
  --alphaq-dense joint --alphaq-router-cache native-router.json \
  --alphaq-cache alphaq-cache --devices 0,1
```

`native-router.json` follows the model-bound `mfq.alphaq-router-frequency.v1`
schema described below. The evaluated cache contains 1,048,576 calibration
tokens, 512 experts per layer, and ten selected experts per token. Router
frequency weights expert importance; the imatrix weights candidate fitting.
The allocation objective uses analytical AlphaQ distortion. Measured weighted
SSE is not an extra multiplier in this selected method. The built-in candidate
set is NVQ1-S, NVQ1-L, NVQ2J, NVQ2J-L, NVQ2J-XL, NVQ3J, NVQ3J-512, NVQ3J-L,
NINT4, NINT5, NINT6 and NINT8, using their canonical group definitions.

In the Qwen3.8-Flash-Next evaluation, 73,728 expert projections and 768 dense
matrices participate. Four dense matrices without matching imatrix entries
retain source BF16. Dense matrices also have a native-precision candidate.
PLE remains at the selected tier's mapped precision and consumes the same
whole-file budget. PLE frequency-based allocation is outside this workflow.
For this model, `--alphaq-dense joint` is necessary: `auto` retains the earlier
recipe policy for compatibility, as detailed below.

The following frozen experimental allocations completed full WikiText-2 test
forward evaluation on 580 independent contexts of length 512. Each context
scores positions 256 through 510, totaling 147,900 positions. KLD is
`KL(reference || candidate)`; Top1 is agreement with the reference argmax.
Reference PPL is 4.7181834093. All rows fit their corresponding pinned UD
whole-file byte caps; labels identify budget tiers rather than uniform formats.

| Budget tier | Actual whole-file BPW | KLD | Top1 agreement | PPL |
|---|---:|---:|---:|---:|
| IQ1_S | 3.279974 | 0.275743 | 81.8026% | 5.055572 |
| IQ1_M | 3.370049 | 0.251235 | 82.9135% | 4.987575 |
| Q2_K_XL | 3.565835 | 0.184769 | 85.2231% | 4.863464 |
| IQ3_XXS | 3.705662 | 0.148228 | 86.6315% | 4.830583 |
| Q3_K_XL | 4.068465 | 0.105772 | 88.6430% | 4.753334 |
| IQ4_XS | 4.235579 | 0.092440 | 89.3313% | 4.764935 |
| Q4_K_XL | 5.033566 | 0.047344 | 92.3239% | 4.719553 |
| Q5_K_XL | 7.155960 | 0.028423 | 94.1028% | 4.714414 |
| Q6_K_XL | 7.647549 | 0.025251 | 94.4131% | 4.728415 |

Matched fixed-dense controls exist for IQ1_S, IQ1_M and Q2_K_XL; joint allocation
improves KLD and Top1 in all three comparisons. Comparisons against the earlier
UD-mapped recipes also change the four missing-imatrix matrices to BF16 and
therefore do not isolate dense allocation. Q5_K_XL and Q6_K_XL improve all three
metrics over those earlier recipes; improvement across every budget is not
claimed.

These measurements use retained canonical candidate views. The normal CLI was
separately checked for allocator equivalence, candidate-byte preservation and
complete-file export; a fresh refit is not claimed to reproduce these exact
metrics. The final joint result summary SHA256 is
`b9ad8fd29f8a85f4cb4b1664cc3b69672189e7bae39c58a5d002f16b91e07309`.
Its evidence audit SHA256 is
`4fa849533cd399b796df7da5daa23bc38bd5bfcf0873ed1d0d49b94138af52a8`.
The audit reaggregates 21 complete results from 1,491 raw metric batches,
including nine joint results, nine earlier references and three matched controls.

## Whole-file quantization

```bash
mfq quantize model-hf output.mfq --imatrix calibration.gguf \
  --target-bpw 3.56583659 --backend cuda
```

This HF path collects or reuses AlphaQ statistics, allocates canonical profiles,
trains or reuses every selected NVQ-JSC table, and calls the normal converter.
Its ceiling is `floor(target_bpw * floating_model_parameters / 8)` bytes.
PLE, native tensors, integer runtime constants, runtime assets, header and
tensor table all consume that ceiling. Integer constants do not inflate the
parameter denominator. The main text model is included; vision and MTP are
excluded. The writer checks actual serialized size before publishing output.

`--alphaq-dense auto` uses pinned UD nonexpert recipes for the evaluated
Qwen3.8-Flash-Next configuration, choosing the closest of the nine UD whole-file
budgets. These recipes retain revision
`38bb39ee97821de2c9009abb7e93950eec396e66`; their expert precisions are replaced
by AlphaQ. This reflects the Q2 experiment's selected recipe fallback and does
not claim optimality at every budget or on other models. An explicit `--recipe`
selects another compatible nonexpert recipe. Other model configurations use
joint allocation by default.

`--alphaq-dense joint` allocates all non-PLE rank-two matrices with matching
imatrix entries together with routed experts. Missing-imatrix matrices keep
their source precision; small matrices have a native candidate without a
hardcoded precision floor. `--alphaq-dense recipe` retains the complete
nonexpert recipe, including entries without imatrix, matching the separate
recipe experiment. PLE is outside allocation: `--ple-dtype native`, `NINT4`,
or `NINT8` controls its lookups; the default `recipe` uses the selected recipe
or native storage when no recipe is supplied.

Router frequency comes from the imatrix's actual per-expert and router token
counts, normalized globally. `--alphaq-router none` disables this weighting.
An optional `--alphaq-router-cache` reads a model-matched native frequency cache
(`mfq.alphaq-router-frequency.v1`, source identity, topk, and per-layer integer
counts/tokens). Such a run uses additional calibration observations and must
be distinguished from the imatrix-only workflow.

`--alphaq-cache DIR` reuses model statistics across budgets and stores trained
tables by imatrix checksum. Each allocation has its own directory. The default
is `output.mfq.alphaq` beside the requested output. Stale identities or changed
table checksums fail explicitly. A completed `encoded-candidates.json`
manifest in that directory also enables direct reuse of encoded tensor and
expert candidates. The normal writer streams the selected bytes into one
atomic output without refitting weights or staging a second full model.
Entries bind source, imatrix, exact precision options, trained tables and
candidate-file identity; a missing or mismatched entry fails explicitly.
Weighted and unweighted NINT8 candidates are distinct. The cache keeps the
producer's checksum evidence and checks file size/modification time on reuse;
it does not reread every unused candidate to rehash the whole candidate bank.
Source candidates are preserved after export.

Without an encoded candidate manifest, selected weights are fitted with the
normal converter's codecs and retained under the cache's `fitted` directory.
Each completed tensor records its source, imatrix, exact precision/table
identity and checksum. Restarting the same command verifies and reuses these
tensors automatically; it never treats an incomplete tensor as complete.
Automatic row chunks start at the full tensor geometry and shrink only after
an actual allocation failure. An explicit `--row-chunk` remains explicit.
CUDA writers transfer packed NVQ and uniform-NINT fields directly, avoiding
the transfer and repacking of their expanded integer arrays.

The allocation is also checkpointed before table training or weight fitting,
so a later failure does not force another solve. Source/statistics/budget and
allocator identities must match; required checkpoint errors propagate.
Commands sharing one cache use an operating-system lock released when its
owner exits. Successful export retains fitted payloads for future reuse.

For multiple CUDA devices, add `--devices 0,1,2` (indices within
`CUDA_VISIBLE_DEVICES`). This explicit selection starts one persistent worker
per selected device. Available workers take the next tensor dynamically;
statistics have independent bank checkpoints. After allocation, each worker
trains or reuses one tensor's tables and immediately fits that tensor before
taking another. There is no barrier waiting for every table to finish before
weight fitting begins. A failed device is retired without retrying its task;
healthy workers continue saving their remaining work, and the command reports
the failure without publishing an incomplete model. Rerunning verifies and
reuses completed statistics, tables and weights. Existing encoded candidate
caches still export directly. `--device cuda:N` selects a single device and
cannot combine with `--devices`.

`--dry-run` checks scope and file budget
without collecting statistics or fitting weights. CPU end-to-end tests cover
normal model writing, byte limits, native retention, recipe policy, and reuse;
the full-size final artifact still requires its own complete WT2 validation.
