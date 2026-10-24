# Benchmark reporting workflow

Status: standalone inference workflow implemented, 2026-10-23.
Source review: `d3d6ebcb`. The [usage guide](inference-charts.md) documents the
runner hook, saved-result importer, eight chart groups, offline reports and
immutable local archive. Reporting dependencies are pinned and regression
tests run in CI. The broader data contract, Studio integration and additional
benchmark families below remain a design proposal; use the usage guide for
commands that are available now.

## Baseline before this implementation

The source review found scripts for selected published figures, but no single
workflow from a new benchmark run to a validated, shareable report. The gaps
below describe that baseline.

| Source | Existing output | Gap |
| --- | --- | --- |
| [`bench/model_bench.py`](../../bench/model_bench.py) | Raw runs, JSON summary, Markdown table | No plot command; limited run identity; summary uses per-run medians |
| [Studio evaluations](../../MFQStudio/src/features/evaluations/EvaluationDetails.tsx) | History, comparison tables, JSON export | No figure export; inference summary uses total tokens / total time |
| [Inference service](../../mfq/server/services/inference_benchmark.py) | Measured runs, warmups, parameters, comparison key | Failed jobs do not produce a completed evaluation; hardware/build identity needs more detail |
| [Figure generators](../figures/generate-inference-speed.py) | English and Chinese SVGs from fixed snapshots | Input files, layouts and some axis limits are specific to one published comparison |
| [Metal comparison](qwen38-flash-next-metal-comparison.md) | Source revisions, timing notes, regeneration command | Some source results lack repetitions or use different timing definitions |
| [CUDA format benchmarks](../../bench/cuda_format_comparison/README.md) | Timing samples and numerical comparison JSON | Separate kernel scope; no shared reporting contract |

The older figure scripts share a [Studio theme](../figures/studio-figure-theme.json).
New benchmark reports use a separate white, blue, rose and cyan
[benchmark theme](../../bench/reporting-theme.json). The older figure scripts
remain separate from the new reporting environment and its CI checks.

## Workflow and scope

Use this sequence for each report:

1. Define the workload, expected cases, baseline and fields allowed to vary.
2. Run the benchmark and save raw data plus effective runtime metadata locally.
3. Import raw data through a source adapter and validate each measurement.
4. Form comparable groups and compute statistics from measured samples.
5. Create a public bundle using an explicit metadata allowlist.
6. Render figures, tables and a methods report from that same bundle.
7. Verify data consistency, rendering and privacy before sharing the bundle.

Start with the standalone runner and Studio inference exports. Add WT2
quality and task accuracy adapters next. Keep kernel timings in a separate
report family. Do not combine kernel event timing, model throughput and
concurrent request throughput into one score.

Reporting must run offline on a CPU machine. It must not load a model, start
a runtime or rerun a GPU measurement. Run results remain immutable; rendering
can be repeated without changing the measurements.

## Data contract

Introduce a versioned normalized format, `mfq.benchmark-report.v1`, shared by
the command-line report tool and a later Studio report action. Separate
private import data from the public bundle.

| Section | Required information |
| --- | --- |
| `schema`, `report_id`, `family` | Schema version; report ID; inference, quality, accuracy or kernel |
| `sources` | Adapter and version; private source digest; public bundle digest; evidence level |
| `experiment` | Expected cases and measured repetitions; warmup policy; case order; baseline; variable fields |
| `identities` | Model and tokenizer digests; quantization; runtime revision and binary digest; backend; GPU type/count/VRAM; driver/runtime versions; relevant memory policies |
| `protocol` | Timing definition; input fixture identity and token lengths; context; output limit; sampling; batch/concurrency; cache policy; MTP configuration; cooldown |
| `samples` | Case ID; run ID/order; warmup flag; status; actual counts; timing values and units; MTP counters; normal finish information when available |
| `summary` | Valid, failed, warmup and missing counts; statistics and aggregation method for each group |
| `validation` | Stable reason codes; missing metadata; comparison eligibility; publication eligibility |

Store durations in milliseconds and sizes in bytes; convert only for display.
Use explicit metric names such as `decode_tps`, `ttft_ms`, `weight_bytes` and
`peak_device_bytes`. Model weight size, resident weights, peak device memory
and process memory are different quantities and must have different labels.
Never infer quantization or model identity from a filename alone.

Unknown values are `null` with a reason. Unknown is not zero and two unknown
identities do not establish a match. Retain per-run data in the public bundle
when it contains only approved numeric values and safe IDs.

### Import boundaries

- The standalone adapter reads `results[].runs`, not `summary`. The current
  writer needs a schema version, explicit requested modes, protocol identity,
  effective context/settings, finish reason, run ID and build/hardware/model
  metadata. Keep its existing fields for current consumers.
- The Studio adapter reads `metrics.runs` and `metrics.warmup_runs`, plus
  parameters and identities. It must not treat `metrics.series` as raw samples.
  Existing `comparison_key` is useful input, but is not a complete hardware
  or build comparison check.
- Record the expected case matrix before execution. A standalone failure may
  stop later cases; a failed Studio job may have no evaluation export.
  Preserve job failure status and any validated progress separately. Never
  fabricate successful samples for those cases.
- Put readers for existing unversioned exports under `compat/`. Label them
  `legacy_partial` when required facts are absent. A local metadata sidecar
  can supply facts with recorded provenance; it must not invent missing runs,
  finish reasons, token identities or uncertainty estimates.
- Existing published snapshots remain usable as `published_snapshot` evidence.
  Preserve their original statistic and timing definition. Render them with
  that label; exclude them from automated performance regression decisions.

New writers should retain the validated native result needed by
[`validate_benchmark_result`](../../mfq/evaluation/inference.py). Importers
for older flattened exports perform only the checks supported by their data;
they must not create a fake native response to pass validation.

## Validation and comparison

Require finite numeric values, positive measured timing intervals, exact input
sizing, no prefix reuse for a cold-prefill protocol, and the requested MTP mode
actually executing. Require unique run IDs and counts consistent with the
experiment matrix. Reject duplicate JSON keys and unsupported schema versions.

Preserve failed, timeout, skipped and not-run cases as separate states. Show
their reason codes and counts in the report. They have no numeric bar.
An incomplete experiment can produce a diagnostic report, but not a passing
publication or regression result. A malformed file fails import before
writing output.

The standard fixed-workload speed report requires the requested number of
completion tokens. Early EOS is a valid runtime outcome, but is marked
`short_output` and excluded from that comparison. A separate natural-stop
protocol can retain it with actual lengths. One completion token has no
decode interval, so decode throughput is unavailable in the report.

Use two explicit comparison modes:

- **Regression:** same model, tokenizer, hardware, driver, workload, timing
  definition and effective runtime settings. Only declared candidate fields,
  usually runtime revision/binary digest, may differ.
- **Configuration comparison:** explicitly vary quantization, backend,
  hardware or runtime. Display each differing field. Compare whole tested
  configurations; do not claim to isolate a kernel or runtime change.

MTP on/off is an explicit variable. Keep draft policy, draft weights and
maximum depth visible. Keep the requested output length and actual lengths
visible too. Separate hardware into panels by default. Never pool different
prompt lengths, devices, models, cache policies or timing scopes.

A comparison key is a canonical digest of all required invariants after
removing only the declared variables. Inspect the underlying values as well
as the digest. A different `prefill` definition, such as time to first token
versus engine prompt time, requires a separate panel and metric label.

For quality results, match dataset revision, split, tokenization, scored
positions, context and reference identity. For task accuracy, also match
task/scorer version, question set, demonstrations, sampling, output limit and
pass@k definition. Do not average scores from unrelated tasks into a default
overall score.

## Statistics and run profiles

Use measured runs only; retain warmups as diagnostics. Recompute per-run
prefill throughput as `prompt_tokens * 1000 / prefill_ms` and decode throughput
as `(completion_tokens - 1) * 1000 / decode_ms`, using the native interval
definition. Do not divide all generated tokens by decode time.

The default inference plot shows median and the 25th–75th percentile interval,
with sample count and optional individual points. Compute quantiles by linear
interpolation at `(n - 1) * p` on sorted samples. This interval describes
spread; it is not a confidence interval. Never fabricate error bars for a
summary-only source.

Also put total tokens / total time in the table as `pooled_tps`. This preserves
the current Studio statistic while making its difference from median explicit.
MTP acceptance is `sum(accepted) / sum(drafted)`, not a mean of per-run ratios.
Use full precision for calculations and round only display values.

Proposed profiles:

| Profile | Warmups per case | Measured runs per case | Intended use |
| --- | ---: | ---: | --- |
| `smoke` | 1 | 2 | Check data collection and report generation; label as smoke |
| `standard` | 1 | 5 | Default comparison report with median, spread and raw points |
| `regression` | 1 | At least 5 per revision | Alternate baseline/candidate runs; keep run order and pairing |

These are reporting policy defaults, not claims of statistical significance.
Five runs do not guarantee stable performance. Preserve cooldown, device
sharing, power mode and run order; expose missing controls. For a release
decision, use repeat experiments and project-selected thresholds rather than
a universal percentage threshold. A smoke result cannot pass as `standard`.

Show a regression ratio as candidate median / baseline median for throughput,
or baseline median / candidate median for latency; in both cases above 1 is
better. Label the formula. Do not present this ratio as paired statistical
evidence unless the experiment recorded paired runs.

Quality uses token-weighted aggregates and task accuracy uses the documented
scorer's numerator/denominator. Do not apply inference repetition statistics
to either family. Show uncertainty only when the source data and method
support it.

## Standard figure set

| Figure | Axes and grouping | Availability |
| --- | --- | --- |
| Inference speed | Input tokens versus prefill/decode tok/s in separate panels; decode and MTP series | First version |
| First-token latency | Input tokens versus TTFT ms; same grouping and spread | First version |
| MTP behavior | Acceptance percentage and speed ratio in separate panels; actual draft settings | When both modes are valid and comparable |
| Regression | Per-case speed ratio around a reference line at 1; missing/failed case table | When a valid baseline is supplied |
| Quality versus size | Weight bytes versus KLD, Top1 agreement or perplexity, each on its own axis | Quality adapter |
| Task accuracy | Score per task/category, with denominator and metric definition | Accuracy adapter |
| Kernel comparison | Event latency by operation/shape, with timing scope and correctness status | Separate kernel adapter |

Every figure includes model/configuration, hardware, runtime revision, workload,
statistic, sample count, metric unit and report ID. The report supplies full
methods and source details. Use dynamic limits, zero-based bars, stable colors
and marker shapes. Show missing cases explicitly. Paginate crowded figures;
do not shrink labels until unreadable. Use a log scale only with an explicit
label and valid positive data; KLD zero remains visible in a linear view.

Use the separate blue/rose [benchmark theme](../../bench/reporting-theme.json)
with a white background; keep mode colors consistent across charts. Generate English and Simplified Chinese
variants to match the existing published figures; keep text in a locale map
so Traditional Chinese can be added without changing calculations.
SVG is the primary shareable output, with PNG previews and optional PDF.
Numeric data and alt text must remain available beside the figures.

## Command interface and output

Proposed interface, not an existing command:

```sh
python -m mfq.benchmark_report build \
  --manifest .mfq/benchmarks/report.local.json \
  --output .mfq/benchmark-reports/current

python -m mfq.benchmark_report verify \
  .mfq/benchmark-reports/current

python -m mfq.benchmark_report render \
  --bundle .mfq/benchmark-reports/current/bundle.json \
  --output .mfq/benchmark-reports/rebuilt
```

The local manifest names input adapters/files, metadata sidecars, expected
cases, profile, baseline and variable fields. It is private and must not be
copied into the public output. `build` validates, normalizes, sanitizes and
renders. `render` only consumes the public bundle. `verify` validates the
schema, eligibility, checksums, plot/table agreement and public field policy.

Output consists of `bundle.json`, `summary.csv`, `report.md`,
`validation.json`, `checksums.json` and `figures/`. `report.md` records methods,
comparison limits, failed/missing cases and generator versions, and links to
each figure. No output depends on the original local source paths.

Write to a temporary directory and publish the complete bundle atomically.
Refuse a nonempty destination by default. Distinguish exit statuses for success,
incomplete evidence, invalid input, rendering failure and privacy failure.
An incomplete diagnostic bundle must retain its non-passing status on rerender.

Lock Python/reporting dependencies and fonts in a separate reporting
environment. Record tool, theme and font versions. Remove timestamps from
image metadata and fix SVG IDs. In that pinned environment, rebuilding the
same bundle must produce the same artifact hashes. Missing required fonts
must fail rendering rather than produce missing CJK glyphs.

## Public metadata policy

The public bundle is constructed from approved fields, not a copy of arbitrary
input dictionaries. Allow hardware specifications, public model aliases and
revisions, numeric results, protocol settings, stable reason codes and safe
build identifiers. Keep raw source digests and local source mappings private;
publish the digest of the sanitized bundle for verification.

Exclude hostnames, user names, IP addresses, device serials/UUIDs, absolute
paths, raw environment variables, command lines, runtime log tails, stack
traces, prompts, generated answers and authentication data. Explicitly map
performance settings into typed fields instead of exporting `env`. Hashes
are not anonymization: publish fixture/model hashes only for approved public
artifacts, and use opaque aliases for private inputs. Such aliases alone
cannot establish cross-report identity.

Validate free-text labels and source links as well as schema keys. Check JSON,
Markdown, CSV and SVG metadata for seeded secrets and path-like values; escape
markup and spreadsheet-formula prefixes. Do not attach raw errors to charts.
Keep diagnostic detail in a private file outside the public output directory.

Generated bundles stay under ignored `.mfq/` or CI artifact storage by default,
in line with the [contribution rules](../../CONTRIBUTING.md). This design does
not add generated measurements or figures to source control. Existing curated
figures remain readable; a future migration must keep their source caveats.

## Delivery stages and acceptance

1. Add `mfq/benchmark_report/` with contract, validation, comparison, statistics
   and public-bundle modules. Put old-format readers in `compat/`. Add new
   metadata emission to the standalone runner and inference service.
2. Add the two inference adapters and the CPU renderer. Reuse the theme;
   extract common drawing helpers without replacing historical data.
3. Add CPU CI fixtures for the full import-to-report path, plus a controlled
   GPU smoke check that feeds real raw output through that same path.
4. Add WT2 and accuracy adapters, then a Studio report download backed by the
   same contract and renderer. Preserve JSON export for raw inspection.
5. Add a kernel adapter only after its operation, fixture, timing and numerical
   correctness contracts are explicit.

Focused acceptance cases:

- Equivalent standalone and Studio samples yield identical normalized
  inference statistics; the tests include unequal durations so median and
  pooled throughput cannot be confused.
- Warmups never affect results; duplicate runs, NaN/Inf, empty intervals,
  incorrect token counts, MTP mismatch, early EOS and timeouts are exercised.
- A failed first case leaves later expected cases marked not-run. A failed
  Studio job without an evaluation remains visible in the diagnostic report.
- Differing hardware, prompt identity, cache policy or timing scope blocks a
  regression comparison; declared configuration comparisons display changes.
- Old snapshots retain their statistic and evidence label. Missing metadata
  or repeats cannot be upgraded by merely rerendering a bundle.
- CSV/JSON values and SVG plotted values agree. Axes include all finite data;
  empty and crowded reports are readable. Render and visually inspect both
  locales, including long labels, missing rows and error bars.
- The same sanitized bundle renders twice with matching hashes in the pinned
  environment. Seeded paths, secrets and prompts never enter any public file.
- A GPU smoke run proves collection, validation, export and CPU rendering end
  to end; it does not establish a performance improvement.

The first usable milestone is a local command that accepts fresh standalone
or Studio inference data, emits the speed/TTFT report and a verified public
bundle, and requires no manual transfer of numbers into a plotting script.
