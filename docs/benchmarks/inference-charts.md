# Inference chart reports

[`bench/render_model_report.py`](../../bench/render_model_report.py) renders
saved `bench/model_bench.py` results without loading a model or using a GPU.
The report uses a white background, blue for ordinary decode, rose for MTP,
and cyan for GPU telemetry. It has its own
[theme](../../bench/reporting-theme.json), separate from Studio UI colors.

```sh
uv run --no-project --with matplotlib==3.11.0 python bench/render_model_report.py \
  .mfq/benchmarks/results.json \
  --metadata .mfq/benchmarks/metadata.json \
  --output .mfq/benchmark-reports/run-01
```

The raw JSON is required. Metadata is optional; without it, hardware and
runtime revision are labeled as unrecorded. Metadata can supply public
`model`, `hardware`, `backend`, `source_revision`, `runtime_sha256` and
`protocol` fields. Do not use local paths or hostnames as public labels.
Arbitrary raw metadata, environment variables, filenames and logs are not
copied into the output. Counts in metadata must match the raw run.

The default expected modes are `decode mtp`. For an ordinary-only run, pass
`--modes decode`. For a multi-model input, select one with `--model-index`.
This first renderer requires a complete fixed-workload matrix. It rejects
failed, missing, duplicate-count or short-output cells instead of hiding
them in a successful comparison. It does not claim to implement the full
[reporting workflow design](reporting-workflow.md).

Optional GPU inputs are `--gpu-samples`, `--gpu-before` and `--gpu-after`.
Samples use the saved form `[{"time": ..., "gpu": {...}}]`; readings contain
`GPU use (%)`, `VRAM Total Used Memory (B)` and `VRAM Total Memory (B)`.
Only those counters and relative sample times enter the public report.
Before/after inputs are readings; before can also be a list of readings.
These device-wide samples cannot establish per-request memory use or the
true instantaneous peak.

The output contains an offline `index.html` gallery, eight PNG/SVG figures,
an eight-page `charts.pdf`, `bundle.json`, `samples.csv`, `summary.csv`,
`coverage.csv`, `theme.json`, `report.md` and `checksums.json`.
An existing output directory is not overwritten.

| Chart | Metrics |
| --- | --- |
| Throughput | Prefill, decode, whole-request generation tok/s |
| Latency | TTFT, prefill ms, decode ms, derived generation duration and mean decode interval |
| MTP | Pooled acceptance, drafted/accepted/rejected counts, throughput ratio and TTFT cost |
| Stability | Every measured request's deviation from its case median |
| Warmup | Warmup and measured requests in saved order; timing on a log scale |
| Aggregation | Median versus pooled throughput for all three throughput metrics |
| GPU | Sampled utilization and allocated VRAM; capacity and before/after readings |
| Validation | Requested/actual input, prefill and completion counts; sample counts and MTP mode |

All source run fields have an entry in the coverage table. KLD, Top1,
perplexity, task accuracy, pass@1, MC1–MC3, power, energy and per-token tail
latencies stay marked as not measured when they are absent. Results from a
different model or experiment are not used to fill those gaps.

Warmups are excluded from statistics. Bars show per-request medians; whiskers
show interpolated 25th–75th percentiles. The spread is not a confidence
interval. Decode uses `(completion_tokens - 1) / decode_time`. Generation
throughput covers the full native request interval; its duration is derived
as `completion_tokens / generation_tps`. Mean decode interval is
`decode_ms / (completion_tokens - 1)`, not a per-token latency distribution.
MTP acceptance uses total accepted / total drafted tokens.

Plots keep the native metric scopes separate. The report is local and does
not insert results into Studio's SQLite database. Generated files belong in
ignored local storage or artifact storage, not source control.

## Automatic reports after each benchmark

Use Python 3.11 or newer with the pinned reporting environment. Run one model
per command. The report hook is enabled by `--report-root`:

```sh
uv run --no-project --python 3.11 \
  --with-requirements bench/requirements-reporting.txt \
  python bench/model_bench.py models/example.mfq \
  --runtime build/hip/mfq-runtime \
  --prefill-lengths 32,128,512 --decode-tokens 64 \
  --repetitions 5 --warmup-runs 1 --ctx-size 1024 --mtp \
  --report-metadata bench/report-metadata.example.json \
  --report-root .mfq/benchmarks
```

Copy the example metadata file into ignored local storage and set the public
model/hardware labels for the actual run. The runner checks metadata and
report dependencies before loading the model. It records the runtime file's
SHA-256 and the effective workload, context size, seed, temperature, batch
and cache settings. A source revision must describe the binary's build;
it is not inferred from the current checkout. No connection details or model
paths belong in the metadata labels.

The hook saves the raw results, checks all expected samples, renders the
eight chart groups and validates the manifest, checksums, links and PDF page
count. It then adds the result to the local report archive. A run or report
failure exits nonzero and keeps the previous latest-success pointer.

```text
.mfq/benchmarks/
  private/<run-id>/
    raw.json
    metadata.json
    status.json
    error.txt             # only on failure
  public/
    index.html            # browsable history
    latest.json           # latest successful report
    <run-id>/             # immutable report and data bundle
```

Only `public/` is suitable for sharing. Raw environment overrides, error text
and any local identifiers stay in `private/`, which has owner-only access.
Neither the hook nor the report importer uploads data. GPU telemetry is
optional input; the hook does not start a device monitoring process.

Each invocation gets a distinct run ID; retrying an import creates a new
record. Historical report directories are not overwritten. Publications are
serialized with `.publish-lock`; another writer fails with a retryable error.
If a process is killed while holding that lock, first confirm no publisher
is running, then remove the empty lock directory before retrying. A catalog
write failure can leave a verified report that is not the latest report;
its private status records the failure. Importing the saved raw data again
recovers without rerunning the GPU workload.

To import a completed run, including previously saved GPU telemetry:

```sh
uv run --no-project --python 3.11 \
  --with-requirements bench/requirements-reporting.txt \
  python bench/report_workflow.py .mfq/results.json \
  --metadata .mfq/metadata.json \
  --gpu-samples .mfq/gpu-samples.json \
  --gpu-before .mfq/gpu-before.json --gpu-after .mfq/gpu-after.json \
  --archive-root .mfq/benchmarks
```

Omit the GPU arguments if no telemetry was saved. Pass `--modes decode` for
an ordinary-only run and `--model-index` to select a model in an older
multi-model export. Quality, task and power metrics still require their own
measurements; the workflow does not infer them from throughput.

## Layout and regression checks

The HTML template adapts Kami's long-document typography, spacing and table
rules. It keeps the requested white/blue/rose/cyan palette, uses local font
fallbacks, and makes no font or script network requests. The original Kami
MIT notice is retained in the template. This is an offline chart report,
not a dependency on the installed Kami plugin. Narrow screens scroll charts
and coverage tables within the page rather than shrinking plot labels.

The PDF remains the eight-page chart appendix. The HTML adds a measurement
overview, computed MTP comparisons, test settings, navigation and downloads.
These two formats share the same normalized measurements.

```sh
uv run --no-project --python 3.11 \
  --with-requirements bench/requirements-reporting.txt --with pytest \
  python -m pytest -q tests/test_benchmark_report.py tests/test_report_workflow.py
```

The benchmark CI job also runs these tests. They cover invalid and partial
runs, warmup exclusion, pooled statistics, absent optional metrics, private
field removal, report failures, archive locking and a full stdio runner to
report test. The fixture runtime is synthetic and is not a GPU performance
measurement. Report dependencies are pinned; font and platform differences
can still change output bytes between machines.
