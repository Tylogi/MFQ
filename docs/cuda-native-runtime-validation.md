# Native CUDA runtime validation

MFQ's production CUDA inference executable is `mfq-runtime`. It owns tensor
storage, CUDA streams and events, CUDA Graphs, and generic graph operations
without Python, PyTorch, ATen, or LibTorch. The packed MFQ CUDA kernels are the
same kernels used by the optional reference executable.

`mfq-runtime-torch` is an opt-in migration target. It exists only for A/B
validation and is excluded from normal builds and packages.

## Operator regression targets

`mfq-native-tensor-cuda-test` covers padded, sliced, transposed and broadcast
matrix products (including zero contraction and CUDA Graph capture), small-K
selection with stable ties, and tiled grouped-query attention. The attention
path bounds score storage to 128 query rows and shares the K/V head storage.

`mfq-packed-prefill-test` checks FP8-SQ CPU decoding and packed slicing,
compares MXFP4-SQ Tensor Core execution with the direct packed kernel, and
checks native TopK/graph behavior while printing timings for retained shapes.
`mfq-mxfp4-sq-test` consumes independently decoded wire fixtures, including
packed column/row slicing and CPU decoding. Generate them and check SQ linear
CPU execution and shard composition with:

```shell
python bench/cuda_mxfp4_sq_fixtures.py build/sq-fixtures
build/cuda-native/mfq-mxfp4-sq-test build/sq-fixtures
build/cuda-native/mfq-sq-linear-test build/sq-fixtures
```

SQ tensor-parallel partitions preserve the native scale blocks: MXFP4-SQ input
partitions align to 32 columns; FP8-SQ partitions align to the format's row and
column scale block boundaries. The single-GPU shard test exercises both output
gather and input reduction; it does not certify inter-device transport.

The CUDA workflow is manual and requires a dedicated `mfq-cuda` runner and the
`MFQ_CI_GPU_UUID` repository variable. It executes GPU numerical and graph tests;
source-contract tests alone do not establish CUDA correctness. The macOS
workflow also runs Qwen3.5 and MiniCPM-o text-prefill chunking tests.

MXFP4-SQ uses Tensor Core tiles by default for `M >= 32` and `M*N >= 131072`,
where the retained RTX 3090 Ti comparisons showed gains; smaller projections
keep direct packed execution. `MFQ_FORCE_SQ_TENSOR_CORE=1` tests that kernel on
smaller shapes. `MFQ_DISABLE_SQ_TENSOR_CORE=1`, `MFQ_DISABLE_NATIVE_STRIDED_BATCH_MATMUL=1`, and
`MFQ_DISABLE_NATIVE_TILED_SDPA=1` retain comparison paths for their corresponding
optimizations. These operator checks do not establish model-level token parity
or replace full-model quality evaluation.

## Build and dependency checks

Configure the production runtime on both Linux and Windows:

```shell
cmake -S cpp_runtime -B build/cuda-native \
  -DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=OFF \
  -DBUILD_TESTING=ON
cmake --build build/cuda-native --config Release -j
ctest --test-dir build/cuda-native -C Release --output-on-failure
```

The resulting executable must not import or link Python, Torch, ATen, c10, or
LibTorch. Check the final dependency table with `ldd` on Linux and
`dumpbin /DEPENDENTS` on Windows. CUDA runtime, CUDA driver, cuBLAS, the host C++
runtime, and optional NCCL are expected.

For an A/B build, configure a separate tree with LibTorch discoverable and
`-DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=ON`. This adds `mfq-runtime-torch`; it does
not alter `mfq-runtime`.

Execution policy environment variables are parsed when a
`CudaExecutionContext` is constructed. Each Engine keeps its own typed
`CudaExecutionConfig`; changing the process environment afterwards does not
reconfigure an existing Engine. Diagnostic A/B paths must override the target
Engine's config directly or construct a fresh Engine.

### SSD-backed mixed experts on Linux

With `--moe-gpu-cache-gb`, `MFQ_MOE_SSD_CACHE_DIR=/path/on/ssd` puts
NINT/NVQ and other cacheable packed expert fields in temporary file mappings.
This avoids retaining the whole expert model in anonymous RAM. MXFP4 keeps
its existing direct MFQ range reader. Choose an SSD directory, not a RAM disk;
allow space for another copy of the packed runtime expert fields. Files are
unlinked immediately and released with their last tensor view.

Loading still converts one projection at a time. Cache misses copy selected
experts through bounded pinned staging; `MFQ_MOE_MAPPED_GATHER` is disabled for
these mappings so CUDA cannot pin the whole model. Prefill uses the cache when
its selected experts fit, otherwise it stages one full projection. Statistics
report `file_backed_bytes` separately from retained `host_bytes`; resident file
pages are reclaimable OS cache and may still appear in process RSS.

`mfq-moe-host-store-test` checks exact bytes, offsets, empty fields, view
lifetime and invalid layouts. `mfq-mfe-decode-test MODEL TENSOR` compares every
FP16 output against the separate pool kernels for all experts, using both
shared and routed inputs at 1, 2, 3, 4 and 8 tokens. An optional third argument
writes all outputs as raw FP16 for byte comparison between builds. Setting
`MFQ_MOE_SSD_CACHE_DIR` also checks the SSD path with a 512MiB GPU cache. For example:

```shell
MFQ_MOE_SSD_CACHE_DIR=/path/on/ssd build/cpp_runtime/cuda/mfq-mfe-decode-test \
  model.mfq model.block.1.mlp.experts.down.weight
```

Validate a real model's generation and peak RAM separately; an individual
projection check does not prove full-model parity.

### Qwen3.8 Flash Next V4-XS measurement (2026-10-05)

Measured the complete five-shard `Qwen3.8-Flash-Next-EWQ-MFQ-V4-XS` model
on Linux with 32 GB system RAM, an RTX 4090 with 48 GiB VRAM, PCIe 4 x16,
and a Samsung 9100 PRO NVMe SSD. The native Release build used CUDA 13.2
and CUDA architecture 86. No Metal implementation was changed.

Both builds used a 30 GiB expert GPU cache and the same SSD mappings,
normal read-ahead, packed-parameter loading and shared-arena validation fixes.
The compute baseline used four NVQ warps for K=640, four routed NINT warps,
the original Qwen expert-reduction loop and one scatter block per transfer.
The candidate also splits large cache transfers across CUDA blocks.

The downloaded artifact marks MTP as removed and sets
`text_config.mtp_num_hidden_layers=0`, but its embedded
`text_config.mtp.num_hidden_layers` is still 1. For both builds an external
`--config` copy sets that nested count to 0 and `text_config.mtp.layer_types`
to `[]`. Weight files were not edited. The loader's predictor consistency
check remains enabled.

The prompt is “Explain why the sky is blue in one sentence.”, rendered with
the bundled chat template and thinking disabled: 23 input tokens, 33 greedy
output tokens, context capacity 128. Decode timing covers the 32 steps after
prefill and excludes model loading. Each process starts with an empty expert
cache. Run with the first shard path, not the containing directory:

```shell
MFQ_MOE_SSD_CACHE_DIR=/path/on/ssd MFQ_REPORT_CUDA_MEMORY=1 \
  build/cpp_runtime/mfq-diagnostics \
  --model /models/Qwen3.8-Flash-Next-EWQ-MFQ-V4-XS-00001-of-00005.mfq \
  --config corrected-config.json --ids-file prompt.i32 \
  --gen 33 --ctx-size 128 --moe-gpu-cache-gb 30
```

The two unprofiled runs per variant produced:

| Variant | Decode tokens/s |
| --- | ---: |
| Baseline | 3.750, 3.828 |
| NVQ/NINT/reduction changes, before parallel scatter | 3.955, 3.977 |
| Final candidate, including parallel scatter | 4.508, 4.529 |

The final mean is about 19% above the baseline mean. Loading took about
134–138 seconds and prefill about 9.5–9.8 seconds. These are short-prompt SSD
measurements, not resident kernel timing or a long-context throughput claim.

The final candidate matches the baseline byte for byte across all 51 dumped
stages: embedding, 48 blocks, final norm and complete logits, totaling
17,310,720 FP32 values. All 33 generated token IDs also match. Separate real
projection checks compare every FP16 output across all 512 experts for
1, 2, 3, 4 and 8 tokens with shared and routed inputs, including SSD cache
misses. Scatter tests cover mixed transfer sizes, zero-length copies,
unaligned pointers, vector boundaries and guard bytes up to 5 MiB per field.

The expert mappings occupy 52.59 GiB of temporary SSD space, with about
31.3 MiB of persistent host metadata. Non-profiled generation stayed below
22 GiB peak process RSS and about 2.1 GiB peak anonymous RAM. File-backed RSS
is reclaimable page cache. GPU use after prefill was about 38.33 GiB. A test
runner monitored memory every 0.5 seconds and would stop below 2 GiB available
RAM or above 22 GiB anonymous RAM; no run reached either threshold.

The resident single-token down-projection microbenchmarks improved from
51.08 to 42.89 microseconds and from 51.75 to 41.58 microseconds (medians of
five 3,000-iteration runs). At this stage the SSD cache used per-pool NVQ
dispatch, so these results exclude the heterogeneous NVQ speedup.
The exact Qwen reduction preserves separate FP32 product/add rounding before
the final FP16 cast; it does not silently replace those operations with FMA.

Nsight Systems captured only the 32 decode steps, with CUDA tracing and CPU
sampling disabled. Comparing the candidate before and after parallel scatter:

| Decode measurement | One scatter block | Parallel scatter |
| --- | ---: | ---: |
| Scatter kernel time, 3,381 calls | 1.1929 s | 0.0479 s |
| All kernel time, 318,389 calls | 2.5700 s | 1.5093 s |
| H2D copy time, 6.962 GB | 0.2825 s | 0.2829 s |
| CPU time inside `cudaEventSynchronize`, 4,608 calls | 1.7519 s | 0.7112 s |
| First-to-last GPU activity span | 8.3990 s | 7.2918 s |

Use the unprofiled runs for throughput: the diagnostic timer also includes
profiler start/stop overhead when capture is enabled. After the scatter fix,
the union of GPU kernels, copies and memsets is 1.8036 seconds, or 24.7% of
the captured activity span. The remaining gaps include host preparation,
dispatch and I/O; a CUDA trace alone cannot assign all of them to SSD faults.
The follow-up below addresses repeated routing, grouped normalization and
NVQ dispatch for cached weights.

#### Follow-up: route reuse, grouped RMSNorm and cached NVQ dispatch

Commits `db384ed8`, `8e07a077` and `7a711f87` share one route plan across
gate/up/down, fuse native Qwen grouped RMSNorm, and initialize the existing
heterogeneous NVQ dispatcher against GPU cache arenas. Fixed expert-to-pool
ownership is separate from changing expert-to-slot indices; slot updates and
invalidations travel in the same transfer batch as expert data.

With the same model, prompt and 30 GiB cache, interleaved unprofiled runs
of the previous implementation and this follow-up produced:

| Variant | Decode tokens/s | Mean |
| --- | ---: | ---: |
| Previous implementation rerun (`85e29053`) | 4.592, 4.671 | 4.632 |
| Follow-up (`7a711f87`) | 4.920, 4.944 | 4.932 |

The follow-up is **6.5% faster than the rerun baseline mean**. Comparing with
the older recorded 4.518 mean gives 9.2%, but the fresh comparison accounts
for the higher baseline in this session. Candidate loading took 132.5–136.6
seconds and prefill 8.98–9.24 seconds. Route reuse alone measured 4.687
tokens/s in one run; this ablation was not a repeated speed estimate.

Both generation runs reproduce all 33 token IDs. All 51 full-model trace
stages remain byte-identical to the original compute baseline. Norm checks
cover 360 dtype/shape/layout/magnitude combinations, an epsilon regression,
and 24 graph replay cases. The fused kernel preserves the mean reduction
tree, separate FP32 products, FP64 epsilon addition and final cast. Keeping
epsilon in FP64 matters for small activations. The corrected F32 decode norm
median is 5.26 microseconds versus 18.42 for the tensor chain (five 1,000-call
runs, width 10,240 and group size 2,560).

Three real gate/down projection checks use a 192 MiB cache. All 1,392 output
comparisons pass, with 4,729–4,807 evictions per projection and no full-weight
fallback. Disabling heterogeneous dispatch also preserves the baseline bytes.
Ten resident fixtures cover eleven NVQ/NPQ families and widths 96–4,096.
Their seven-row packed layout fails the SSD cache's expert-major field guard
in both builds, so SSD validation uses the real model projections instead.

The new 32-step Nsight decode capture gives:

| Measurement | After parallel scatter | Follow-up |
| --- | ---: | ---: |
| All kernel launches | 318,389 | 268,565 |
| NVQ kernel launches | 35,232 | 4,608 |
| NVQ kernel time | 0.2175 s | 0.0976 s |
| All kernel time | 1.5093 s | 1.3797 s |
| `cudaMallocAsync` calls | 239,050 | 219,850 |
| `cudaEventSynchronize` calls | 4,608 | 1,536 |
| Time inside `cudaEventSynchronize` | 0.7112 s | 0.3923 s |
| H2D bytes during decode | 6,962,423,360 | 6,968,106,992 |
| First-to-last GPU activity span | 7.2918 s | 7.5630 s |

The extra H2D bytes carry NVQ slot maps. Expert payload traffic, cache
hits/misses and evictions are unchanged. Route readback over prefill and
generation drops from 316,800 to 105,600 bytes. The profiled activity span
does not reproduce the unprofiled throughput ordering; use the unprofiled
A/B runs for speed, not the profiler timer or a single captured span.

GPU activity covers 1.6751 seconds, 22.1% of the new captured span. Host
preparation, dispatch and I/O remain the main area to investigate; this trace
cannot separate their contributions to the gaps. Next targets are the
remaining gated-residual elementwise chains and overlap of NINT/NVQ SSD reads
with compute. Mixed-expert mmap fields still use synchronous host copies;
the existing deferred read pool serves MXFP4 ranges. Strata's
`FileExpertSource::prefetch` and POSIX `DirectFile` batch reads into bounded
staging buffers and are useful references for extending that path.

Candidate generation peaked at 21.67 GiB RSS and 2.02 GiB anonymous RAM,
with at least 23.56 GiB system RAM available; neither guard fired. GPU use
after prefill was 38.33 GiB. Profiling adds instrumentation memory and peaked
at 2.95 GiB anonymous RAM.

Local raw binaries, memory samples, full output comparisons and Nsight
reports are retained under `/tmp/mfq-qwen38-perf`. The complete native build
and the host-store, NINT q8, packed NINT rows, storage-offset/scatter and
Flash Next CTests passed.

Run the model-backed scheduler and execution-isolation gates on a machine with
enough device memory. The second command loads and generates with two complete
engines concurrently and compares each result with its serial oracle:

```shell
mfq-diagnostics --model model.mfq --check-continuous-batching
mfq-diagnostics --model model.mfq --check-engine-isolation
```

The continuous-batching gate covers stable slot retirement, paged-KV reuse and
release, CUDA Graph capture/replay, callback cancellation, and prefix reuse.

#### Follow-up: sustained decode bottleneck and speed ceiling

The 32-step measurements above include a cold expert cache, so they understate
sustained decode throughput. A longer run used the same model, prompt, corrected
MTP-disabled config and binary, with context capacity 256 and 99 timed decode
steps. Its generated prefix matches the 32-step runs. Physical read traffic was
sampled from `/proc/<pid>/io` every 0.5 seconds; phase totals below interpolate
the samples at the runtime's load, prefill and decode boundaries.

This fixed-step diagnostic continues after EOS. The one-sentence answer ends
at generated token 38; subsequent steps produce additional chat turns. The
99-step results below describe that fixed trace, not a natural 99-token answer.

| Expert cache | Timed window | Decode tokens/s | Time/token |
| --- | --- | ---: | ---: |
| 30 GiB | First 32 steps, mean of two cold runs | 4.932 | 202.8 ms |
| 30 GiB | Complete 99-step run | 6.199 | 161.3 ms |
| 30 GiB | Last 67 steps, inferred from the matching prefix | 7.067 | 141.5 ms |
| 36 GiB | Complete 99-step run | 6.347 | 157.6 ms |

The last-67-step value subtracts the mean 32-step decode time from the 99-step
time. It is a sustained-window estimate rather than per-token instrumentation.
The post-answer diagnostic continuation reaches about 7 tokens/s in this
estimated window; 4.932 tokens/s describes the cold 32-step window before EOS.
Increasing the cache budget by 20% improved the complete run by only 2.4%.
It reduced recorded evictions from 1,236 to 27, but misses only fell from
30,590 to 30,296 and H2D traffic from 25.885 GB to 25.534 GB. The 36 GiB run
left 3.04 GiB free after prefill, so further cache growth also leaves little
room for larger contexts. Cache size is not the primary limiter for this
workload, although other prompts can have different expert locality.

The two cold 32-step runs physically read 8.593 GB and 8.541 GB during decode,
or about 267.7 MB/token. Subtracting their mean from the matching 99-step run
gives 9.725 GB for the additional 67 steps, or 145.2 MB/token. Over the same
window the expert cache submitted 117.5 MB/token to the GPU, recorded 1,296.8
demand hits/token and still recorded 143.2 demand misses/token. The incremental
window took 141.5 ms/token and delivered only 1.03 GB/s of physical reads.

The SSD preparation path is the bottleneck. PCIe H2D copied 217.8 MB/token in
8.97 ms/token in the 32-step CUDA trace, an effective 24.3 GB/s. The cache's
`submit_transfers` loop instead copies
each mixed MFE field synchronously from its file mapping into pinned staging.
Missing pages therefore block the caller on storage. The eight-worker deferred
range reader only serves range-backed MXFP4 fields; all runs reported zero
range-read calls for these mixed experts.

The same trace contains 5.888 seconds of gaps between GPU activities. Gaps whose
next activity is a memcpy account for 5.557 seconds, or 94.4%. It also records
48 `cudaEventSynchronize` calls/token for route readback. Qwen builds its route
inside the routed-expert branch, after shared-expert computation, so neither
route readback nor expert preparation overlaps that shared branch. The generic
CUDA FFN already contains the required early-route and projection-bundle
prefetch pattern.

The measured CUDA work provides two useful, workload-specific ceilings:

| Work retained per token | Measured time | No-idle estimate |
| --- | ---: | ---: |
| Kernels, copies and memsets | 52.35 ms | 19.1 tokens/s |
| Kernels only | 43.12 ms | 23.2 tokens/s |

These are no-idle estimates for the current operations, not theoretical GPU
limits. Reaching 19.1 tokens/s with the warm-cache traffic requires roughly
2.8 GB/s of effective expert reads and enough overlap to hide them. At 2 GB/s,
145.2 MB/token alone imposes a 72.6 ms I/O floor, or 13.8 tokens/s before other
unhidden work. A practical first target for parallel, overlapped reads is
therefore 10–14 tokens/s. Approximately 20 tokens/s is the next ceiling without
changing the current GPU work, and exceeding 23 tokens/s requires reducing that
work as well.

Within the traced GPU work, NINT is the largest named MFE family at
13.24 ms/token; NVQ uses 3.05 ms/token and cache scatter 1.54 ms/token. Other
kernels total 25.19 ms/token. The runtime launches about 8,393 kernels and
performs about 6,870 `cudaMallocAsync` plus 6,829 `cudaFreeAsync` calls per
token. CUDA Graph replay currently excludes both Flash-Next models and engines
with an expert cache. Kernel fusion, allocator churn and NINT tuning become the
next limits after the storage stalls are removed; optimizing them first cannot
recover the roughly 90 ms/token currently lost above the measured GPU work.

The shortest path to the next throughput tier is:

1. Batch mixed-field reads through the existing bounded read pool instead of
   faulting file mappings serially in `submit_transfers`.
2. Build the Qwen route immediately after TopK and start route readback and
   projection-bundle preparation before shared-expert computation.
3. Re-profile after I/O overlap, then address NINT and the remaining launch and
   allocation fragmentation shown by the new trace.

#### Follow-up: Qwen gate and router fusion

Two decode-shaped CUDA paths were still paying avoidable per-token work. The
36 Qwen linear-attention blocks built the GDN decay and beta tensors through a
chain of scalar, unary, binary and transpose kernels. They now use the existing
fused gate/beta kernel with the Qwen stable-softplus evaluation preserved. The
48 MoE blocks also recomputed every expert's softmax weight during each of the
ten selection ranks. A one-warp `512 -> Top10` path computes those weights once
and keeps the existing score ordering and selected-weight normalization.

The same 32-step Nsight capture used above gives the following comparison. The
route, cache statistics, H2D bytes and generated token sequence are unchanged.

| Decode capture metric | Before | Fused operators | Change |
| --- | ---: | ---: | ---: |
| CUDA kernels | 268,565 | 257,045 | -11,520 (-4.29%) |
| `cudaMallocAsync` calls | 219,850 | 209,482 | -10,368 (-4.72%) |
| `cudaFreeAsync` calls | 218,532 | 208,164 | -10,368 (-4.74%) |
| Qwen `512 -> Top10` kernel time | 47.691 ms | 6.970 ms | -85.4% |
| Sum of kernel time | 1.3797 s | 1.3048 s | -5.43% |
| Kernels, copies and memsets | 1.6751 s | 1.6001 s | -4.48% |

The gate fusion replaces 12,672 generic launches with 1,152 fused launches,
one per linear-attention block and decode step. The specialized TopK keeps one
launch per MoE block but lowers its mean time from 31.05 to 4.54 microseconds.
Measured GPU work falls by 2.35 ms/token. This moves the no-idle estimate from
19.1 to 20.0 tokens/s for all GPU activity and from 23.2 to 24.5 tokens/s for
kernels alone.

Two unprofiled runs averaged 4.937 tokens/s versus the preceding 4.932-token/s
baseline, a 0.09% difference inside the cold SSD variation. The optimization is
visible in GPU work rather than end-to-end wall time while storage stalls remain
dominant. The GDN prefill path is bit-identical across 51 trace stages and
17,310,720 FP32 values. Both decode runs generated the same 33-token sequence;
the CUDA TopK smoke check also preserves expert IDs and bounds weight error at
2e-6. The activation and Flash-Next native CUDA tests pass.

#### Follow-up: parallel mixed-expert SSD staging (2026-10-05)

Host timers now distinguish route-event waits, staging-slot acquisition, and
source materialization into pinned staging. Generation prints cache counters
after prefill and after decode; subtract them to measure decode alone. These
are caller wall times, including page faults and worker completion, and must
not be added to GPU activity times because the work can overlap.

With the preceding fused-operator binary, a fresh 32-step run spent 4.827 s of
its 6.560 s decode window in staging (73.6%). Route waits took another 0.667 s;
slot acquisition took only 1.77 ms across all 32 steps. The bottleneck is
materializing SSD-backed fields, rather than waiting for a free staging slot.

Strata's `FileExpertSource::fill_many` in
`references/Strata/src/core/expert_source.cpp` uses parallel mapped reads to keep
multiple page-fault reads outstanding. The Metal expert cache also uses bounded
workers and staging ownership. CUDA now sends immutable packed fields from
mixed-expert mappings through its existing eight-worker range-read pool, sharing
the existing four pinned staging slots. Mutable expert-to-slot maps stay on the
caller; already materialized deferred MXFP4 buffers use the existing copy path.
The batch completes before H2D submission, preserving the existing slot/event
ordering. `MFQ_MOE_SSD_IO_WORKERS=1` selects serial mixed-field staging for A/B
checks; the default remains eight workers.

The model, 23-token prompt, MTP-disabled config, greedy sampling and 30 GiB GPU
expert-cache budget match the preceding measurements. Each run starts a fresh
process and creates fresh SSD backing files on ext4/NVMe. The short run times
32 decode steps; the long run times 99 steps with context capacity 256.

| Measurement | Serial staging | Parallel staging | Change |
| --- | ---: | ---: | ---: |
| Cold 32-step decode | 4.878 tokens/s | 6.837 / 6.811 tokens/s | +39.9% vs two-run mean |
| Complete 99-step decode | 6.197 tokens/s | 8.518 tokens/s | +37.5% |
| Cold prefill | 8.924 s | 3.273 / 3.261 s | -63.4% vs two-run mean |
| Cold staging time/token | 150.85 ms | 87.72 / 88.50 ms | -41.6% |
| 99-step staging time/token | 107.06 ms | 64.23 ms | -40.0% |
| 99-step route wait/token | 20.01 ms | 17.90 ms | -10.5% |
| 99-step stage acquisition, total | 4.55 ms | 4.22 ms | Negligible |

The long measurement has one run per variant, so it is an observed improvement,
not a distribution or steady-state limit. The two short candidate runs are
within 0.4%. Model loading is unchanged: 132.7–135.3 s for the serial runs and
135.1–136.0 s for the parallel runs. Decode still spends 54.7% of its long-run
wall time staging experts. Early Qwen route readback and overlapping shared
computation remain the next scheduling work; the preceding approximately
20-token/s GPU no-idle estimate is still a ceiling, not achieved throughput.

There is no additional resident RAM expert cache. The mappings cover 52.59 GiB
on SSD; `host_bytes=32,774,144` accounts for 31.26 MiB of retained expert
metadata, not the process's full RAM use. Existing pinned staging, model-load
temporaries and runtime allocations are also present. Sampled process memory
is as follows; anonymous and RSS peaks can occur at different times.

| Process memory | Serial 32 steps | Parallel 32 steps | Serial 99 steps | Parallel 99 steps |
| --- | ---: | ---: | ---: | ---: |
| Peak RSS | 21.06 GiB | 21.55 / 21.54 GiB | 21.63 GiB | 22.83 GiB |
| Peak anonymous RSS | 2.19 GiB | 2.05 / 2.03 GiB | 2.00 GiB | 1.90 GiB |
| RSS minus anonymous at sampled peak | 19.74 GiB | 19.79 / 20.15 GiB | 20.48 GiB | 21.60 GiB |

The last row is an estimate that also includes shared/pinned mappings, not
just file pages. The expert-file mappings remain reclaimable and are not
registered or locked with CUDA. The parallel long run used about 1.2 GiB more
peak RSS without growing anonymous RAM; the following measurement reports
file/shared RSS and pinned-stage allocation separately.
Strata's `pin_cache_complement` requires a fully filled GPU cache before building
its RAM complement. This CUDA cache is populated on demand and evicts experts,
so that resident-cache policy cannot be copied directly into this path.

Physical reads, estimated from 0.5-second `/proc/<pid>/io` samples, were
8.49 GB for serial cold decode and 8.98 / 9.20 GB for parallel decode. Effective
read throughput rose from 1.29 to 1.92 / 1.96 GB/s, with modest additional
readahead traffic. Long-run reads were 18.10 versus 18.87 GB, or 1.13 versus
1.62 GB/s. Logical expert H2D bytes are identical: 6,949,218,824 for 32 steps
and 14,818,792,272 for 99 steps. Routes, demand hits/misses and evictions also
match, and the complete generated token sequences match in both windows.
`range_read_bytes/calls/ms` now include mapped worker copies as well as explicit
MXFP4 ranges; they describe logical source materialization, not physical SSD
traffic. Mapped copies open no additional files.

Validation: the full native build and all 55 CTest tests pass. The new worker
check covers disjoint unaligned mapped slices, an empty destination, accounting
and rejection of ambiguous mapped/range sources. A real 512-expert projection
with a 192 MiB GPU cache passes 464 full-output bit-equality cases, including
4,729 evictions. Full-model prefill is bit-identical to the preceding binary
across 51 trace stages and 17,310,720 FP32 values.

The remote-master check also found `c738ac9b`, merged as `41f779e5` (Metal
hybrid decode and long-context QSA). Its changes concern native dispatch,
small-batch fusion and attention, and do not change the SSD expert reader.
The staging optimization above follows the measured host bottleneck; the newer
Metal operator changes remain references for subsequent GPU profiling.

#### Follow-up: early Qwen route readback and bounded projection bundles

The routed loader now retains its packed weight handles alongside the callable.
Qwen can therefore use the existing cache-prefetch operations: it builds the
route after TopK, begins its small D2H copy before shared computation, and
prepares Gate/Up/Down together after enqueuing the shared branch. Packed split
Gate/Up and fused Gate/Up both use the existing bundle helper; ordinary dense
projections retain their original forward path. Cached forward still checks
admission and waits for transfer completion. Gate, Up and Down retain their
projection roles so the existing exact-range MXFP4 overlap path can identify
them.

Bundles are bounded to at most eight rows. An initial all-size implementation
improved the fixed 99-step decode from 8.518 to 10.080 tokens/s, but increased
23-token prefill from 3.242 to 3.816 s and device use after prefill from
39,250.8 to 40,820.8 MiB. Bounding bundles retains the decode gain and lets larger
prefill use the existing separate staging sequence. The current final run gives:

| Fixed short-prompt measurement | Parallel staging only | Early readback + bounded bundles |
| --- | ---: | ---: |
| First 32 decode steps, all before EOS | 6.837 / 6.811 tokens/s | 9.120 tokens/s (+33.6% vs mean) |
| Complete fixed 99-step trace, including post-EOS steps | 8.518 tokens/s | 10.302 tokens/s (+20.9%) |
| Prefill | 3.242 s | 3.188 s |
| 32-step staging time/token | 87.72 / 88.50 ms | 47.79 ms |
| 99-step staging time/token | 64.23 ms | 38.76 ms |
| 99-step route wait/token | 17.90 ms | 21.33 ms |
| 99-step H2D submissions | 8,580 | 2,979 |

The gain is primarily from larger source batches and fewer transfer submissions.
Route-event wait increases in this trace, so the measurement does not establish
that early route readback alone improves wall time. The cold 32-step H2D payload
is identical at 6,949,218,824 bytes. The fixed 99-step payload falls by one cache
miss (1,380,488 bytes) to 14,817,411,784 bytes, with 1,235 versus 1,236 evictions.
Demand misses become zero because the same reads are now counted as prefetch
misses: 8,058 for 32 steps and 17,650 for 99 steps. This does not mean zero SSD
reads. Both generated sequences match their preceding baselines.

Cache statistics now report retained `pinned_stage_bytes` and
`device_stage_bytes`. Both are 805,306,368 bytes (768 MiB) in the bounded run.
Its peak process RSS is 22.70 GiB and peak anonymous RSS is 2.01 GiB. At the
sampled RSS peak, `/proc/<pid>/status` reports 20.78 GiB file RSS, 0.76 GiB shared
RSS and 1.16 GiB anonymous RSS; these peaks need not coincide. Pinned staging
remains separate from the 31.26 MiB retained expert metadata. Device use after
prefill is 39,346.8 MiB, leaving 9,163.25 MiB free with the same 30 GiB expert
cache budget.

Full-model prefill still matches across 51 trace stages and 17,310,720 FP32
values. The native cache-binding regression now exercises early readback and
both bundle layouts with host-backed and exact-range sources, retained cache
lifetime and empty-bundle rejection. The full native build and all 55 CTest
tests pass.

A second workload asks for eight numbered sections on autoregressive inference,
with at least 100 words each, covering tokenization, attention, KV cache, expert
routing, GPU kernels, CPU memory, SSD offload and bottlenecks. It uses the bundled
tokenizer and chat format with thinking disabled, 80 input tokens, context
capacity 512 and the same 30 GiB expert cache. Both variants generate the same
100 tokens; none is EOS, so all 99 timed steps precede answer termination.

| Natural long-answer measurement | Parallel staging only | Bounded bundles |
| --- | ---: | ---: |
| Decode | 10.625 tokens/s | 11.650 tokens/s (+9.65%) |
| Prefill | 5.937 s | 6.313 s |
| Prefill + timed decode | 15.255 s | 14.811 s (-2.91%) |
| Decode staging time/token | 40.15 ms | 23.72 ms |
| Decode route wait/token | 16.12 ms | 18.60 ms |
| Decode expert H2D bytes | 8,360,180,688 | 8,350,022,544 |
| Peak RSS | 23.04 GiB | 23.04 GiB |
| Peak anonymous RSS | 1.88 GiB | 1.92 GiB |

These are single-run A/B measurements. The candidate's longer prefill corresponds
to 4.697 versus 4.342 s of source staging with the same prefill H2D payload and
separate staging sequence; no prefill speedup is claimed. Its retained pinned
and device staging are each 1 GiB. The longer prompt warms more experts, so
its absolute decode speed cannot be compared directly with the 23-token prompt.

## Tensor and expert parallel execution

The native runtime accepts either a rank count or an ordered CUDA device list.
Split weights stay attached to that device order:

```shell
mfq-runtime --model model.mfq --tensor-parallel 4
mfq-runtime --model moe.mfq \
  --expert-parallel 0,1,2,3 --expert-split 1,1,1,1
mfq-runtime --model moe.mfq \
  --tensor-parallel 0,1,2,3 --tensor-split 1,1,1,1 \
  --expert-parallel 0,1,2,3 --expert-split 1,1,2,4
```

Combined tensor and expert parallel execution currently uses one NCCL rank
group, so both ordered device lists must match. Their split weights may differ.
For compatibility with older launches, routed experts inherit the tensor
parallel device group when `--expert-parallel` is omitted.

Four- and eight-rank changes must pass `mfq-tensor-parallel-test` before a CUDA
run. A real multi-GPU gate must then cover dense tensor shards, routed expert
ownership, NCCL reduction, eager execution, and CUDA Graph replay. Eight-rank
code is not considered device-validated until that physical gate runs.

## Required A/B matrix

Run every row with fixed model files, prompts, seeds, cache limits, and sampling
parameters. Start each executable in a fresh process.

| Area | Required coverage |
| --- | --- |
| Architectures | dense causal LM, GQA, multimodal MiniCPM-o, and routed MoE |
| Formats | dense BF16/F16, NINT, NVQ/NPQ/NEPQ, MXFP8, and MXFP4 where the architecture permits them |
| Shapes | single-token decode, short and long prefill, odd sizes, batched inputs, and GQA head broadcasting |
| State | empty cache, reused prefix cache, context rollover, session reset, and interrupted generation |
| Sampling | greedy, temperature, top-k, top-p, min-p, repetition/presence penalties, and fixed random seed |
| Execution | eager, CUDA Graph capture/replay, one and multiple CUDA streams, one GPU, and tensor/expert parallel when NCCL is present |
| Media | text, image, video, audio, half duplex, and full duplex |

Capture raw logits from both executables at prefill and every decode step. The
packed kernel bodies are unchanged and keep their existing numerical contract.
Generic reductions and cuBLAS may use a different reduction tree from ATen.
Accept results only when logits stay within the model's established tolerance
and greedy tokens match. Explain any token divergence before merging.

Also compare peak device memory, host memory, model-load time, prefill speed,
and decode speed. The native runtime is rejected if it introduces a material
regression or silently falls back to host computation.

## Packaging gate

Create clean Windows and Linux packages from the native build. Install each in
an environment that has no `torch` package and no LibTorch files, then verify:

1. empty-server startup and model discovery;
2. loading and unloading representative dense and MoE models;
3. OpenAI-compatible streaming and non-streaming generation;
4. multimodal processing through MFQd;
5. process shutdown without leaked workers; and
6. package size and dependency inventory.

The legacy MiniCPM-o diagnostic `.pt` filenames use MFQ's `MFQTNSR1` tensor
envelope in the native executable. Python pickle files created by `torch.save`
are intentionally handled only by `mfq-runtime-torch`; production HTTP media
requests do not cross this file boundary.

## CUDA backend modularization (2026-10-03)

Kernel declarations now live in domain headers for norm, activation, attention,
linear attention, MoE, quant, cache, and sampling. Callers include the domains
that they use. Each Engine load owns its MoE cache; target and predictor weights
retain that cache through aliasing source handles, including copied projection
operations. Cache registration and lifecycle APIs take cache resources directly.

Qwen full/linear attention, GLM DSA indexer/MLP choices, MiniCPM text/TTS norm
semantics, and common dense/MoE parameter assembly live in shared models.
CUDA loaders retain quantization layout, projection fusion, important-neuron
materialization, CPU placement, and TP/EP choices.

The build boundaries use the existing targets:

| Target | Responsibility |
| --- | --- |
| `mfq-models`, `mfq-engine` | Backend-neutral model definitions/configuration and Engine execution |
| `mfq-cuda-ops` | CUDA numerical operations and execution resources; no Engine/model-family/transport dependency |
| `mfq-cuda-loading` | Weight materialization, cache loading and model-load bindings |
| `mfq-cuda-runtime` | Concrete Engine/model instantiation and native execution plans |

Native and optional Torch executables use the same loading and instantiation
source lists. Only runtime executables link transport. The generated Ninja
order dependencies for `mfq-cuda-ops` contain only shared model-graph and
model-source targets, with no Engine, model-family, scheduler, or transport.

Incremental build probes used the existing Release/Ninja/CUDA 13.2 build. After
a completed build, touch each source and run `cmake --build build/cpp_runtime`.
Each probe compiled exactly one translation unit before and after the boundary
change; no unrelated CUDA/model compilation occurred.

| Touched source | Before (seconds) | After (seconds) | Recompiled target |
| --- | ---: | ---: | --- |
| `backends/cuda/ops/format.cpp` | 0.96 | 0.99 | `mfq-cuda-ops` |
| `engine/src/generation_policy.cpp` | 0.56 | 0.57 | `mfq-engine` |
| `models/qwen35/config.cpp` | 2.11 | 2.21 | `mfq-models` |

These are single warm-build observations including required relinks, not a
performance benchmark. There is no measured benefit supporting additional
library targets, so none were added. Splitting targets would not remove the
concrete instantiations' actual template-header dependencies.

Validation: full native build and all 52 CTest tests passed on the local RTX
4090. The offset regression compares embedding and KV writers against CPU
assignment with nonzero aligned/unaligned offsets, dispatch thresholds, FP16,
BF16 and FP32, independent K/V/cache misalignment, per-batch positions, ring
wraparound and untouched storage guards. The cache regression covers independent
instances, execution reset, retained projections and final destruction. Shared
assembly tests cover full/linear attention, dense/split/fused MoE, GLM indexers,
MiniCPM norm variants and malformed shapes/partial expert representations.

`pytest -q tests/test_cpp_runtime_*` passed 203 tests with 11 skips. Full Python
execution with `--continue-on-collection-errors` had 1012 passes, 146 skips,
17 failures and 56 collection errors. Its limitations remain separate from
this refactor: Torch/MLX/SciPy are missing,
and the health-metrics and prefill-timing source contracts already fail at the
starting revision `96be1553`. Real model numerical/performance A/B, multi-GPU
communication and the LibTorch build remain subject to the gates above; these
modularization checks do not substitute for them.
