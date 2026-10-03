# Flash-Next EWQ on native Metal

## NINT MHC projections

Flash-Next's MHC down, up and residual-injection matrices accept dense weights
or canonical NINT weights, including mixed per-row q/k layouts. Each NINT MHC
matrix is decoded once to resident FP16 through the shared native NINT decoder,
then consumed by the existing gated-residual operators. Dense weights retain
their original representation and arithmetic; this is not a packed MHC matmul
kernel or a new quantization policy.

Materialization is scoped to these small MHC projection matrices. Norms and
integer/hash metadata keep their strict dense loader. PLE tables keep their
mmap-backed selected-row decoder, and routed experts retain their MFE path.
This does not enable global weight predequantization.

Build `mfq-metal-nint-rows-test` and run
`python -m pytest tests/test_native_nint_ple.py` for matched dense/NINT MHC
graphs. Coverage includes uniform and adaptive q/k, mixed dense/packed MHC,
the final mixer, NINT PLE, prefill, incremental decode, EOS boundaries and
two-batch execution. Negative cases reject quantized norm and integer metadata.
Fixture equality does not establish full-model quality or throughput.

## EWQ residency and first-token latency

The server excludes Flash-Next's `position_embedding.ngram.shard.*.weight`
payloads from the always-resident weight estimate: both FP8 and NINT PLE tables
are mmap-backed and only selected rows are decoded. PLE scale/hash metadata and
projection matrices still count as resident. This is not free memory: touched
file pages, selected rows and runtime buffers still consume memory, and the
host-memory admission limit still applies.

When the resident estimate fits the current budget, experts remain fully
resident. When it does not, the automatic expert cache uses the remaining
budget after dense weights and runtime headroom, rather than a fixed small
cache. Explicit cache settings are preserved. Model-load logs report the
selected policy, budget, checkpoint size, row-streamed bytes and expert bytes.

Before allocating weights, the native Metal executable sets MLX's wired-memory
ceiling to the device's recommended working-set size. This lets allocated GPU
buffers remain resident instead of being compressed during model loading and
decompressed on the first request. The ceiling does not allocate that amount of
memory, replace the admission/cache budget, or change system-wide settings.
If the installed MLX/macOS cannot enable wiring, startup warns and continues.

Native, unrotated two-projection MFE weights finalize
their packed Gate/Up concatenation during Flash-Next loading. The finalized
weight rebases its projection views onto those same buffers and releases its
separate source pools, avoiding a first-request copy and duplicate retained
weights without changing split dispatch. Standalone-kernel and unsupported split layouts
keep their existing path; weights are not dequantized to dense storage.

If expert paging is selected, Metal supports NVQ1-S and NVQ1-L (implicit and
embedded codebooks) alongside NINT, NVQ-JSC and MX. Packed state/index/delta
streams are sliced at expert row boundaries, including non-byte-aligned
boundaries. An unsupported cohort now fails the Flash-Next load explicitly
instead of silently loading an entire layer outside the requested cache.
The NVQ1-L direct kernel also masks incomplete final activation vectors.

Flash-Next's reported prefill time includes host paging, route synchronization
and graph construction as well as GPU evaluation. It no longer reports just a
subset of evaluated operations. MFE hit/miss/I/O counters remain unimplemented;
their zero values must not be interpreted as absence of paging.

Build and run `mfq-metal-moe-test` for mixed-format paging, NVQ1 bit boundaries,
custom codebooks, malformed payload rejection, eviction/lifetime, scalar
numerical references and packed-projection memory ownership. Run
`python -m pytest tests/test_server_models.py` for catalog/budget regressions.
