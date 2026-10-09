# Portable Metal kernel checks

Set `MFQ_METAL_DISABLE_NAX=1` before starting a native process to select MFQ's
non-NAX MFE, dense NINT and sparse block attention kernels. The override also
disables M5-only MFE row-packing policy and takes precedence over
`MFQ_METAL_MIXED_PREFILL_NAX`. It does not change the reported GPU or MLX's
internal matmul/gather-QMM dispatch, and is not a simulator for another Apple GPU.

The portable grouped MFE prefill path specializes fixed tensor group sizes and
format families. Quantization bits remain per-row data. Input token count and
route count remain runtime parameters rather than compilation constants.
Complete weight tiles skip redundant shared-memory clearing. The portable path
uses 64-row blocks for larger groups, with a separate 32-row kernel for every
expert tail of at most 32 rows. Thus its padded matrix work never exceeds the
32-row plan, including skewed routing. Small projections and groups keep 32.

Portable dense NINT prefill uses 32-column matrix tiles with vectorized input
loads and direct decoded writes to shared memory. Contractions and short inputs
use 64-row tiles; expansions with at least 128 input rows use 128-row tiles to
reuse decoded weights. Projections with at most eight outputs instead use SIMD
reductions, sharing each decoded weight across four input rows without padding
the output width to a matrix tile. Long prefills retain transient dequantization
and MLX matmul, including its split-K and matrix-vector dispatch. All input
lengths remain runtime parameters, and load-time preparation
compiles every portable tile variant for each tensor group size. Decode and
small-M verification kernels are unchanged.

For portable dense NINT matrices with at least one million weights and at
least 32 outputs, contractions use transient dequantization plus MLX matmul
from 64 input rows; expansions switch at 512. Smaller matrices and the NAX
path retain the 2048-row crossover. This lets MLX use its split-K kernels for
long-K contractions and avoids repeatedly decoding weights on medium prefills.
The selection checks allocator, wired and serving-budget headroom for the
decoded matrix plus input/output buffers. Insufficient headroom keeps packed
execution without evicting resident experts or prefix cache for this optional
optimization. Neither path retains a decoded weight copy; both packed and
dequantization kernels are included in load-time compilation.

`mfq-metal-mfe-prefill-benchmark MODEL PREFIX ROWS REPETITIONS [BLOCK_ROWS [uniform|skewed]]` exercises real
Gate, Up, Down and fused Gate/Up records from one MoE layer. The model must
contain split Gate/Up records and an embedded model configuration. Routing is
deterministic, either uniform or concentrated on a hot subset; the model supplies top-k.
Omitting `BLOCK_ROWS` uses the production recommendation; an explicit value
isolates block geometry for matched operator measurements.
The benchmark warms each projection, reports every timed round, and checks
finite outputs with an output hash. Timings include graph construction,
evaluation and synchronization, but exclude loading, compilation and building
the shared route plan. They are projection latency measurements, not physical
DRAM bandwidth or whole-model throughput.

Compare the same model, layer, shapes and routing on baseline and candidate
executables with NAX disabled in both. Do not run another inference workload
or a model load concurrently. Results obtained on M5 remain M5 non-NAX results;
device-specific tile tuning still requires validation on the target GPU.
