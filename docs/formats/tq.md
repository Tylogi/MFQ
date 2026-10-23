# TQ

TQ uses a fixed procedural Gaussian codebook and independent 1–8 bit precision
for each output row. It applies no rotation. Each 256-weight ring uses a
16-bit Trellis state, with FP16 row scales and either 4-bit subgroup gains or
shared FP16 input-channel scales. The binary payload remains `RTRELL05`.

The CUDA encoder uses two-pass Viterbi search and supports diagonal imatrix
weighting. GEMV and SwiGLU FFN dispatch precision per row on the GPU, without
sorting or grouping rows. Inference reconstructs FP16 weights, accumulates in
FP32, and accepts FP16 or BF16 activations. FFN uses two kernels, including
intermediate activation rounding and memory traffic.

For 1–4 bit fitting, the default search uses paired FP16 path costs, sampled
scale trials and weighted row/subgroup scale refinement. It retains the full
Trellis state space. `quantize(..., search="full")` uses the original FP32
encoder and exhaustive scale trials. Both produce the same TQ format.

```python
import numpy as np
import torch

from mfq.formats.tq import TQTensor
from mfq.kernels.cuda.tq import Projection
from mfq.quantize.tq import quantize

weight = torch.randn(2048, 4096, device="cuda")
bits = (np.arange(2048) % 8 + 1).astype(np.uint8)
importance = torch.ones(4096, device="cuda")
tensor = quantize(weight, bits, imatrix=importance, groupsize=512)
tensor.save("projection.tq")

loaded = TQTensor.load("projection.tq")
projection = Projection(loaded, input_columns=weight.shape[1])
x = torch.randn(4096, device="cuda", dtype=torch.float16)
y = projection.gemv(x)
```

`SwiGLU(gate, up)` fuses Gate, Up and the activation.
`SwiGLUFFN(gate, up, down)` adds Down projection. Preallocated outputs and
workspace support allocation-free CUDA graph capture.

The Metal API is `mfq.kernels.metal.tq.Projection`. It consumes the same payload,
supports ordinary and routed projections, and fuses independent Gate/Up tensors
without precision sorting. Native Metal loading accepts the `TQ` record dtype;
`tq_shapes` in the MFQ header records logical `[output, input]` or
`[experts, output, input]` dimensions. Stored rows are expert-major. TQ execution
does not pass through MFE. The Metal expert cache for Flash-Next and DeepSeek-V4.1 keeps the three
TQ banks in shared compressed byte pools, with per-expert offsets and LRU
eviction. Expert reads use vectored file I/O; repeated routes share one load.
Small streamed batches with sufficient expert overlap also reuse decoded rows
across up to four routed inputs. This uses route IDs already available for I/O;
the fully resident path does not add a host synchronization to construct a plan.
Changing an active cache budget rebuilds only its compressed expert pools;
dense weights and request KV state remain loaded. Invalid budgets leave the
current pools unchanged. Resident TQ experts can switch to the same streaming
cache when the weight budget decreases, without replacing the model or request KV.
Pool allocations never exceed the complete compressed expert payload plus its
runtime metadata and alignment, even when the configured budget is larger.
For subgroup sizes of at least 256, Metal prepares FP32 group-scale metadata at
load time; encoded weights and their FP16 reconstruction remain unchanged.
Each MoE layer stores three TQ tensors: Gate, Up and Down, with logical
`[experts, output, input]` dimensions and independent row bit widths, scales and
codes. Gate/Up fusion reads the two original tensors together without repacking
them or creating precision cohorts. Ordinary projection batches share one Metal dispatch across
independent TQ tensors, including different output widths and scale layouts.
On M5, batched projections decode directly into NAX fragments; routed prefill
uses a TQ row-block plan to reuse each expert's weights across tokens.
Long dense prefill can decode once into a temporary matrix before GEMM,
with a 256 MiB staging limit. FP16 activation paths use FP16 temporary weights;
FP32 and BF16 paths promote the exact reconstructed FP16 values to FP32.
This does not require NAX, and the expanded matrix is not cached as a model
weight. Longer routed batches share decoded tiles in threadgroup memory.
Without NAX, sufficiently populated routed batches use SIMD-group matrix
multiply with the same compressed tiles and row-block plan.
Narrow-output, long-input projections split the input dimension into FP32
partials before their final reduction. These dispatches depend on geometry,
not row precision or model identity.
Short-input routed Down projections fuse the weighted expert reduction; longer
inputs keep the independently measured projection and reduction path.
PLE and Engram row stores gather the selected TQ rows without expanding their
complete tables. Selected rows with different subgroup sizes or column scales
share one decode dispatch; metadata is prepared only for those rows.
Tables larger than one payload can use ordered row shards without changing the
TQ binary format. The MFQ header's `tq_row_shards` object maps each logical table
name to an ordered array of physical TQ record names. These records concatenate
along the row axis and share the logical input width; each keeps its own row
precisions and scale layout. Physical names append `.tq_row_shard.00000`,
`.tq_row_shard.00001`, and so on to the logical name. The logical name must not
also be a physical record. Row stores resolve and cache only the requested rows
across shards; they do not concatenate or expand the full table in memory.

Tests: `pytest tests/test_formats/test_tq.py tests/test_tq_cuda.py tests/test_tq_metal.py`.
