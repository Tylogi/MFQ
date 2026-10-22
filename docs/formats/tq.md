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

These APIs operate on ordinary two-dimensional tensors. Model-container and
routed-expert-cache integration are separate from this format implementation.

Tests: `pytest tests/test_formats/test_tq.py tests/test_tq_cuda.py`.
