"""Packed TQ projections and fixed-weight input gradients on ROCm."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from pathlib import Path

import torch

from mfq.kernels.cuda.tq import Projection as _Projection, ProjectionView, _dtype, _overlap


@lru_cache(None)
def library():
    from torch.utils.cpp_extension import load

    if not torch.version.hip:
        raise RuntimeError("TQ HIP kernels require a ROCm build of PyTorch")
    path = load(name="mfq_tq_ops_hip", sources=[str(Path(__file__).with_name("tq_ops.hip"))],
                extra_cuda_cflags=["-O3", "-ffp-contract=off"], is_python_module=False)
    lib = ct.CDLL(path)
    P, I = ct.c_void_p, ct.c_int
    for name in ("mfq_tq_forward", "mfq_tq_backward"):
        getattr(lib, name).argtypes = [ProjectionView, P, P, I, I, P]
        getattr(lib, name).restype = I
    lib.mfq_tq_gemv.argtypes = [ProjectionView, P, P, I, I, I, P]
    lib.mfq_tq_gemv.restype = I
    return lib


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, projection):
        ctx.projection = projection
        return projection._run(x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        return ctx.projection.backward(gradient.contiguous()), None


class Projection(_Projection):
    """A fixed compressed matrix; backward returns gradients for activations."""

    _library = staticmethod(library)

    def _run(self, x, out=None, *, backward=False):
        width, result_width = ((self.rows, self.input_columns) if backward else
                               (self.input_columns, self.rows))
        if (x.device != self.device or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or x.ndim < 1 or x.size(-1) != width or not x.is_contiguous()):
            raise ValueError(f"Expected contiguous FP16/BF16/FP32 activations with width {width} on {self.device}")
        shape = (*x.shape[:-1], result_width)
        if out is None:
            out = torch.empty(shape, dtype=x.dtype, device=self.device)
        if (out.shape != shape or out.dtype != x.dtype or out.device != self.device
                or not out.is_contiguous() or _overlap(x, out)):
            raise ValueError("Output must match the result shape, dtype and device without input overlap")
        tokens = x.numel() // width
        function = self.lib.mfq_tq_backward if backward else self.lib.mfq_tq_forward
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device).cuda_stream
            for start in range(0, tokens, 65535 * 16):
                code = function(self.view, x.data_ptr() + start * width * x.element_size(),
                                out.data_ptr() + start * result_width * out.element_size(),
                                min(65535 * 16, tokens - start), _dtype(x.dtype), stream)
                if code:
                    raise RuntimeError(f"TQ HIP operator returned {code}")
        return out

    def __call__(self, x):
        return _Linear.apply(x, self)

    def gemv(self, x, out=None):
        if x.ndim != 1:
            raise ValueError("GEMV requires a one-dimensional activation vector")
        return self.forward(x, out)

    def forward(self, x, out=None):
        """Run decode or prefill; explicit output buffers do not track gradients."""
        if out is None:
            return self(x)
        if torch.is_grad_enabled() and x.requires_grad:
            raise ValueError("Use projection(x) when activation gradients are required")
        return self._run(x, out)

    def backward(self, gradient, out=None):
        """Compute dX = dY @ W directly from the compressed matrix."""
        return self._run(gradient, out, backward=True)


class SwiGLU:
    def __init__(self, gate, up):
        if (gate.rows != up.rows or gate.input_columns != up.input_columns or gate.device != up.device):
            raise ValueError("Gate and Up must share their geometry and device")
        self.gate, self.up = gate, up

    def __call__(self, x):
        return torch.nn.functional.silu(self.gate(x)) * self.up(x)


class SwiGLUFFN(SwiGLU):
    def __init__(self, gate, up, down):
        super().__init__(gate, up)
        if down.input_columns != gate.rows or down.device != gate.device:
            raise ValueError("Down must match the Gate/Up output width and device")
        self.down = down

    def __call__(self, x):
        return self.down(super().__call__(x))
