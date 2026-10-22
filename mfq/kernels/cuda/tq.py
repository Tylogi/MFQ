"""Fixed-book, row-dynamic Trellis operators for ordinary 2D projections.

FP16 reconstructed weights, FP32 accumulation, FP16 or BF16 activations.
Supplying outputs/workspace makes calls allocation-free and CUDA-graph safe.
"""
import ctypes as ct
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from mfq.formats.tq import RELATIVE, TQTensor


class ProjectionView(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in
                ('words', 'descriptors', 'scales', 'relative', 'bias')] + [
                    (name, ct.c_int) for name in
                    ('rows', 'stored_columns', 'input_columns', 'gslog', 'columns')]


@lru_cache(None)
def library():
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    path = load(
        name="mfq_tq_ops",
        sources=[str(directory / "tq_ops.cu")],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        is_python_module=False,
    )
    lib = ct.CDLL(path)
    P, I, V = ct.c_void_p, ct.c_int, ProjectionView
    lib.mfq_tq_gemv.argtypes = [V, P, P, I, I, I, P]
    lib.mfq_tq_swiglu.argtypes = [V, V, P, P, I, I, I, P]
    lib.mfq_tq_ffn.argtypes = [V, V, V, P, P, P, I, P]
    for name in ('mfq_tq_gemv', 'mfq_tq_swiglu', 'mfq_tq_ffn'):
        getattr(lib, name).restype = I
    return lib


def _status(code):
    if code:
        raise RuntimeError(f'Trellis operator returned {code}')


def _vector(x, length, device, dtype=None):
    if (x.device != device or x.dtype not in (torch.float16, torch.bfloat16) or
            x.ndim != 1 or x.numel() != length or not x.is_contiguous() or
            (dtype is not None and x.dtype != dtype)):
        raise ValueError(f'Expected contiguous {dtype or "FP16/BF16"} vector of length {length} on {device}')


def _overlap(x, y):
    return max(x.data_ptr(), y.data_ptr()) < min(
        x.data_ptr() + x.numel() * x.element_size(),
        y.data_ptr() + y.numel() * y.element_size())


class Projection:
    def __init__(self, data, input_columns=None, bias=None, device='cuda'):
        if isinstance(data, (str, Path)):
            data = TQTensor.load(data)
        if not isinstance(data, TQTensor):
            raise TypeError('Expected a TQTensor or RTRELL05 path')
        data = data.components()
        self.rows, self.stored_columns = int(data['rows']), int(data['width'])
        self.input_columns = self.stored_columns if input_columns is None else int(input_columns)
        gs, columns = int(data['gs']), bool(data['columns'])
        if (self.rows <= 0 or self.stored_columns % 256 or
                not 0 < self.input_columns <= self.stored_columns or
                (not columns and (gs < 16 or gs > 2**20 or gs & (gs - 1)))):
            raise ValueError('Invalid projection geometry or subgroup size')
        target = torch.device(device)
        if target.type != 'cuda':
            raise ValueError('Trellis CUDA operators require a CUDA device')
        with torch.cuda.device(target):
            self.words = torch.as_tensor(np.array(data['packed'], copy=True), device=target)
            self.descriptors = torch.as_tensor(np.array(data['descriptors'], copy=True), device=target)
            self.scales = torch.as_tensor(np.array(data['scales'], copy=True), device=target)
            self.relative = torch.tensor(RELATIVE.copy(), device=target)
            self.device = self.words.device
            self.bias = None if bias is None else torch.as_tensor(
                bias, dtype=torch.float32, device=self.device).contiguous()
        if self.bias is not None and self.bias.shape != (self.rows,):
            raise ValueError('Bias length must equal projection output rows')
        if self.descriptors.numel() != self.rows:
            raise ValueError('Row descriptor count differs')
        expected_scales = self.stored_columns if columns else (
            self.rows * ((self.stored_columns + gs - 1) // gs) + 1) // 2
        if self.scales.numel() != expected_scales:
            raise ValueError('Scale payload length differs')
        self.view = ProjectionView(self.words.data_ptr(), self.descriptors.data_ptr(),
            self.scales.data_ptr(), self.relative.data_ptr(),
            self.bias.data_ptr() if self.bias is not None else None,
            self.rows, self.stored_columns, self.input_columns,
            0 if columns else gs.bit_length() - 1, int(columns))
        with torch.cuda.device(self.device):
            self.lib = library()

    def gemv(self, x, out=None, *, warps=0, rows_per_block=0):
        _vector(x, self.input_columns, self.device)
        if out is None:
            out = torch.empty(self.rows, dtype=x.dtype, device=self.device)
        _vector(out, self.rows, self.device, x.dtype)
        if _overlap(x, out):
            raise ValueError('GEMV input and output must not overlap')
        with torch.cuda.device(self.device):
            _status(self.lib.mfq_tq_gemv(self.view, x.data_ptr(), out.data_ptr(),
                int(x.dtype == torch.bfloat16), warps, rows_per_block,
                torch.cuda.current_stream(self.device).cuda_stream))
        return out


class SwiGLU:
    def __init__(self, gate, up):
        if (gate.rows != up.rows or gate.input_columns != up.input_columns or
                gate.device != up.device):
            raise ValueError('Gate/Up shapes, devices and operator libraries must match')
        self.gate, self.up = gate, up

    def __call__(self, x, out=None, *, warps=0, rows_per_block=0):
        g, u = self.gate, self.up
        _vector(x, g.input_columns, g.device)
        if out is None:
            out = torch.empty(g.rows, dtype=x.dtype, device=g.device)
        _vector(out, g.rows, g.device, x.dtype)
        if _overlap(x, out):
            raise ValueError('SwiGLU input and output must not overlap')
        with torch.cuda.device(g.device):
            _status(g.lib.mfq_tq_swiglu(g.view, u.view, x.data_ptr(), out.data_ptr(),
                int(x.dtype == torch.bfloat16), warps, rows_per_block,
                torch.cuda.current_stream(g.device).cuda_stream))
        return out


class SwiGLUFFN(SwiGLU):
    """Two CUDA kernels: fused Gate/Up/SwiGLU, then Down GEMV."""
    def __init__(self, gate, up, down):
        super().__init__(gate, up)
        if (down.input_columns != gate.rows or down.device != gate.device):
            raise ValueError('Down input width, device and operator library must match Gate/Up')
        self.down = down

    def __call__(self, x, out=None, *, workspace=None):
        g, u, d = self.gate, self.up, self.down
        _vector(x, g.input_columns, g.device)
        if out is None:
            out = torch.empty(d.rows, dtype=x.dtype, device=g.device)
        if workspace is None:
            workspace = torch.empty(g.rows, dtype=x.dtype, device=g.device)
        _vector(out, d.rows, g.device, x.dtype)
        _vector(workspace, g.rows, g.device, x.dtype)
        if _overlap(workspace, x) or _overlap(workspace, out):
            raise ValueError('FFN workspace must not overlap input/output')
        with torch.cuda.device(g.device):
            _status(g.lib.mfq_tq_ffn(g.view, u.view, d.view, x.data_ptr(),
                workspace.data_ptr(), out.data_ptr(), int(x.dtype == torch.bfloat16),
                torch.cuda.current_stream(g.device).cuda_stream))
        return out
