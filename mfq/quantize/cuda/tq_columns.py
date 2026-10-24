"""Shared-column, 16-state-bit TQ fitting with sparse scale repair."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from pathlib import Path

import torch
from mfq.formats.tq import fixed_codebook


@lru_cache(None)
def _library():
    from torch.utils.cpp_extension import load
    path = load(name="mfq_tq_columns_v1",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "--fmad=false"], is_python_module=False)
    lib = ct.CDLL(path)
    lib.half_wave.argtypes = [ct.c_int] * 2
    lib.half_wave.restype = ct.c_int
    lib.half_encode.argtypes = [ct.c_void_p] * 6 + [ct.c_int] * 4 + [ct.c_void_p]
    lib.half_encode.restype = ct.c_int
    return lib


def encode(x, metric, bits):
    if (bits not in range(1, 9) or x.ndim != 2 or x.shape[1] != 256 or not len(x)
            or not x.is_cuda or x.dtype != torch.float32 or metric.dtype != torch.float32
            or x.shape != metric.shape or x.device != metric.device
            or not x.is_contiguous() or not metric.is_contiguous()):
        raise ValueError("Expected matching contiguous FP32 [rings,256] tensors and 1..8 bits")
    if torch.version.hip:
        from .tq_encoder import encode as reference_encode
        table = torch.tensor(fixed_codebook(), device=x.device)
        return reference_encode(x, metric, table, bits)
    with torch.cuda.device(x.device):
        lib = _library()
        wave = lib.half_wave(len(x), bits)
        if wave <= 0:
            raise RuntimeError(f"TQ column fitter occupancy failed: {wave}")
        wave = min(wave, 4096)
        divisor = 8 if bits == 1 else 2 if bits in (2, 3) else 1
        history = torch.empty((wave, 256 * (1 << (16 - bits)) // divisor),
                              device=x.device, dtype=torch.uint8)
        y = torch.empty_like(x)
        states = torch.empty_like(x, dtype=torch.int16)
        bounds = torch.empty(len(x), device=x.device, dtype=torch.int32)
        code = lib.half_encode(*(v.data_ptr() for v in (x, metric, y, states, history, bounds)),
            len(x), bits, wave, 16, torch.cuda.current_stream(x.device).cuda_stream)
        if code:
            raise RuntimeError(f"TQ column fitter returned CUDA error {code}")
        return y, states, bounds


@torch.inference_mode()
def prepare_columns(weight):
    """One precision-independent scale vector shared by every row/candidate."""
    w = weight.float()
    rms = torch.tensor(fixed_codebook(), device=w.device).float().square().mean().sqrt()
    columns = (w.square().mean(0).sqrt() / w.square().mean().sqrt().clamp_min(2**-24)).clamp_min(.01).half()
    for _ in range(2):
        anchor = ((w / columns.float()).square().mean(-1).sqrt() / rms).clamp_min(2**-24).half()
        columns = ((w / anchor.float()[:, None]).square().mean(0).sqrt() / rms).clamp_min(.01).half()
    return columns


@torch.inference_mode()
def fit_columns(weight, k, h, columns, *, fast=False, repair_scales=True):
    """Fit row anchors against a fixed, precision-independent column vector."""
    w = weight.float()
    rows, width = w.shape
    rms = torch.tensor(fixed_codebook(), device=w.device).float().square().mean().sqrt()
    row_rms = (w / columns.float()).square().mean(-1).sqrt() / rms
    anchor = row_rms.clamp_min(2**-24).half()
    factors = ((.65, .7, .75, .8, .85, .9, .95, 1., 1.05, 1.1, 1.2, 1.3)
               if k == 1 else (1., .9, 1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.))

    def trial(target, a):
        scale = a.float()[:, None] * columns.float()[None, :]
        metric = scale * h.sqrt()[None, :]
        metric /= metric.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
        y, state, _ = encode((target / scale).reshape(-1, 256).contiguous(),
                             metric.reshape(-1, 256).contiguous(), k)
        basis = y.reshape_as(target) * columns.float()[None, :]
        proposed = ((target * basis * h).sum(-1) /
                    (basis.square() * h).sum(-1).clamp_min(1e-30)).clamp_min(2**-24).half()
        old = ((target - (basis * a.float()[:, None]).half().float()).square() * h).sum(-1)
        new = ((target - (basis * proposed.float()[:, None]).half().float()).square() * h).sum(-1)
        return state.reshape_as(target), torch.where(new < old, proposed, a), torch.minimum(new, old)

    ids = torch.linspace(0, rows - 1, min(rows, 16 if fast else 64), device=w.device).long()
    factor_table = torch.tensor(factors, device=w.device)
    sample_a = (anchor[ids].float()[None, :] * factor_table[:, None]).half().flatten()
    _, _, loss = trial(w[ids].repeat(len(factors), 1).contiguous(), sample_a)
    order = loss.reshape(len(factors), len(ids)).sum(-1).argsort()
    best = torch.full((rows,), torch.inf, device=w.device)
    states = torch.empty_like(w, dtype=torch.int16)
    anchors = anchor.clone()
    for index in order[:1 if fast else 2]:
        state, a, loss = trial(w, (anchor.float() * factor_table[index]).half().clamp_min(2**-24))
        take = loss < best
        best[take], anchors[take], states[take] = loss[take], a[take], state[take]
    if not fast:
        state, a, loss = trial(w, anchors.clone())
        take = loss < best
        best[take], anchors[take], states[take] = loss[take], a[take], state[take]
    if repair_scales:
        from .tq_scale_repair import propose_scales
        proposals = propose_scales(w, h, columns, anchors, best)
        ids = torch.nonzero(proposals > anchors).flatten()
        if len(ids):
            state, a, loss = trial(w[ids].contiguous(), proposals[ids])
            take = loss < best[ids]
            selected = ids[take]
            anchors[selected], states[selected] = a[take], state[take]
    anchors.masked_fill_(row_rms == 0, 0)
    return states, anchors
