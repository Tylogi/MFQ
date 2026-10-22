"""Fixed-book CUDA Trellis fitting with optional diagonal imatrix weighting.

``quantize(weight, bits, imatrix=...)`` accepts a scalar width or one width
per output row. Scale-factor trials select each row by the actual FP16
reconstruction error. Offline grouping by width never changes row order.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from mfq.formats.tq import RELATIVE, TQTensor, fixed_codebook, pack_states, unpack_states
from mfq.quantize.cuda.tq_encoder import encode


@torch.inference_mode()
def quantize(
    weight: torch.Tensor,
    bits: int | np.ndarray,
    *,
    imatrix: torch.Tensor | None = None,
    groupsize: int = 512,
    scale_factors: tuple[float, ...] | None = None,
) -> TQTensor:
    """Fit a CUDA 2D weight tensor; return a standalone host payload.

    By default use the retained ten-factor search (twelve factors at 1 bit).
    Logical widths are padded to whole rings and subgroups; retain the
    original width when constructing the CUDA ``Projection``.
    """
    if not weight.is_cuda or weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("Expected a nonempty CUDA 2D weight tensor")
    if groupsize < 16 or groupsize > 2**20 or groupsize & (groupsize - 1):
        raise ValueError("groupsize must be a power of two in [16, 2**20]")
    rows, logical = weight.shape
    widths = np.broadcast_to(np.asarray(bits), (rows,))
    if not np.isin(widths, np.arange(1, 9)).all():
        raise ValueError("Each row must select 1..8 bits")
    if scale_factors is not None and (
        not scale_factors or any(not math.isfinite(f) or f <= 0 for f in scale_factors)
    ):
        raise ValueError("Scale factors must be finite and positive")
    width = math.ceil(logical / max(256, groupsize)) * max(256, groupsize)
    with torch.cuda.device(weight.device):
        w = torch.nn.functional.pad(weight.float(), (0, width - logical))
        if not torch.isfinite(w).all():
            raise ValueError("Weights must be finite")
        h = torch.ones(logical, device=w.device) if imatrix is None else imatrix.to(w.device).float()
        if h.shape != (logical,) or not torch.isfinite(h).all() or (h < 0).any():
            raise ValueError("imatrix must be a finite nonnegative input-channel vector")
        if not (h > 0).any():
            raise ValueError("imatrix must contain a positive weight")
        h = torch.nn.functional.pad(h, (0, width - logical))
        lut = torch.tensor(fixed_codebook(), device=w.device)
        relative = torch.tensor(RELATIVE, device=w.device).float()
        code_rms = lut.float().square().mean().sqrt()
        row_payloads = [None] * rows
        descriptors = np.zeros(rows, dtype=np.uint64)
        selectors = np.zeros((rows, width // groupsize), dtype=np.uint8)
        for k in np.unique(widths):
            k = int(k)
            host_ids = np.flatnonzero(widths == k)
            ids = torch.tensor(host_ids, device=w.device)
            target = w[ids]
            anchor = (target.square().mean(-1).sqrt() / code_rms).clamp_min(2**-24).half()
            wanted = target.reshape(len(ids), -1, groupsize).square().mean(-1).sqrt()
            wanted = wanted / code_rms / anchor[:, None].float()
            codes = (wanted.clamp_min(2**-24).log2()[..., None] - relative.log2()).abs().argmin(-1)
            factors = scale_factors
            if factors is None:
                factors = ((.65, .7, .75, .8, .85, .9, .95, 1., 1.05, 1.1, 1.2, 1.3)
                           if k == 1 else (1., .9, 1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.))
            best_loss = torch.full((len(ids),), float("inf"), device=w.device)
            best_anchor = torch.empty_like(anchor)
            best_states = torch.empty((len(ids), width), dtype=torch.int16, device=w.device)
            for factor in factors:
                trial = (anchor.float() * factor).half().clamp_min(2**-24)
                expanded = (trial[:, None].float() * relative[codes]).repeat_interleave(groupsize, -1)
                x = (target / expanded).reshape(-1, 256).contiguous()
                metric = expanded * h.sqrt()[None, :]
                metric = (metric / metric.square().mean(-1, keepdim=True).sqrt()).reshape(-1, 256).contiguous()
                values, states, _ = encode(x, metric, lut, k)
                reconstruction = (values.reshape_as(target) * expanded).half().float()
                loss = ((target - reconstruction).square() * h).sum(-1)
                take = loss < best_loss
                best_loss[take] = loss[take]
                best_anchor[take] = trial[take]
                best_states[take] = states.reshape_as(best_states)[take]
            states = best_states.cpu().numpy().view(np.uint16).reshape(-1, 256)
            packed = pack_states(states, k)
            if not np.array_equal(unpack_states(packed, k), states):
                raise RuntimeError("Encoder tail-biting states do not match the packed payload")
            packed = packed.reshape(len(ids), -1)
            anchors = best_anchor.cpu().numpy().view(np.uint16).astype(np.uint64)
            descriptors[host_ids] = np.uint64(k - 1) << np.uint64(32) | anchors << np.uint64(40)
            selectors[host_ids] = codes.byte().cpu().numpy()
            for i, row in enumerate(host_ids):
                row_payloads[row] = packed[i]
        lengths = width * widths.astype(np.uint64) // 32
        descriptors |= np.r_[np.uint64(0), np.cumsum(lengths[:-1], dtype=np.uint64)]
        selectors = selectors.reshape(-1)
        if len(selectors) & 1:
            selectors = np.pad(selectors, (0, 1))
        scales = selectors[::2] | (selectors[1::2] << 4)
        return TQTensor(rows, width, groupsize, False, descriptors, scales,
                             np.concatenate(row_payloads))
