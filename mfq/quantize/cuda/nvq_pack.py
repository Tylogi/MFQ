"""CUDA serializers for validated NVQ streams; no expanded per-bit CPU arrays."""

from __future__ import annotations

import numpy as np
import torch

from ._ext import ext


def pack_bits(values: np.ndarray, bits: int, device: str) -> bytes:
    value = torch.as_tensor(np.ascontiguousarray(values, dtype=np.int32), device=device)
    return ext().nvq_pack_bits(value, bits).cpu().numpy().tobytes()


def pack_group64(state, indices, signs, neuron_len: int, device: str) -> bytes:
    tensors = [
        torch.as_tensor(np.ascontiguousarray(x, dtype=np.int32), device=device)
        for x in (state, indices, signs)
    ]
    return ext().nvq_pack_group64(*tensors, neuron_len).cpu().numpy().tobytes()
