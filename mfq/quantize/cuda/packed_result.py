"""Serialize CUDA quantizer fields directly into canonical MFQ byte streams.

Only compressed bytes cross to the CPU. Returning separate stream parts
lets callers write and hash them without another complete payload copy.
"""
import math
import struct

import torch

from ._ext import ext


def bits(value, width):
    return ext().nvq_pack_bits(value.to(torch.int32).contiguous(), width).cpu().numpy().tobytes()


def fp16(value):
    return value.to(torch.float16).cpu().numpy().tobytes()


def nvq1(spec, width, anchor, scale, indices, delta, codebook, *, small):
    from mfq.formats import nvq1_l as large, nvq1_s as short
    rows = anchor.numel()
    nvec = math.ceil(width / 8)
    if small:
        header = short._HEADER.pack(short._MAGIC, short._VERSION, spec.sub_bits, 24, 0, width, 2)
        tables = short.pack_nvq1_s_banked_codebook(codebook)
    else:
        profile = large._PROFILE_IQ1S_GRID if codebook is None else large._PROFILE_CUSTOM_TERNARY
        header = large._HEADER.pack(large._MAGIC, profile, spec.sub_bits, 24, 0, width, 2)
        tables = b"" if codebook is None else large.pack_ternary_codebook(codebook)
    return (header + struct.pack("<2qI", rows, width, rows) + tables, fp16(anchor),
            bits(scale, spec.sub_bits), bits(indices.reshape(rows, -1)[:, :nvec], spec.index_bits),
            bits(delta, 1))


def jsc(spec, width, anchor, state, indices, signs, alpha, bank, codebooks):
    from mfq.formats import nvq
    rows, ng = anchor.numel(), math.ceil(width / 24)
    nvec, nsign = math.ceil(width / spec.vector_size), math.ceil(width / 8)
    layout = nvq.resolve_jsc_storage_layout(spec)
    header = (nvq._HEADER.pack(nvq._MAGIC, nvq._CODEBOOK_ID[spec.codebook] | nvq._JSC_FLAG,
                              4, 24, 0, width, 2) + struct.pack("<2qI", rows, width, rows)
              + nvq.pack_jsc_tables(alpha, bank, codebooks, storage_layout=layout))
    state = state.reshape(rows, ng)
    indices = indices.reshape(rows, -1)[:, :nvec]
    signs = signs[:, :nsign]
    if layout == nvq._JSC_GROUP64_LAYOUT_NAME:
        fields = [x.to(torch.int32).contiguous() for x in (state, indices, signs)]
        streams = (ext().nvq_pack_group64(*fields, width).cpu().numpy().tobytes(),)
    else:
        streams = (bits(state, 4), bits(indices, spec.index_bits), bits(signs, 7))
    return (header, fp16(anchor), *streams)


def nint(spec, width, anchor, minimum, scale, sub_min, q):
    from mfq.formats.io import _NINT_HDR
    from mfq.formats.nint import NINT_ADAPTIVE_FLAG, NINT_K_SELECTOR_BITS, NINT_Q_SELECTOR_BITS
    rows, ng = anchor.numel(), math.ceil(width / spec.groupsize)
    header = (_NINT_HDR.pack(spec.bits | NINT_ADAPTIVE_FLAG, spec.sub_bits, spec.groupsize, 0, width)
              + struct.pack("<I2qII", 2, rows, width, rows, ng))
    k = torch.ones(rows, dtype=torch.int32, device=anchor.device)
    qb = torch.full_like(k, spec.bits - 1)
    return (header, fp16(anchor), fp16(minimum), bits(k, NINT_K_SELECTOR_BITS),
            bits(scale, spec.sub_bits), bits(sub_min, spec.sub_bits), bits(qb, NINT_Q_SELECTOR_BITS),
            bits(q, spec.bits))
