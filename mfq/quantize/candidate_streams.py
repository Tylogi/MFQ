"""Byte-preserving selection/assembly of saved canonical candidate streams."""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from mfq.formats import nvq, nvq1_l, nvq1_s
from mfq.formats.io import _NINT_HDR
from mfq.formats.nint import NINT_ADAPTIVE_FLAG


@dataclass(frozen=True)
class Layout:
    prefix: bytes
    rows: int
    columns: int
    shape_offset: int
    nint: bool
    groups: int
    field_bits: tuple[int, ...]

    def header(self, rows):
        prefix = bytearray(self.prefix)
        if self.nint:
            struct.pack_into("<2qII", prefix, self.shape_offset, rows, self.columns, rows, self.groups)
        else:
            struct.pack_into("<2qI", prefix, self.shape_offset, rows, self.columns, rows)
        return bytes(prefix)


def layout(blob, dtype):
    view = memoryview(blob)
    if dtype == "NINT":
        qb, kb, gs, axis, columns = _NINT_HDR.unpack_from(view)
        if not qb & NINT_ADAPTIVE_FLAG or axis != 0:
            raise ValueError("candidate bank requires canonical axis0 NINTv2")
        qb &= ~NINT_ADAPTIVE_FLAG
        ndim = struct.unpack_from("<I", view, _NINT_HDR.size)[0]
        shape_offset = _NINT_HDR.size + 4
        rows, width, out, ng = struct.unpack_from("<2qII", view, shape_offset)
        if ndim != 2 or width != columns or rows != out or ng != (columns+gs-1)//gs:
            raise ValueError("NINT candidate shape differs")
        off = shape_offset + 24
        fields = (16, 16, 2, ng*kb, ng*kb, 3, ng*gs*qb)
        is_nint = True
    elif dtype == "NVQ":
        magic, profile, sub, gs, axis, columns, ndim = nvq._HEADER.unpack_from(view)
        if axis != 0 or ndim != 2 or gs != 24:
            raise ValueError("candidate bank requires rank2 gs24 axis0 NVQ")
        shape_offset = nvq._HEADER.size
        rows, width, out = struct.unpack_from("<2qI", view, shape_offset)
        if width != columns or rows != out:
            raise ValueError("NVQ candidate shape differs")
        off = shape_offset + 20
        ng = (columns+23)//24
        if magic == nvq1_s._MAGIC:
            if profile != nvq1_s._VERSION or sub != 4 or columns % 8:
                raise ValueError("NVQ1-S candidate profile differs")
            off += nvq1_s.NVQ1_S_TABLE_BYTES
            fields = (16, ng*4, (columns//8)*9, ng)
        elif magic == nvq1_l._MAGIC:
            if profile not in (nvq1_l._PROFILE_IQ1S_GRID, nvq1_l._PROFILE_CUSTOM_TERNARY):
                raise ValueError("NVQ1-L table profile differs")
            off += 4096 if profile == nvq1_l._PROFILE_CUSTOM_TERNARY else 0
            fields = (16, ng*sub, ((columns+7)//8)*11, ng)
        elif magic == nvq._MAGIC and profile & nvq._JSC_FLAG:
            spec = nvq.NvqSpec(nvq._ID_CODEBOOK[profile & ~nvq._JSC_FLAG], groupsize=24, sub_bits=4, sign_mode="even")
            _, _, _, used, storage = nvq._unpack_jsc_metadata_profile(
                view[off:], vector_size=spec.vector_size, codebook_entries=spec.codebook_entries)
            off += used
            fields = ((16, ng*64) if storage == nvq._JSC_GROUP64_LAYOUT_NAME else
                      (16, ng*4, ((columns+spec.vector_size-1)//spec.vector_size)*spec.index_bits,
                       ((columns+7)//8)*7))
        else:
            raise ValueError("unsupported candidate NVQ layout")
        is_nint = False
    else:
        raise ValueError(f"unsupported candidate dtype {dtype}")
    if off + sum((rows*b+7)//8 for b in fields) != len(view):
        raise ValueError("candidate stream lengths differ")
    return Layout(bytes(view[:off]), rows, columns, shape_offset, is_nint, ng, fields)


def select_parts(blob, dtype, experts, rows_per_expert):
    """Select whole byte-aligned experts, preserving order and every stored bit."""
    view = memoryview(blob)
    description = layout(view, dtype)
    selected = np.asarray(experts, dtype=np.int64)
    if (rows_per_expert <= 0 or description.rows % rows_per_expert or not len(selected)
            or selected.min() < 0 or selected.max() >= description.rows//rows_per_expert
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("invalid candidate expert selection")
    if any(rows_per_expert*b % 8 for b in description.field_bits):
        raise ValueError("candidate expert boundaries must be byte aligned")
    result = [description.header(len(selected)*rows_per_expert)]
    off = len(description.prefix)
    for bits in description.field_bits:
        unit = rows_per_expert*bits//8
        count = description.rows//rows_per_expert
        data = np.frombuffer(view, dtype=np.uint8, count=count*unit, offset=off).reshape(count, unit)
        result.append(np.ascontiguousarray(data[selected]).tobytes())
        off += count*unit
    return tuple(result)


def write_merged(output, cohorts, rows_per_expert):
    """Merge selections from chunks of one profile without decoding/refitting.

    cohorts contains (read-only buffer, dtype, local expert positions).
    The output is a flat canonical tensor payload suitable for one MFE pool.
    """
    descriptions = [layout(blob, dtype) for blob, dtype, _ in cohorts]
    first = descriptions[0]
    if any(d.header(0) != first.header(0) or d.field_bits != first.field_bits for d in descriptions):
        raise ValueError("candidate chunks have inconsistent codebooks or formats")
    selected_count = sum(len(ids) for _, _, ids in cohorts)
    output.write(first.header(selected_count*rows_per_expert))
    for field, bits in enumerate(first.field_bits):
        if rows_per_expert*bits % 8:
            raise ValueError("candidate expert boundaries must be byte aligned")
        unit = rows_per_expert*bits//8
        for (blob, _, selected), description in zip(cohorts, descriptions, strict=True):
            count = description.rows//rows_per_expert
            indices = np.asarray(selected, dtype=np.int64)
            if (description.rows % rows_per_expert or indices.min() < 0 or indices.max() >= count
                    or len(np.unique(indices)) != len(indices)):
                raise ValueError("invalid merged candidate expert selection")
            off = len(description.prefix) + sum((description.rows*b+7)//8 for b in description.field_bits[:field])
            data = np.frombuffer(blob, dtype=np.uint8, count=count*unit, offset=off).reshape(count, unit)
            output.write(memoryview(np.ascontiguousarray(data[indices])).cast("B"))
    return len(first.prefix) + sum(selected_count*rows_per_expert*b//8 for b in first.field_bits)
