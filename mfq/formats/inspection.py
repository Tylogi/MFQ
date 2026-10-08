from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

from mfq.formats.compat import canonical_dtype
from mfq.formats.io import unpack_bits
from mfq.formats.nepq import nepq_spec


@dataclass(frozen=True)
class TensorInspection:
    shape: tuple[int, ...]
    family: str
    row_bytes: np.ndarray | None = None
    granularity: str = "tensor"


def _shape(blob: memoryview, offset: int, rank: int) -> tuple[int, ...]:
    if not 1 <= rank <= 16:
        raise ValueError("invalid tensor rank")
    shape = struct.unpack_from(f"<{rank}q", blob, offset)
    if any(value <= 0 for value in shape):
        raise ValueError("invalid tensor dimensions")
    return shape


def _selectors(blob: memoryview, offset: int, count: int, bits: int) -> np.ndarray:
    if count > 16_777_216 or offset + (count * bits + 7) // 8 > len(blob):
        raise ValueError("invalid or oversized row selector table")
    return unpack_bits(blob, offset, count, bits)[0].astype(np.int64)


def _row_storage(cost: np.ndarray, size: int) -> np.ndarray:
    remainder = size - float(cost.sum())
    if remainder < -1e-6:
        raise ValueError("row storage exceeds tensor payload")
    return cost + max(0.0, remainder) / len(cost)


def inspect_tensor(dtype: str, blob: memoryview, *, rows: bool = False) -> TensorInspection:
    dtype = canonical_dtype(dtype)
    if dtype == "NINT":
        raw_q, k, gs, axis, width = struct.unpack_from("<BBiii", blob)
        rank = struct.unpack_from("<I", blob, 14)[0]
        shape = _shape(blob, 18, rank)
        offset = 18 + rank * 8
        outputs, groups = struct.unpack_from("<II", blob, offset)
        if not 0 <= axis < rank or shape[axis] != outputs or gs <= 0 or groups != math.ceil(width / gs):
            raise ValueError("invalid NINT geometry")
        q = raw_q & 127
        if not 1 <= q <= 8 or not 1 <= k <= 8 or math.prod(shape) != outputs * width:
            raise ValueError("invalid NINT precision or shape")
        row_bytes = None
        if rows:
            if raw_q & 128:
                offset += 8 + outputs * 4
                ks = _selectors(blob, offset, outputs, 2) + k - 1
                if np.any(ks < 1) or np.any(ks > 8):
                    raise ValueError("invalid NINT subgroup selector")
                offset += (outputs * 2 + 7) // 8
                offset += sum(2 * ((int(np.count_nonzero(ks == value)) * groups * value + 7) // 8)
                              for value in range(1, 9))
                qs = _selectors(blob, offset, outputs, 3) + 1
                row_bytes = _row_storage((32 + 5 + 2 * groups * ks + groups * gs * qs) / 8, len(blob))
            else:
                row_bytes = np.full(outputs, len(blob) / outputs)
        return TensorInspection(shape, "NINTv2" if raw_q & 128 else f"NINT{q}", row_bytes,
                                "row" if raw_q & 128 and rows else "cohort")
    if dtype in {"NVQ", "NPQ"}:
        magic, profile, _sub_bits, _gs, _axis, _width, rank = struct.unpack_from("<4sBBHiiI", blob)
        shape = _shape(blob, 20, rank)
        families = {b"NQ1L": "NVQ1-L", b"NQ1S": "NVQ1-S", b"NPQL": "NPQ0-L", b"NPQS": "NPQ0-S"}
        family = families.get(magic)
        if magic in {b"NVQ1", b"NIQ1"}:
            codebook = profile & 31
            if profile & 32:
                family = {1: "NVQ2J", 2: "NVQ3J", 3: "NVQ3J-512", 4: "NVQ2J-L", 5: "NVQ2J-XL", 6: "NVQ3J-L"}.get(codebook)
            else:
                family = {1: "NVQ2", 2: "NVQ3"}.get(codebook)
        if family is None:
            raise ValueError("unknown NVQ/NPQ profile")
        return TensorInspection(shape, family, granularity="cohort")
    if dtype == "NINT8-0":
        magic, _axis, _width, rank = struct.unpack_from("<4siiI", blob)
        if magic != b"NI80":
            raise ValueError("invalid NINT8-0 magic")
        return TensorInspection(_shape(blob, 16, rank), dtype, granularity="cohort")
    if dtype == "NEPQ":
        magic, version, profile, _groups, _flags, experts, outputs, width, _banks, _rotation, _seed = struct.unpack_from("<4sBBBBIIIIIQ", blob)
        if magic != b"NEP1" or version != 1:
            raise ValueError("unsupported NEPQ header")
        shape = (experts, outputs, width) if experts > 1 else (outputs, width)
        if any(value <= 0 for value in shape):
            raise ValueError("invalid NEPQ geometry")
        return TensorInspection(shape, nepq_spec(profile).label, granularity="cohort")
    if dtype in {"MXFP4", "MXFP8"}:
        magic, _version, _kind, _reserved, outputs, width = struct.unpack_from("<4sBBHQQ", blob)
        if magic != b"MXT1" or min(outputs, width) <= 0:
            raise ValueError("invalid MX geometry")
        return TensorInspection((outputs, width), dtype, granularity="cohort")
    if dtype == "MXFP4-SQ":
        magic, _version, _base, _reserved, outputs, width = struct.unpack_from("<4sBBHQQ", blob)
        if min(outputs, width) <= 0 or magic not in {b"SQV2", b"SQ2\0", b"SQ3\0", b"SQ31"}:
            raise ValueError("unsupported MXFP4-SQ header")
        row_bytes = None
        if rows and magic == b"SQV2":
            qs = _selectors(blob, 24, outputs, 2) + 1
            cost = (width * qs + 2) / 8 + np.where(qs == 4, width / 32, width / 256 + 7)
            row_bytes = _row_storage(cost, len(blob))
        return TensorInspection((outputs, width), dtype, row_bytes, "row" if row_bytes is not None else "cohort")
    if dtype in {"MXFP8-SQ", "FP8-128SQ"}:
        magic, _version, _scale, _flags, _reserved, _br, _bc, outputs, width, _sr, _sc = struct.unpack_from("<4sBBBBHHQQQQ", blob)
        if magic not in {b"M8SQ", b"F8SQ"} or min(outputs, width) <= 0:
            raise ValueError("invalid FP8-SQ geometry")
        row_bytes = None
        if rows:
            qs = _selectors(blob, 44, outputs, 3) + 1
            row_bytes = _row_storage(np.ceil(width * qs / 8) + 3 / 8, len(blob))
        return TensorInspection((outputs, width), dtype, row_bytes, "row" if rows else "cohort")
    if dtype in {"BF16", "F16", "F32", "F8_E4M3", "I32", "I64"}:
        rank = struct.unpack_from("<I", blob)[0]
        shape = _shape(blob, 4, rank) if rank else ()
        size = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I32": 4, "I64": 8}[dtype]
        if 4 + 8 * rank + math.prod(shape) * size != len(blob):
            raise ValueError("dense shape does not match payload")
        return TensorInspection(shape, dtype, granularity="cohort")
    raise ValueError(f"unsupported tensor dtype: {dtype}")
