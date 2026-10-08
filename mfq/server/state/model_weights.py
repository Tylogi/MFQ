from mfq.formats.compat import canonical_dtype


def estimated_resident_weight_bytes(
    weight_bytes: int, by_dtype: dict[str, int],
) -> int:
    indexed = sum(by_dtype.values())
    if indexed > weight_bytes:
        return weight_bytes + (weight_bytes + 9) // 10
    reserve = weight_bytes - indexed
    for dtype, size in by_dtype.items():
        dtype = canonical_dtype(dtype)
        if dtype not in {'F8_E4M3', 'F16', 'BF16', 'F32', 'F64', 'I8', 'I16', 'I32', 'I64',
                           'U8', 'U16', 'U32', 'U64', 'BOOL'}:
            reserve += size
    return weight_bytes + (reserve + 9) // 10
