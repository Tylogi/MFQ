import re

from mfq.formats.compat import canonical_dtype


def weight_role(name: str) -> str:
    if re.search(r'\.mlp\.experts\.(?:\d+\.)?(?:gate|up|down|gate_up)(?:_proj)?\.(?:weight|weight_scale)$', name) or any(
        marker in name for marker in ('.ffn_gate_exps.', '.ffn_up_exps.', '.ffn_gate_up_exps.', '.ffn_down_exps.')
    ):
        return 'experts'
    if name.startswith(('model.token_embedding.', 'token_embd.')) or '.embed_tokens.' in name or name == 'embed_tokens.weight':
        return 'embedding'
    return 'dense'


def estimated_weight_roles(by_role: dict[str, dict[str, int]]) -> dict[str, int]:
    return {role: estimated_resident_weight_bytes(sum(dtypes.values()), dtypes)
            for role, dtypes in by_role.items()}


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
