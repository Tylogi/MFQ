"""Model scope, imatrix routing and exact container costs for full-file AlphaQ."""
from __future__ import annotations

import io
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path

import numpy as np

from mfq.calibration.ew_solver import EwItemKey
from mfq.formats.assets import ASSET_DTYPE, RuntimeAsset
from mfq.formats.header import FileHeader
from mfq.formats.nint import NintSpec
from mfq.formats.shards import _write_header_and_table
from mfq.quantize.imatrix import ImportanceMatrix
from mfq.quantize.standard_presets import TensorScope, describe_tensor
from mfq.tools.quantize_hf_to_mfq import (
    BlobRecord, TensorPlan, _hf_imatrix_names, _plan_blob_nbytes,
)


_BANK = re.compile(r'^model\.block\.(\d+)\.mlp\.experts\.(gate|up|down|gate_up)\.weight$')


@dataclass(frozen=True)
class AlphaQModelScope:
    plans: tuple[TensorPlan, ...]
    experts: tuple[TensorPlan, ...]
    dense: tuple[TensorPlan, ...]
    native_dense: tuple[TensorPlan, ...]
    ple: tuple[TensorPlan, ...]
    imatrix_entries: dict[str, str]
    model_weight_count: int


def model_scope(plans: Sequence[TensorPlan], imatrix: ImportanceMatrix) -> AlphaQModelScope:
    """Use all non-PLE rank-2 matrices with imatrix, including small routers.

    Plans must come from the normal converter with the same fused/split layout
    that will consume the eventual scheme. Missing dense imatrix preserves
    source precision; missing routed-expert imatrix is an input error.
    """
    plans = tuple(plans)
    if not plans or len({p.name for p in plans}) != len(plans):
        raise ValueError('model plans must have distinct tensor names')
    experts, dense, native, ple, bindings = [], [], [], [], {}
    count = 0
    for item in plans:
        if min(item.shape, default=1) <= 0:
            raise ValueError('model plans require positive tensor dimensions')
        # Integer runtime constants occupy bytes but are not float parameters.
        if item.source_dtype not in {'I8', 'I16', 'I32', 'I64', 'U8', 'BOOL'} or item.source_quantization:
            count += math.prod(item.shape)
        descriptor = describe_tensor(item.source_name or item.name, item.shape,
                                     item.source_dtype, canonical_name=item.name)
        if descriptor.scope == TensorScope.PLE:
            ple.append(item)
            continue
        bank = _BANK.fullmatch(item.name)
        if bank is not None:
            if len(item.shape) != 3:
                raise ValueError(f'routed expert plan must have rank 3: {item.name}')
            match = imatrix.find(_hf_imatrix_names(item))
            if match is None:
                raise ValueError(f'missing expert imatrix: {item.name}')
            name, entry = match
            if entry.width != item.shape[-1] or entry.matrices != item.shape[0]:
                raise ValueError(f'expert imatrix shape differs: {item.name}')
            bindings[item.name] = name
            experts.append(item)
        elif len(item.shape) == 2 and (
            descriptor.quantizable or item.source_dtype in {'BF16', 'F16', 'F32'}
        ):
            if item.source_dtype not in {'BF16', 'F16', 'F32'} or item.source_quantization:
                raise ValueError(f'joint native candidate requires BF16/F16/F32 source: {item.name}')
            match = imatrix.find(_hf_imatrix_names(item))
            if match is None:
                native.append(item)
            else:
                name, entry = match
                if entry.width != item.shape[-1] or entry.matrices not in (1, item.shape[0]):
                    raise ValueError(f'dense imatrix shape differs: {item.name}')
                bindings[item.name] = name
                dense.append(item)
    if not experts or count <= 0:
        raise ValueError('full-file AlphaQ requires routed experts and float parameters')
    return AlphaQModelScope(plans, tuple(experts), tuple(dense), tuple(native), tuple(ple), bindings, count)


def imatrix_router_multipliers(
    scope: AlphaQModelScope,
    imatrix: ImportanceMatrix,
    router_entries: Mapping[int, str],
    topk: int,
    *,
    square_root: bool = False,
) -> tuple[dict[EwItemKey, float], dict[str, float]]:
    """Counts / actual router tokens, with one global normalization.

    Per-projection counts must agree and cover exactly top-k selections for
    each token. No confidence or gradient observations are inferred.
    """
    if type(topk) is not int or topk < 1:
        raise ValueError('topk must be a positive integer')
    observed, rows, exposures = {}, {}, {}
    for item in scope.experts:
        layer, projection = _BANK.fullmatch(item.name).groups()
        layer = int(layer)
        if layer not in router_entries or router_entries[layer] not in imatrix.entries:
            raise ValueError(f'missing actual router token counts for layer {layer}')
        tokens = np.asarray(imatrix.entries[router_entries[layer]].counts)
        counts = np.asarray(imatrix.entries[scope.imatrix_entries[item.name]].counts)
        if (tokens.shape != (1,) or not np.isfinite(tokens).all() or tokens[0] <= 0
                or not np.equal(tokens, tokens.astype(np.int64)).all()):
            raise ValueError('router token counts must be positive integers')
        if (counts.shape != (item.shape[0],) or not np.isfinite(counts).all() or (counts < 0).any()
                or not np.equal(counts, counts.astype(np.int64)).all()
                or topk > item.shape[0] or counts.sum() != tokens[0]*topk):
            raise ValueError('expert counts differ from full topk coverage')
        if layer in observed and not np.array_equal(observed[layer], counts):
            raise ValueError('expert projection routing counts disagree')
        observed[layer] = counts
        rows[layer] = counts.astype(np.float64)/tokens[0]
        exposures[item.name] = topk/item.shape[0]
    if set(router_entries) != set(rows):
        raise ValueError('router token entries must cover exactly the routed layers')
    if square_root:
        rows = {layer: np.sqrt(values) for layer, values in rows.items()}
    mean = sum(v.sum() for v in rows.values())/sum(v.size for v in rows.values())
    if mean <= 0:
        raise ValueError('routing frequencies have no observed mass')
    result = {}
    for item in scope.experts:
        layer, projection = _BANK.fullmatch(item.name).groups()
        layer = int(layer)
        result.update({EwItemKey(item.name, layer, projection, i): float(value/mean)
                       for i, value in enumerate(rows[layer])})
    return result, exposures


@dataclass(frozen=True)
class AlphaQFileBudget:
    maximum_file_bytes: int
    fixed_file_bytes: int
    model_weight_count: int
    native_overrides: dict[str, str]
    header_and_table_bytes: int
    fixed_payload_bytes: int


def full_file_budget(
    scope: AlphaQModelScope,
    header: FileHeader,
    assets: Sequence[RuntimeAsset],
    *,
    target_bpw: str | float | Decimal,
    joint_dense: bool = True,
    preserve_missing_imatrix: bool = True,
    artifact_root: str | Path | None = None,
) -> AlphaQFileBudget:
    """Reserve the real header/table, assets, PLE and native/fixed weights.

    Four-byte dense dtype tags conservatively cover NINT/NVQ/BF16/F16/F32.
    Expert payload headers/pools belong to the allocator's variable cost.
    The supplied header/assets must also be used by the final normal writer.
    """
    try:
        bpw = Decimal(str(target_bpw))
    except InvalidOperation as error:
        raise ValueError('target BPW must be finite and positive') from error
    if not bpw.is_finite() or bpw <= 0:
        raise ValueError('target BPW must be finite and positive')
    cap = int((bpw*scope.model_weight_count/8).to_integral_value(rounding=ROUND_FLOOR))
    names = [a.name for a in assets]
    if len(set(names)) != len(names) or set(names) & {p.name for p in scope.plans}:
        raise ValueError('runtime asset names must be distinct from model tensors')
    variable_experts = {p.name for p in scope.experts}
    variable_dense = {p.name for p in scope.dense} if joint_dense else set()
    if joint_dense and not preserve_missing_imatrix:
        raise ValueError('joint allocation must preserve matrices without imatrix')
    native = ({p.name: p.source_dtype for p in scope.native_dense}
              if preserve_missing_imatrix else {})
    records = [BlobRecord(a.name, ASSET_DTYPE, len(a.data), Path('.')) for a in assets]
    fixed_payload = sum(len(a.data) for a in assets)
    for item in scope.plans:
        if item.name in variable_experts:
            dtype, size = 'MFE', 0
        elif item.name in variable_dense:
            dtype, size = 'NINT', 0
        else:
            if item.name in native:
                item = replace(item, target_dtype=native[item.name], target_spec=None,
                               target_precision=None, target_options=())
            size = _plan_blob_nbytes(item, NintSpec(), artifact_root)
            dtype = 'NINT' if item.target_dtype.startswith('NINT') and item.target_dtype != 'NINT8-0' else item.target_dtype
            fixed_payload += size
        records.append(BlobRecord(item.name, dtype, size, Path('.')))
    stream = io.BytesIO()
    _write_header_and_table(stream, replace(header, num_tensors=len(records)), records)
    fixed = stream.tell()+fixed_payload
    if cap <= fixed:
        raise ValueError('full-file cap leaves no space for allocated weights')
    return AlphaQFileBudget(cap, fixed, scope.model_weight_count, native, stream.tell(), fixed_payload)
