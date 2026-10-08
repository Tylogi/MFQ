from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import torch

from mfq._alphaq_profiles import ALPHAQ_PROFILES
from mfq.calibration.alphaq import AlphaQTensorStatistics, alphaq_builtin_candidates, alphaq_candidates
from mfq.calibration.alphaq_fit import FittedPayloadCache
from mfq.calibration.alphaq_joint import DenseChoice, allocate_joint, dense_choices
from mfq.calibration.alphaq_model import AlphaQModelScope, full_file_budget
from mfq.calibration.alphaq_output import runtime_metadata
from mfq.calibration.artifact import load_scheme, save_scheme
from mfq.calibration.artifact import ExpertPrecision
from mfq.calibration.ew_solver import EwCandidate, EwCandidateTable, EwItemKey, EwTensorSpec
from mfq.formats.assets import ASSET_DTYPE, RuntimeAsset, is_asset_record
from mfq.formats.shards import _write_header_and_table, write_blob_record_shards
from mfq.formats.mx import mx_header_bytes
from mfq.quantize.backend import resolve_quant_backend
from mfq.quantize.imatrix import ImportanceMatrix, load_importance_matrix
from mfq.quantize.workbench_candidates import WORKBENCH_CANDIDATES, SQ_CANDIDATES
from mfq.quantize.standard_presets import TensorRole, TensorScope, describe_tensor
from mfq.tools import quantize_hf_to_mfq as q

SOURCE_FLOAT_DTYPES = frozenset({'BF16', 'F16', 'F32', 'F64', 'F8_E4M3', 'F8_E5M2', 'MXFP4', 'MXFP8'})
NATIVE_DTYPES = frozenset({'BF16', 'F16', 'F32', 'F8_E4M3', 'MXFP4', 'MXFP8'})


class RecipeValidationError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def emit(**event):
    print(json.dumps(event, allow_nan=False), flush=True)


class BatchGate:
    def __init__(self, interactive=False):
        self.interactive = interactive
        self.tensor = ''

    def __call__(self, requested=1024):
        if not self.interactive:
            return requested
        emit(event='workbench_batch', tensor=self.tensor, requested_rows=requested)
        raw = sys.stdin.readline()
        if not raw:
            raise InterruptedError('workbench scheduler disconnected')
        rows = json.loads(raw)['rows']
        if type(rows) is not int or rows < 1 or rows > requested:
            raise ValueError('invalid batch permit')
        return rows


class CooperativeRows:
    def __init__(self, source, gate, native_writer=None, sq_writer=None):
        self.source, self.gate = source, gate
        self.write_native = native_writer
        self.sq_writer = sq_writer

    def __getattr__(self, name):
        return getattr(self.source, name)

    def batch_rows(self, requested):
        return self.gate(requested)

    def read_rows(self, *args, **kwargs):
        return self.source.read_rows(*args, **kwargs)

    def read_expert_rows(self, expert, start, end, *, device='cpu'):
        if hasattr(self.source, 'read_expert_rows'):
            return self.source.read_expert_rows(expert, start, end, device=device)
        rows = self.source.rows_per_expert if hasattr(self.source, 'rows_per_expert') else self.source.shape[-2]
        return self.source.read_rows(expert * rows + start, expert * rows + end, device=device)

    def write_mxfp4_sq_expert_pool(self, expert, path, *, q):
        return self.sq_writer(expert, path, q, None)

    def write_fp8_sq_expert_pool(self, expert, path, *, q, neuron_importance=None):
        return self.sq_writer(expert, path, q, neuron_importance)

    def __getitem__(self, value):
        return self.source[value]

    def tensor(self):
        self.gate(8)
        return self.source.tensor()


class ModelSource:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.checkpoint = None
        if self.path.is_dir():
            self.inventory = q._hf_source_inventory(self.path)
            self.config = q.load_hf_model_config(self.path)
            shards = {self.path / item.shard for item in self.inventory.values()}
            self.files = tuple(sorted(shards | {self.path / 'config.json'} | set(self.path.glob('*token*json'))))
        else:
            from mfq.quantize.mfq_source import FullPrecisionMfqCheckpoint
            self.checkpoint = FullPrecisionMfqCheckpoint(self.path)
            self.config = self.checkpoint.model_config()
            self.inventory = {name: q.SourceTensorMetadata(name, f'mfq-{item.source_index:05d}', item.shape, item.dtype, item.nbytes)
                for name, item in self.checkpoint.infos.items()}
            self.files = tuple(self.checkpoint.store.paths)
        try:
            if 'nvfp4' in json.dumps(self.config).lower() or any('NVFP4' in item.dtype.upper() for item in self.inventory.values()):
                raise ValueError('NVFP4 is not supported as a quantization source')
            encodings, self.auxiliaries = q._source_quantizations(self.inventory, self.config)
            self.encodings = encodings
            labels = {'BF16': 'BF16', 'F16': 'FP16', 'F32': 'FP32', 'F64': 'FP64', 'F8_E4M3': 'FP8', 'F8_E5M2': 'FP8', 'MXFP4': 'MXFP4', 'MXFP8': 'MXFP8'}
            self.source_precisions = sorted({
                'MXFP4' if encodings.get(name) and encodings[name].scheme == 'mxfp4_block32' else
                'MXFP8' if encodings.get(name) and encodings[name].scheme.startswith('mxfp8_') else labels[item.dtype]
                for name, item in self.inventory.items() if name not in self.auxiliaries and (name in encodings or item.dtype in labels)})
            invalid = [f'{name}={item.dtype}' for name, item in self.inventory.items() if name not in self.auxiliaries and name not in encodings and (
                item.dtype not in SOURCE_FLOAT_DTYPES | q._HF_INTEGER_DTYPES or
                (name.endswith('.weight') and item.dtype not in SOURCE_FLOAT_DTYPES and name not in encodings))]
            if invalid:
                raise ValueError('source weights have no supported native decoder: ' + ', '.join(invalid[:5]))
            self.plans = self.plan()
            if not self.parameters():
                raise ValueError('source contains no supported floating weights')
            self.initial_identity = self.identity()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.checkpoint is not None:
            self.checkpoint.close()

    def identity(self):
        return {'files': [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in self.files],
            'config_sha256': hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()}

    def plan(self, scheme=None, overrides=None):
        plans = q._plan(self.path, False, None, 'f32', scheme,
            source_inventory=self.inventory, source_config=self.config, quantize_ple=False, quantize_mtp=True, quantize_vision=True)
        plans = q._normalize_hf_expert_storage(plans)
        if self.checkpoint is not None:
            normalized = []
            for item in plans:
                if item.source_dtype == 'MXFP8' and item.expert_source_names is None:
                    block = self.checkpoint.tensor_source(item.source_name or item.name)._mx_layout.block_shape
                    geometry = str(block[0]) if block[0] == block[1] else f'{block[0]}x{block[1]}'
                    item = replace(item, source_quantization=f'mxfp8_block{geometry}', source_scale_dtype='F8_E8M0')
                normalized.append(item)
            plans = normalized
        selected = set(scheme.selections) | set(scheme.expert_selections) if scheme else set()
        preserve = {item.name: self.native_dtype(item) for item in plans if item.name not in selected}
        preserve.update(overrides or {})
        return tuple(q._apply_tensor_precision_overrides(plans, preserve))

    def native_dtype(self, item):
        dtype = item.source_dtype
        if dtype in {'BF16', 'F16', 'F32'} | q._HF_INTEGER_DTYPES:
            return dtype
        if item.transform or item.row_start is not None or item.row_end is not None:
            return 'F32'
        if self.checkpoint is not None and dtype in {'MXFP4', 'MXFP8'}:
            return dtype
        if dtype == 'MXFP4':
            return dtype
        if len(item.shape) == 2 and item.source_quantization in q._MXFP8_SOURCE_BLOCKS:
            return 'MXFP8'
        descriptor = describe_tensor(item.source_name or item.name, item.shape, dtype, canonical_name=item.name)
        if dtype == 'F8_E4M3' and descriptor.role == TensorRole.PLE_EMBEDDING and item.source_quantization == 'fp8_tensor_scale':
            return dtype
        return 'F32'

    def sq_precision(self, item, profile, expert=0):
        if profile not in SQ_CANDIDATES or item.transform and not item.transform.startswith('expert_') and item.expert_source_names is None:
            return None
        family, width = profile.rsplit('-', 1)
        bits = 4 if width == 'F' else int(width)
        if item.expert_source_names is not None:
            schemes = item.expert_source_quantizations[expert]
            dtypes = item.expert_source_scale_dtypes[expert]
            source_dtypes = {self.encodings[name].logical_dtype or self.inventory[name].dtype if name in self.encodings else self.inventory[name].dtype for name in item.expert_source_names[expert]}
        else:
            schemes = (item.source_quantization,)
            dtypes = (item.source_scale_dtype,)
            source_dtypes = {item.source_dtype}
        if family == 'MXFP4-SQ':
            if source_dtypes != {'MXFP4'}:
                return None
            return ExpertPrecision(family, options=(('q', bits),))
        if source_dtypes == {'MXFP8'} and self.checkpoint is not None:
            names = item.expert_source_names[expert] if item.expert_source_names else (item.source_name or item.name,)
            contracts = [('MXFP8-SQ', self.checkpoint.tensor_source(name)._mx_layout.block_shape) for name in names]
            dtypes = ('F8_E8M0',) * len(names)
        else:
            if source_dtypes != {'F8_E4M3'}:
                return None
            contracts = [q._fp8_sq_contract(scheme, dtype) for scheme, dtype in zip(schemes, dtypes, strict=True)]
        if not contracts or any(contract is None or contract != contracts[0] for contract in contracts):
            return None
        resolved, block = contracts[0]
        if resolved != ('FP8-128SQ' if family == 'FP8-SQ' else family):
            return None
        if len(item.shape) == 2:
            offset = item.row_start or 0
        else:
            offset = item.shape[1] if item.transform == 'expert_up' else 0
        if offset % block[0] and item.shape[-2] > block[0] - offset % block[0]:
            return None
        if len(set(dtypes)) != 1:
            return None
        if item.expert_source_names is not None:
            row_offset = 0
            for name in item.expert_source_names[expert][:-1]:
                row_offset += self.inventory[name].shape[-2]
                if row_offset % block[0]:
                    return None
        return q._fp8_sq_precision(resolved, block, dtypes[0], bits)

    def eligible_candidates(self):
        available = set(ALPHAQ_PROFILES)
        for item in self.plans:
            descriptor = describe_tensor(item.source_name or item.name, item.shape, item.source_dtype, canonical_name=item.name)
            if descriptor.scope == TensorScope.PLE or not (len(item.shape) == 2 or len(item.shape) == 3 and descriptor.role == TensorRole.ROUTED_EXPERT):
                continue
            for name in SQ_CANDIDATES - available:
                if any(self.sq_precision(item, name, expert) is not None for expert in range(item.shape[0] if len(item.shape) == 3 else 1)):
                    available.add(name)
        return [name for name in WORKBENCH_CANDIDATES if name in available]

    def parameters(self, plans=None):
        return sum(math.prod(item.shape) for item in (self.plans if plans is None else plans)
            if item.source_dtype in SOURCE_FLOAT_DTYPES and (item.source_name or item.name) not in self.auxiliaries)

    def storage_bits(self):
        return sum(path.stat().st_size for path in self.files) * 8

    def write_native(self, item, path, chunk, gate):
        if item.target_dtype in q._MATRIX_LOCAL_SQ_FAMILIES:
            return item.target_dtype, self.write_sq(item, path, item.target_precision, None, gate)
        if item.target_dtype not in {'MXFP4', 'MXFP8', 'F8_E4M3'}:
            return None
        name = item.source_name or item.name
        if item.transform or item.row_start is not None or item.row_end is not None:
            raise ValueError(f'native preservation does not support transformed matrix: {item.name}')
        if self.checkpoint is not None:
            record = self.checkpoint.store.records[name]
            if record.dtype != item.target_dtype:
                raise ValueError(f'native source dtype differs: {name}')
            view = self.checkpoint.store.blob_view(record)
            try:
                with path.open('wb') as output:
                    for start in range(0, len(view), 4 * 1024 * 1024):
                        gate(8)
                        output.write(view[start:start + 4 * 1024 * 1024])
            finally:
                view.release()
        else:
            weight = q._RawSafeTensorSlice(self.path / item.shard, name)
            if item.target_dtype == 'F8_E4M3':
                size = q._write_float8_e4m3_axis0_blob(CooperativeRows(weight, gate), item.shape, path, chunk)
                return item.target_dtype, size
            scale = q._RawSafeTensorSlice(self.path / item.source_scale_shard, item.source_scale_name)
            gate(8)
            if item.target_dtype == 'MXFP8':
                q._write_native_mxfp8_blob(weight, scale, item.shape, item.source_quantization, path)
            else:
                q._Mxfp4TensorSlice(weight, scale)
                with path.open('wb') as output:
                    output.write(mx_header_bytes('MXFP4', item.shape, weight.shape, scale.shape))
                    q._copy_raw_safetensor_payload(weight, output)
                    q._copy_raw_safetensor_payload(scale, output)
        return item.target_dtype, path.stat().st_size

    def write_sq(self, item, path, precision, expert, gate, neuron_importance=None):
        gate(8)
        bits = int(precision.option('q', 4))
        label = ('FP8-SQ' if precision.family == 'FP8-128SQ' else precision.family) + '-' + ('F' if precision.family == 'MXFP4-SQ' and bits == 4 else str(bits))
        expected = self.sq_precision(item, label, expert or 0)
        if expected != precision:
            raise ValueError(f'SQ precision differs from the native source contract: {item.name}')
        if item.expert_source_names is not None and self.checkpoint is None:
            reader = q._SeparateExpertRowSource(
                self.path, item.expert_shape or item.shape, item.expert_source_names, item.expert_source_shards,
                item.expert_source_quantizations, item.expert_source_scale_names, item.expert_source_scale_shards, item.expert_source_scale_dtypes)
            try:
                if precision.family == 'MXFP4-SQ':
                    return reader.write_mxfp4_sq_expert_pool(expert, path, q=bits)
                return reader.write_fp8_sq_expert_pool(expert, path, q=bits, neuron_importance=neuron_importance)
            finally:
                reader.close()
        if self.checkpoint is not None:
            names = item.expert_source_names[expert] if item.expert_source_names else (item.source_name or item.name,)
            scale_names = item.expert_source_scale_names[expert] if item.expert_source_names else (item.source_scale_name,)
            codes, multipliers = [], []
            for name, scale_name in zip(names, scale_names, strict=True):
                reader = self.checkpoint.tensor_source(name)
                if reader.dtype_name in {'MXFP4', 'MXFP8'}:
                    raw, scale = reader._mx_arrays()
                    codes.append(np.array(raw, copy=True)); multipliers.append(np.array(scale, copy=True))
                else:
                    codes.append(reader.read_rows(0, reader.rows, device='cpu').contiguous().view(torch.uint8).numpy())
                    scale = self.checkpoint.tensor_source(scale_name).tensor()
                    multipliers.append(scale.view(torch.uint16).numpy() if scale.dtype == torch.bfloat16 else scale.numpy())
            values, scales = np.concatenate(codes), np.concatenate(multipliers)
            if item.expert_source_names is None:
                rows = item.shape[-2]
                offset = item.row_start or 0
                if expert is not None:
                    offset = expert * reader.shape[-2] + (rows if item.transform == 'expert_up' else 0)
                values = values.reshape(-1, values.shape[-1])[offset:offset + rows]
                block_rows = int(precision.option('block_rows', 1 if precision.family == 'MXFP4-SQ' else 128))
                first = offset // block_rows
                if expert is not None:
                    first = expert * scales.shape[-2] + (rows // block_rows if item.transform == 'expert_up' else 0)
                scales = scales.reshape(-1, scales.shape[-1])[first:first + math.ceil(rows / block_rows)]
        else:
            weight = q._RawSafeTensorSlice(self.path / item.shard, item.source_name or item.name)
            scale = q._RawSafeTensorSlice(self.path / item.source_scale_shard, item.source_scale_name)
            rows = item.shape[-2]
            offset = item.row_start or 0
            if expert is not None:
                source_rows = weight.shape[-2]
                offset = expert * source_rows + (rows if item.transform == 'expert_up' else 0)
            values = weight.read_rows(offset, offset + rows, device='cpu').contiguous().view(torch.uint8).numpy()
            if precision.family == 'MXFP4-SQ':
                scales = scale.read_rows(offset, offset + rows, device='cpu').contiguous().view(torch.uint8).numpy()
            else:
                block_rows = int(precision.option('block_rows', 128))
                first = offset // block_rows
                if expert is not None:
                    first = expert * scale.shape[-2] + (rows // block_rows if item.transform == 'expert_up' else 0)
                scales = scale.read_rows(first, first + math.ceil(rows / block_rows), device='cpu').contiguous()
                scales = scales.view(torch.uint8).numpy() if scale.dtype_name == 'F8_E8M0' else scales.view(torch.uint16).numpy() if scale.dtype_name == 'BF16' else scales.numpy()
        if precision.family == 'MXFP4-SQ':
            return q.write_mxfp4_sq_blob(path, values, scales, row_q_bits=bits)
        if precision.family == 'MXFP8-SQ':
            return q.write_mxfp8_sq_blob(path, values, scales, row_q_bits=bits,
                block_shape=(int(precision.option('block_rows', 128)), int(precision.option('block_columns', 128))), neuron_importance=neuron_importance)
        return q.write_fp8_128_sq_blob(path, values, scales, row_q_bits=bits,
            scale_dtype=str(precision.option('scale_dtype', 'BF16')), neuron_importance=neuron_importance)

    @contextmanager
    def rows(self, item, gate):
        if item.expert_source_names is not None:
            raw = q._MfqGlmExpertRowSource(self.checkpoint, item.expert_shape or item.shape,
                item.expert_source_names, item.expert_source_quantizations, item.expert_source_scale_names) if self.checkpoint else q._SeparateExpertRowSource(
                self.path, item.expert_shape or item.shape, item.expert_source_names, item.expert_source_shards,
                item.expert_source_quantizations, item.expert_source_scale_names, item.expert_source_scale_shards, item.expert_source_scale_dtypes)
        elif self.checkpoint is not None:
            raw = self.checkpoint.tensor_source(item.source_name or item.name)
            if item.source_dtype in q._HF_FP8_DTYPES and item.source_quantization:
                raw = q._ScaledFp8TensorSlice(raw, self.checkpoint.tensor_source(item.source_scale_name), item.source_quantization)
        else:
            raw = q._raw_source_for_plan(self.path, item)
        try:
            source = raw
            if item.transform and item.transform.startswith('expert_'):
                source = q._PackedExpertProjectionSource(raw, item)
            elif len(item.shape) == 2:
                source = q._HfPlanRowSource(raw, item)
            def sq_writer(expert, path, bits, importance):
                profile = 'MXFP4-SQ-' + ('F' if bits == 4 else str(bits)) if self.sq_precision(item, 'MXFP4-SQ-F', expert) else (
                    'MXFP8-SQ-' if self.sq_precision(item, 'MXFP8-SQ-8', expert) else 'FP8-SQ-') + str(bits)
                precision = self.sq_precision(item, profile, expert)
                if precision is None:
                    raise ValueError(f'SQ candidate is incompatible with the native source: {item.name}/{expert}')
                return self.write_sq(item, path, precision, expert, gate, importance)
            yield CooperativeRows(source, gate, lambda plan, path, chunk: self.write_native(plan, path, chunk, gate), sq_writer)
        finally:
            if hasattr(raw, 'close'):
                raw.close()

    def metadata(self, policy):
        if self.checkpoint is None:
            return runtime_metadata(self.path, self.plans, imatrix=None, imatrix_sha256='', policy=policy)
        assets = tuple(RuntimeAsset(name, 'application/octet-stream', self.checkpoint.store.read_blob(name))
            for name in self.checkpoint.store.records if is_asset_record(name))
        extra = {**self.checkpoint.header.extra, 'quantization_workbench': policy}
        return replace(self.checkpoint.header, extra=extra), assets


def source_summary(path):
    source = ModelSource(path)
    try:
        text = source.config.get('text_config', source.config)
        return {'path': str(source.path), 'architecture': text.get('model_type', source.config.get('model_type', 'unknown')),
            'tensors': len(source.plans), 'parameters': source.parameters(),
            'source_precisions': source.source_precisions, 'eligible_candidates': source.eligible_candidates(),
            'imatrix_supported': source.path.is_dir() and text.get('model_type', source.config.get('model_type', '')).removesuffix('_text') in {'qwen3_5', 'qwen3_5_moe', 'gemma4'},
            'format': 'hf' if source.path.is_dir() else 'mfq', 'full_precision': True}
    finally:
        source.close()


def matrix_statistics(source, rows, columns, device, gate):
    spectra, total, squared, start = [], 0., 0., 0
    block_rows = rows // 128 * 128 if min(rows, columns) >= 128 else rows
    if min(rows, columns) < 128:
        gate(max(8, rows))
        full = source.read_rows(0, rows, device=device).float()
        if not bool(torch.isfinite(full).all()):
            raise ValueError('source weights contain NaN or infinity')
        variance = float(full.cpu().double().var(correction=0))
        eigen = torch.linalg.svdvals(full).square().cpu().double().numpy()
    else:
        while start < rows:
            permitted = gate(4096)
            count = max(128, permitted // 128 * 128)
            end = min(rows, start + count)
            value = source.read_rows(start, end, device=device).float()
            if not bool(torch.isfinite(value).all()):
                raise ValueError('source weights contain NaN or infinity')
            cpu = value.cpu().double()
            total += float(cpu.sum()); squared += float(cpu.square().sum())
            use = min(end, block_rows) - start
            if use > 0:
                blocks = value[:use, :columns // 128 * 128].reshape(use // 128, 128, columns // 128, 128).permute(0, 2, 1, 3).reshape(-1, 128, 128)
                spectra.append(torch.linalg.svdvals(blocks).square().cpu().double().numpy().reshape(-1))
            start = end
        variance = max(0., squared / (rows * columns) - (total / (rows * columns)) ** 2)
        eigen = np.concatenate(spectra)
    eigen.sort()
    if eigen.size < 2:
        return 1., variance
    tail = min(eigen.size - 1, max(10, int(eigen.size * .1)))
    threshold = max(eigen[-tail - 1], 1e-12)
    denominator = max(float(np.log(np.maximum(eigen[-tail:], 1e-12) / threshold).sum()), 1e-12)
    return 1. + tail / denominator, variance


def data_free_scheme(scheme):
    def precision(value):
        return replace(value, options=tuple({**dict(value.options), 'imatrix_weighted': False}.items())) if value.nint_spec else value
    return replace(scheme, selections={name: replace(item, precision=precision(item.descriptor)) for name, item in scheme.selections.items()},
        expert_selections={name: replace(item, selections=tuple(replace(entry, precision=precision(entry.descriptor)) for entry in item.selections))
            for name, item in scheme.expert_selections.items()})


def workbench_expert_candidates(source, statistics, plans, names):
    from mfq.formats.io import _MFE_HDR
    base = tuple(name for name in names if name in ALPHAQ_PROFILES)
    table = alphaq_builtin_candidates(statistics, base) if base else EwCandidateTable({
        stat.name: EwTensorSpec(stat.name, stat.name, stat.layer, stat.projection, *stat.shape, _MFE_HDR.size * 8, None)
        for stat in statistics}, (), None, {})
    choices = list(table.candidates)
    for stat, item in zip(statistics, plans, strict=True):
        for expert in range(stat.shape[0]):
            for name in names:
                precision = source.sq_precision(item, name, expert)
                if precision is None:
                    continue
                size = q._mixed_moe_blob_nbytes((1, *stat.shape[1:]), (precision,), None) - _MFE_HDR.size
                choices.append(EwCandidate(EwItemKey(stat.name, stat.layer, stat.projection, expert), name,
                    precision, size * 8, '', 0, 0., 0., size * 8 / math.prod(stat.shape[1:])))
    expected = {EwItemKey(stat.name, stat.layer, stat.projection, expert) for stat in statistics for expert in range(stat.shape[0])}
    if expected != {choice.key for choice in choices}:
        raise ValueError('selected candidates do not cover every expert native format; include NVQ/NINT candidates')
    return alphaq_candidates(replace(table, candidates=tuple(choices)))


def workbench_dense_candidates(source, statistics, plans, names):
    from mfq.formats.nint import NintSpec
    choices = list(dense_choices(statistics, {item.name: source.native_dtype(item) for item in plans},
        tuple(name for name in names if name in ALPHAQ_PROFILES), native_plans={item.name: item for item in plans}))
    for stat, item in zip(statistics, plans, strict=True):
        for name in names:
            precision = source.sq_precision(item, name)
            if precision is None:
                continue
            plan = replace(item, target_dtype=precision.family, target_precision=precision, target_spec=None)
            bits = int(precision.option('q', 4))
            exact = bits == (4 if precision.family == 'MXFP4-SQ' else 8)
            choices.append(DenseChoice(EwItemKey(stat.name, stat.layer, stat.projection, 0), name,
                8 * q._plan_blob_nbytes(plan, NintSpec()), 0. if exact else 2. ** (-2 * bits), precision))
    return tuple(choices)


def recipe_arguments(target_bpw, candidates):
    try:
        names = tuple(candidates)
    except TypeError as error:
        raise RecipeValidationError('invalid_candidates', 'Candidates must be a list of canonical format names.') from error
    if not names:
        raise RecipeValidationError('quantization_empty_candidates', 'Select at least one source-compatible candidate.')
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names) or set(names) - set(WORKBENCH_CANDIDATES):
        raise RecipeValidationError('invalid_candidates', 'Select distinct canonical NVQ/NINT/native-SQ candidates.')
    try:
        target = Decimal(str(target_bpw))
    except InvalidOperation as error:
        raise RecipeValidationError('quantization_invalid_target', 'Target bpw must be finite and in (0, 32].') from error
    if not target.is_finite() or not 0 < target <= 32:
        raise RecipeValidationError('quantization_invalid_target', 'Target bpw must be finite and in (0, 32].')
    return names, target


def prepare_recipe(source, names, target, *, check_target=True):
    from mfq.formats.io import _MFE_HDR
    from mfq.formats.nint import NINT_K_SELECTOR_BITS, NINT_Q_SELECTOR_BITS
    incompatible = set(names) - set(source.eligible_candidates())
    if incompatible:
        raise RecipeValidationError('quantization_incompatible_candidates',
            'Candidates require matching native source weights: ' + ', '.join(sorted(incompatible)))
    experts, dense, ple = [], [], []
    for item in source.plans:
        descriptor = describe_tensor(item.source_name or item.name, item.shape, item.source_dtype, canonical_name=item.name)
        if descriptor.scope == TensorScope.PLE:
            ple.append(item)
        elif descriptor.role == TensorRole.ROUTED_EXPERT and len(item.shape) == 3:
            experts.append(item)
        elif len(item.shape) == 2 and item.source_dtype in SOURCE_FLOAT_DTYPES:
            dense.append(item)
    if not experts and not dense:
        raise RecipeValidationError('quantization_no_matrices', 'Source has no allocatable matrices.')
    parameters = source.parameters()
    source_bpw = source.storage_bits() / parameters
    scope = AlphaQModelScope(source.plans, tuple(experts), tuple(dense), (), tuple(ple), {}, parameters)
    policy = {'method': 'DF-V1-AlphaQ', 'target_bpw': f'{float(target):.17e}', 'candidates': list(names), 'calibration_tokens': 0}
    header, assets = source.metadata(policy)
    dense_tags = {item.name: max(['NINT', source.native_dtype(item), *(
        precision.family for name in names if (precision := source.sq_precision(item, name)) is not None)], key=len) for item in dense}
    budget = full_file_budget(scope, header, assets, target_bpw=target, dense_dtype_tags=dense_tags, check_capacity=False)
    nint_profiles = sum(name.startswith('NINT') for name in names)
    packing_slack = ((1 << NINT_Q_SELECTOR_BITS) - 1 + 2 * ((1 << NINT_K_SELECTOR_BITS) - 1)) * (
        len(experts) * nint_profiles + (len(dense) if nint_profiles else 0))
    budget = replace(budget, fixed_file_bytes=budget.fixed_file_bytes + packing_slack)
    minimum = maximum = budget.fixed_file_bytes * 8
    base = tuple(name for name in names if name in ALPHAQ_PROFILES)
    for item in experts:
        stat = AlphaQTensorStatistics(item.name, 0, 'expert', (1, *item.shape[1:]), (1.,), (1.,))
        choices = alphaq_builtin_candidates((stat,), base).candidates if base else ()
        pools = {entry.pool_key: entry.pool_storage_bits for entry in choices}
        minimum += _MFE_HDR.size * 8 + sum(pools.values())
        maximum += _MFE_HDR.size * 8
        used_pools = set()
        for expert in range(item.shape[0]):
            options = [(entry.variable_storage_bits, entry.distortion, entry.pool_key) for entry in choices]
            for name in names:
                precision = source.sq_precision(item, name, expert)
                if precision is not None:
                    bits = int(precision.option('q', 4))
                    exact = bits == (4 if precision.family == 'MXFP4-SQ' else 8)
                    size = q._mixed_moe_blob_nbytes((1, *item.shape[1:]), (precision,), None) - _MFE_HDR.size
                    options.append((size * 8, 0. if exact else 2. ** (-2 * bits), ''))
            if not options:
                raise RecipeValidationError('quantization_candidate_coverage',
                    'Selected candidates do not cover every expert native format; include NVQ/NINT candidates.')
            minimum += min(entry[0] for entry in options)
            best = min(options, key=lambda entry: (entry[1], entry[0], entry[2]))
            maximum += best[0]
            used_pools.add(best[2])
        maximum += sum(pools.get(key, 0) for key in used_pools)
    for item in dense:
        stat = AlphaQTensorStatistics(item.name, 0, 'dense', (1, *item.shape), (1.,), (1.,))
        choices = workbench_dense_candidates(source, (stat,), (item,), names)
        minimum += min(entry.variable_storage_bits for entry in choices)
        maximum += min(choices, key=lambda entry: (entry.distortion, entry.variable_storage_bits)).variable_storage_bits
    details = {'target_bpw': float(target), 'minimum_bpw': minimum / parameters,
        'maximum_bpw': maximum / parameters, 'source_bpw': source_bpw,
        'minimum_file_bytes': (minimum + 7) // 8, 'maximum_file_bytes': maximum // 8,
        'source_file_bytes': source.storage_bits() // 8}
    if not check_target:
        return scope, budget, policy, details
    if budget.maximum_file_bytes > details['source_file_bytes']:
        raise RecipeValidationError('quantization_target_above_source',
            f'Target {target} bpw exceeds the source model average precision {source_bpw:.6f} bpw.', details)
    if details['minimum_file_bytes'] > min(details['maximum_file_bytes'], details['source_file_bytes'], parameters * 4):
        raise RecipeValidationError('quantization_no_feasible_budget',
            'The selected candidates have no feasible budget within the source precision ceiling; change the candidate set.', details)
    if budget.maximum_file_bytes * 8 < minimum:
        raise RecipeValidationError('quantization_target_below_candidates',
            f'Target {target} bpw is below the selected candidate minimum budget {minimum / parameters:.6f} bpw.', details)
    if budget.maximum_file_bytes * 8 > maximum:
        raise RecipeValidationError('quantization_target_above_candidates',
            f'Target {target} bpw exceeds the highest-precision candidate plan {maximum / parameters:.6f} bpw.', details)
    return scope, budget, policy, details


def validate_recipe(model, target_bpw, candidates):
    names, target = recipe_arguments(target_bpw, candidates)
    source = ModelSource(model)
    try:
        return prepare_recipe(source, names, target)[3]
    finally:
        source.close()


def generate_recipe(model, output, target_bpw, candidates, *, backend='auto', gate=None):
    gate = gate or BatchGate()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    names, target = recipe_arguments(target_bpw, candidates)
    source = ModelSource(model)
    try:
        scope, budget, policy, _ = prepare_recipe(source, names, target)
        experts, dense, parameters = scope.experts, scope.dense, scope.model_weight_count
        device = resolve_quant_backend(backend).device
        if device == 'mps':
            device = 'cpu'
        statistics = []
        for index, item in enumerate((*experts, *dense)):
            gate.tensor = item.name
            shape = item.shape if len(item.shape) == 3 else (1, *item.shape)
            alpha, variance = [], []
            with source.rows(item, gate) as reader:
                for expert in range(shape[0]):
                    class Matrix:
                        def read_rows(self, start, end, *, device):
                            return reader.read_rows(expert * shape[1] + start, expert * shape[1] + end, device=device)
                    a, v = matrix_statistics(Matrix(), shape[1], shape[2], device, gate)
                    alpha.append(a); variance.append(v)
            match = re.match(r'model\.block\.(\d+)\.', item.name)
            statistics.append(AlphaQTensorStatistics(item.name, int(match[1]) if match else 0,
                item.name.split('.')[-2] if len(item.shape) == 3 else 'dense', shape, tuple(alpha), tuple(variance)))
            emit(event='workbench_progress', phase='statistics', completed=index + 1, total=len(experts) + len(dense), tensor=item.name)
        expert_stats, dense_stats = statistics[:len(experts)], statistics[len(experts):]
        expert_choices = workbench_expert_candidates(source, expert_stats, experts, names)
        topk = source.config.get('text_config', source.config).get('num_experts_per_tok', 1)
        allocation = allocate_joint(expert_stats, expert_choices, dense_stats,
            workbench_dense_candidates(source, dense_stats, dense, names),
            model_weight_count=parameters, maximum_file_bytes=budget.maximum_file_bytes, fixed_file_bytes=budget.fixed_file_bytes,
            expert_exposure={item.name: min(1., topk / item.shape[0]) for item in experts})
        scheme = data_free_scheme(allocation.scheme)
        metadata = {**scheme.metadata, 'workbench': {**policy, 'source': source.initial_identity,
            'tensor_shapes': {item.name: list(item.shape) for item in source.plans},
            'native_overrides': allocation.native_overrides, 'maximum_file_bytes': budget.maximum_file_bytes,
            'estimated_bpw': allocation.report['model_bpw'], 'parameters': parameters}}
        scheme = replace(scheme, metadata=metadata, target_profile='DF-V1-AlphaQ')
        if source.identity() != source.initial_identity:
            raise ValueError('source changed during allocation')
        save_scheme(output, scheme)
        result = {'output': str(output), 'method': 'DF-V1-AlphaQ', 'target_bpw': float(target),
            'estimated_bpw': allocation.report['model_bpw'], 'tensors': len(scheme.selections) + len(scheme.expert_selections)}
        emit(event='workbench_result', **result)
        return result
    finally:
        source.close()


def load_recipe(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate recipe key: {key}')
            result[key] = value
        return result
    json.loads(Path(path).read_text(), object_pairs_hook=unique)
    scheme = load_scheme(path)
    for item in [*scheme.selections.values(), *(entry for bank in scheme.expert_selections.values() for entry in bank.selections)]:
        if not math.isfinite(item.train_loss) or not math.isfinite(item.validation_loss) or min(item.train_loss, item.validation_loss) < 0:
            raise ValueError('recipe losses must be finite and nonnegative')
        artifact = item.descriptor.artifact
        if artifact and not (Path(scheme.path).parent / artifact).resolve().is_file():
            raise ValueError('recipe references an unavailable quantizer artifact')
    return scheme


def recipe_plans(source, scheme):
    shapes = {item.name: item.shape for item in source.plans}
    workbench = scheme.metadata.get('workbench', {})
    if workbench.get('source') and workbench['source'] != source.initial_identity:
        raise ValueError('recipe was generated for a different source checkpoint')
    if workbench.get('tensor_shapes') and workbench['tensor_shapes'] != {name: list(shape) for name, shape in shapes.items()}:
        raise ValueError('recipe tensor inventory differs from this model')
    for name, item in scheme.selections.items():
        if shapes.get(name) != (item.rows, item.columns):
            raise ValueError(f'recipe tensor shape differs: {name}')
    for name, item in scheme.expert_selections.items():
        if shapes.get(name) != (item.n_experts, item.rows_per_expert, item.columns):
            raise ValueError(f'recipe expert shape differs: {name}')
    overrides = workbench.get('native_overrides', {})
    if not isinstance(overrides, dict) or set(overrides) - shapes.keys() or any(value not in NATIVE_DTYPES for value in overrides.values()):
        raise ValueError('invalid native tensor overrides')
    if set(overrides) & (scheme.selections.keys() | scheme.expert_selections.keys()):
        raise ValueError('native overrides conflict with quantized selections')
    return source.plan(scheme, overrides)


def fit_model(model, recipe, output, *, imatrix=None, backend='auto', row_chunk=1024, gate=None, cache=None):
    gate = gate or BatchGate()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = ModelSource(model)
    try:
        scheme = load_recipe(recipe)
        plans = recipe_plans(source, scheme)
        workbench = scheme.metadata.get('workbench', {})
        data_free = workbench.get('method') == 'DF-V1-AlphaQ'
        if data_free and imatrix:
            raise ValueError('DF-V1-AlphaQ does not consume an imatrix')
        if not imatrix and any(q._hf_plan_requires_imatrix(item) for item in plans):
            raise ValueError('this recipe requires an imatrix; import one before fitting')
        matrix = load_importance_matrix(imatrix) if imatrix else ImportanceMatrix(None, {}, (), 0, 0, False)
        policy = {key: workbench[key] for key in ('method', 'target_bpw', 'candidates', 'calibration_tokens')} if data_free else {
            'method': scheme.target_profile, 'recipe_sha256': hashlib.sha256(Path(recipe).read_bytes()).hexdigest()}
        header, assets = source.metadata(policy)
        directory = Path(cache or output.parent / (output.name + '.fit-cache'))
        backend_config = resolve_quant_backend(backend)
        payloads = FittedPayloadCache(source.path, directory, matrix,
            hashlib.sha256(Path(imatrix).read_bytes()).hexdigest() if imatrix else 'data-free', backend_config,
            row_chunk=row_chunk, artifact_root=Path(recipe).parent,
            source_factory=lambda item: source.rows(item, gate), identity=source.identity, data_free=data_free, adaptive_rows=True)
        records = []
        for index, item in enumerate(plans):
            gate.tensor = item.name
            gate(8)
            records.append(payloads.require(item))
            emit(event='workbench_progress', phase='fitting', completed=index + 1, total=len(plans), tensor=item.name)
        from mfq.calibration.alphaq_payloads import EncodedRecord
        records.extend(EncodedRecord(asset.name, ASSET_DTYPE, len(asset.data), lambda handle, data=asset.data: handle.write(data)) for asset in assets)
        header = replace(header, num_tensors=len(records))
        buffer = io.BytesIO(); _write_header_and_table(buffer, header, records)
        expected = buffer.tell() + sum(item.nbytes for item in records)
        maximum = workbench.get('maximum_file_bytes')
        if maximum is not None and expected > maximum:
            raise ValueError(f'fitted model exceeds recipe byte budget: {expected} > {maximum}')
        def validate(paths):
            payloads.validate()
            if source.identity() != source.initial_identity or len(paths) != 1 or paths[0].stat().st_size != expected:
                raise ValueError('source or output changed during assembly')
        gate(8)
        emit(event='workbench_progress', phase='assembly', completed=len(plans), total=len(plans))
        write_blob_record_shards(output, header, records, before_publish=validate)
        parameters = source.parameters(plans)
        result = {'output': str(output), 'total_bytes': expected, 'actual_bpw': expected * 8 / parameters, 'tensors': len(plans)}
        emit(event='workbench_result', **result)
        return result
    finally:
        source.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['recipe', 'fit'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-bpw', type=float)
    parser.add_argument('--candidates', default=','.join(ALPHAQ_PROFILES))
    parser.add_argument('--recipe')
    parser.add_argument('--imatrix')
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--row-chunk', type=int, default=1024)
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args(argv)
    gate = BatchGate(args.interactive)
    torch.set_num_threads(2 if args.interactive else max(2, torch.get_num_threads()))
    if args.operation == 'recipe':
        generate_recipe(args.input, args.output, args.target_bpw, args.candidates.split(','), backend=args.backend, gate=gate)
    else:
        fit_model(args.input, args.recipe, args.output, imatrix=args.imatrix, backend=args.backend, row_chunk=args.row_chunk, gate=gate)


if __name__ == '__main__':
    main()
