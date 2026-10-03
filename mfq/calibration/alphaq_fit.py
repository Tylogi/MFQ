"""Recoverable selected-weight fitting for normal AlphaQ quantization."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from mfq.calibration.alphaq_cache import rows_for_plan, source_identity
from mfq.calibration.alphaq_output import file_sha256
from mfq.calibration.alphaq_payloads import checked_file, plan_identity, precision_identity
from mfq.calibration.alphaq_source import _atomic_json, _progress
from mfq.formats.compat import canonical_dtype
from mfq.formats.nint import NintSpec


class FittedPayloadCache:
    """Keep completed tensors across failure and assemble the final file once.

    Automatic row chunks start at the full tensor geometry. Only an actual
    allocation failure reduces them. The normal converter owns all codecs.
    """

    def __init__(self, root, directory, imatrix, imatrix_sha256, backend,
                 *, row_chunk=0, artifact_root=None):
        if row_chunk < 0:
            raise ValueError('row chunk must be non-negative')
        self.root, self.directory = Path(root), Path(directory)/'fitted'
        self.imatrix, self.imatrix_sha256 = imatrix, imatrix_sha256
        self.backend, self.row_chunk = backend, int(row_chunk)
        self.artifact_root = artifact_root
        self.source = source_identity(self.root)
        self.checked = []
        self.verified = {}

    def accept_verified(self, item, record, signature, info):
        """Consume a just-verified worker result without hashing its payload twice."""
        if signature != self.signature(item) or record.name != item.name:
            raise ValueError('AlphaQ worker payload identity differs')
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        expected = self.directory/key/'tensor.blob'
        if (record.path != expected or Path(info['path']) != expected
                or record.nbytes != info['bytes']):
            raise ValueError('AlphaQ worker payload path or size differs')
        checked_file(info)
        self.verified[item.name] = (item, record, signature, info)

    def signature(self, item):
        identities = {}
        for index, precision in enumerate(item.expert_precisions or ()):
            if precision not in identities:
                identities[precision] = dict(precision=precision_identity(precision), experts=[])
            identities[precision]['experts'].append(index)
        return dict(format='mfq.alphaq-fitted-tensor.v1', source=self.source,
                    imatrix_sha256=self.imatrix_sha256, plan=plan_identity(item),
                    expert_pools=list(identities.values()))

    def require(self, item):
        from mfq.tools import quantize_hf_to_mfq as q

        signature = self.signature(item)
        if item.name in self.verified:
            saved_item, record, saved_signature, info = self.verified[item.name]
            if item != saved_item or signature != saved_signature:
                raise ValueError('AlphaQ worker payload plan changed before writing')
            checked_file(info)
            self.checked.append(info)
            return record
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        directory = self.directory/key
        marker, payload = directory/'complete.json', directory/'tensor.blob'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved.get('signature') != signature or saved.get('complete') is not True:
                raise ValueError(f'AlphaQ fitted tensor checkpoint identity differs: {item.name}')
            checked_file(saved['file'])
            if file_sha256(payload) != saved['file']['sha256']:
                raise ValueError(f'AlphaQ fitted tensor checkpoint checksum differs: {item.name}')
            _progress(stage='alphaq_fit_reused', tensor=item.name, bytes=saved['file']['bytes'])
        else:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory/'tensor.partial'
            binding = q._bind_hf_imatrix(self.imatrix, [item]).get(item.name)
            rows = math.prod(item.shape[:-1]) if len(item.shape) > 1 else item.shape[0]
            chunk = self.row_chunk or max(8, (rows+7)//8*8)
            failures, started = [], time.monotonic()
            while True:
                try:
                    with rows_for_plan(self.root, item) as source:
                        dtype, size = self._fit(item, source, binding, temporary, chunk)
                except torch.OutOfMemoryError:
                    if chunk <= 8 or self.row_chunk:
                        raise
                    failures.append(chunk)
                else:
                    break
                # Release the failed call's traceback and temporaries before
                # retrying the same tensor; completed checkpoints are untouched.
                chunk = max(8, chunk//16*8)
                gc.collect()
                if self.backend.name == 'cuda':
                    with torch.cuda.device(self.backend.device):
                        torch.cuda.empty_cache()
                elif self.backend.name == 'metal':
                    torch.mps.empty_cache()
                _progress(stage='alphaq_fit_oom', tensor=item.name,
                          failed_rows=failures[-1], next_rows=chunk)
            if temporary.stat().st_size != size:
                raise ValueError(f'AlphaQ fitted tensor size differs: {item.name}')
            if self.signature(item) != signature or source_identity(self.root) != self.source:
                raise ValueError('AlphaQ source or tables changed during weight fitting')
            os.replace(temporary, payload)
            info = payload.stat()
            saved = dict(complete=True, signature=signature, dtype=dtype,
                         file=dict(path=str(payload), bytes=size, mtime_ns=info.st_mtime_ns,
                                   sha256=file_sha256(payload)), row_chunk=chunk,
                         oom_rows=failures, elapsed_seconds=time.monotonic()-started)
            _atomic_json(marker, saved)
            _progress(stage='alphaq_fit_complete', tensor=item.name, bytes=size,
                      row_chunk=chunk, elapsed_seconds=saved['elapsed_seconds'])
        self.checked.append(saved['file'])
        return q.BlobRecord(item.name, saved['dtype'], saved['file']['bytes'], payload)

    def _fit(self, item, source, binding, path, chunk):
        from mfq.tools import quantize_hf_to_mfq as q

        name, device = self.backend.name, self.backend.device
        if item.target_dtype == 'MFE':
            size = q._write_mixed_moe_axis0_blob(source, item.shape, item.expert_shape,
                item.expert_precisions, path, chunk, name, device, self.artifact_root,
                importance=q._hf_expert_importance(item, binding),
                neuron_importance=q._hf_neuron_importance(item, binding))
            dtype = 'MFE'
        elif item.target_dtype.startswith('NINT'):
            size = q._write_nint_axis0_blob(source, item.shape, q._spec_for_plan(item, NintSpec()),
                path, chunk, name, device,
                importance_rows=None if binding is None else binding.input_rows or binding.rows,
                neuron_importance_rows=None if binding is None else binding.neuron_rows,
                allocation_group_rows=None if binding is None else binding.allocation_group_rows)
            dtype = q._nint_blob_public_dtype(path)
        elif item.target_precision is not None and item.target_dtype.startswith('NVQ'):
            size = q._write_flat_family_axis0_blob(source, item.shape, item.target_precision,
                path, chunk, name, device, self.artifact_root,
                importance_rows=None if binding is None else binding.input_rows or binding.rows)
            dtype = canonical_dtype(item.target_dtype)
        elif len(item.shape) >= 2 and item.target_dtype in ('BF16', 'F16', 'F32'):
            size = q._write_dense_axis0_blob(source, item.shape, path, item.target_dtype, chunk)
            dtype = item.target_dtype
        else:
            value = source.tensor()
            if item.row_start is not None or item.row_end is not None:
                value = value[item.row_start:item.row_end]
            size = q._dense_blob_from_tensor(value, path, item.target_dtype)
            dtype = item.target_dtype
        return dtype, size

    def validate(self):
        if source_identity(self.root) != self.source:
            raise ValueError('AlphaQ source changed during weight fitting')
        for part in self.checked:
            checked_file(part)
