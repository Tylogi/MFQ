"""Consume identity-checked encoded candidates through the normal MFQ writer."""
from __future__ import annotations

import hashlib
import json
import mmap
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mfq.calibration.alphaq_cache import source_identity
from mfq.calibration.alphaq_output import file_sha256
from mfq.calibration.alphaq_source import _progress
from mfq.calibration.artifact import precision_document
from mfq.formats.io import _MFE_HDR, _MFE_POOL_HDR, _MFE_MAGIC
from mfq.formats.shards import copy_sparse_range
from mfq.quantize.candidate_streams import layout, write_merged


def precision_identity(precision):
    value = precision_document(precision)
    artifact = value.pop('artifact', None)
    value['artifact_sha256'] = file_sha256(Path(artifact)) if artifact else None
    return value


def plan_identity(item):
    """Bind both the source row view and the complete fitting precision policy."""
    value = asdict(item)
    value.pop('expert_precisions', None)
    value.pop('target_precision', None)
    # Recipe labels can differ when aliases have the same actual precision.
    value.pop('gguf_type', None)
    value['precision'] = precision_identity(item.target_precision) if item.target_precision else None
    return json.loads(json.dumps(value))


def checked_file(part):
    path = Path(part['path'])
    stat = path.stat()
    if stat.st_size != part['bytes'] or stat.st_mtime_ns != part['mtime_ns']:
        raise ValueError(f'encoded candidate file changed: {path}')
    return path


@dataclass(frozen=True)
class EncodedRecord:
    name: str
    dtype: str
    nbytes: int
    emitter: object
    path: Path = Path('.')

    def write_to(self, target):
        self.emitter(target)
        _progress(stage='alphaq_cached_payload', tensor=self.name, bytes=self.nbytes)


class PayloadCache:
    """Cache entries are bound to source, imatrix, precision, tables and files.

    File SHA-256 and producer evidence are retained in the manifest. Reuse
    checks the size and modification time pinned by the producer's completed
    checksum audit; a changed file fails instead of silently refitting it.
    """
    def __init__(self, path, root, imatrix_sha256):
        self.path = Path(path)
        self.doc = json.loads(self.path.read_text())
        if (self.doc.get('format') != 'mfq.alphaq-encoded-candidates.v1'
                or self.doc.get('source') != source_identity(root)
                or self.doc.get('imatrix_sha256') != imatrix_sha256):
            raise ValueError('encoded candidate cache belongs to another source or imatrix')
        self.digest = file_sha256(self.path)

    def require(self, item):
        if item.target_dtype == 'MFE':
            return self._experts(item)
        choices = self.doc.get('fixed', {}).get(item.name, [])
        identity = plan_identity(item)
        matches = [c for c in choices if c['identity'] == identity]
        if len(matches) != 1:
            raise ValueError(f'encoded cache has no unique matching tensor policy: {item.name}')
        entry = matches[0]
        part = entry['file']; checked_file(part)
        offset, size = int(entry.get('offset', 0)), int(entry['nbytes'])
        if offset < 0 or size <= 0 or offset+size > part['bytes']:
            raise ValueError('encoded fixed tensor range is invalid')
        def emit(target):
            with checked_file(part).open('rb') as source:
                copy_sparse_range(source, target, size, offset=offset)
            checked_file(part)
        return EncodedRecord(item.name, entry['dtype'], size, emit)

    def _experts(self, item):
        saved = self.doc['experts'].get(item.name)
        if saved is None or tuple(saved['shape']) != item.shape:
            raise ValueError(f'encoded cache expert scope differs: {item.name}')
        if item.expert_precisions is None or len(item.expert_precisions) != item.shape[0]:
            raise ValueError('expert precision count differs')
        groups, identities = defaultdict(list), {}
        for expert, precision in enumerate(item.expert_precisions):
            profile = f'NINT{precision.nint_spec.bits}' if precision.nint_spec else precision.family
            groups[profile].append(expert)
            if profile not in identities:
                identities[profile] = precision_identity(precision)
            elif precision != item.expert_precisions[groups[profile][0]]:
                raise ValueError('same candidate profile has conflicting precision policies')
        pools = []
        experts, rows, columns = item.shape
        for profile, ids in sorted(groups.items()):
            candidate = saved['profiles'].get(profile)
            if candidate is None or candidate['precision'] != identities[profile]:
                raise ValueError(f'encoded expert precision/tables differ: {item.name}/{profile}')
            pieces, gathered, description, dtype = [], [], None, None
            for part in candidate['parts']:
                selected = [i for i in ids if part['first'] <= i < part['stop']]
                if not selected:
                    continue
                path = checked_file(part)
                with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    current = layout(mapped, part['dtype'])
                    if current.rows != (part['stop']-part['first'])*rows or current.columns != columns:
                        raise ValueError('encoded expert chunk shape differs')
                    if description is None:
                        description, dtype = current, part['dtype']
                    elif current.header(0) != description.header(0) or part['dtype'] != dtype:
                        raise ValueError('encoded expert chunks contain different tables or layouts')
                pieces.append((part, [i-part['first'] for i in selected])); gathered.extend(selected)
            if gathered != ids:
                raise ValueError('encoded expert chunks omit or reorder experts')
            if any(rows*b % 8 for b in description.field_bits):
                raise ValueError('encoded expert fields require byte-aligned expert boundaries')
            size = len(description.prefix)+sum(len(ids)*rows*b//8 for b in description.field_bits)
            pools.append((ids, dtype, size, pieces))
        size = _MFE_HDR.size+sum(_MFE_POOL_HDR.size+4*len(ids)+len(dtype)+size
                                 for ids, dtype, size, _ in pools)
        def emit(target):
            target.write(_MFE_HDR.pack(_MFE_MAGIC, experts, rows, columns, len(pools)))
            for ids, dtype, size, pieces in pools:
                target.write(_MFE_POOL_HDR.pack(len(ids),len(dtype),size,0))
                target.write(np.asarray(ids,dtype='<i4').tobytes());target.write(dtype.encode('ascii'))
                with ExitStack() as stack:
                    cohorts = []
                    for part, selected in pieces:
                        stream = stack.enter_context(checked_file(part).open('rb'))
                        mapped = stack.enter_context(mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ))
                        cohorts.append((mapped,part['dtype'],selected))
                    if write_merged(target,cohorts,rows) != size:
                        raise ValueError('encoded expert output size differs')
                    del cohorts,mapped
                for part, _ in pieces:
                    checked_file(part)
        return EncodedRecord(item.name,'MFE',size,emit)

    def validate(self):
        if file_sha256(self.path) != self.digest:
            raise ValueError('encoded candidate manifest changed during conversion')
