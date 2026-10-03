"""Reusable statistics and trained tables for the normal AlphaQ command."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import errno
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from mfq.calibration.alphaq import METHOD, AlphaQTensorStatistics, alphaq_weight_statistics
from mfq.calibration.alphaq_output import file_sha256, model_files
from mfq.calibration.alphaq_source import _atomic_json, _progress


@contextmanager
def cache_lock(path):
    """Coordinate commands sharing one cache; the OS releases failed owners."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'\0'); stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(.1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def rows_for_plan(root, item):
    from mfq.tools import quantize_hf_to_mfq as q
    if item.expert_source_names is not None:
        raw = q._SeparateExpertRowSource(root, item.expert_shape or item.shape,
            item.expert_source_names, item.expert_source_shards,
            item.expert_source_quantizations, item.expert_source_scale_names,
            item.expert_source_scale_shards, item.expert_source_scale_dtypes)
    else:
        raw = q._raw_source_for_plan(root, item)
    try:
        source = raw
        if item.transform and item.transform.startswith('expert_'):
            source = q._PackedExpertProjectionSource(raw, item)
        elif len(item.shape) == 2:
            source = q._HfPlanRowSource(raw, item)
        yield source
    finally:
        if hasattr(raw, 'close'):
            raw.close()


def source_identity(root):
    return {'model': str(root.resolve()), 'files': [list(x) for x in model_files(root)]}


def collect_statistics(root, plans, cache, device, *, mapper=None):
    """Resume independent complete banks/matrices; never sample spectral blocks."""
    import torch
    path = cache/'statistics.json'
    identity = {**source_identity(root), 'method': METHOD, 'collector': 'canonical-full-v1'}
    saved = json.loads(path.read_text()) if path.exists() else {
        'format': 'mfq.alphaq-model-statistics.v1', 'identity': identity, 'tensors': {}}
    if saved.get('format') != 'mfq.alphaq-model-statistics.v1' or saved.get('identity') != identity:
        raise ValueError('AlphaQ statistics cache belongs to another model or method')
    if mapper is not None:
        missing = [item for item in plans if item.name not in saved['tensors']]
        if missing:
            # Validate existing entries before starting any missing GPU work.
            collect_statistics(root, [p for p in plans if p.name in saved['tensors']], cache, device)
            for item, value in mapper('statistics', missing):
                if value.name != item.name:
                    raise ValueError('AlphaQ worker returned statistics for another tensor')
                saved['tensors'][item.name] = asdict(value)
                _atomic_json(path, saved)
            return collect_statistics(root, plans, cache, device)
    results, started = [], time.monotonic()
    for item in plans:
        shape = item.shape if len(item.shape) == 3 else (1, *item.shape)
        match = re.match(r'^model\.block\.(\d+)\.', item.name)
        layer = int(match[1]) if match else 0
        projection = item.name.split('.')[-2] if len(item.shape) == 3 else 'dense'
        if item.name in saved['tensors']:
            doc = saved['tensors'][item.name]
            value = AlphaQTensorStatistics(doc['name'], doc['layer'], doc['projection'],
                tuple(doc['shape']), tuple(doc['alpha']), tuple(doc['variance']))
            if (value.name, value.layer, value.projection, value.shape) != (item.name, layer, projection, shape):
                raise ValueError(f'AlphaQ statistics identity mismatch: {item.name}')
        else:
            alpha, variance, start, batch = [], [], 0, shape[0]
            with rows_for_plan(root, item) as source:
                while start < shape[0]:
                    count = min(batch, shape[0]-start)
                    try:
                        a, v = _collect(source, start, count, shape, device)
                    except torch.OutOfMemoryError:
                        if count == 1:
                            raise
                        batch = max(1, count//2)
                        continue
                    alpha.extend(a.tolist()); variance.extend(v.tolist()); start += count
            value = AlphaQTensorStatistics(item.name, layer, projection, shape, tuple(alpha), tuple(variance))
            saved['tensors'][item.name] = asdict(value)
            _atomic_json(path, saved)
        results.append(value)
        _progress(stage='alphaq_statistics', completed=len(results), total=len(plans),
                  tensor=item.name, elapsed_seconds=time.monotonic()-started)
    return tuple(results)


def _collect(source, start, count, shape, device):
    weight = source.read_rows(start*shape[1], (start+count)*shape[1], device=device)
    return alphaq_weight_statistics(weight.reshape(count, *shape[1:]))


def training_rows(item, seed=20260716):
    if len(item.shape) == 3:
        match = re.fullmatch(r'model\.block\.(\d+)\.mlp\.experts\.(\w+)\.weight', item.name)
        if match is None:
            raise ValueError(f'unsupported canonical expert identity: {item.name}')
        name = f'blk.{match[1]}.ffn_{match[2]}_exps.weight'
    else:
        name = item.name
    integer = int.from_bytes(hashlib.blake2b(f'{seed}:{name}'.encode(), digest_size=8).digest(), 'little')
    rng = np.random.default_rng(integer)
    if len(item.shape) == 3:
        experts, rows, _ = item.shape
        rng.permutation(experts); rng.permutation(experts)
        # Sixteen rows from each expert, matching the evaluated tensor tables.
        return np.sort(np.concatenate([e*rows+rng.permutation(rows)[:16] for e in range(experts)]))
    return np.sort(rng.permutation(item.shape[0])[:min(8192, item.shape[0])])


def jsc_config(profile):
    from mfq.quantize.nvq_jsc import NvqJscConfig
    from mfq.tools import quantize_hf_to_mfq as q
    return NvqJscConfig(banks=4 if profile.startswith('NVQ2') else 2,
        iterations=4, assignment_refine_steps=2, search_steps=19,
        learned_scale_lut=profile.startswith('NVQ2'), group_chunk=4096,
        raw_multiplier=8, seed=20260716, codebook_storage='int8', spec=q._NVQ_SPECS[profile])


def table_signature(root, item, profile, imatrix_sha256, *, source=None, rows_sha256=None):
    # JSON normalization also turns nested immutable tuples into cache arrays.
    return json.loads(json.dumps(dict(source=source if source is not None else source_identity(root), tensor=item.name,
        shape=item.shape, profile=profile, config=asdict(jsc_config(profile)),
        training_row_ids_sha256=(rows_sha256 if rows_sha256 is not None else
                                hashlib.sha256(training_rows(item).tobytes()).hexdigest()),
        imatrix_sha256=imatrix_sha256, objective='imatrix_weighted_sse')))


def trained_tables(root, plans, scheme, imatrix, imatrix_sha256, cache, device):
    """Attach trained JSC tables to every selected cohort before normal writing."""
    from mfq.quantize.nvq_jsc import train_nvq_jsc, jsc_tables_from_tensor
    from mfq.tools import quantize_hf_to_mfq as q
    by_name = {p.name: p for p in plans}
    identity = source_identity(root)
    sampled_rows = {}
    trained = {}
    tasks = set()
    for name, selection in scheme.expert_selections.items():
        tasks.update((name, p.family) for p in selection.precisions if 'J' in p.family)
    tasks.update((name, s.descriptor.family) for name, s in scheme.selections.items()
                 if 'J' in s.descriptor.family)
    for completed, (name, profile) in enumerate(sorted(tasks), 1):
        item = by_name[name]
        path = cache/'codebooks'/imatrix_sha256/name/(profile+'.npz')
        marker = path.with_suffix('.json')
        if name not in sampled_rows:
            sampled_rows[name] = training_rows(item)
        ids = sampled_rows[name]
        signature = table_signature(root, item, profile, imatrix_sha256, source=identity,
                                    rows_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        if marker.exists():
            doc = json.loads(marker.read_text())
            if (doc.get('signature') != signature or not doc.get('complete')
                    or file_sha256(path) != doc.get('sha256')):
                raise ValueError(f'AlphaQ trained table cache mismatch: {name}/{profile}')
        else:
            binding = q._bind_hf_imatrix(imatrix, [replace(item, target_dtype=profile,
                target_precision=None)])[name]
            with rows_for_plan(root, item) as source:
                weight = source.read_rows(ids, device=device)
            fitted, history = train_nvq_jsc(weight, importance=binding.selected(ids),
                                           config=jsc_config(profile), device=device)
            tables = jsc_tables_from_tensor(fitted)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, scale_lut=tables.scale_lut, bank_for_state=tables.bank_for_state,
                         codebooks=tables.codebooks)
            os.replace(temporary, path)
            _atomic_json(marker, dict(complete=True, signature=signature, sha256=file_sha256(path),
                                      history=[asdict(x) for x in history]))
            del weight, fitted, tables
        trained[name, profile] = str(path.resolve())
        _progress(stage='alphaq_trained_tables', completed=completed, total=len(tasks), tensor=name, profile=profile)

    if source_identity(root) != identity:
        raise ValueError('AlphaQ source changed during table training/reuse')

    def descriptor(name, precision):
        artifact = trained.get((name, precision.family))
        if artifact is None:
            return precision
        options = {**dict(precision.options), 'group_chunk': 4096, 'iterations': 4,
                   'assignment_refine_steps': 2, 'search_steps': 19}
        return replace(precision, artifact=artifact, options=tuple(options.items()))

    return replace(scheme,
        expert_selections={name: replace(s, selections=tuple(replace(x,
            precision=descriptor(name, x.descriptor)) for x in s.selections))
            for name, s in scheme.expert_selections.items()},
        selections={name: replace(s, precision=descriptor(name, s.descriptor))
                    for name, s in scheme.selections.items()})
