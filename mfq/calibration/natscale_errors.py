"""Deterministic imatrix-weighted format errors measured with the export codecs."""
from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from mfq.calibration.alphaq import ALPHAQ_PROFILES, AlphaQTensorStatistics, alphaq_builtin_candidates
from mfq.calibration.alphaq_cache import (
    _LatentRows, cache_lock, jsc_config, rows_for_plan, source_identity, trained_tables,
)
from mfq.calibration.alphaq_fit import FittedPayloadCache
from mfq.calibration.alphaq_joint import _imatrix_precision
from mfq.calibration.alphaq_output import file_sha256
from mfq.calibration.alphaq_source import _atomic_json, _progress
from mfq.calibration.artifact import CalibrationScheme, ExpertSelection, ExpertTensorSelection


def error_signature(root, plans, imatrix_sha, profiles, sample_layers, sample_experts):
    if (type(sample_layers) is not int or sample_layers <= 0
            or type(sample_experts) is not int or sample_experts <= 0):
        raise ValueError('NatScale sample sizes must be positive integers')
    if (not profiles or len(set(profiles)) != len(profiles)
            or set(profiles)-set(ALPHAQ_PROFILES)):
        raise ValueError('NatScale errors require distinct canonical candidate profiles')
    package = Path(__file__).parent.parent
    paths = {Path(__file__), *[package/name for name in (
        'calibration/alphaq_fit.py', 'calibration/alphaq_cache.py',
        'calibration/alphaq.py', 'calibration/evaluator.py', '_alphaq_profiles.py',
        'tools/quantize_hf_to_mfq.py', 'quantize/expert_nint.py',
        'quantize/nint_quant.py', 'quantize/nint_quant_torch.py',
        'quantize/nvq_jsc.py', 'quantize/nvq1_s_quant.py', 'quantize/nvq1_l_quant.py')]}
    paths.update((package/'formats').glob('*.py'))
    paths.update((package/'quantize').glob('*.py'))
    return dict(source=source_identity(root), imatrix_sha256=imatrix_sha,
        tensors=[dict(name=p.name, shape=list(p.shape)) for p in plans], profiles=list(profiles),
        sample_layers=sample_layers, sample_experts=sample_experts, seed=20260716,
        tables={p: asdict(jsc_config(p)) for p in profiles if 'J' in p},
        fitting='normal MFQ export codecs; NINT imatrix_weighted; full-bank JSC tables',
        implementation={str(p.relative_to(package)): file_sha256(p) for p in sorted(paths)})


def load_errors(path, root, plans, imatrix_sha, profiles, sample_layers, sample_experts):
    path = Path(path)
    if not path.exists():
        raise ValueError(f'measured NatScale errors are missing: {path}; run mfq calibrate natscale-errors')
    doc = json.loads(path.read_text())
    signature = error_signature(root, plans, imatrix_sha, profiles, sample_layers, sample_experts)
    if (doc.get('format') != 'mfq.natscale-errors.v1' or doc.get('signature') != signature
            or doc.get('complete') is not True):
        raise ValueError('NatScale error cache is incomplete or belongs to different inputs/settings/codecs')
    errors = doc['errors']
    layers = sorted({int(p.name.split('.')[2]) for p in plans})
    expected = set(int(x) for x in np.array(layers)[even_indices(len(layers), sample_layers)])
    if {int(l) for l in errors} != expected:
        raise ValueError('NatScale error cache has missing sampled layers')
    for projections in errors.values():
        if set(projections) != {'gate', 'up', 'down'}:
            raise ValueError('NatScale errors must cover all three projections')
        for values in projections.values():
            if set(values) != set(profiles) or any(not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError('NatScale errors must be finite, nonnegative and cover every profile')
    return errors


def even_indices(count, samples):
    return np.unique(np.rint(np.linspace(0, count-1, min(count, samples))).astype(int))


def _scheme(item, precision):
    n, rows, columns = item.shape
    selected = ExpertTensorSelection(item.name, item.name, n, rows, columns,
        tuple(ExpertSelection(e, precision.nint_spec, 8, 0., 0., precision) for e in range(n)))
    return CalibrationScheme(None, 'NatScale-errors', 0, {}, {}, {},
                             expert_selections={item.name: selected})


def measure_errors(root, scope, imatrix, cache, backend, *, profiles=ALPHAQ_PROFILES,
                   sample_layers=6, sample_experts=3, row_chunk=0, output=None):
    """Fit every profile on evenly spaced exposed experts in sampled MoE layers.

    Tables are learned on exactly the same full-bank rows/options as final
    export, in the shared codebook cache. No model forward/backward is run.
    """
    from mfq.formats.io import unpack_tensor_payload
    from mfq.quantize.expert_nint import dequantize_expertwise
    from mfq.tools import quantize_hf_to_mfq as q

    root, cache = Path(root), Path(cache)
    output = Path(output) if output else cache/'natscale-errors.json'
    sha = file_sha256(imatrix.path)
    signature = error_signature(root, scope.experts, sha, profiles, sample_layers, sample_experts)
    saved = json.loads(output.read_text()) if output.exists() else dict(
        format='mfq.natscale-errors.v1', signature=signature, complete=False, samples={})
    if saved.get('format') != 'mfq.natscale-errors.v1' or saved.get('signature') != signature:
        raise ValueError('NatScale measurement cache does not match source/imatrix/options/codecs')
    if saved.get('complete'):
        return load_errors(output, root, scope.experts, sha, profiles, sample_layers, sample_experts)
    layers = sorted({int(p.name.split('.')[2]) for p in scope.experts})
    sampled = set(np.array(layers)[even_indices(len(layers), sample_layers)])
    targets = [p for p in scope.experts if int(p.name.split('.')[2]) in sampled]
    errors = {}
    for item in targets:
        layer, projection = item.name.split('.')[2], item.name.split('.')[-2]
        entry_name = scope.imatrix_entries[item.name]
        entry = imatrix.entries[entry_name]
        positive = np.flatnonzero(entry.counts > 0)
        if not positive.size:
            raise ValueError(f'no exposed experts to measure: {item.name}')
        expert_ids = positive[even_indices(len(positive), sample_experts)]
        # Candidate descriptors come from the allocator's catalog, not a
        # separately maintained approximation of final quantizer settings.
        n, rows, columns = item.shape
        stat = AlphaQTensorStatistics(item.name, int(layer), projection,
                                      item.shape, (1.,)*n, (1.,)*n)
        table = alphaq_builtin_candidates([stat], profiles)
        descriptors = {c.profile: _imatrix_precision(c.precision)
                       for c in table.candidates if c.key.expert_id == 0}
        for profile, precision in descriptors.items():
            missing = [int(e) for e in expert_ids
                       if f'{item.name}/{e}/{profile}' not in saved['samples']]
            if missing:
                trained = trained_tables(root, [item], _scheme(item, precision), imatrix,
                                         sha, cache, backend.device)
                precision = trained.expert_selections[item.name].precisions[0]
            for expert in missing:
                with rows_for_plan(root, item) as source:
                    weight = source.read_rows(expert*rows, (expert+1)*rows, device=backend.device).float()
                moments = entry.values[expert:expert+1]
                subset = replace(imatrix, entries={entry_name: replace(entry,
                    values=moments, counts=entry.counts[expert:expert+1], row_importance=None,
                    allocation_groups=None)})
                sample = replace(item, shape=(1, rows, columns), expert_shape=(1, rows, columns),
                                 target_dtype='MFE', expert_precisions=(precision,))
                binding = q._bind_hf_imatrix(subset, [sample]).get(item.name)
                fitter = FittedPayloadCache(root, cache, subset, sha, backend)
                cache.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='rho-', dir=cache) as temporary:
                    payload = Path(temporary)/'tensor.blob'
                    dtype, size = fitter._fit(sample, _LatentRows(weight.unsqueeze(0)), binding,
                                             payload, row_chunk or max(8, (rows+7)//8*8))
                    if payload.stat().st_size != size:
                        raise ValueError('NatScale measured payload is truncated')
                    decoded = dequantize_expertwise(unpack_tensor_payload(dtype, payload.read_bytes()))[0]
                native = weight.cpu().double().numpy()
                denominator = float(np.sum(native**2*moments))
                numerator = float(np.sum((decoded-native)**2*moments))
                if not math.isfinite(numerator) or not math.isfinite(denominator):
                    raise ValueError('nonfinite NatScale measured error')
                saved['samples'][f'{item.name}/{expert}/{profile}'] = dict(
                    numerator=numerator, denominator=denominator, expert=expert,
                    tensor=item.name, layer=layer, projection=projection, profile=profile)
                _atomic_json(output, saved)
                _progress(stage='natscale_error_sample', tensor=item.name, expert=expert, profile=profile)
            values = [saved['samples'][f'{item.name}/{e}/{profile}'] for e in expert_ids]
            denominator = sum(v['denominator'] for v in values)
            numerator = sum(v['numerator'] for v in values)
            errors.setdefault(layer, {}).setdefault(projection, {})[profile] = (
                numerator/denominator if denominator > 0 else 0.)
    if source_identity(root) != signature['source'] or file_sha256(imatrix.path) != sha:
        raise ValueError('NatScale measurement inputs changed during fitting')
    saved.update(complete=True, errors=errors)
    _atomic_json(output, saved)
    return load_errors(output, root, scope.experts, sha, profiles, sample_layers, sample_experts)


def run(args):
    from mfq.calibration.alphaq_cache import collect_statistics
    from mfq.calibration.alphaq_model import model_scope
    from mfq.calibration.alphaq_workflow import planned, routing
    from mfq.quantize.backend import resolve_quant_backend
    from mfq.quantize.imatrix import load_importance_matrix
    from mfq.tools import quantize_hf_to_mfq as q

    root, cache = Path(args.model).resolve(), Path(args.cache).resolve()
    imatrix = load_importance_matrix(Path(args.imatrix).resolve())
    config = q.load_hf_model_config(root)
    plans = planned(root, None, CalibrationScheme(None, 'NatScale', 0, {}, {}, {}), {}, 'native')
    scope = model_scope(plans, imatrix, exclude_scaled_dense=True)
    routing(scope, imatrix, config, 'imatrix', '', root, absolute=True)
    backend = resolve_quant_backend(args.backend, args.device)
    with cache_lock(cache/'.quantize.lock'):
        if args.calibration_stage == 'natscale-statistics':
            collect_statistics(root, scope.experts, cache, backend.device)
        else:
            measure_errors(root, scope, imatrix, cache, backend,
                profiles=tuple(args.profiles.split(',')) if args.profiles else ALPHAQ_PROFILES,
                sample_layers=args.sample_layers, sample_experts=args.sample_experts,
                row_chunk=args.row_chunk, output=args.output or None)
    return 0


def add_parsers(stages):
    for name, help_text in (('natscale-errors', 'measure imatrix-weighted format errors with export codecs'),
                            ('natscale-statistics', 'collect weight spectra and column energies in one pass')):
        parser = stages.add_parser(name, help=help_text)
        parser.add_argument('--model', required=True)
        parser.add_argument('--imatrix', required=True)
        parser.add_argument('--cache', required=True)
        parser.add_argument('--backend', choices=('auto', 'cpu', 'cuda', 'metal'), default='auto')
        parser.add_argument('--device', default='')
        if name == 'natscale-errors':
            parser.add_argument('--profiles', default='')
            parser.add_argument('--sample-layers', type=int, default=6)
            parser.add_argument('--sample-experts', type=int, default=3)
            parser.add_argument('--row-chunk', type=int, default=0)
            parser.add_argument('--output', default='')
        parser.set_defaults(_impl=run)
