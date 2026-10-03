"""Explicit-device scheduling for normal AlphaQ statistics and train/fit work."""
from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from mfq.calibration.alphaq_source import _progress


def execution_devices(requested, backend, single_device):
    """Only use explicitly selected visible CUDA indices; never discover idle GPUs."""
    if not requested:
        return ()
    if backend.name != 'cuda' or single_device not in ('', 'cuda'):
        raise ValueError('--devices requires CUDA and cannot combine with --device INDEX')
    import torch
    parts = str(requested).split(',')
    if any(re.fullmatch(r'(?:cuda:)?[0-9]+', p.strip()) is None for p in parts):
        raise ValueError('--devices needs comma-separated visible CUDA indices')
    indices = [int(p.strip().removeprefix('cuda:')) for p in parts]
    if len(set(indices)) != len(indices) or any(i >= torch.cuda.device_count() for i in indices):
        raise ValueError('--devices contains duplicate or unavailable CUDA indices')
    return tuple(f'cuda:{i}' for i in indices)


def dynamic_map(executors, function, tasks):
    """One task per device. Failed devices retire; healthy work still checkpoints."""
    iterator, pending, failures = iter(tasks), {}, []

    def submit(executor):
        try:
            task = next(iterator)
        except StopIteration:
            return
        try:
            pending[executor.submit(function, task)] = executor, task
        except Exception as error:
            failures.append(error)

    for executor in executors:
        submit(executor)
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            executor, task = pending.pop(future)
            try:
                value = future.result()
            except Exception as error:
                failures.append(error)
                # A CUDA fault must not be retried or followed by more work in
                # the same process. Other devices continue their unique tasks.
                continue
            yield task, value
            submit(executor)
    if failures:
        raise RuntimeError(f'{len(failures)} AlphaQ worker(s) failed; completed checkpoints retained') from failures[0]


_worker = None


def _initialize(root, cache, imatrix_path, imatrix_sha, backend, device, row_chunk, artifact_root):
    global _worker
    import torch
    from mfq.quantize.backend import QuantBackend
    if backend == 'cuda':
        torch.cuda.set_device(device)
    _worker = dict(root=Path(root), cache=Path(cache), imatrix_path=Path(imatrix_path),
        imatrix_sha=imatrix_sha, backend=QuantBackend(backend, device),
        row_chunk=row_chunk, artifact_root=Path(artifact_root), imatrix=None)


def _run_job(job):
    from mfq.calibration.alphaq_cache import collect_statistics, trained_tables
    from mfq.calibration.alphaq_fit import FittedPayloadCache
    from mfq.calibration.alphaq_output import file_sha256
    from mfq.quantize.imatrix import load_importance_matrix

    kind, item, scheme = job
    state, started = _worker, time.monotonic()
    device = state['backend'].device
    if kind == 'statistics':
        # Each bank is independently durable even if its parent exits before
        # merging statistics.json. Workers never overwrite that shared file.
        key = hashlib.sha256(item.name.encode()).hexdigest()
        directory = state['cache']/'statistics-parts'/key
        value = collect_statistics(state['root'], (item,), directory, device)[0]
    elif kind == 'fit':
        if state['imatrix'] is None:
            if file_sha256(state['imatrix_path']) != state['imatrix_sha']:
                raise ValueError('AlphaQ worker imatrix changed before loading')
            state['imatrix'] = load_importance_matrix(state['imatrix_path'])
        trained = trained_tables(state['root'], (item,), scheme, state['imatrix'],
            state['imatrix_sha'], state['cache'], device)
        if item.name in trained.expert_selections:
            item = replace(item, expert_precisions=trained.expert_selections[item.name].precisions)
        elif item.name in trained.selections:
            item = replace(item, target_precision=trained.selections[item.name].descriptor)
        fitted = FittedPayloadCache(state['root'], state['cache'], state['imatrix'],
            state['imatrix_sha'], state['backend'], row_chunk=state['row_chunk'],
            artifact_root=state['artifact_root'])
        record = fitted.require(item)
        fitted.validate()
        value = (item, trained, record, fitted.signature(item), fitted.checked[-1])
    else:
        raise ValueError(f'unknown AlphaQ worker operation: {kind}')
    return dict(value=value, device=device, pid=os.getpid(), seconds=time.monotonic()-started)


class DeviceWorkers:
    """Persistent processes, one per selected device, with dynamic task assignment."""

    def __init__(self, root, cache, imatrix_path, imatrix_sha, backend, devices,
                 *, row_chunk, artifact_root):
        self.arguments = (root, cache, imatrix_path, imatrix_sha, backend.name)
        self.tail = row_chunk, artifact_root
        self.devices, self.executors, self.receipts = tuple(devices), [], []

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        for executor in self.executors:
            executor.shutdown(wait=True)

    def map(self, kind, items):
        tasks = list(items)
        if not tasks:
            return
        if not self.devices:
            raise ValueError('parallel AlphaQ work requires explicit devices')
        if not self.executors:
            context = multiprocessing.get_context('spawn')
            self.executors = [ProcessPoolExecutor(max_workers=1, mp_context=context,
                initializer=_initialize, initargs=(*self.arguments, device, *self.tail))
                for device in self.devices]
        jobs = [(kind, item, None) for item in tasks] if kind == 'statistics' else [
            (kind, item, scheme) for item, scheme in tasks]
        # Start larger tensors first; each available device then takes the next
        # whole bank. The count derives from work, not an inherited batch cap.
        jobs.sort(key=lambda job: math.prod(job[1].shape), reverse=True)
        for completed, (job, result) in enumerate(dynamic_map(self.executors, _run_job, jobs), 1):
            receipt = dict(stage=kind, tensor=job[1].name,
                           **{k: v for k, v in result.items() if k != 'value'})
            self.receipts.append(receipt)
            _progress(stage='alphaq_parallel', operation=kind, completed=completed,
                      total=len(jobs), **{k: v for k, v in receipt.items() if k != 'stage'})
            yield job[1], result['value']


def train_and_fit(workers, root, plans, scheme, imatrix, imatrix_sha, cache, backend,
                  *, row_chunk, artifact_root):
    """Finish each tensor's tables and fit together; no all-tables barrier."""
    from mfq.calibration.alphaq_fit import FittedPayloadCache
    payloads = FittedPayloadCache(root, cache, imatrix, imatrix_sha, backend,
                                  row_chunk=row_chunk, artifact_root=artifact_root)
    tasks = [(item, replace(scheme, path=None, metadata={}, candidate_table={},
        expert_selections={item.name: scheme.expert_selections[item.name]}
            if item.name in scheme.expert_selections else {},
        selections={item.name: scheme.selections[item.name]}
            if item.name in scheme.selections else {})) for item in plans]
    experts, dense = {}, {}
    for _, (item, trained, record, signature, info) in workers.map('fit', tasks):
        payloads.accept_verified(item, record, signature, info)
        experts.update(trained.expert_selections)
        dense.update(trained.selections)
    if set(experts) != set(scheme.expert_selections) or set(dense) != set(scheme.selections):
        raise ValueError('parallel AlphaQ fitting did not cover the complete scheme')
    return replace(scheme, expert_selections=experts, selections=dense), payloads
