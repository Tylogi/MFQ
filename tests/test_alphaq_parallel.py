from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from mfq.calibration.alphaq_parallel import dynamic_map, execution_devices
from mfq.cli import main
from mfq.quantize.backend import QuantBackend
from tests.test_alphaq_fit import command
from tests.test_alphaq_workflow import fixture


def test_fast_device_keeps_working_while_another_device_is_busy():
    release = Event()
    def work(task):
        if task == 'slow':
            assert release.wait(10)
        elif task == 'last':
            release.set()
        return task
    with ThreadPoolExecutor(1) as slow, ThreadPoolExecutor(1) as fast:
        results = list(dynamic_map([slow, fast], work, ['slow', 'fast', 'next', 'last']))
    assert {task for task, _ in results} == {'slow', 'fast', 'next', 'last'}
    assert next(i for i, (task, _) in enumerate(results) if task == 'next') < next(
        i for i, (task, _) in enumerate(results) if task == 'slow')


def test_failed_device_is_not_retried_and_healthy_tasks_finish():
    calls = []
    def work(task):
        calls.append(task)
        if task == 0:
            raise RuntimeError('simulated device failure')
        return task * 2
    results = []
    with ThreadPoolExecutor(1) as bad, ThreadPoolExecutor(1) as healthy:
        with pytest.raises(RuntimeError, match='completed checkpoints retained') as error:
            for result in dynamic_map([bad, healthy], work, range(8)):
                results.append(result)
    assert 'simulated device failure' in str(error.value.__cause__)
    assert sorted(calls) == list(range(8))
    assert sorted(results) == [(n, n*2) for n in range(1, 8)]


def test_devices_are_explicit_distinct_visible_cuda_indices(monkeypatch):
    monkeypatch.setattr('torch.cuda.device_count', lambda: 3)
    backend = QuantBackend('cuda', 'cuda')
    assert execution_devices('', backend, 'cuda') == ()
    assert execution_devices('0,cuda:2', backend, 'cuda') == ('cuda:0', 'cuda:2')
    for value in ('0,0', '0,3', '0,', '-1', 'all'):
        with pytest.raises(ValueError):
            execution_devices(value, backend, 'cuda')
    with pytest.raises(ValueError):
        execution_devices('0,1', backend, 'cuda:0')
    with pytest.raises(ValueError):
        execution_devices('0,1', QuantBackend('cpu', 'cpu'), 'cuda')


def test_normal_command_two_spawned_workers_matches_serial_and_resumes(tmp_path, monkeypatch):
    import json
    from mfq.calibration import alphaq_parallel
    root, imatrix, _ = fixture(tmp_path)
    baseline, output = tmp_path/'serial.mfq', tmp_path/'parallel.mfq'
    cache = tmp_path/'parallel-cache'
    assert main(command(root, imatrix, tmp_path/'serial-cache', baseline)) == 0
    # Real spawned CPU processes exercise portable scheduling and IPC; CUDA
    # identity/real geometry is checked separately on the allowed devices.
    monkeypatch.setattr(alphaq_parallel, 'execution_devices', lambda *a: ('cpu', 'cpu'))
    receipts, original = [], alphaq_parallel.DeviceWorkers.map
    def traced(self, *args):
        yield from original(self, *args)
        receipts.extend(self.receipts)
    monkeypatch.setattr(alphaq_parallel.DeviceWorkers, 'map', traced)
    argv = [*command(root, imatrix, cache, output), '--devices', '0,1']
    assert main(argv) == 0
    assert output.read_bytes() == baseline.read_bytes()
    assert len({r['pid'] for r in receipts}) == 2
    assert {r['stage'] for r in receipts} == {'statistics', 'fit'}
    parts = list((cache/'statistics-parts').glob('*/statistics.json'))
    assert len(parts) == 5
    parts_before = {p: p.stat().st_mtime_ns for p in parts}
    markers = list((cache/'fitted').glob('*/complete.json'))
    before = {p: p.read_bytes() for p in markers}
    assert len(markers) == 6
    assert main([*argv, '--overwrite']) == 0
    assert output.read_bytes() == baseline.read_bytes()
    assert {p: p.read_bytes() for p in markers} == before
    # If the parent dies before merging its statistics file, independent bank
    # checkpoints recover without repeating their spectral calculation.
    (cache/'statistics.json').unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('equivalent merged statistics must reuse allocation')
    monkeypatch.setattr('mfq.calibration.alphaq_workflow.allocate_joint', forbidden)
    assert main([*argv, '--overwrite']) == 0
    assert output.read_bytes() == baseline.read_bytes()
    assert {p: p.read_bytes() for p in markers} == before
    assert len(json.loads((cache/'statistics.json').read_text())['tensors']) == 5
    assert {p: p.stat().st_mtime_ns for p in parts} == parts_before


def test_same_allocation_can_be_verified_after_implementation_change(tmp_path, monkeypatch):
    from mfq.calibration import alphaq_workflow
    root, imatrix, _ = fixture(tmp_path)
    cache, output = tmp_path/'cache', tmp_path/'output.mfq'
    argv = command(root, imatrix, cache, output)
    assert main(argv) == 0
    before = output.read_bytes()
    signature = alphaq_workflow.allocation_signature
    def changed(*args):
        value = signature(*args)
        value['implementation']['test_revision'] = 'new version'
        return value
    monkeypatch.setattr(alphaq_workflow, 'allocation_signature', changed)
    assert main([*argv, '--overwrite']) == 0
    assert output.read_bytes() == before


def test_worker_fits_a_bank_before_requesting_the_next_bank(tmp_path, monkeypatch):
    from mfq.calibration import alphaq_parallel, alphaq_cache
    from mfq.calibration.alphaq_fit import FittedPayloadCache
    from mfq.calibration.artifact import CalibrationScheme
    from mfq.calibration.alphaq_output import file_sha256
    from mfq.tools.quantize_hf_to_mfq import _plan, BlobRecord
    root, imatrix, _ = fixture(tmp_path)
    cache = tmp_path/'cache'
    scheme = CalibrationScheme(None, 'AlphaQ', 0, {}, {}, {})
    plans = _plan(root, True, None, 'F32', scheme, exclude_mtp=True)
    events = []
    def train(root, plans, scheme, *args):
        events.append(('train', plans[0].name))
        return scheme
    def fit(self, item):
        events.append(('fit', item.name))
        self.checked.append({})
        return BlobRecord(item.name, 'BF16', 0, tmp_path/'unused')
    monkeypatch.setattr(alphaq_cache, 'trained_tables', train)
    monkeypatch.setattr(FittedPayloadCache, 'require', fit)
    monkeypatch.setattr(FittedPayloadCache, 'validate', lambda self: None)
    alphaq_parallel._initialize(root, cache, imatrix, file_sha256(imatrix),
                                'cpu', 'cpu', 0, cache)
    for item in plans[:2]:
        alphaq_parallel._run_job(('fit', item, scheme))
    assert events == [(stage, p.name) for p in plans[:2] for stage in ('train', 'fit')]
