import json
import os
import io

import numpy as np

import pytest
import torch

from mfq.calibration.alphaq_fit import FittedPayloadCache
from mfq.cli import main
from mfq.formats.io import open_mmap
from tests.test_alphaq_workflow import fixture


@pytest.mark.parametrize('bits', [1, 2, 3, 4, 5, 6, 7, 8])
def test_cuda_packed_region_accepts_final_padding_only(bits):
    from mfq.tools.quantize_hf_to_mfq import _PackedBitRegionWriter, pack_bits
    values = (np.arange(19, dtype=np.uint16) % (1 << bits)).astype(np.uint8)
    output = io.BytesIO()
    writer = _PackedBitRegionWriter(output, 0, bits, len(values))
    writer.append_packed(pack_bits(values[:16], bits), 16)
    writer.append_packed(pack_bits(values[16:], bits), 3)
    writer.finish()
    assert output.getvalue() == pack_bits(values, bits)
    with pytest.raises(ValueError, match='alignment or length'):
        writer.append_packed(b'\0', 1)


def command(root, imatrix, cache, output):
    return ['quantize', str(root), str(output), '--imatrix', str(imatrix),
            '--alphaq-cache', str(cache), '--target-bpw', '12',
            '--alphaq-profiles', 'NINT4,NINT8', '--backend', 'cpu']


def test_failed_fit_resumes_completed_tensors_and_rejects_corrupt_payload(tmp_path, monkeypatch):
    root, imatrix, _ = fixture(tmp_path)
    cache, output = tmp_path/'cache', tmp_path/'output.mfq'
    original, calls = FittedPayloadCache._fit, []

    def interrupted(self, item, *args):
        calls.append(item.name)
        if len(calls) == 2:
            raise RuntimeError('interrupted second tensor')
        return original(self, item, *args)

    monkeypatch.setattr(FittedPayloadCache, '_fit', interrupted)
    with pytest.raises(RuntimeError, match='interrupted second tensor'):
        main(command(root, imatrix, cache, output))
    assert not output.exists()
    markers = list((cache/'fitted').glob('*/complete.json'))
    assert len(markers) == 1
    first = calls[0]
    assert main(command(root, imatrix, cache, output)) == 0
    assert calls.count(first) == 1

    def forbidden(*args, **kwargs):
        raise AssertionError('a complete payload must not be refitted')
    monkeypatch.setattr(FittedPayloadCache, '_fit', forbidden)
    before = output.read_bytes()
    assert main([*command(root, imatrix, cache, output), '--overwrite']) == 0
    assert output.read_bytes() == before
    part = json.loads(markers[0].read_text())['file']
    stat = os.stat(part['path'])
    with open(part['path'], 'r+b') as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([byte[0] ^ 1]))
    os.utime(part['path'], ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='checksum differs'):
        main([*command(root, imatrix, cache, output), '--overwrite'])
    assert output.read_bytes() == before


def test_only_real_oom_reduces_full_geometry_and_preserves_output_bytes(tmp_path, monkeypatch):
    root, imatrix, _ = fixture(tmp_path)
    baseline, retry = tmp_path/'baseline.mfq', tmp_path/'retry.mfq'
    assert main(command(root, imatrix, tmp_path/'baseline-cache', baseline)) == 0
    original, attempts = FittedPayloadCache._fit, []

    def allocation_failure(self, item, source, binding, path, chunk):
        if item.target_dtype == 'MFE':
            attempts.append((item.name, chunk))
            if chunk > 8:
                raise torch.OutOfMemoryError('injected actual allocation failure')
        return original(self, item, source, binding, path, chunk)

    monkeypatch.setattr(FittedPayloadCache, '_fit', allocation_failure)
    cache = tmp_path/'retry-cache'
    assert main(command(root, imatrix, cache, retry)) == 0
    assert attempts[0][1] == 24
    assert retry.read_bytes() == baseline.read_bytes()
    for marker in (cache/'fitted').glob('*/complete.json'):
        doc = json.loads(marker.read_text())
        if doc['dtype'] == 'MFE':
            assert doc['oom_rows'] == [24] and doc['row_chunk'] == 8


def test_required_fit_checkpoint_failure_does_not_publish_model(tmp_path, monkeypatch):
    root, imatrix, _ = fixture(tmp_path)
    output = tmp_path/'output.mfq'

    def denied(*args, **kwargs):
        raise PermissionError('required fitted tensor checkpoint denied')

    monkeypatch.setattr('mfq.calibration.alphaq_fit._atomic_json', denied)
    with pytest.raises(PermissionError, match='required fitted tensor checkpoint denied'):
        main(command(root, imatrix, tmp_path/'cache', output))
    assert not output.exists()


def test_allocation_survives_failure_before_table_training(tmp_path, monkeypatch):
    from mfq.calibration import alphaq_workflow as workflow
    root, imatrix, _ = fixture(tmp_path)
    cache, output = tmp_path/'cache', tmp_path/'output.mfq'
    original = workflow.trained_tables
    def fail(*args, **kwargs):
        raise RuntimeError('table stage interrupted')
    monkeypatch.setattr(workflow, 'trained_tables', fail)
    with pytest.raises(RuntimeError, match='table stage interrupted'):
        main(command(root, imatrix, cache, output))
    assert not output.exists()
    def forbidden(*args, **kwargs):
        raise AssertionError('completed allocation must be reused')
    monkeypatch.setattr(workflow, 'allocate_joint', forbidden)
    monkeypatch.setattr(workflow, 'trained_tables', original)
    assert main(command(root, imatrix, cache, output)) == 0
    before = output.read_bytes()
    directory = next((cache/'allocations').iterdir())
    (directory/'untrained-scheme.json').write_text('{}')
    with pytest.raises(ValueError, match='checkpoint checksum differs'):
        main([*command(root, imatrix, cache, output), '--overwrite'])
    assert output.read_bytes() == before
