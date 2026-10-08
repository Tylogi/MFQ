import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mfq.calibration.wt2 import ReferenceWriter, write_head


@pytest.mark.parametrize('vocab', [19, 20])
def test_v3_binary_matches_dense_reference_and_streams_head(tmp_path, vocab):
    torch.manual_seed(42)
    weights = torch.randn(vocab, 8)
    hidden = torch.randn(15, 8)
    targets = torch.arange(15) % vocab
    calls = []
    def tensor(name, **kwargs):
        calls.append((kwargs['row_start'], kwargs['row_end']))
        return weights[kwargs['row_start']:kwargs['row_end']]
    backend = SimpleNamespace(device=torch.device('cpu'), teacher_dtype=torch.float32,
        index=SimpleNamespace(weight_map={'lm_head.weight': 'weights'}, tensor=tensor))
    path = tmp_path / 'ref.logits'
    tokens = np.arange(64, dtype='<i4').reshape(2, 32)
    with ReferenceWriter(path, tokens, vocab) as writer:
        for chunk in range(2):
            nll = write_head(backend, hidden, targets, writer, chunk, row_chunk=7, token_chunk=5)
        writer.commit()
    assert max(end - start for start, end in calls) <= 7
    raw = path.read_bytes()
    assert raw[:8] == b'_logit3_'
    assert struct.unpack_from('<II', raw, 8) == (vocab, 2)
    assert struct.unpack_from('<III', raw, 16) == (32, 17, 15)
    assert np.array_equal(np.frombuffer(raw, dtype='<i4', count=64, offset=40).reshape(2, 32), tokens)
    exact = torch.log_softmax(hidden @ weights.T, -1).numpy()
    expected = exact[np.arange(15), targets.numpy()]
    target_offset = 40 + 64 * 4
    actual_targets = np.frombuffer(raw, dtype='<f4', count=30, offset=target_offset).reshape(2, 15)
    np.testing.assert_allclose(actual_targets, np.stack([expected, expected]), atol=2e-6)
    assert nll == pytest.approx(-expected.sum(), abs=2e-5)
    elements = 2 * ((vocab + 1) // 2) + 4
    rows = np.frombuffer(raw, dtype='<u2', offset=target_offset + 30 * 4).reshape(2, 15, elements)
    header = rows[:, :, :4].copy().view('<f4')
    decoded = rows[:, :, 4:4 + vocab] * header[:, :, :1] + header[:, :, 1:2]
    unclipped = exact >= exact.max(-1, keepdims=True) - 16
    np.testing.assert_allclose(decoded[0][unclipped], exact[unclipped], atol=16 / 65535)
    np.testing.assert_array_equal(decoded[0].argmax(-1), exact.argmax(-1))
    if vocab % 2:
        assert np.all(rows[:, :, -1] == 0)


def test_resident_head_produces_the_same_file(tmp_path):
    weights, hidden = torch.randn(23, 4), torch.randn(15, 4)
    index = SimpleNamespace(weight_map={'lm_head.weight': 'weights'}, tensor=lambda name, **kwargs: weights[kwargs['row_start']:kwargs['row_end']])
    backend = SimpleNamespace(device=torch.device('cpu'), teacher_dtype=torch.float32, index=index)
    for name, resident in [('streamed', None), ('resident', weights)]:
        with ReferenceWriter(tmp_path / name, np.zeros((1, 32), dtype='<i4'), 23) as writer:
            write_head(backend, hidden, torch.arange(15), writer, 0, row_chunk=8, resident_weight=resident)
            writer.commit()
    assert (tmp_path / 'streamed').read_bytes() == (tmp_path / 'resident').read_bytes()


def test_failed_reference_does_not_leave_a_complete_or_partial_file(tmp_path):
    path = tmp_path / 'ref'
    with pytest.raises(RuntimeError), ReferenceWriter(path, np.zeros((1, 32), dtype='<i4'), 2):
        raise RuntimeError('cancelled')
    assert not path.exists()
    assert not path.with_name('ref.partial').exists()


def test_reference_never_overwrites_existing_outputs(tmp_path):
    path = tmp_path / 'ref'
    path.write_bytes(b'existing')
    with pytest.raises(FileExistsError), ReferenceWriter(path, np.zeros((1, 32), dtype='<i4'), 2):
        pass
    assert path.read_bytes() == b'existing'
