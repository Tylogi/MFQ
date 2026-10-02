"""Gram spectra preserve the full-block Hill statistic and degenerate cases."""
import numpy as np
import pytest
import torch

from mfq.calibration.alphaq import alphaq_weight_statistics
from mfq.calibration.alphaq_spectrum import symmetric_eigenvalues


@pytest.mark.parametrize('kind', ['random', 'zero', 'constant', 'identity', 'rank_one', 'tiny'])
def test_gram_matches_svd_with_degenerate_tail_recovery(kind):
    generator = torch.Generator().manual_seed(37)
    weight = torch.randn((2, 257, 129), generator=generator)
    if kind == 'zero':
        weight.zero_()
    elif kind == 'constant':
        weight.fill_(.25)
    elif kind == 'identity':
        weight = torch.eye(128).expand(2, -1, -1).clone()
    elif kind == 'rank_one':
        weight = torch.randn((2, 128, 1), generator=generator)*torch.randn((2, 1, 256), generator=generator)
    elif kind == 'tiny':
        weight = weight[:, :4]
    expected = alphaq_weight_statistics(weight, backend='svd')
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        actual = alphaq_weight_statistics(weight, backend='gram')
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-4, atol=0)
    np.testing.assert_array_equal(actual[1], expected[1])


def test_eigenvalues_preserve_input_and_reject_invalid_geometry():
    value = torch.eye(4).expand(3, -1, -1).clone()
    before = value.clone()
    torch.testing.assert_close(symmetric_eigenvalues(value), torch.ones((3, 4)))
    torch.testing.assert_close(value, before)
    for invalid in (value.double(), value[0], value[:, :, :2]):
        with pytest.raises(ValueError, match='square'):
            symmetric_eigenvalues(invalid)


def test_backend_typo_is_not_silently_accepted():
    with pytest.raises(ValueError, match='backend'):
        alphaq_weight_statistics(torch.ones((1, 2, 3)), backend='grma')


def test_gram_oom_retries_only_unfinished_blocks(monkeypatch):
    import mfq.calibration.alphaq_spectrum as spectrum
    weight = torch.randn((2, 128, 384), generator=torch.Generator().manual_seed(91))
    expected = alphaq_weight_statistics(weight, backend='gram')
    original = spectrum.gram_eigenvalues
    calls = []
    def limited(blocks):
        calls.append(len(blocks))
        if len(blocks) > 2:
            raise torch.OutOfMemoryError('injected allocation failure')
        return original(blocks)
    monkeypatch.setattr(spectrum, 'gram_eigenvalues', limited)
    actual = alphaq_weight_statistics(weight, backend='gram')
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-4)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert calls[0] == 6 and sum(c for c in calls if c <= 2) == 6
