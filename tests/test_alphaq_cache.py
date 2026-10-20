from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mfq.calibration import alphaq_cache


@pytest.fixture
def bank(tmp_path, monkeypatch):
    weight = torch.randn(3, 4, 5, generator=torch.Generator().manual_seed(17))
    item = SimpleNamespace(name='model.block.0.mlp.experts.gate.weight', shape=tuple(weight.shape))
    cache = tmp_path / 'cache'
    cache.mkdir()

    @contextmanager
    def rows(*_):
        yield alphaq_cache._TensorRows(weight)

    monkeypatch.setattr(alphaq_cache, 'rows_for_plan', rows)
    monkeypatch.setattr(alphaq_cache, 'source_identity', lambda _: {'model': 'test', 'files': []})
    return tmp_path, item, cache, weight


def forbidden(*_, **__):
    pytest.fail('completed spectral statistics must not be recomputed')


def test_ordinary_alphaq_reuses_spectral_only_cache(bank, monkeypatch):
    root, item, cache, _ = bank
    original = alphaq_cache.collect_statistics(root, [item], cache, 'cpu')
    manifest = (cache / 'statistics.json').read_bytes()
    assert 'column_energy' not in json.loads(manifest)['tensors'][item.name]
    monkeypatch.setattr(alphaq_cache, '_collect', forbidden)
    monkeypatch.setattr(alphaq_cache, '_collect_energy', forbidden)
    assert alphaq_cache.collect_statistics(root, [item], cache, 'cpu') == original
    assert (cache / 'statistics.json').read_bytes() == manifest


def test_natscale_upgrades_old_cache_without_repeating_svd(bank, monkeypatch):
    root, item, cache, weight = bank
    original = alphaq_cache.collect_statistics(root, [item], cache, 'cpu')
    monkeypatch.setattr(alphaq_cache, '_collect', forbidden)
    upgraded = alphaq_cache.collect_statistics(root, [item], cache, 'cpu', column_energy=True)
    assert upgraded == original
    np.testing.assert_allclose(upgraded[0].column_energy, weight.square().sum(dim=1).numpy())
    monkeypatch.setattr(alphaq_cache, '_collect_energy', forbidden)
    reused = alphaq_cache.collect_statistics(root, [item], cache, 'cpu', column_energy=True)
    np.testing.assert_array_equal(reused[0].column_energy, upgraded[0].column_energy)


def test_optional_energy_corruption_does_not_break_ordinary_alphaq(bank):
    root, item, cache, _ = bank
    original = alphaq_cache.collect_statistics(root, [item], cache, 'cpu', column_energy=True)
    document = json.loads((cache / 'statistics.json').read_text())
    energy = cache / document['tensors'][item.name]['column_energy']['path']
    energy.write_bytes(b'corrupt')
    assert alphaq_cache.collect_statistics(root, [item], cache, 'cpu') == original
    with pytest.raises(ValueError, match='checksum differs'):
        alphaq_cache.collect_statistics(root, [item], cache, 'cpu', column_energy=True)


def test_energy_cache_rejects_paths_outside_cache(bank):
    root, item, cache, _ = bank
    alphaq_cache.collect_statistics(root, [item], cache, 'cpu', column_energy=True)
    manifest = cache / 'statistics.json'
    document = json.loads(manifest.read_text())
    document['tensors'][item.name]['column_energy']['path'] = '../outside.npy'
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='path escapes'):
        alphaq_cache.collect_statistics(root, [item], cache, 'cpu', column_energy=True)


def test_parallel_statistics_request_includes_energy(bank):
    root, item, cache, weight = bank
    operations = []

    def mapper(operation, items):
        operations.append(operation)
        worker_cache = cache / 'worker'
        worker_cache.mkdir(exist_ok=True)
        for plan in items:
            value = alphaq_cache.collect_statistics(
                root, [plan], worker_cache, 'cpu', column_energy=operation == 'statistics-energy'
            )[0]
            yield plan, value

    result = alphaq_cache.collect_statistics(
        root, [item], cache, 'cpu', mapper=mapper, column_energy=True
    )
    assert operations == ['statistics-energy']
    np.testing.assert_allclose(result[0].column_energy, weight.square().sum(dim=1).numpy())


def test_parallel_statistics_validates_cached_energy_before_worker_dispatch(bank):
    root, item, cache, _ = bank
    alphaq_cache.collect_statistics(root, [item], cache, 'cpu', column_energy=True)
    document = json.loads((cache / 'statistics.json').read_text())
    energy = cache / document['tensors'][item.name]['column_energy']['path']
    energy.write_bytes(b'corrupt')
    missing = SimpleNamespace(name=item.name.replace('.gate.', '.up.'), shape=item.shape)
    with pytest.raises(ValueError, match='checksum differs'):
        alphaq_cache.collect_statistics(
            root, [item, missing], cache, 'cpu', mapper=forbidden, column_energy=True
        )


def test_parallel_energy_upgrade_uses_worker_without_repeating_svd(bank, monkeypatch):
    root, item, cache, weight = bank
    original = alphaq_cache.collect_statistics(root, [item], cache, 'cpu')[0]
    monkeypatch.setattr(alphaq_cache, '_collect', forbidden)
    monkeypatch.setattr(alphaq_cache, '_collect_energy', forbidden)
    operations = []
    energy = weight.square().sum(dim=1).numpy()

    def mapper(operation, items):
        operations.append(operation)
        assert items == [item]
        yield item, replace(original, column_energy=energy)

    result = alphaq_cache.collect_statistics(
        root, [item], cache, 'cpu', mapper=mapper, column_energy=True
    )
    assert operations == ['statistics-energy']
    assert result[0] == original
    np.testing.assert_array_equal(result[0].column_energy, energy)


def test_statistics_batch_uses_requested_devices_free_memory(monkeypatch):
    queries = []

    def memory(device):
        queries.append(device)
        return 2 * 48 * 32 * 64 * 3, 2**40

    monkeypatch.setattr(torch.cuda, 'mem_get_info', memory)
    assert alphaq_cache._initial_batch((8, 32, 64), 'cuda:2') == 3
    assert queries == ['cuda:2']
    assert alphaq_cache._initial_batch((8, 32, 64), 'cpu') == 8
    assert queries == ['cuda:2']


def test_oom_backoff_preserves_every_expert(monkeypatch):
    attempts = []
    monkeypatch.setattr(alphaq_cache, '_initial_batch', lambda *_: 3)

    def collect(_, start, count, *__):
        attempts.append((start, count))
        if count > 1:
            raise torch.OutOfMemoryError('simulated allocation failure')
        return start

    assert list(alphaq_cache._collect_batches(None, (3, 4, 5), 'cpu', collect)) == [0, 1, 2]
    assert attempts == [(0, 3), (0, 1), (1, 1), (2, 1)]
