import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from mfq.calibration import loading
from mfq.server.protocol.quantization import WorkbenchImatrixPayload, Wt2ReferencePayload


class Backend:
    def __init__(self, failure=None):
        self.device = torch.device('cpu')
        self.teacher_dtype = torch.float32
        self.root = '/source'
        self.num_layers = 3
        self.hidden_size = 4
        self.config = SimpleNamespace(model_type='qwen3_5')
        self.index = SimpleNamespace(tensor=lambda *args, **kwargs: torch.arange(32).reshape(8, 4).float())
        self.calls = []
        self.references = []
        self.failure = failure

    @contextmanager
    def layer(self, index, *, quantized):
        assert not quantized
        self.calls.append(index)
        if index == 1 and self.failure:
            error, self.failure = self.failure, None
            raise error
        layer = torch.nn.Linear(4, 4)
        self.references.append(weakref.ref(layer))
        yield layer


def test_payloads_default_layerwise():
    assert WorkbenchImatrixPayload(model='x', corpus='x', output='x').layerwise
    assert Wt2ReferencePayload(model='x', dataset_id='00000000-0000-0000-0000-000000000001', output='x').layerwise


def test_layerwise_releases_previous_layer():
    backend = Backend()
    with loading.SourceLayers(backend) as source:
        for index in range(backend.num_layers):
            with source.layer(index) as layer:
                assert layer is not None
                assert sum(reference() is not None for reference in backend.references) == 1
            del layer
            gc.collect()
            assert all(reference() is None for reference in backend.references)


@pytest.mark.parametrize('allowed', [False, True])
def test_resident_memory_check_and_actual_residency(monkeypatch, allowed):
    monkeypatch.setattr(loading, 'loading_plan', lambda *args, **kwargs: {'resident_allowed': allowed})
    backend = Backend()
    with loading.SourceLayers(backend, layerwise=False) as source:
        assert source.layerwise is not allowed
        assert source.fallback_reason == (None if allowed else 'insufficient_memory')
        assert backend.calls == ([0, 1, 2] if allowed else [])
        for _ in range(2):
            with source.layer(0) as layer:
                assert layer is not None
            del layer
        assert backend.calls == ([0, 1, 2] if allowed else [0, 0])
    assert all(reference() is None for reference in backend.references)


def test_out_of_memory_clears_partially_loaded_layers_and_falls_back(monkeypatch):
    monkeypatch.setattr(loading, 'loading_plan', lambda *args, **kwargs: {'resident_allowed': True})
    backend = Backend(torch.OutOfMemoryError('allocation failed'))
    with loading.SourceLayers(backend, layerwise=False) as source:
        assert source.layerwise
        assert source.fallback_reason == 'insufficient_memory'
        assert all(reference() is None for reference in backend.references)
        with source.layer(1) as layer:
            assert layer is not None
        del layer


def test_other_model_errors_are_not_mislabeled_as_memory_pressure(monkeypatch):
    monkeypatch.setattr(loading, 'loading_plan', lambda *args, **kwargs: {'resident_allowed': True})
    backend = Backend(RuntimeError('bad model weights'))
    with pytest.raises(RuntimeError, match='bad model weights'), loading.SourceLayers(backend, layerwise=False):
        pass
    assert all(reference() is None for reference in backend.references)


def test_loading_plan_uses_free_memory_and_expanded_weights(monkeypatch):
    from mfq.quantize import workbench
    source = SimpleNamespace(path=SimpleNamespace(is_dir=lambda: True), config={'hidden_size': 16, 'num_attention_heads': 2},
        parameters=lambda: 1_000_000_000, close=lambda: None)
    monkeypatch.setattr(workbench, 'ModelSource', lambda path: source)
    monkeypatch.setattr(loading.psutil, 'virtual_memory', lambda: SimpleNamespace(total=128 << 30, available=1 << 30))
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch.mps, 'recommended_max_memory', lambda: 96 << 30)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 0)
    plan = loading.loading_plan('/source', backend='metal', context_size=32)
    assert plan['weights_bytes'] == 2_000_000_000
    assert plan['available_bytes'] == 0
    assert not plan['resident_allowed']
    monkeypatch.setattr(loading.psutil, 'virtual_memory', lambda: SimpleNamespace(total=128 << 30, available=32 << 30))
    assert loading.loading_plan('/source', backend='metal', context_size=32)['resident_allowed']


def test_unavailable_device_cannot_pass_memory_check(monkeypatch):
    from mfq.quantize import workbench
    source = SimpleNamespace(path=SimpleNamespace(is_dir=lambda: True), config={'hidden_size': 16}, parameters=lambda: 1, close=lambda: None)
    monkeypatch.setattr(workbench, 'ModelSource', lambda path: source)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert not loading.loading_plan('/source', backend='metal')['resident_allowed']
