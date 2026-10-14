import math
from types import SimpleNamespace

import pytest

from mfq.server.protocol.models import KvMemorySettings, KvQuantizationSettings
from mfq.server.runtime.kv_budget import kv_resident_fraction, kv_tier_budget, checkpoint_kv_budget, check_ssd_capacity
from mfq.server.services.model_memory import cache_profile


def test_auto_defaults_and_smooth_budget():
    settings = KvMemorySettings()
    assert settings.placement == 'auto' and settings.ram_preallocated and settings.gpu_budget_bytes is None
    for context, fraction in [(0, 1), (131072, .6), (262144, .3), (1048576, .1)]:
        assert kv_resident_fraction(context) == pytest.approx(fraction)
        if context > 0:
            assert abs(kv_resident_fraction(context - 1) - fraction) < 1e-5
            assert abs(kv_resident_fraction(context + 1) - fraction) < 1e-5
    samples = [kv_resident_fraction(context) for context in range(0, 1200000, 128)]
    assert all(a >= b for a, b in zip(samples, samples[1:]))
    plan = kv_tier_budget(131072, 1024, 10000)
    assert plan.gpu_budget_bytes == 7024 and plan.ram_required_bytes == 4000
    assert kv_tier_budget(1048576, 1024, 10000, 1024).ram_required_bytes == 10000
    with pytest.raises(ValueError, match='kv_budget_below_indexer'):
        kv_tier_budget(131072, 1024, 10000, 1023)


@pytest.mark.parametrize('placement', ['auto', 'auto_ssd', 'vram', 'ram'])
@pytest.mark.parametrize('preallocated', [True, False])
@pytest.mark.parametrize('gpu_budget', [None, 123456])
@pytest.mark.parametrize('ram_budget', [None, 0, 123456])
def test_storage_configuration_matrix(placement, preallocated, gpu_budget, ram_budget):
    payload = dict(placement=placement, ram_preallocated=preallocated, gpu_budget_bytes=gpu_budget, ram_budget_bytes=ram_budget)
    if not preallocated and placement != 'auto_ssd':
        with pytest.raises(ValueError, match='kv_ssd_required'):
            KvMemorySettings(**payload)
    else:
        result = KvMemorySettings(**payload)
        assert KvMemorySettings.model_validate(result.model_dump()) == result


@pytest.mark.parametrize('bits', [None, 2, 2.5, 3, 3.5, 4, 6, 8])
def test_cuda_geometry_matches_packed_rows_and_full_indexer(monkeypatch, bits):
    profile = cache_profile(dict(model_type='qwen4_exp', max_position_embeddings=1048576, num_hidden_layers=4,
        layer_types=['linear_attention'] * 3 + ['full_attention'], num_key_value_heads=2, head_dim=256,
        indexer_head_dim=128, indexer_compress_ratio=8), include_recurrent=False, predictor_layers=1, cuda=True)
    assert profile is not None
    monkeypatch.setattr('mfq.server.services.model_memory.checkpoint_cache_profile', lambda *a, **kw: profile)
    plan = checkpoint_kv_budget(None, 65537, KvMemorySettings(), KvQuantizationSettings(enabled=bits is not None, bits=bits or 4))
    assert plan.indexer_bytes == 2 * 128 * (65537 + math.ceil(65537 / 8)) * 2
    row_bytes = 256 * 4 if bits is None else (2 + math.ceil(256 * math.floor(bits)/32) + math.ceil(256 * math.ceil(bits)/32))*4
    assert plan.raw_bytes == 65537 * 2 * 2 * row_bytes


def test_ssd_check_rejects_capacity_shortfall_without_creating_directory(tmp_path, monkeypatch):
    monkeypatch.setattr('shutil.disk_usage', lambda path: SimpleNamespace(free=1024))
    directory = tmp_path / 'not-created' / 'kv'
    check_ssd_capacity(directory, 1024)
    with pytest.raises(ValueError, match='kv_ssd_insufficient'):
        check_ssd_capacity(directory, 1025)
    assert not directory.exists()


@pytest.mark.parametrize('placement', ['auto', 'auto_ssd', 'vram', 'ram'])
@pytest.mark.parametrize('allocation', ['preallocated', 'dynamic'])
@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('bits', [None, 2.5, 4])
def test_placement_allocation_and_precision_budget(monkeypatch, placement, allocation, explicit, bits):
    profile = cache_profile(dict(model_type='qwen4_exp', max_position_embeddings=1048576, num_hidden_layers=1,
        layer_types=['full_attention'], num_key_value_heads=1, head_dim=256,
        indexer_head_dim=128, indexer_compress_ratio=8), include_recurrent=False, cuda=True)
    monkeypatch.setattr('mfq.server.services.model_memory.checkpoint_cache_profile', lambda *a, **kw: profile)
    settings = KvMemorySettings(placement=placement, allocation=allocation, gpu_budget_bytes=(1 << 30) if explicit else None)
    quantization = KvQuantizationSettings(enabled=bits is not None, bits=bits or 4)
    if allocation == 'dynamic' and placement in {'ram', 'auto_ssd'}:
        with pytest.raises(ValueError, match='kv_dynamic_unavailable'):
            checkpoint_kv_budget(None, 131072, settings, quantization)
        return
    plan = checkpoint_kv_budget(None, 131072, settings, quantization)
    if explicit:
        assert plan.gpu_budget_bytes == 1 << 30
    elif placement == 'vram' or allocation == 'dynamic':
        assert plan.gpu_budget_bytes == plan.indexer_bytes + plan.raw_bytes
    elif placement == 'ram':
        assert plan.gpu_budget_bytes == plan.indexer_bytes
    else:
        assert plan.gpu_budget_bytes == plan.indexer_bytes + math.ceil(plan.raw_bytes * .6)
    if placement == 'ram':
        assert plan.ram_required_bytes == plan.raw_bytes
    elif explicit or allocation == 'dynamic' or placement == 'vram':
        assert plan.ram_required_bytes == 0


@pytest.mark.parametrize('preallocated,ram_limit,expected_ssd', [(True, None, 64 << 20), (True, 0, 72 << 20),
    (True, 2 << 20, 70 << 20), (False, None, 72 << 20), (False, 0, 72 << 20)])
def test_ssd_admission_uses_ram_preallocation(tmp_path, monkeypatch, preallocated, ram_limit, expected_ssd):
    from mfq.server.protocol.models import ModelLoadRequest
    from mfq.server.runtime.runtime_pool import RuntimePool
    from mfq.server.runtime.kv_budget import KvTierBudget
    plan = KvTierBudget(1 << 20, 64 << 20, 57 << 20, 8 << 20, .6)
    monkeypatch.setattr('mfq.server.runtime.runtime_pool.checkpoint_kv_budget', lambda *a: plan)
    checks = []
    monkeypatch.setattr('mfq.server.runtime.runtime_pool.check_ssd_capacity', lambda path, size: checks.append(size))
    pool = SimpleNamespace(resolve_context_size=lambda *a, **kw: 131072,
        _kv_ram_limit=lambda *a: (16 << 20) if ram_limit is None else ram_limit,
        _prefix_cache_directory=lambda: tmp_path)
    request = ModelLoadRequest(model='fixture', kv_memory=dict(placement='auto_ssd', ram_preallocated=preallocated, ram_budget_bytes=ram_limit))
    RuntimePool._validate_kv_memory_budget(pool, None, request)
    assert checks == [expected_ssd]
