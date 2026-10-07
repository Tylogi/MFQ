from __future__ import annotations

import asyncio
import json
import struct
from datetime import datetime, timezone

import httpx
import pytest

from mfq.server.protocol.models import HubModelFile, HubModelInfo
from mfq.server.services.hub import HubCatalog, _model_variants, system_profile
from mfq.server.services.hub_metadata import (
    IncompleteHeaderError,
    estimated_resident_weight_bytes,
    inspect_mfq_header,
    read_mfq_metadata,
)


def mfq_index(records, config=None):
    def text(value):
        encoded = value.encode()
        return struct.pack('<I', len(encoded)) + encoded
    data = b'MFQ1' + struct.pack('<I', 2) + text('qwen4_exp')
    data += struct.pack('<I', 1 if config else 0)
    if config:
        data += text('hf_config') + text(json.dumps(config))
    data += struct.pack('<I', len(records))
    for name, dtype, size in records:
        data += text(name) + text(dtype) + struct.pack('<Q', size)
    return data, len(data) + sum(size for _, _, size in records)


def test_mfq_index_excludes_assets_and_streamed_ple_without_reading_payloads():
    data, size = mfq_index([
        ('__mfq_asset__/hf/tokenizer.json', 'BLOB', 1024),
        ('model.block.1.position_embedding.ngram.shard.0.weight', 'NINT', 3000),
        ('model.block.0.mlp.experts.gate.weight', 'MFE', 6000),
        ('model.block.1.position_embedding.key.weight', 'NINT', 100),
    ])
    result = inspect_mfq_header(data, size)
    assert result.weight_bytes == 6100 and result.ssd_ple_bytes == 3000
    assert result.weight_bytes_by_dtype == {'MFE': 6000, 'NINT': 100}
    assert not result.has_mtp_weights
    assert estimated_resident_weight_bytes(result.weight_bytes, result.weight_bytes_by_dtype) == 6710
    with pytest.raises(IncompleteHeaderError):
        inspect_mfq_header(data[:30], size)
    with pytest.raises(ValueError, match='file size'):
        inspect_mfq_header(data, size + 1)


@pytest.mark.parametrize('status', [200, 206])
def test_range_reads_are_bounded_and_never_accept_whole_weight_downloads(status):
    data, size = mfq_index([('model.weight', 'NINT', 1000000)])
    def respond(request):
        assert request.headers['range'] == 'bytes=0-65535'
        return httpx.Response(status, content=data + bytes(65536 - len(data)), headers={'content-range': f'bytes 0-65535/{size}'})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            if status == 200:
                with pytest.raises(ValueError, match='bounded'):
                    await read_mfq_metadata(client, 'https://modelscope.cn/file', None, size)
            else:
                result = await read_mfq_metadata(client, 'https://modelscope.cn/file', None, size)
                assert result.weight_bytes == 1000000
    asyncio.run(run())


def test_variant_estimate_excludes_ple_and_drives_memory_recommendation():
    profile = system_profile(backend='metal', runtime_memory_budget_bytes=80 << 30)
    variants = _model_variants([
        HubModelFile(name='model-00001-of-00002.mfq', byte_size=60 << 30, weight_bytes=30 << 30, ssd_ple_bytes=30 << 30),
        HubModelFile(name='model-00002-of-00002.mfq', byte_size=50 << 30, weight_bytes=50 << 30, ssd_ple_bytes=0),
    ], profile)
    assert variants[0].resident_weight_bytes == 80 << 30
    assert variants[0].ssd_ple_bytes == 30 << 30
    estimated = sum(size + (size + 9) // 10 for size in (30 << 30, 50 << 30))
    assert variants[0].estimated_resident_weight_bytes == estimated
    assert variants[0].configuration.required_memory_bytes == estimated
    assert variants[0].configuration.recommendation == 'caution'
    unknown = _model_variants([HubModelFile(name='model.mfq', byte_size=20 << 30)], profile)[0]
    assert unknown.resident_weight_bytes is None
    assert unknown.ssd_ple_bytes is None
    assert unknown.configuration.required_memory_bytes == (20 << 30) + ((20 << 30) + 9) // 10
    assert 'tensor metadata is unavailable' in unknown.configuration.reasons[-1]


@pytest.mark.parametrize('dtype', ['NINT', 'NINTv2', 'NINT5', 'MFE', 'NVQ3J-L', 'MXFP8', 'FP8-128SQ'])
def test_packed_weights_receive_storage_allowance_without_dense_expansion(dtype):
    assert estimated_resident_weight_bytes(1000, {dtype: 1000}) == 1100


def test_dense_weights_are_not_marked_up_and_missing_metadata_is_conservative():
    assert estimated_resident_weight_bytes(1000, {'F8_E4M3': 200, 'BF16': 300, 'NINT': 500}) == 1050
    assert estimated_resident_weight_bytes(1000, {'F32': 200, 'I32': 300}) == 1050
    assert estimated_resident_weight_bytes(1000, {}) == 1100
    assert estimated_resident_weight_bytes(1000, {'F16': 2000}) == 1100
    assert estimated_resident_weight_bytes(0, {}) == 0
    with pytest.raises(ValueError):
        HubModelFile(name='model.mfq', weight_bytes_by_dtype={'NINT': -1})


def test_dtype_sizes_canonicalize_and_exclude_fp8_ple():
    data, size = mfq_index([
        ('model.block.1.position_embedding.ngram.shard.0.weight', 'F8_E4M3', 3000),
        ('model.weight', 'NINT5', 1000),
        ('model.norm.weight', 'BF16', 100),
    ])
    metadata = inspect_mfq_header(data, size)
    assert metadata.weight_bytes_by_dtype == {'NINT': 1000, 'BF16': 100}
    assert metadata.ssd_ple_bytes == 3000
    profile = system_profile(backend='metal', runtime_memory_budget_bytes=1200)
    variant = _model_variants([HubModelFile(
        name='model.mfq', byte_size=size, weight_bytes=metadata.weight_bytes,
        weight_bytes_by_dtype=metadata.weight_bytes_by_dtype, ssd_ple_bytes=metadata.ssd_ple_bytes,
    )], profile)[0]
    assert variant.resident_weight_bytes == 1100
    assert variant.estimated_resident_weight_bytes == 1200
    assert variant.configuration.recommendation == 'three_stars'
    assert variant.ssd_ple_bytes == 3000


def test_partial_shard_metadata_uses_whole_file_fallback():
    variant = _model_variants([
        HubModelFile(name='model-00001-of-00002.mfq', byte_size=600, weight_bytes=300, ssd_ple_bytes=300),
        HubModelFile(name='model-00002-of-00002.mfq', byte_size=500),
    ], system_profile(backend='metal', runtime_memory_budget_bytes=2000))[0]
    assert variant.resident_weight_bytes is None and variant.ssd_ple_bytes is None
    assert variant.estimated_resident_weight_bytes == 1210


def test_mtp_detection_uses_real_predictor_records_not_config_declarations():
    data, size = mfq_index([('model.weight', 'NINT', 100)], {'mtp_num_hidden_layers': 1})
    assert not inspect_mfq_header(data, size).has_mtp_weights
    data, size = mfq_index([('predictor.block.0.attention.query.weight', 'NINT', 100)])
    assert inspect_mfq_header(data, size).has_mtp_weights


def test_embedded_config_reads_only_the_small_asset_range():
    config = {'model_type': 'deepseek_v4', 'num_hidden_layers': 43}
    payload = json.dumps(config).encode()
    data, size = mfq_index([
        ('blk.45.ffn_down.weight', 'MFE', 1000000),
        ('__mfq_asset__/model_config.json', 'BLOB', len(payload)),
    ])
    start = len(data) + 1000000
    calls = []
    def respond(request):
        calls.append(request.headers['range'])
        if len(calls) == 1:
            return httpx.Response(206, content=data + bytes(65536 - len(data)), headers={'content-range': f'bytes 0-65535/{size}'})
        assert request.headers['range'] == f'bytes={start}-{start + len(payload) - 1}'
        return httpx.Response(206, content=payload, headers={'content-range': f'bytes {start}-{start + len(payload) - 1}/{size}'})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            metadata = await read_mfq_metadata(client, 'https://modelscope.cn/file', None, size)
            assert metadata.config == config
            assert metadata.last_legacy_block == 45
            assert metadata.weight_bytes == 1000000
            assert len(calls) == 2
    asyncio.run(run())


def test_parameter_metadata_and_mtp_survive_catalog_snapshot(tmp_path):
    from mfq.server.protocol.models import ModelCacheProfile, ModelParameterBreakdown
    import time

    catalog = HubCatalog(cache_path=tmp_path / 'hub.json')
    key = ('modelscope', 'Tylogi/Qwen3.8-Flash-Next-EWQ-V1-MFQ')
    parameters = ModelParameterBreakdown(total=176, dense=4, routed_experts=121, ple=51, active=6)
    catalog._official_cache[key] = HubModelInfo(
        provider=key[0], repo_id=key[1], revision='master', files=[],
        parameter_breakdown=parameters, mtp_supported=True,
        cache_profile=ModelCacheProfile(max_context=32768, fixed_bytes=128),
    )
    catalog._source_cache_times[key] = time.monotonic()
    catalog._save_cache()
    restored = HubCatalog(cache_path=tmp_path / 'hub.json')
    model = next(m for m in restored._official_specs() if m.name == 'Qwen3.8-Flash-Next')
    public = restored._official_model(model, system_profile())
    assert public.parameter_breakdown == parameters
    assert public.mtp_supported is True
    assert public.cache_profile == ModelCacheProfile(max_context=32768, fixed_bytes=128)


def test_default_source_prefers_inspected_metadata_and_keeps_source_order_on_ties():
    from mfq.server.protocol.models import ModelCacheProfile

    catalog = HubCatalog()
    spec = next(item for item in catalog._official_specs() if item.name == 'DeepSeek-V4-Flash-0731')
    repo = spec.sources[0].repo_id
    hf = HubModelInfo(provider='huggingface', repo_id=repo, revision='main',
                      files=[HubModelFile(name='S4.mfq', byte_size=100)])
    ms = HubModelInfo(provider='modelscope', repo_id=repo, revision='master',
                      files=[HubModelFile(name='S4.mfq', byte_size=100, weight_bytes=90)],
                      cache_profile=ModelCacheProfile(max_context=32768, fixed_bytes=128), mtp_supported=False)
    catalog._official_cache[('huggingface', repo)] = hf
    catalog._official_cache[('modelscope', repo)] = ms
    public = catalog._official_model(spec, system_profile())
    assert public.selected_source.provider == 'modelscope'
    assert public.cache_profile == ms.cache_profile
    assert public.mtp_supported is False
    catalog._official_cache[('huggingface', repo)] = ms.model_copy(update={'provider': 'huggingface'})
    assert catalog._official_model(spec, system_profile()).selected_source.provider == 'huggingface'


def test_dynamic_model_identity_description_and_publication_sort(monkeypatch):
    catalog = HubCatalog()
    repo = 'Tylogi/Qwen3.8-Flash-Next-EWQ-V1-MFQ'
    catalog._discovered_sources.add(('modelscope', repo))
    catalog._official_cache[('modelscope', repo)] = HubModelInfo(
        provider='modelscope', repo_id=repo, revision='master',
        architectures=['Qwen4ExpForConditionalGeneration'], ple_parameter_count=51200000000,
        published_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        files=[HubModelFile(name='model.mfq', byte_size=100, weight_bytes=60, ssd_ple_bytes=40)],
    )
    async def refresh(profile):
        pass
    monkeypatch.setattr(catalog, '_refresh_official_cache', refresh)
    async def run():
        result = await catalog.official()
        assert result.data[0].name == 'Qwen3.8-Flash-Next'
        assert result.data[0].family == 'Qwen3.8-Flash-Next'
        assert '51.2B' in result.data[0].description_zh
        assert 'PLE' in result.data[0].description and 'SSD' in result.data[0].description
        await catalog.aclose()
    asyncio.run(run())
