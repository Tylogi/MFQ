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


def test_variant_weight_baseline_excludes_ple_and_has_no_arbitrary_headroom():
    profile = system_profile(backend='metal', runtime_memory_budget_bytes=80 << 30)
    variants = _model_variants([
        HubModelFile(name='model-00001-of-00002.mfq', byte_size=60 << 30, weight_bytes=30 << 30, ssd_ple_bytes=30 << 30),
        HubModelFile(name='model-00002-of-00002.mfq', byte_size=50 << 30, weight_bytes=50 << 30, ssd_ple_bytes=0),
    ], profile)
    assert variants[0].resident_weight_bytes == 80 << 30
    assert variants[0].ssd_ple_bytes == 30 << 30
    assert variants[0].configuration.required_memory_bytes == 80 << 30
    assert variants[0].configuration.recommendation == 'three_stars'
    unknown = _model_variants([HubModelFile(name='model.mfq', byte_size=20 << 30)], profile)[0]
    assert unknown.resident_weight_bytes is None
    assert unknown.configuration.required_memory_bytes == 20 << 30


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
