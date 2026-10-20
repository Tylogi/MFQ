from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from mfq.server.protocol.models import HubModelFile, HubModelInfo, ModelCacheProfile, ModelParameterBreakdown
from mfq.server.services import hub
from mfq.server.services.hub_metadata import MfqMetadata


REPO = 'Tylogi/Flash-Next-MFQ'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(hub, '_metadata_client', lambda *args, **kwargs:
                        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500))))


def model(files):
    return HubModelInfo(provider='huggingface', repo_id=REPO, revision='a' * 40, files=files,
                        architectures=['qwen4_exp'], config_inspected=True,
                        parameter_breakdown=ModelParameterBreakdown(total=100, dense=40, routed_experts=50, ple=10, active=45),
                        cache_profile=ModelCacheProfile(max_context=32768), mtp_supported=True)


def test_header_deadline_starts_after_queue_wait_and_does_not_limit_repository(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        info = model([HubModelFile(name=f'model-{index + 1:05d}-of-00028.mfq', byte_size=100) for index in range(28)])
        active = maximum = completed = 0
        async def listing(*args):
            return info
        async def read(*args):
            nonlocal active, maximum, completed
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.025)
                completed += 1
                return MfqMetadata('qwen4_exp', {}, 60, 40)
            finally:
                active -= 1
        monkeypatch.setattr(hub, '_METADATA_DEADLINE', 0.06)
        monkeypatch.setattr(catalog, '_info_huggingface', listing)
        monkeypatch.setattr(hub, 'read_mfq_metadata', read)
        result = await catalog._fetch_info('huggingface', REPO, None, hub.system_profile())
        assert completed == 28 and maximum == 8
        assert all(item.ssd_ple_bytes == 40 for item in result.files)
        assert result.variants[0].ssd_ple_bytes == 28 * 40
        await catalog.aclose()
    asyncio.run(run())


def test_more_than_256_headers_are_not_silently_skipped(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        info = model([HubModelFile(name=f'{index}.mfq', byte_size=100) for index in range(270)])
        async def read(*args):
            return MfqMetadata('qwen4_exp', {}, 60, 40)
        monkeypatch.setattr(hub, 'read_mfq_metadata', read)
        result = await catalog._inspect_mfq_files(info, hub.system_profile())
        assert all(item.ssd_ple_bytes == 40 for item in result.files)
        assert len(result.variants) == 270
        await catalog.aclose()
    asyncio.run(run())


def test_headers_publish_incrementally_and_close_cancels_inflight_reads(tmp_path, monkeypatch):
    async def run():
        catalog = hub.HubCatalog(cache_path=tmp_path / 'hub.json')
        info = model([HubModelFile(name='first.mfq', byte_size=100), HubModelFile(name='slow.mfq', byte_size=100)])
        blocked = asyncio.Event()
        canceled = asyncio.Event()
        async def listing(*args):
            return info
        async def read(client, url, params, size):
            if 'slow.mfq' in url:
                blocked.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    canceled.set()
            return MfqMetadata('qwen4_exp', {}, 60, 40)
        monkeypatch.setattr(catalog, '_info_huggingface', listing)
        monkeypatch.setattr(hub, 'read_mfq_metadata', read)
        task = asyncio.create_task(catalog._fetch_info('huggingface', REPO, None, hub.system_profile()))
        catalog._refresh_task = task
        await blocked.wait()
        async def wait_for_progress():
            while catalog._official_cache[('huggingface', REPO)].files[0].ssd_ple_bytes is None:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait_for_progress(), 0.5)
        partial = await asyncio.wait_for(catalog.official(), 0.5)
        assert partial.refreshing and not task.done()
        assert catalog._official_cache[('huggingface', REPO)].files[1].ssd_ple_bytes is None
        await catalog.aclose()
        assert canceled.is_set() and task.cancelled()
    asyncio.run(run())


def test_incomplete_snapshot_repairs_only_missing_header_without_waiting_for_catalog_ttl(tmp_path, monkeypatch):
    async def run():
        path = tmp_path / 'hub.json'
        catalog = hub.HubCatalog(cache_path=path)
        info = model([
            HubModelFile(name='model-00001-of-00002.mfq', byte_size=100, sha256='a' * 64),
            HubModelFile(name='model-00002-of-00002.mfq', byte_size=100, sha256='b' * 64),
        ])
        async def listing(*args):
            return info
        async def partial_read(client, url, params, size):
            if '-00002-of-' in url:
                raise ValueError('temporary invalid range response')
            return MfqMetadata('qwen4_exp', {}, 60, 40)
        monkeypatch.setattr(catalog, '_info_huggingface', listing)
        monkeypatch.setattr(hub, 'read_mfq_metadata', partial_read)
        await catalog._fetch_info('huggingface', REPO, None, hub.system_profile())
        catalog._official_cache_time = time.monotonic()
        catalog._source_cache_times[('huggingface', REPO)] = time.monotonic()
        catalog._save_cache()
        restored = hub.HubCatalog(cache_path=path)
        calls = []
        async def repair(client, url, params, size):
            calls.append(url)
            return MfqMetadata('qwen4_exp', {}, 100, 0)
        async def no_listing(*args):
            pytest.fail('fresh repository listing should not be fetched again')
        monkeypatch.setattr(restored, '_fetch_info', no_listing)
        monkeypatch.setattr(hub, 'read_mfq_metadata', repair)
        assert (await restored.official()).refreshing
        task = restored._refresh_task
        assert (await restored.official()).refreshing and restored._refresh_task is task
        await task
        assert len(calls) == 1 and '-00002-of-' in calls[0]
        result = restored._official_cache[('huggingface', REPO)]
        assert result.variants[0].ssd_ple_bytes == 40
        assert result.variants[0].resident_weight_bytes == 160
        assert result.parameter_breakdown == info.parameter_breakdown
        assert result.cache_profile == info.cache_profile and result.mtp_supported is True
        assert not (await restored.official()).refreshing
        reloaded = hub.HubCatalog(cache_path=path)
        assert not (await reloaded.official()).refreshing
        assert reloaded._official_cache[('huggingface', REPO)].files[0].ssd_ple_bytes == 40
        await catalog.aclose()
        await restored.aclose()
        await reloaded.aclose()
    asyncio.run(run())


def test_failed_details_are_throttled_and_retried_without_refetching_successes(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        key = ('huggingface', REPO)
        catalog._official_cache[key] = model([
            HubModelFile(name='good.mfq', byte_size=100, weight_bytes=60, ssd_ple_bytes=40),
            HubModelFile(name='bad.mfq', byte_size=100),
        ])
        catalog._official_cache_time = time.monotonic()
        calls = []
        async def read(client, url, params, size):
            calls.append(url)
            raise httpx.ReadTimeout('offline')
        monkeypatch.setattr(hub, 'read_mfq_metadata', read)
        assert (await catalog.official()).refreshing
        await catalog._refresh_task
        assert not (await catalog.official()).refreshing
        assert len(calls) == 1 and 'bad.mfq' in calls[0]
        catalog._detail_attempt_times[key] -= hub._MFQ_DETAIL_RETRY_TTL
        assert (await catalog.official()).refreshing
        await catalog._refresh_task
        assert len(calls) == 2 and catalog._official_cache[key].files[0].ssd_ple_bytes == 40
        await catalog.aclose()
    asyncio.run(run())


def test_a_stuck_header_is_bounded_without_canceling_other_headers(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        info = model([HubModelFile(name='stuck.mfq', byte_size=100), HubModelFile(name='good.mfq', byte_size=100)])
        canceled = asyncio.Event()
        async def read(client, url, params, size):
            if 'stuck.mfq' in url:
                try:
                    await asyncio.Event().wait()
                finally:
                    canceled.set()
            return MfqMetadata('qwen4_exp', {}, 60, 40)
        monkeypatch.setattr(hub, '_METADATA_DEADLINE', 0.02)
        monkeypatch.setattr(hub, 'read_mfq_metadata', read)
        result = await catalog._inspect_mfq_files(info, hub.system_profile())
        assert canceled.is_set()
        assert result.files[0].ssd_ple_bytes is None and result.files[1].ssd_ple_bytes == 40
        await catalog.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('revision,sha256,reuse', [
    ('b' * 40, 'a' * 64, True), ('b' * 40, 'b' * 64, False),
    ('main', None, False), ('a' * 40, None, True), ('b' * 40, None, False),
])
def test_file_reuse_checks_hash_or_immutable_revision_not_just_name_and_size(revision, sha256, reuse):
    catalog = hub.HubCatalog()
    original = model([HubModelFile(name='model.mfq', byte_size=100, sha256='a' * 64 if sha256 else None,
                                   weight_bytes=60, ssd_ple_bytes=40)])
    if revision == 'main':
        original = original.model_copy(update={'revision': 'main'})
    catalog._official_cache[('huggingface', REPO)] = original
    incoming = model([HubModelFile(name='model.mfq', byte_size=100, sha256=sha256)]).model_copy(update={'revision': revision})
    result = catalog._reuse_mfq_files(incoming)
    assert (result.files[0].ssd_ple_bytes == 40) is reuse


def test_verified_cross_provider_reuse_recomputes_variants_without_network(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        info = model([HubModelFile(name='model.mfq', byte_size=100, sha256='a' * 64)])
        cached = info.model_copy(update={'provider': 'modelscope', 'revision': 'master',
                                        'files': [info.files[0].model_copy(update={'weight_bytes': 60, 'ssd_ple_bytes': 40})]})
        catalog._official_cache[('modelscope', REPO)] = cached
        async def no_read(*args):
            pytest.fail('SHA256-identical file should reuse the inspected header')
        monkeypatch.setattr(hub, 'read_mfq_metadata', no_read)
        result = await catalog._inspect_mfq_files(catalog._reuse_mfq_files(info), hub.system_profile())
        assert result.variants[0].ssd_ple_bytes == 40 and result.variants[0].resident_weight_bytes == 60
        assert result.cache_profile == info.cache_profile
        await catalog.aclose()
    asyncio.run(run())


def test_complete_file_bytes_with_missing_model_metadata_repair_and_persist(tmp_path, monkeypatch):
    from tests.test_flash_next_architecture import _qwen_config
    async def run():
        path = tmp_path / 'hub.json'
        catalog = hub.HubCatalog(cache_path=path)
        key = ('huggingface', REPO)
        files = [HubModelFile(name=f'model-{i:05d}-of-00043.mfq', byte_size=100,
            weight_bytes=60, ssd_ple_bytes=40, sha256=f'{i:064x}') for i in range(1, 44)]
        catalog._official_cache[key] = HubModelInfo(provider='huggingface', repo_id=REPO, revision='a' * 40, files=files)
        catalog._official_cache_time = time.monotonic()
        calls = []
        async def read(client, url, params, size):
            calls.append(url)
            return MfqMetadata('qwen4_exp', {**_qwen_config(), 'language_model_only': True}, 60, 40, has_mtp_weights='-00043-of-' in url)
        monkeypatch.setattr(hub, 'read_mfq_metadata', read)
        assert (await catalog.official()).refreshing
        await catalog._refresh_task
        repaired = catalog._official_cache[key]
        assert len(calls) == 2
        assert repaired.parameter_breakdown is not None and repaired.cache_profile is not None
        assert repaired.attention_profile.model_dump() == dict(sparse=True, compressed=False, hybrid=True)
        assert repaired.mtp_supported and repaired.config_inspected
        assert all(f.weight_bytes == 60 and f.ssd_ple_bytes == 40 for f in repaired.files)
        restored = hub.HubCatalog(cache_path=path)
        assert not (await restored.official()).refreshing
        assert restored._official_cache[key].parameter_breakdown == repaired.parameter_breakdown
        await catalog.aclose()
        await restored.aclose()
    asyncio.run(run())


def test_empty_model_metadata_donor_does_not_hide_complete_donor():
    catalog = hub.HubCatalog()
    file = HubModelFile(name='model.mfq', byte_size=100, sha256='a' * 64, weight_bytes=60, ssd_ple_bytes=40)
    empty = HubModelInfo(provider='huggingface', repo_id=REPO, revision='a' * 40, files=[file])
    complete = model([file]).model_copy(update={'provider': 'modelscope'})
    catalog._official_cache = {('huggingface', REPO): empty, ('modelscope', REPO): complete}
    result = catalog._reuse_mfq_files(empty)
    assert result.parameter_breakdown == complete.parameter_breakdown
    assert result.config_inspected and result.architectures == complete.architectures
