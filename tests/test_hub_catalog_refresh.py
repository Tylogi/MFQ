from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from mfq.server.protocol.models import HubModelFile, HubModelInfo, HubSystemProfile
from mfq.server.services import hub


PROFILE = HubSystemProfile(platform='test', machine='test', backend='unknown')
OLD = 'Tylogi/Qwen3.8-Flash-Next-EWQ-V1-MFQ'
NEW = 'Tylogi/Qwen3.8-Flash-Next-EW-MFQ'


def model(provider, repo):
    return HubModelInfo(provider=provider, repo_id=repo, revision='master',
        files=[HubModelFile(name='S4.mfq', byte_size=42, weight_bytes=30, ssd_ple_bytes=12)])


def remember(catalog, provider, repo):
    key = (provider, repo)
    catalog._discovered_sources.add(key)
    catalog._official_cache[key] = model(provider, repo)
    catalog._source_cache_times[key] = time.monotonic()
    catalog._detail_attempt_times[key] = time.monotonic()
    catalog._official_cache_time = time.monotonic()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(hub, '_metadata_client', lambda *args, **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))))


def test_rename_refresh_merges_sources_and_does_not_restore_old_card(tmp_path, monkeypatch):
    async def run():
        path = tmp_path / 'hub.json'
        catalog = hub.HubCatalog(cache_path=path)
        remember(catalog, 'huggingface', NEW)
        remember(catalog, 'modelscope', OLD)
        catalog._save_cache()
        assert len((await catalog.official(profile=PROFILE)).data) == 2

        async def discover(provider):
            return [NEW]

        async def fetch(provider, repo, revision, profile):
            if repo == OLD:
                raise hub._RepositoryNotFound('repository renamed')
            return model(provider, repo)

        monkeypatch.setattr(catalog, '_discover_official', discover)
        monkeypatch.setattr(catalog, '_fetch_info', fetch)
        pending = await catalog.official(profile=PROFILE, refresh=True)
        task = catalog._refresh_task
        assert pending.refreshing
        assert (await catalog.official(profile=PROFILE, refresh=True)).refreshing
        assert catalog._refresh_task is task
        await task
        result = await catalog.official(profile=PROFILE)
        assert not result.refreshing and len(result.data) == 1
        assert {(source.provider, source.repo_id) for source in result.data[0].sources} == {
            ('huggingface', NEW), ('modelscope', NEW)}
        assert all(source.available for source in result.data[0].sources)
        assert ('modelscope', OLD) not in catalog._official_cache
        assert ('modelscope', OLD) not in catalog._discovered_sources
        assert ('modelscope', OLD) not in catalog._source_cache_times
        assert ('modelscope', OLD) not in catalog._detail_attempt_times
        restored = hub.HubCatalog(cache_path=path)
        assert len((await restored.official(profile=PROFILE)).data) == 1
        assert ('modelscope', OLD) not in restored._official_cache
        await catalog.aclose()
        await restored.aclose()
    asyncio.run(run())


def test_complete_listing_removes_absent_repo_after_inflight_metadata_finishes(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        remember(catalog, 'modelscope', OLD)
        listed = asyncio.Event()

        async def discover(provider):
            listed.set()
            return []

        async def fetch(provider, repo, revision, profile):
            await listed.wait()
            await asyncio.sleep(0)
            catalog._official_cache[(provider, repo)] = model(provider, repo)
            return model(provider, repo)

        monkeypatch.setattr(catalog, '_discover_official', discover)
        monkeypatch.setattr(catalog, '_fetch_info', fetch)
        await catalog._refresh_official_cache(PROFILE)
        assert not catalog._discovered_sources and not catalog._official_cache
        await catalog.aclose()
    asyncio.run(run())


def test_failed_provider_listing_keeps_its_cache_while_other_provider_is_reconciled(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        remember(catalog, 'huggingface', NEW)
        remember(catalog, 'modelscope', OLD)
        old_time = catalog._source_cache_times[('huggingface', NEW)]

        async def discover(provider):
            if provider == 'huggingface':
                raise hub.HubError('temporary outage')
            return []

        async def fetch(*args):
            raise hub.HubError('temporary outage')

        monkeypatch.setattr(catalog, '_discover_official', discover)
        monkeypatch.setattr(catalog, '_fetch_info', fetch)
        await catalog._refresh_official_cache(PROFILE)
        assert catalog._discovered_sources == {('huggingface', NEW)}
        assert set(catalog._official_cache) == {('huggingface', NEW)}
        assert catalog._source_cache_times[('huggingface', NEW)] == old_time
        await catalog.aclose()
    asyncio.run(run())


def test_confirmed_missing_repo_is_removed_even_when_author_listing_is_offline(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        remember(catalog, 'modelscope', OLD)

        async def discover(*args):
            raise hub.HubError('temporary author-list outage')

        async def fetch(*args):
            raise hub._RepositoryNotFound('repository deleted')

        monkeypatch.setattr(catalog, '_discover_official', discover)
        monkeypatch.setattr(catalog, '_fetch_info', fetch)
        await catalog._refresh_official_cache(PROFILE)
        assert not catalog._discovered_sources and not catalog._official_cache
        await catalog.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('repo', [
    'Tylogi/Qwen3.6-27B-MFQ', 'Tylogi/Qwen3.8-27B-MFQ',
    'Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ', 'Tylogi/MiniCPM-o-4_5-MFQ',
])
@pytest.mark.parametrize('status', [401, 404])
def test_deleted_repositories_have_no_exemptions_and_stay_absent_after_restart(tmp_path, monkeypatch, repo, status):
    async def run():
        path = tmp_path / 'hub.json'
        catalog = hub.HubCatalog(cache_path=path)
        remember(catalog, 'huggingface', repo)
        catalog._save_cache()
        catalog = hub.HubCatalog(cache_path=path)

        async def discover(*args):
            return []

        monkeypatch.setattr(catalog, '_discover_official', discover)
        monkeypatch.setattr(hub, '_metadata_client', lambda *args, **kwargs:
            httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status))))
        await catalog._refresh_official_cache(PROFILE)
        assert not catalog._official_cache and not catalog._discovered_sources
        assert not catalog._source_cache_times and not catalog._detail_attempt_times
        current = await catalog.official(profile=PROFILE)
        assert not current.data and not current.refreshing
        restored = hub.HubCatalog(cache_path=path)
        current = await restored.official(profile=PROFILE)
        assert not current.data and not current.refreshing
        await restored.aclose()
        await catalog.aclose()
    asyncio.run(run())


def test_empty_catalog_has_no_models_before_discovery_or_when_offline(monkeypatch):
    async def run():
        catalog = hub.HubCatalog()
        assert not catalog._official_specs()
        async def offline(*args):
            raise hub.HubError('offline')
        monkeypatch.setattr(catalog, '_discover_official', offline)
        assert not (await catalog.official(profile=PROFILE)).data
        await catalog._refresh_task
        result = await catalog.official(profile=PROFILE)
        assert not result.data and not result.refreshing
        await catalog.aclose()
    asyncio.run(run())


def test_deleting_one_provider_preserves_other_provider_and_metadata(tmp_path, monkeypatch):
    async def run():
        path = tmp_path / 'hub.json'
        catalog = hub.HubCatalog(cache_path=path)
        for provider in ('huggingface', 'modelscope'):
            remember(catalog, provider, NEW)
        async def discover(provider):
            return [NEW] if provider == 'modelscope' else []
        async def fetch(provider, repo, revision, profile):
            if provider == 'huggingface':
                raise hub.HubError('401')
            return model(provider, repo)
        monkeypatch.setattr(catalog, '_discover_official', discover)
        monkeypatch.setattr(catalog, '_fetch_info', fetch)
        await catalog._refresh_official_cache(PROFILE)
        restored = hub.HubCatalog(cache_path=path)
        result = await restored.official(profile=PROFILE)
        assert len(result.data) == 1
        assert [(s.provider, s.available) for s in result.data[0].sources] == [('modelscope', True)]
        assert result.data[0].variants[0].ssd_ple_bytes == 12
        await catalog.aclose()
        await restored.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('provider', ['huggingface', 'modelscope'])
@pytest.mark.parametrize('status', [404, 410, 401, 403, 429, 503])
def test_repository_http_errors_distinguish_missing_from_temporary_or_auth_failures(monkeypatch, provider, status):
    monkeypatch.setattr(hub, '_metadata_client', lambda *args, **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status))))
    catalog = hub.HubCatalog()
    expected = hub._RepositoryNotFound if status in {404, 410} else hub.HubError
    with pytest.raises(expected) as caught:
        asyncio.run(catalog._fetch_info(provider, NEW, None, PROFILE))
    assert isinstance(caught.value, hub._RepositoryNotFound) == (status in {404, 410})


@pytest.mark.parametrize('provider', ['huggingface', 'modelscope'])
def test_incomplete_pagination_is_not_treated_as_a_complete_author_listing(monkeypatch, provider):
    def respond(request):
        if provider == 'huggingface':
            return httpx.Response(200, json=[{'id': NEW}],
                headers={'Link': '<https://huggingface.co/api/models?cursor=next>; rel="next"'})
        return httpx.Response(200, json={'success': True, 'data': {
            'models': [{'id': NEW}] * 20, 'total_count': 100000}})

    monkeypatch.setattr(hub, '_metadata_client', lambda *args, **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    with pytest.raises(hub.HubError, match='pagination did not complete'):
        asyncio.run(hub.HubCatalog._discover_official(provider))


@pytest.mark.parametrize('provider,payload', [
    ('huggingface', {}), ('huggingface', [None]), ('huggingface', [{}]),
    ('modelscope', []),
    ('modelscope', {'success': True}),
    ('modelscope', {'success': True, 'data': {'models': {}}}),
    ('modelscope', {'success': True, 'data': {'models': [{}]}}),
])
def test_invalid_author_response_cannot_erase_the_catalog(monkeypatch, provider, payload):
    monkeypatch.setattr(hub, '_metadata_client', lambda *args, **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))))
    with pytest.raises(hub.HubError, match='Invalid'):
        asyncio.run(hub.HubCatalog._discover_official(provider))


def test_missing_file_listing_does_not_mark_an_existing_repository_deleted(monkeypatch):
    def respond(request):
        if request.url.path.endswith('/repo/files'):
            return httpx.Response(404)
        return httpx.Response(200, json={'success': True, 'data': {}})

    monkeypatch.setattr(hub, '_metadata_client', lambda *args, **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    with pytest.raises(hub.HubError) as caught:
        asyncio.run(hub.HubCatalog._info_modelscope(NEW, None, PROFILE))
    assert not isinstance(caught.value, hub._RepositoryNotFound)
