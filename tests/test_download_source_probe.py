import asyncio
from types import SimpleNamespace

import httpx
import pytest

from mfq.server.api import network
from mfq.server.protocol.models import HubDownloadSource, HubModelFile
from mfq.server.services.hub import HubCatalog


@pytest.mark.parametrize('direct_fails', [False, True])
def test_route_probe_follows_file_redirects_and_reuses_winner_for_download(monkeypatch, direct_fails):
    actual_client = httpx.AsyncClient
    calls = []
    async def respond(request):
        assert request.headers['range'] == 'bytes=0-1048575'
        if request.url.host == 'hub.test':
            return httpx.Response(302, headers={'location': 'https://cdn.test/weights'})
        return httpx.Response(206, stream=httpx.ByteStream(b'x' * 1048576))
    def client(**kwargs):
        proxy = kwargs.get('proxy')
        calls.append(proxy)
        async def handle(request):
            if not proxy and direct_fails:
                raise httpx.ConnectError('direct unavailable', request=request)
            if proxy and not direct_fails:
                await asyncio.sleep(.05)
            return await respond(request)
        return actual_client(transport=httpx.MockTransport(handle), follow_redirects=True)
    environment = {'HTTPS_PROXY': 'http://proxy.test:8080', 'NO_PROXY': 'localhost'}
    monkeypatch.setattr(network, 'system_proxy_environment', lambda: dict(environment))
    monkeypatch.setattr(network.httpx, 'AsyncClient', client)
    network._DOWNLOAD_ROUTES.clear()
    async def run():
        use_proxy, speed = await network.probe_download_route('https://hub.test/file')
        assert use_proxy is direct_fails and speed > 0
        selected, proxy = await network.download_environment('https://hub.test', sample_url='https://hub.test/other-file')
        assert proxy is direct_fails
        assert selected['NO_PROXY'] == ('localhost' if direct_fails else '*')
        assert len(calls) == 2
    try:
        asyncio.run(run())
    finally:
        network._DOWNLOAD_ROUTES.clear()


def test_catalog_probes_weight_payloads_instead_of_metadata(monkeypatch):
    catalog = HubCatalog()
    calls = []
    async def files(provider, repo, revision):
        return SimpleNamespace(revision=revision or 'main', files=[
            HubModelFile(name='config.json', byte_size=4), HubModelFile(name='S4/model.mfq', byte_size=2**30)])
    async def probe(url):
        calls.append(url)
        return 'huggingface' in url, 100.0 if 'huggingface' in url else 500.0
    monkeypatch.setattr(catalog, 'download_files', files)
    monkeypatch.setattr(network, 'probe_download_route', probe)
    sources = [HubDownloadSource(provider=provider, repo_id='owner/model') for provider in ('huggingface', 'modelscope')]
    result = asyncio.run(catalog.probe_download_sources(sources))
    assert [item['bytes_per_second'] for item in result['data']] == [100, 500]
    assert all('S4/model.mfq' in url for url in calls)
    assert all('config.json' not in url for url in calls)
