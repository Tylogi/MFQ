from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.protocol.models import HubModelFile
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_download_directory import make_handlers
from tests.test_server_service import FakeBackend


TABLE = 'Engram/model-EW-Engram-MXFP8-00002-of-00003.mfq'
PRIMARY = 'Q3/model-EW-Q3-00001-of-00003.mfq'


@pytest.mark.parametrize('provider', ['modelscope', 'huggingface'])
def test_reuse_preview_reports_only_complete_files_and_updates_after_deletion(tmp_path, provider):
    async def run():
        handlers = make_handlers(tmp_path)
        destination = f'models/{provider}/owner/model'
        folder = handlers._download_output(destination, None)
        table = folder / TABLE
        table.parent.mkdir(parents=True)
        table.write_bytes(b'12345')
        primary = folder / PRIMARY
        primary.parent.mkdir(parents=True)
        primary.with_suffix('.mfq.incomplete').write_bytes(b'1234567')
        files = [HubModelFile(name=TABLE, byte_size=5), HubModelFile(name=PRIMARY, byte_size=7)]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        store = SessionStore(tmp_path / 'state.db')
        service = ServerService(store, FakeBackend(), hub_catalog=hub, tool_handlers=handlers)
        body = dict(provider=provider, repo_id='owner/model', revision='main', destination=destination)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
            response = await client.post('/api/v1/hub/download-reuse', json=body)
            assert response.status_code == 200
            assert response.json()['reusable_files'] == {TABLE: 5}
            primary.write_bytes(b'bad')
            assert (await client.post('/api/v1/hub/download-reuse', json=body)).json()['reusable_files'] == {TABLE: 5}
            primary.write_bytes(b'1234567')
            assert (await client.post('/api/v1/hub/download-reuse', json=body)).json()['reusable_files'] == {TABLE: 5, PRIMARY: 7}
            table.unlink()
            assert (await client.post('/api/v1/hub/download-reuse', json=body)).json()['reusable_files'] == {PRIMARY: 7}
        assert not store.list_jobs()
    asyncio.run(run())


def test_reuse_preview_respects_selected_directory_and_creates_no_folders(tmp_path):
    async def run():
        handlers = make_handlers(tmp_path)
        files = [HubModelFile(name=TABLE, byte_size=5)]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        service = ServerService(SessionStore(tmp_path / 'state.db'), FakeBackend(), hub_catalog=hub, tool_handlers=handlers)
        body = dict(provider='modelscope', repo_id='owner/model', destination='models/modelscope/owner/model')
        default = handlers.download_directories.default_directory
        default.rmdir()  # Simulate a default folder that has not been created yet.
        assert not default.exists()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
            response = await client.post('/api/v1/hub/download-reuse', json={**body, 'download_directory': str(default)})
            assert response.status_code == 200 and response.json()['reusable_files'] == {}
            assert not default.exists()
            chosen = tmp_path / 'chosen'
            chosen.mkdir()
            await handlers.download_directories.configure(chosen)
            table = chosen / 'modelscope/owner/model' / TABLE
            table.parent.mkdir(parents=True)
            table.write_bytes(b'12345')
            assert (await client.post('/api/v1/hub/download-reuse', json={**body, 'download_directory': str(chosen)})).json()['reusable_files'] == {TABLE: 5}
            assert (await client.post('/api/v1/hub/download-reuse', json={**body, 'download_directory': str(default)})).json()['reusable_files'] == {}
            assert not default.exists()
            invalid = await client.post('/api/v1/hub/download-reuse', json={**body, 'download_directory': str(tmp_path / 'unregistered')})
            assert invalid.status_code == 422
    asyncio.run(run())


def test_reuse_preview_rejects_paths_outside_download_destination(tmp_path):
    async def run():
        handlers = make_handlers(tmp_path)
        file = tmp_path / 'outside.mfq'
        file.write_bytes(b'12345')
        files = [HubModelFile(name='../../../../outside.mfq', byte_size=5)]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        service = ServerService(SessionStore(tmp_path / 'state.db'), FakeBackend(), hub_catalog=hub, tool_handlers=handlers)
        body = dict(provider='modelscope', repo_id='owner/model', destination='models/modelscope/owner/model')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
            response = await client.post('/api/v1/hub/download-reuse', json=body)
            assert response.status_code == 200 and response.json()['reusable_files'] == {}
            response = await client.post('/api/v1/hub/download-reuse', json={**body, 'destination': '../escape'})
            assert response.status_code == 422
    asyncio.run(run())
