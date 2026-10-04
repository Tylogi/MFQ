import asyncio
import json

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.api.auth import required_scope
from mfq.server.services.service import ServerService, ServiceError
from mfq.server.state.storage import SessionStore
from tests.test_server_openai_compat import _Backend


class Backend(_Backend):
    async def aclose(self):
        pass

    async def runtime_models(self):
        return {'data': [{'id': 'model-one'}, {'id': 'model-two'}]}


def test_aliases_persist_and_route_both_completion_modes(tmp_path):
    async def run():
        store = SessionStore(tmp_path / 'server.sqlite3')
        backend = Backend()
        service = ServerService(store, backend)
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            result = await client.put('/api/v1/runtime/model-aliases', json={'aliases': {'model-one': 'fast-model'}})
            assert result.status_code == 200
            assert store.runtime_model_aliases() == {'model-one': 'fast-model'}
            assert (await client.get('/api/v1/runtime/model-aliases')).json() == result.json()
            assert [item['id'] for item in (await client.get('/v1/models')).json()['data']] == ['fast-model', 'model-two']
            for stream in (False, True):
                response = await client.post('/v1/chat/completions', json={
                    'model': 'fast-model', 'stream': stream, 'messages': [{'role': 'user', 'content': 'hello'}]})
                assert response.status_code == 200
                assert backend.requests[-1]['model'] == 'model-one'
                if stream:
                    chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
                    assert all(chunk['model'] == 'fast-model' for chunk in chunks)
                else:
                    assert response.json()['model'] == 'fast-model'
            assert ServerService(store, Backend()).resolve_model_alias('fast-model') == 'model-one'
            assert service.resolve_model_alias('model-one') == 'model-one'
            assert (await client.post('/api/v1/jobs', json={'kind': 'runtime.memory.configure', 'payload': {}})).status_code == 403
        await service.aclose()
    asyncio.run(run())


def test_aliases_reject_collisions_and_unknown_models_without_changing_saved_state(tmp_path):
    async def run():
        service = ServerService(SessionStore(tmp_path / 'server.sqlite3'), Backend())
        await service.configure_model_aliases({'model-one': 'fast'})
        for aliases in ({'missing': 'alias'}, {'model-one': 'model-two'},
                        {'model-one': 'same', 'model-two': 'same'}, {'model-one': 'bad\nname'}):
            with pytest.raises(ServiceError):
                await service.configure_model_aliases(aliases)
            assert service.model_aliases == {'model-one': 'fast'}
        await service.configure_model_aliases({'model-one': ''})
        assert service.model_aliases == {}
        await service.aclose()
    asyncio.run(run())


def test_runtime_configuration_requires_admin_scope():
    for path in ('model-aliases', 'memory-policy', 'listener'):
        assert required_scope('PUT', f'/api/v1/runtime/{path}') == 'admin'
