import asyncio
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.server.api import create_app
from mfq.server.api.auth import required_scope
from mfq.server.protocol.models import CreateGenerationPresetRequest, ModelArtifactResource, ModelLoadRequest, ResponseRequestSettings, RuntimeContextPolicy, RuntimeInstanceState
from mfq.server.runtime.backend import BackendError
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.services.jobs import JobContext
from mfq.server.services.model_context import declared_context_size, yarn_context_limits
from mfq.server.services.service import ServerService
from mfq.server.state.catalog import DiscoveredModel, ModelCatalog
from mfq.server.state.storage import SCHEMA_VERSION, SessionStore


@pytest.mark.parametrize('config,tokenizer,expected', [
    ({'max_position_embeddings': 262144}, {}, 262144),
    ({'text_config': {'max_position_embeddings': 131072}}, {}, 131072),
    ({'language_config': {'seq_length': 65536}}, {}, 65536),
    ({'n_positions': 2048, 'text_config': {'max_seq_len': 4096}}, {}, 2048),
    ({}, {'model_max_length': 120000}, 120000),
    ({'max_position_embeddings': True}, {'model_max_length': 10**30}, None),
    ({'max_position_embeddings': -1}, {'model_max_length': '32768'}, None),
])
def test_declared_context(config, tokenizer, expected):
    assert declared_context_size(config, tokenizer) == expected


def artifact(tmp_path, capacity=262144):
    return DiscoveredModel(ModelArtifactResource(id='a' * 32, name='model', architecture='qwen3_5',
        context_capacity=capacity, total_bytes=1024, shard_count=1, tensor_count=1, complete=True, loadable=True,
        record_count=1, dtypes=['F16'], modified_at=datetime.now(timezone.utc)), tmp_path / 'model.mfq')


class ReloadBackend:
    def __init__(self):
        self.calls = []
        self.error = None
        self.actual = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def reload_runtime(self, size):
        self.calls.append(size)
        self.started.set()
        await self.release.wait()
        if self.error:
            raise self.error
        return {'max_context': self.actual or size}

    async def aclose(self):
        pass


def managed_runtime(tmp_path):
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', automatic_memory_budget=False)
    backend = ReloadBackend()
    instance = _Runtime(id=uuid4(), artifact=artifact(tmp_path), backend=backend,
        process=None, port=0, context_size=32768, context_capacity=262144, state=RuntimeInstanceState.READY)
    pool._instances[instance.id] = instance
    pool._load_requests['model'] = ModelLoadRequest(model='model')
    service = ServerService(SessionStore(tmp_path / 'state.sqlite3'), pool, runtime_manager=pool)
    return pool, instance, backend, service


def test_automatic_resolution_and_launch_arguments(tmp_path, monkeypatch):
    pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', backend='metal')
    model = artifact(tmp_path)
    assert ModelLoadRequest(model='model').context_size is None
    assert pool.resolve_context_size(model) == 262144
    pool.context_policy = RuntimeContextPolicy(max_context_size=65536, model_overrides={'model': 120000})
    assert pool.resolve_context_size(model) == 120000
    assert pool.resolve_context_size(model, 8192) == 8192
    assert pool.resolve_context_size(model, ignore_override=True) == 65536
    assert pool.resolve_context_size(artifact(tmp_path, None), ignore_override=True) == 32768
    monkeypatch.setattr('mfq.server.runtime.runtime_pool.native_tokenizer_arguments', lambda *_: [])
    argv, _ = pool._launch_configuration(model, ModelLoadRequest(model='model'), port=0)
    assert '120000' in argv and 'None' not in argv


def test_catalog_reads_context_from_embedded_configuration(tmp_path):
    save(tmp_path / 'model.mfq', FileHeader(model_arch='qwen3_5',
        extra={'hf_config': {'text_config': {'max_position_embeddings': 262144}}}),
        {'weight': np.ones((2, 2), dtype=np.float16)})
    result = ModelCatalog._inspect(tmp_path, tmp_path / 'model.mfq')
    assert result.resource.loadable and result.resource.context_capacity == 262144


def test_reload_persists_overrides_and_auto_removes_them(tmp_path):
    async def run():
        pool, instance, backend, service = managed_runtime(tmp_path)
        await pool.reload_runtime(120000, instance.id)
        assert instance.context_size == 120000 and instance.context_job is None
        assert service.store.runtime_context_policy()['model_overrides'] == {'model': 120000}
        await pool.configure_context_policy(RuntimeContextPolicy(max_context_size=65536))
        assert pool.context_policy.model_overrides == {'model': 120000}
        restored = RuntimePool(ModelCatalog([]), tmp_path / 'other-runtime')
        ServerService(service.store, restored, runtime_manager=restored)
        assert restored.resolve_context_size(instance.artifact) == 120000
        await pool.reload_runtime(None, instance.id)
        assert instance.context_size == 65536 and backend.calls == [120000, 65536]
        assert service.store.runtime_context_policy()['model_overrides'] == {}
        assert instance.control_leases == 0
        pool._instances.clear()
        await service.aclose()
        await restored.aclose()
    asyncio.run(run())

@pytest.mark.parametrize('override,expected', [(None, 65536), (131072, 131072)])
def test_offload_load_uses_normal_global_or_per_model_context(tmp_path, override, expected):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime', backend='metal')
        model = artifact(tmp_path)
        pool.catalog.resolve = AsyncMock(return_value=model)
        pool.context_policy = RuntimeContextPolicy(max_context_size=65536,
            model_overrides={} if override is None else {'model': override})
        pool._validate_qsa_kv_policy = AsyncMock(side_effect=RuntimeError('validation boundary'))
        with pytest.raises(RuntimeError, match='validation boundary'):
            await pool.load(SimpleNamespace(), {'model': 'model',
                'qsa_kv_offload': {'enabled': True, 'target_context': 262144}})
        assert pool._validate_qsa_kv_policy.await_args.args[1].target_context == expected
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['error', 'mismatch'])
def test_failed_reload_does_not_publish_a_false_setting(tmp_path, failure):
    async def run():
        pool, instance, backend, service = managed_runtime(tmp_path)
        if failure == 'error':
            backend.error = BackendError('reload_failed', 'Weights unavailable')
        if failure == 'mismatch':
            backend.actual = 32768
        with pytest.raises(BackendError):
            await pool.reload_runtime(120000, instance.id)
        assert instance.context_size == 32768 and instance.context_job is None and instance.control_leases == 0
        assert service.store.runtime_context_policy() == {}
        pool._instances.clear()
        await service.aclose()
    asyncio.run(run())


def test_reload_job_returns_before_native_reload_and_routes_progress(tmp_path):
    async def run():
        pool, instance, backend, service = managed_runtime(tmp_path)
        backend.release.clear()
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await asyncio.wait_for(client.post('/api/v1/runtime/context',
                json={'instance_id': str(instance.id), 'context_size': 120000}), 2)
            assert response.status_code == 202
            job_id = UUID(response.json()['id'])
            await asyncio.wait_for(backend.started.wait(), 2)
            assert instance.context_size == 32768
            status = await asyncio.wait_for(client.get('/api/v1/runtime/status'), 2)
            assert status.status_code == 200
            assert status.json()['reloading'] is True and status.json()['max_context'] == 32768
            with pytest.raises(BackendError, match='already being applied'):
                await pool.reload_runtime(8192, instance.id)
            cancelled = await client.post(f'/api/v1/jobs/{job_id}/cancel')
            assert cancelled.status_code == 409
            stream = asyncio.StreamReader()
            stream.feed_data(b'mfq_load_progress completed=50 total=100\nmfq_load_progress stage=warming completed=1 total=2\n')
            stream.feed_eof()
            instance.process = SimpleNamespace(stderr=stream, stdout=stream)
            old = service.store.create_job('model.load', {})
            service.store.claim_job(old.id)
            await pool._pump_output(instance, JobContext(service.store, old.id, asyncio.Event()))
            progress = await client.get(f'/api/v1/jobs/{job_id}')
            assert progress.json()['progress'] == pytest.approx(.97)
            assert progress.json()['progress_data']['phase'] == 'warming'
            assert service.store.get_job(old.id).progress == 0
            backend.release.set()
            await asyncio.wait_for(asyncio.gather(*service.jobs._tasks.values()), 2)
            completed = (await client.get(f'/api/v1/jobs/{job_id}')).json()
            assert completed['status'] == 'succeeded' and completed['result']['max_context'] == 120000
            assert instance.context_size == 120000
            assert (await client.get('/api/v1/runtime/context-policy')).json()['model_overrides'] == {'model': 120000}
            blocked = await client.post('/api/v1/jobs', json={'kind': 'runtime.context.configure', 'payload': {}})
            assert blocked.status_code == 403
        pool._instances.clear()
        await service.aclose()
    asyncio.run(run())


def test_context_policy_permissions():
    assert required_scope('PUT', '/api/v1/runtime/context-policy') == 'admin'
    assert required_scope('POST', '/api/v1/runtime/context') == 'models'


@pytest.mark.parametrize('config,expected', [
    ({'model_type': 'qwen4_exp', 'max_position_embeddings': 262144}, (262144, 1048576, 4.0)),
    ({'text_config': {'model_type': 'qwen3_5_text', 'max_position_embeddings': 262144}}, (262144, 1048576, 4.0)),
    ({'model_type': 'qwen4_exp', 'max_position_embeddings': 1048576, 'rope_parameters': {
        'rope_type': 'yarn', 'factor': 2, 'original_max_position_embeddings': 262144}}, (262144, 524288, 2.0)),
    ({'model_type': 'llama', 'max_position_embeddings': 8192}, (8192, None, None)),
    ({'model_type': 'qwen3_5', 'max_position_embeddings': 8192, 'rope_scaling': {'rope_type': 'linear', 'factor': 4}}, (8192, None, None)),
    ({'model_type': 'qwen4_exp', 'max_position_embeddings': 262144, 'rope_parameters': {'factor': float('nan')}}, (262144, None, None)),
])
def test_yarn_metadata(config, expected):
    assert yarn_context_limits(config) == expected


@pytest.mark.parametrize('enabled,requested,expected,warning', [
    (False, 524288, 262144, 'context_size_exceeded'),
    (True, 524288, 524288, None),
    (True, 2000000, 1048576, 'yarn_context_size_exceeded'),
    (False, 2000000, 262144, 'context_size_exceeded'),
])
def test_yarn_reload_clamps_and_persists(tmp_path, enabled, requested, expected, warning):
    async def run():
        pool, instance, backend, service = managed_runtime(tmp_path)
        instance.artifact = DiscoveredModel(instance.artifact.resource.model_copy(update={
            'yarn_context_capacity': 1048576, 'yarn_max_factor': 4.0}), instance.artifact.path)
        result = await pool.reload_runtime(requested, instance.id, yarn_enabled=enabled)
        assert result['max_context'] == expected and result['warning'] == warning
        assert backend.calls == [expected]
        assert pool.context_policy.model_overrides['model'] == expected
        assert pool.context_policy.model_yarn_enabled.get('model', False) == enabled
        assert (await pool.yarn_context_info(instance.id))['native_context'] == 262144
        restored = RuntimePool(ModelCatalog([]), tmp_path / 'restore-runtime')
        ServerService(service.store, restored, runtime_manager=restored)
        assert restored.resolve_context_size(instance.artifact) == expected
        assert restored.context_policy.model_yarn_enabled.get('model', False) == enabled
        await pool.reload_runtime(None, instance.id, yarn_enabled=False)
        assert instance.context_size == 262144 and pool.context_policy.model_yarn_enabled == {}
        assert pool.context_policy.model_overrides == {}
        pool._instances.clear()
        await service.aclose()
        await restored.aclose()
    asyncio.run(run())


def test_unsupported_yarn_does_not_reload_or_save(tmp_path):
    async def run():
        pool, instance, backend, service = managed_runtime(tmp_path)
        with pytest.raises(BackendError, match='not supported'):
            await pool.reload_runtime(524288, instance.id, yarn_enabled=True)
        assert backend.calls == [] and service.store.runtime_context_policy() == {}
        assert not (await pool.yarn_context_info(instance.id))['supported']
        pool._instances.clear()
        await service.aclose()
    asyncio.run(run())


def test_preset_migration_preserves_explicit_context_and_allows_auto(tmp_path):
    path = tmp_path / 'state.sqlite3'
    store = SessionStore(path)
    saved = store.create_generation_preset(CreateGenerationPresetRequest(name='old', context_size=120000,
        settings=ResponseRequestSettings(sampling={}), metadata={'preserve': True}))
    with sqlite3.connect(path) as db:
        db.execute("UPDATE schema_meta SET value = '18' WHERE key = 'schema_version'")
        db.execute('ALTER TABLE generation_presets RENAME TO preset_fixture')
        db.execute('''CREATE TABLE generation_presets (
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, model TEXT, mode TEXT,
            settings_json TEXT NOT NULL, context_size INTEGER NOT NULL CHECK (context_size >= 512),
            metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
        db.execute('INSERT INTO generation_presets SELECT * FROM preset_fixture')
        db.execute('DROP TABLE preset_fixture')
    migrated = SessionStore(path)
    assert migrated.get_generation_preset(saved.id) == saved
    auto = migrated.create_generation_preset(CreateGenerationPresetRequest(name='auto', settings=ResponseRequestSettings(sampling={})))
    assert auto.context_size is None
    assert SessionStore(path).get_generation_preset(auto.id).context_size is None
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
