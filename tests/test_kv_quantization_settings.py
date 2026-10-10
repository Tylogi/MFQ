import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from mfq.server.protocol.models import (
    KvQuantizationSettings,
    ModelLoadRequest,
    RuntimeContextPolicy,
)
from mfq.server.services.jobs import JobExecutionError
from tests.test_model_context_settings import fixture
from tests.test_server_models import _TestJobContext


@pytest.mark.parametrize('bits', [2, 2.5, 3, 3.5, 4, 6, 8])
def test_quantization_saves_with_context_and_preserves_streaming(tmp_path, bits):
    model, pool, service, instance, _, previous = fixture(tmp_path)
    instance.kv_quantization_supported = True
    quantization = {'enabled': True, 'bits': bits, 'algorithm': 'turboquant'}
    previous.qsa_kv_offload.enabled = True
    pool._validate_qsa_kv_policy = AsyncMock()

    async def run():
        async def load(context, payload):
            assert service.store.runtime_context_policy() == {}
            assert payload['kv_quantization'] == quantization
            assert payload['qsa_kv_offload']['enabled']
            assert payload['qsa_kv_offload']['target_context'] == 524288
            return {'instance_id': str(uuid4()), 'model': model.resource.name, 'context_size': 524288}
        pool.load = AsyncMock(side_effect=load)
        result = await pool.configure_context(_TestJobContext(), {'instance_id': str(instance.id), 'context_size': 524288,
            'yarn_enabled': True, 'kv_quantization': quantization})
        assert result['kv_quantization'] == quantization
        assert service.store.runtime_context_policy()['model_kv_quantization'] == {model.resource.name: quantization}
        pool.load.assert_awaited_once()
        pool._retire_instance.assert_awaited_once_with(instance)
        pool._instances.clear()
        await service.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('bits', [0, 1, 5, 7, 16, '4', float('nan'), float('inf')])
def test_invalid_levels_are_rejected(bits):
    with pytest.raises(ValueError):
        KvQuantizationSettings(enabled=True, bits=bits)


def test_indexer_precision_and_algorithm_are_not_user_overridable():
    for payload in [{'indexer_bits': 8}, {'algorithm': 'affine'}, {'enabled': 'true'}]:
        with pytest.raises(ValueError):
            KvQuantizationSettings.model_validate(payload)


def test_unsupported_candidate_does_not_retire_model(tmp_path):
    _, pool, service, instance, _, _ = fixture(tmp_path)
    instance.kv_quantization_supported = False
    async def run():
        pool.load = AsyncMock()
        with pytest.raises(JobExecutionError):
            await pool.configure_context(_TestJobContext(), {'instance_id': str(instance.id),
                'kv_quantization': {'enabled': True, 'bits': 4}})
        pool._retire_instance.assert_not_awaited()
        pool.load.assert_not_awaited()
        assert service.store.runtime_context_policy() == {}
        pool._instances.clear()
        await service.aclose()
    asyncio.run(run())


def test_failed_reload_recovers_previous_bundle(tmp_path):
    _, pool, service, instance, _, previous = fixture(tmp_path)
    instance.kv_quantization_supported = True
    async def run():
        pool.load = AsyncMock(side_effect=[RuntimeError('candidate failed'), {'context_size': 32768}])
        with pytest.raises(RuntimeError, match='candidate failed'):
            await pool.configure_context(_TestJobContext(), {'instance_id': str(instance.id),
                'kv_quantization': {'enabled': True, 'bits': 3.5}})
        assert pool.load.await_args_list[1].args[1] == previous.model_dump(mode='json')
        assert service.store.runtime_context_policy() == {}
        pool._instances.clear()
        await service.aclose()
    asyncio.run(run())


def test_launch_configuration_does_not_leak_inherited_quantization(tmp_path):
    model, pool, service, _, _, _ = fixture(tmp_path)
    pool.runtime_environment['MFQ_KV_TURBOQUANT_BITS'] = '2'
    for enabled in [False, True]:
        request = ModelLoadRequest(model=model.resource.name, kv_quantization={'enabled': enabled, 'bits': 3.5})
        _, environment = pool._launch_configuration(model, request, port=8105)
        assert float(environment['MFQ_KV_TURBOQUANT_BITS']) == (3.5 if enabled else 0)
    restored = RuntimeContextPolicy.model_validate({'model_kv_quantization': {model.resource.name: {'enabled': True, 'bits': 2.5}}})
    assert restored.model_kv_quantization[model.resource.name].algorithm == 'turboquant'
    pool._instances.clear()
    asyncio.run(service.aclose())
