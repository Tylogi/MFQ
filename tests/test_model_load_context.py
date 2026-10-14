import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mfq.server.protocol.models import ModelLoadRequest
from mfq.server.runtime.load_context import apply_load_context
from mfq.server.services.service import ServerService


@pytest.mark.parametrize('architecture', ['qwen4_exp', 'qwen3_5', 'qwen3.5-moe', 'llama'])
@pytest.mark.parametrize('automatic', [False, True])
@pytest.mark.parametrize('ssd,ram', [(False, True), (True, True), (True, False)])
def test_load_defaults_resolve_by_model(architecture, automatic, ssd, ram):
    request = ModelLoadRequest(model='test', context_size=32768, context_management={
        'automatic': automatic, 'ssd_kv': ssd, 'ram_preallocated': ram,
        'vram_preallocated': False, 'kv_quantization': {'enabled': True, 'bits': 2.5}})
    result = apply_load_context(request, architecture, 'cuda')
    assert result.context_size == 32768
    assert result.kv_memory.placement == ('auto_ssd' if architecture == 'qwen4_exp' and not automatic and ssd else 'auto')
    assert result.kv_memory.ram_preallocated == (ram if architecture == 'qwen4_exp' and not automatic else True)
    assert result.kv_quantization.enabled == (not automatic and architecture != 'llama')
    assert not result.qsa_kv_offload.enabled
    assert {'kv_memory', 'kv_quantization', 'qsa_kv_offload'} <= result.model_fields_set
    # Replayed requests retain the resolved per-model settings.
    assert apply_load_context(ModelLoadRequest.model_validate(result.model_dump()), architecture, 'cuda') == result


def test_non_sparse_model_ignores_sparse_only_invalid_combination():
    request = ModelLoadRequest(model='test', context_management={'automatic': False, 'ssd_kv': False, 'ram_preallocated': False})
    assert apply_load_context(request, 'qwen3_5', 'cuda').kv_memory.ram_preallocated
    with pytest.raises(ValidationError, match='kv_ssd_required'):
        apply_load_context(request, 'qwen4_exp', 'cuda')


def test_explicit_settings_survive_context_defaults_and_legacy_requests_are_unchanged():
    legacy = ModelLoadRequest(model='test')
    assert apply_load_context(legacy, 'qwen4_exp', 'cuda') is legacy
    request = ModelLoadRequest(model='test', context_management={}, kv_memory={'placement': 'vram'},
        kv_quantization={'enabled': True, 'bits': 8})
    result = apply_load_context(request, 'qwen4_exp', 'cuda')
    assert result.kv_memory.placement == 'vram' and result.kv_quantization.bits == 8


def test_load_api_keeps_absent_fields_absent_for_default_resolution():
    service = SimpleNamespace(runtime_manager=object(), create_job=AsyncMock(return_value=SimpleNamespace(id=uuid4())))
    request = ModelLoadRequest(model='test', context_size=32768, context_management={'automatic': False,
        'ssd_kv': True, 'ram_preallocated': False, 'kv_quantization': {'enabled': True, 'bits': 4}})
    asyncio.run(ServerService.load_model(service, request))
    payload = service.create_job.await_args.args[0].payload
    assert 'kv_memory' not in payload and 'kv_quantization' not in payload
    result = apply_load_context(ModelLoadRequest.model_validate(payload), 'qwen4_exp', 'cuda')
    assert result.kv_memory.placement == 'auto_ssd' and not result.kv_memory.ram_preallocated
    assert result.kv_quantization.enabled
