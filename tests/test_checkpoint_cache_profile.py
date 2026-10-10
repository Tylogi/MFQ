import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import MMapTensorStore, save
from mfq.server.api import create_app
from mfq.server.services.model_memory import checkpoint_cache_profile
from mfq.server.services.service import ServerService
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore


def config():
    return {'model_type': 'qwen4_exp_text', 'num_hidden_layers': 4,
        'layer_types': ['linear_attention'] * 3 + ['full_attention'],
        'max_position_embeddings': 262144, 'num_key_value_heads': 2, 'head_dim': 256,
        'indexer_head_dim': 128, 'indexer_compress_ratio': 4, 'indexer_budget': 2048, 'mtp_num_hidden_layers': 1}


def checkpoint(path, predictor=False):
    tensors = {MODEL_CONFIG_ASSET: json.dumps(config()).encode(),
        'model.embedding.weight': np.ones((4, 4), dtype=np.float16)}
    if predictor:
        tensors['predictor.block.0.attention.query.weight'] = np.ones((4, 4), dtype=np.float16)
    save(path, FileHeader(model_arch='qwen4_exp'), tensors)


@pytest.mark.parametrize('predictor', [False, True])
def test_cache_profile_reads_only_config_and_accounts_for_available_predictor(tmp_path, monkeypatch, predictor):
    path = tmp_path / 'model.mfq'
    checkpoint(path, predictor)
    artifact = ModelCatalog._inspect(tmp_path, path)
    original = MMapTensorStore.read_blob

    def read(store, name):
        assert name == MODEL_CONFIG_ASSET
        return original(store, name)

    monkeypatch.setattr(MMapTensorStore, 'read_blob', read)
    monkeypatch.setattr(MMapTensorStore, '__getitem__', lambda *_: pytest.fail('must not decode weights'))
    profile = checkpoint_cache_profile(artifact)
    layers = 2 if predictor else 1
    assert profile.max_context == 262144
    assert profile.fixed_bytes == layers * (9 * 128 * 2 + 2)
    assert [item.name for item in profile.fixed_components] == ['indexer_tail']
    assert [item.bytes_per_row for item in profile.components] == [layers * 2048, layers * 512]
    assert all(item.layers == layers for item in profile.components)
    assert profile.components[0].max_read_rows_per_token == 2048
    assert profile.components[0].bytes_per_row * profile.components[0].max_read_rows_per_token == layers * 4 * 2 ** 20
    assert profile.components[1].max_read_rows_per_token is None


@pytest.mark.parametrize('predictor', [False, True])
def test_hf_metadata_uses_same_profile_without_reading_tensor_payloads(tmp_path, predictor):
    (tmp_path / 'config.json').write_text(json.dumps({'text_config': config()}))
    names = {'model.embed_tokens.weight': 'model-00001.safetensors'}
    if predictor:
        names['mtp.layers.0.self_attn.q_proj.weight'] = 'model-00001.safetensors'
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': names}))
    artifact = SimpleNamespace(path=tmp_path, resource=SimpleNamespace(complete=True, format='hf'))
    profile = checkpoint_cache_profile(artifact)
    assert profile.components[0].layers == (2 if predictor else 1)


def test_unknown_and_incomplete_models_do_not_fabricate_a_profile(tmp_path):
    path = tmp_path / 'unknown.mfq'
    save(path, FileHeader(model_arch='unknown'), {'weight': np.ones(4, dtype=np.float16)})
    artifact = ModelCatalog._inspect(tmp_path, path)
    assert checkpoint_cache_profile(artifact) is None
    artifact.resource.complete = False
    with pytest.raises(ValueError, match='incomplete'):
        checkpoint_cache_profile(artifact)


def test_profile_route_returns_metadata_and_preserves_missing_model_errors(tmp_path):
    path = tmp_path / 'model.mfq'
    checkpoint(path, True)

    class Backend:
        async def aclose(self):
            pass

    async def run():
        catalog = ModelCatalog([tmp_path])
        artifact = (await catalog.list()).data[0]
        service = ServerService(SessionStore(tmp_path / 'server.sqlite3'), Backend(), catalog=catalog)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
                response = await client.get(f'/api/v1/models/{artifact.id}/cache-profile')
                assert response.status_code == 200
                assert response.json()['components'][0]['layers'] == 2
                assert response.json()['components'][0]['max_read_rows_per_token'] == 2048
                assert response.json()['fixed_bytes'] == 2 * (9 * 128 * 2 + 2)
                assert str(tmp_path) not in response.text
                assert (await client.get(f'/api/v1/models/{"0" * 32}/cache-profile')).status_code == 404
        finally:
            await service.aclose()

    asyncio.run(run())
