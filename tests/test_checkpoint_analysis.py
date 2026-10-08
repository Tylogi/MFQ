from __future__ import annotations

import asyncio
import math
from pathlib import Path

import httpx
import numpy as np
import pytest

from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.inspection import inspect_tensor
from mfq.formats.io import MMapTensorStore, pack_nint, save
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.nint import NintTensor, metadata_envelope_spec
from mfq.server.api import create_app
from mfq.server.services.checkpoint_analysis import analyze_checkpoint
from mfq.server.services.service import ServerService
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore
from mfq.tools.split_mfq import split_mfq


def nint(qs: list[int], ks: list[int], width: int = 16, gs: int = 8) -> NintTensor:
    q, k = np.array(qs, dtype=np.uint8), np.array(ks, dtype=np.uint8)
    groups = math.ceil(width / gs)
    rows = len(q)
    return NintTensor(spec=metadata_envelope_spec(gs, q, k), shape=(rows, width), axis=0,
        q=np.zeros((rows, groups, gs), dtype=np.uint8), neuron_scale=np.ones(rows), neuron_min=np.zeros(rows),
        sub_scale=np.ones((rows, groups), dtype=np.uint8), sub_min=np.zeros((rows, groups), dtype=np.uint8),
        neuron_len=width, row_sub_bits=k, row_q_bits=q)


def checkpoint(path: Path, *, fused: bool = False) -> None:
    import json
    outputs = 4 if fused else 2
    quantized = nint([2] * outputs + [8] * outputs, [3] * outputs + [6] * outputs)
    pool = MfeTensor(shape=(2, outputs, 16), pools=(MfePool(np.array([1, 0]), quantized),))
    tensors = {
        MODEL_CONFIG_ASSET: json.dumps({"model_type": "qwen4_exp_text", "num_hidden_layers": 2,
            "hidden_size": 16, "num_experts": 2, "num_experts_per_tok": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "max_position_embeddings": 32768, "layer_types": ["linear_attention", "full_attention"],
            "indexer_head_dim": 4, "indexer_compress_ratio": 4}).encode(),
        "model.block.0.mlp.experts.gate_up.weight" if fused else "model.block.0.mlp.experts.gate.weight": pool,
        "model.block.0.linear_attention.qkv.weight": np.ones((8, 16), dtype=np.float16),
        "model.block.1.attention.indexer.query_key.weight": nint([4, 4], [4, 4]),
        "model.block.1.attention.query.weight": nint([6, 6], [6, 6]),
        "model.embedding.weight": np.ones((16, 16), dtype=np.float16),
        "model.block.0.position_embedding.ngram.layer_multipliers": np.arange(4, dtype=np.int64),
    }
    save(path, FileHeader(model_arch="test", version=2), tensors)


def test_nint_inspection_uses_row_q_and_k_not_nominal_preset() -> None:
    value = nint([2, 2, 8, 8], [3, 3, 6, 6], width=17)
    payload = pack_nint(value)
    with memoryview(payload) as blob:
        result = inspect_tensor("NINT", blob, rows=True)
    assert result.shape == (4, 17)
    assert result.family == "NINTv2"
    assert result.granularity == "row"
    assert result.row_bytes.sum() == pytest.approx(len(payload))
    assert result.row_bytes[2] > result.row_bytes[0]
    assert result.row_bytes[2] - result.row_bytes[0] == (3 * 8 * 6 + 2 * 3 * 3) / 8


@pytest.mark.parametrize("fused", [False, True])
def test_checkpoint_metadata_preserves_expert_ids_and_encoded_budget(tmp_path: Path, monkeypatch, fused: bool) -> None:
    path = tmp_path / "model.mfq"
    checkpoint(path, fused=fused)
    artifact = ModelCatalog._inspect(tmp_path, path)
    monkeypatch.setattr(MMapTensorStore, "__getitem__", lambda *_: pytest.fail("analysis must not decode weights"))
    result = analyze_checkpoint(artifact)
    assert result.layer_count == 2
    assert result.attention_distribution == {"GDN": [0], "QSA": [1]}
    assert result.attention_heads == 2
    assert result.kv_heads == 1
    assert result.cache_profile.max_context == 32768
    assert result.cache_profile.fixed_bytes == 0
    assert {item.group for item in result.cache_profile.components} == {"QSA"}
    assert [item.bytes_per_row for item in result.cache_profile.components] == [32, 8, 16]
    assert not result.warnings
    assert len(result.experts) == (4 if fused else 2)
    first = next(item for item in result.experts if item.expert == 0 and item.projection == "gate")
    second = next(item for item in result.experts if item.expert == 1 and item.projection == "gate")
    assert first.bpw > second.bpw
    routed = [item for item in result.tensors if item.category == "routed_experts"]
    assert sum(item.stored_bytes for item in result.experts) == pytest.approx(sum(item.stored_bytes for item in routed))
    assert sum(item.parameters for item in result.experts) == sum(item.parameters for item in routed)
    assert result.average_bpw == pytest.approx(result.stored_bytes * 8 / result.parameters)
    assert all("layer_multipliers" not in item.name for item in result.tensors)
    assert result.parameters == sum(math.prod(item.shape) for item in result.tensors)


def test_fused_gate_up_does_not_assume_identical_row_precision(tmp_path: Path) -> None:
    path = tmp_path / "fused.mfq"
    value = nint([2, 2, 8, 8], [4, 4, 4, 4])
    pool = MfeTensor(shape=(1, 4, 16), pools=(MfePool(np.array([0]), value),))
    save(path, FileHeader(model_arch="test"), {"model.block.0.mlp.experts.gate_up.weight": pool})
    result = analyze_checkpoint(ModelCatalog._inspect(tmp_path, path))
    assert result.experts[0].projection == "gate"
    assert result.experts[1].projection == "up"
    assert result.experts[1].bpw > result.experts[0].bpw
    assert result.advanced.ffn_activation is None
    assert result.advanced.hc_lowrank is None
    assert result.advanced.active_blocks is None
    assert result.advanced.expert_hidden_size == 2


def test_advanced_qwen_details_distinguish_activations_and_sparse_budget(tmp_path: Path) -> None:
    import json
    path = tmp_path / "details.mfq"
    config = {"model_type": "qwen4_exp", "text_config": {
        "model_type": "qwen4_exp_text", "hidden_size": 16, "hidden_act": "silu", "output_gate_type": "sigmoid",
        "moe_intermediate_size": 2, "shared_expert_intermediate_size": 2, "hc_count": 4, "hc_lowrank": 3,
        "indexer_compress_ratio": 4, "indexer_budget": 2048, "indexer_n_heads": 4, "indexer_kv_heads": 1,
        "indexer_head_dim": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "linear_key_head_dim": 4, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "ngram_size": 3, "heads_per_ngram": 8, "ple_conv_kernel_size": 4, "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 262144, "vocab_size": 16, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}},
        "vision_config": {"hidden_act": "gelu"}}
    save(path, FileHeader(model_arch="test"), {
        MODEL_CONFIG_ASSET: json.dumps(config).encode(),
        "model.block.0.mlp.shared_expert.gate.weight": np.ones((2, 16), dtype=np.float16),
        "model.block.0.linear_attention.qkv.weight": np.ones((8, 16), dtype=np.float16),
        "model.block.0.linear_attention.norm.weight": np.ones(8, dtype=np.float16),
        "model.block.0.attention.mhc.pre.norm.weight": np.ones(64, dtype=np.float16),
        "model.block.0.mlp.mhc.pre.norm.weight": np.ones(64, dtype=np.float16),
        "model.block.1.attention.indexer.query_key.weight": np.ones((4, 16), dtype=np.float16),
        "model.block.1.attention.query_norm.weight": np.ones(8, dtype=np.float16),
        "model.block.1.attention.indexer.key_norm.weight": np.ones(4, dtype=np.float16),
        "model.block.1.position_embedding.key.weight": np.ones((2, 16), dtype=np.float16),
        "model.block.1.position_embedding.conv_norm.weight": np.ones(64, dtype=np.float16),
        "predictor.layers.0.weight": np.ones((2, 16), dtype=np.float16),
        "predictor.hidden_norm.weight": np.ones(64, dtype=np.float16),
    })
    info = analyze_checkpoint(ModelCatalog._inspect(tmp_path, path)).advanced
    assert info.ffn_activation == "silu"
    assert info.router_activation == "softmax"
    assert info.router_normalization == "l1_topk"
    assert info.gdn_conv_activation == "silu"
    assert info.gdn_conv_type == "causal_depthwise_1d"
    assert info.gdn_conv_stride == info.gdn_conv_dilation == 1
    assert info.gdn_gate_activation == "sigmoid"
    assert info.qsa_gate_activation == "sigmoid"
    assert info.residual_gate_activation == "sigmoid"
    assert "attention_gate_activation" not in info.model_dump()
    assert info.shared_expert_count == 1
    assert info.expert_hidden_size == info.shared_expert_hidden_size == 2
    assert (info.hc_count, info.hc_dim, info.hc_lowrank) == (4, 16, 3)
    assert (info.compression_stride, info.active_blocks, info.active_tokens) == (4, 512, 2048)
    assert (info.indexer_query_heads, info.indexer_kv_heads, info.indexer_head_dim) == (4, 1, 128)
    assert (info.linear_key_head_dim, info.linear_value_head_dim, info.linear_conv_kernel) == (4, 8, 4)
    assert info.ple_layer_ids == [1]
    assert info.ple_conv_type == "causal_depthwise_1d"
    assert info.ple_conv_stride == 1
    assert info.ple_conv_dilation == 3
    assert info.ple_conv_activation == "silu"
    assert info.predictor_layers == 1
    assert info.rope_theta == 10000000
    assert info.rope_partial_factor == 0.25
    norms = {(item.component, item.position): item for item in info.normalizations}
    assert norms["gdn", "output"].kind == "GroupedRMSNorm"
    assert norms["gdn", "output"].dimension == 8
    assert norms["gdn", "output"].groups == 4
    assert norms["gdn", "query"].kind == "L2Norm"
    assert norms["gdn", "query"].dimension == 4
    assert norms["gdn", "key"].epsilon == 1e-6
    assert norms["qsa", "query"].kind == "GroupedRMSNorm"
    assert norms["qsa", "query"].dimension == 8
    assert norms["qsa", "query"].groups == 4
    assert norms["indexer", "key"].dimension == 4
    assert norms["indexer", "key"].kind == "GroupedRMSNorm"
    assert norms["indexer", "key"].groups == 1
    for component, position in (("residual", "attention_input"), ("residual", "ffn_input"), ("ple", "conv")):
        assert norms[component, position].kind == "GroupedRMSNorm"
        assert norms[component, position].dimension == 16
        assert norms[component, position].groups == 4
        assert norms[component, position].epsilon == 1e-6
    assert norms["predictor", "hidden_norm"].kind == "RMSNorm"
    assert norms["predictor", "hidden_norm"].dimension == 64
    assert norms["predictor", "hidden_norm"].groups is None


def test_advanced_shared_expert_bank_and_layerwise_compression(tmp_path: Path) -> None:
    import json
    path = tmp_path / "bank.mfq"
    config = {"model_type": "test", "moe_intermediate_size": 2, "n_shared_experts": 3,
        "compress_ratios": [4, 128, 0, False, "4"], "index_topk": 2048, "index_n_heads": 8,
        "index_head_dim": 128, "hc_mult": 4, "hidden_size": 16, "scoring_func": "sqrtsoftplus",
        "mtp_num_hidden_layers": 1, "hidden_act": "gelu"}
    save(path, FileHeader(model_arch="test"), {
        MODEL_CONFIG_ASSET: json.dumps(config).encode(),
        "model.block.0.mlp.shared_expert.gate_up.weight": np.ones((12, 16), dtype=np.float16),
    })
    info = analyze_checkpoint(ModelCatalog._inspect(tmp_path, path)).advanced
    assert (info.shared_expert_count, info.shared_expert_hidden_size) == (3, 2)
    assert info.compression_strides == [4, 128]
    assert info.compression_stride is None
    assert info.active_blocks == 2048
    assert info.active_tokens is None
    assert info.ffn_activation == "gelu"
    assert info.router_activation == "sqrtsoftplus"
    assert info.hc_lowrank is None
    assert info.predictor_layers == 0


def test_gdn_gate_does_not_override_qsa_gate(tmp_path: Path) -> None:
    import json
    path = tmp_path / "gates.mfq"
    save(path, FileHeader(model_arch="test"), {
        MODEL_CONFIG_ASSET: json.dumps({"model_type": "qwen4_exp_text", "hidden_act": "silu", "output_gate_type": "silu"}).encode(),
        "model.block.0.linear_attention.qkv.weight": np.ones((4, 16), dtype=np.float16),
        "model.block.1.attention.indexer.query_key.weight": np.ones((4, 16), dtype=np.float16),
    })
    info = analyze_checkpoint(ModelCatalog._inspect(tmp_path, path)).advanced
    assert info.gdn_gate_activation == "silu"
    assert info.qsa_gate_activation == "sigmoid"
    assert info.residual_gate_activation is None


@pytest.mark.parametrize("architecture", ["qwen3_5_text", "qwen3_5_moe_text"])
def test_qwen35_gdn_output_gate_is_silu(tmp_path: Path, architecture: str) -> None:
    import json
    path = tmp_path / "qwen35.mfq"
    save(path, FileHeader(model_arch="test"), {
        MODEL_CONFIG_ASSET: json.dumps({"model_type": architecture, "hidden_act": "silu", "output_gate_type": "sigmoid",
            "hidden_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4, "linear_key_head_dim": 4,
            "linear_value_head_dim": 8, "rms_norm_eps": 1e-5}).encode(),
        "model.block.0.linear_attention.qkv.weight": np.ones((4, 16), dtype=np.float16),
        "model.block.0.attention.norm.weight": np.ones(16, dtype=np.float16),
        "model.block.0.linear_attention.norm.weight": np.ones(8, dtype=np.float16),
        "model.block.1.attention.query.weight": np.ones((16, 16), dtype=np.float16),
        "model.block.1.attention.query_norm.weight": np.ones(4, dtype=np.float16),
        "model.block.1.attention.key_norm.weight": np.ones(4, dtype=np.float16),
        "predictor.block.0.attention.query_norm.weight": np.ones(4, dtype=np.float16),
    })
    result = analyze_checkpoint(ModelCatalog._inspect(tmp_path, path))
    info = result.advanced
    assert info.gdn_gate_activation == "silu"
    assert info.gdn_conv_activation == "silu"
    assert info.full_attention_gate_activation == "sigmoid"
    assert info.qsa_gate_activation is None
    assert result.attention_distribution == {"GDN": [0], "GQA": [1]}
    norms = {(item.component, item.position): item for item in info.normalizations}
    assert norms["gdn", "input"].dimension == 16
    assert norms["gdn", "output"].epsilon == 1e-5
    assert norms["gdn", "query"].epsilon == norms["gdn", "key"].epsilon == 1e-6
    for position, groups in (("query", 2), ("key", 1)):
        assert norms["attention", position].kind == "GroupedRMSNorm"
        assert norms["attention", position].dimension == 4
        assert norms["attention", position].groups == groups
    assert norms["gdn", "input"].kind == "RMSNorm"
    assert norms["gdn", "input"].groups is None
    assert norms["predictor", "block.0.attention.query_norm"].kind == "GroupedRMSNorm"
    assert norms["predictor", "block.0.attention.query_norm"].groups == 2


def test_deepseek_router_reports_sqrt_softplus_and_selected_l1(tmp_path: Path) -> None:
    import json
    path = tmp_path / "deepseek.mfq"
    save(path, FileHeader(model_arch="test"), {
        MODEL_CONFIG_ASSET: json.dumps({"model_type": "deepseek_v4", "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": True, "hc_mult": 4, "hidden_size": 16, "rms_norm_eps": 1e-6}).encode(),
        "model.block.0.mlp.router.weight": np.ones((4, 16), dtype=np.float16),
        "model.block.0.attention.norm.weight": np.ones(16, dtype=np.float16),
    })
    info = analyze_checkpoint(ModelCatalog._inspect(tmp_path, path)).advanced
    assert info.router_activation == "sqrtsoftplus"
    assert info.router_normalization == "l1_topk"
    assert info.normalizations[0].kind == "RMSNorm"
    assert info.normalizations[0].dimension == 16
    assert info.normalizations[0].groups is None


@pytest.mark.parametrize("query_heads,kv_heads,kind", [(8, 2, "GQA"), (8, 8, "MHA"), (8, None, "Attention"), (8, 3, "Attention")])
def test_full_attention_type_uses_reported_head_geometry(tmp_path: Path, query_heads: int, kv_heads: int | None, kind: str) -> None:
    import json
    path = tmp_path / "attention.mfq"
    save(path, FileHeader(model_arch="test"), {
        MODEL_CONFIG_ASSET: json.dumps({"model_type": "qwen3_5_text", "num_attention_heads": query_heads,
            "num_key_value_heads": kv_heads}).encode(),
        "model.block.0.attention.query.weight": np.ones((16, 16), dtype=np.float16),
    })
    result = analyze_checkpoint(ModelCatalog._inspect(tmp_path, path))
    assert result.attention_distribution == {kind: [0]}
    assert result.layers[0].attention_type == kind


def test_norm_profiles_preserve_layer_variants_and_unknown_types(tmp_path: Path) -> None:
    import json
    path = tmp_path / "norms.mfq"
    save(path, FileHeader(model_arch="test"), {
        MODEL_CONFIG_ASSET: json.dumps({"model_type": "unknown", "norm_topk_prob": False}).encode(),
        "model.block.0.mlp.norm.weight": np.ones(16, dtype=np.float16),
        "model.block.1.mlp.norm.weight": np.ones(16, dtype=np.float16),
        "model.block.2.mlp.norm.weight": np.ones(8, dtype=np.float16),
    })
    info = analyze_checkpoint(ModelCatalog._inspect(tmp_path, path)).advanced
    assert info.router_normalization == "none"
    assert len(info.normalizations) == 2
    assert info.normalizations[0].layers == [0, 1]
    assert info.normalizations[1].layers == [2]
    assert all(item.kind is None and item.epsilon is None for item in info.normalizations)


def test_analysis_api_handles_shards_unknown_ids_and_no_catalog(tmp_path: Path) -> None:
    class Backend:
        async def aclose(self):
            pass

    async def run():
        path = tmp_path / "weights.mfq"
        checkpoint(path)
        output = tmp_path / "library"
        output.mkdir()
        split_mfq(path, output / "split.mfq", split_max_tensors=2)
        catalog = ModelCatalog([output])
        artifact = (await catalog.list()).data[0]
        service = ServerService(SessionStore(tmp_path / "server.sqlite3"), Backend(), catalog=catalog)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                response = await client.get(f"/api/v1/models/{artifact.id}/analysis")
                assert response.status_code == 200
                body = response.json()
                assert body["model_id"] == artifact.id
                assert body["layer_count"] == 2
                assert str(tmp_path) not in response.text
                assert (await client.get(f"/api/v1/models/{'0' * 32}/analysis")).status_code == 404
                sorted(output.glob("*.mfq"))[-1].unlink()
                incomplete = (await catalog.list(refresh=True)).data[0]
                assert not incomplete.complete
                assert (await client.get(f"/api/v1/models/{incomplete.id}/analysis")).status_code == 422
        finally:
            await service.aclose()
        service = ServerService(SessionStore(tmp_path / "empty.sqlite3"), Backend())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                assert (await client.get(f"/api/v1/models/{artifact.id}/analysis")).status_code == 501
        finally:
            await service.aclose()
    asyncio.run(run())
