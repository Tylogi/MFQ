"""End-to-end import of a synthetic Nemotron-H checkpoint.

The fixture keeps every structural feature of the released family (per-expert
routed weights, a two-block MTP head, RADIO plus projector) at a size that
converts in a second.  It is generated per test, never committed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import mfq.tools.quantize_hf_to_mfq as hf_to_mfq
from mfq.formats.assets import MODEL_GRAPH_ASSET
from mfq.formats.io import inspect_mfe_header, load_mmap

HIDDEN = 32
EXPERTS = 3
EXPERT_HIDDEN = 24
LATENT = 8
SHARED_HIDDEN = 40
MAMBA_IN_PROJECTION = 50
MAMBA_CONV = 32
VISION_HIDDEN = 16
PROJECTOR_INPUT = 64


def synthetic_config() -> dict[str, object]:
    """Return a miniature omni config with the released nesting."""

    return {
        "model_type": "nemotron_h_omni",
        "downsample_ratio": 0.5,
        "eos_token_id": 11,
        "img_context_token_id": 18,
        "projector_hidden_size": 64,
        "ps_version": "v2",
        "vit_hidden_size": VISION_HIDDEN,
        "vision_config": {
            "model_type": "radio",
            "hidden_size": VISION_HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "mlp_ratio": 4.0,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-6,
            "qkv_bias": True,
            "use_swiglu_ffn": False,
            "layerscale_value": 1.0,
            "num_channels": 3,
            "patch_size": 4,
            "max_img_size": 8,
            "num_cls_tokens": 1,
            "num_registers": 1,
            "video_temporal_patch_size": 2,
            "summary_idxs": [0],
        },
        "llm_config": {
            "model_type": "nemotron_h",
            "hidden_size": HIDDEN,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 128,
            "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 64,
            "attention_bias": False,
            "mlp_bias": False,
            "mamba_proj_bias": False,
            "use_bias": False,
            "layers_block_type": ["mamba", "moe", "attention", "moe"],
            "mamba_num_heads": 2,
            "mamba_head_dim": 8,
            "ssm_state_size": 4,
            "n_groups": 2,
            "conv_kernel": 4,
            "use_conv_bias": True,
            "mamba_hidden_act": "silu",
            "chunk_size": 4,
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
            "time_step_min": 0.001,
            "time_step_max": 0.1,
            "time_step_floor": 0.0001,
            "intermediate_size": 24,
            "moe_intermediate_size": EXPERT_HIDDEN,
            "moe_latent_size": LATENT,
            "moe_shared_expert_intermediate_size": SHARED_HIDDEN,
            "n_routed_experts": EXPERTS,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 5.0,
            "mlp_hidden_act": "relu2",
            "num_nextn_predict_layers": 1,
            "mtp_layers_block_type": ["attention", "moe"],
            "tie_word_embeddings": False,
        },
    }


def _matrix(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((rows, columns), generator=generator).to(torch.bfloat16)


def _vector(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((size,), generator=generator).to(torch.bfloat16)


def synthetic_tensors() -> dict[str, torch.Tensor]:
    """Return one small checkpoint in the released source naming."""

    tensors: dict[str, torch.Tensor] = {
        "language_model.backbone.embeddings.weight": _matrix(128, HIDDEN, 1),
        "language_model.backbone.norm_f.weight": _vector(HIDDEN, 2),
        "language_model.lm_head.weight": _matrix(128, HIDDEN, 3),
    }

    def mamba_layer(prefix: str, seed: int) -> None:
        tensors[f"{prefix}.norm.weight"] = _vector(HIDDEN, seed)
        tensors[f"{prefix}.mixer.in_proj.weight"] = _matrix(MAMBA_IN_PROJECTION, HIDDEN, seed + 1)
        tensors[f"{prefix}.mixer.out_proj.weight"] = _matrix(HIDDEN, 16, seed + 2)
        tensors[f"{prefix}.mixer.conv1d.weight"] = _matrix(MAMBA_CONV, 4, seed + 3).unsqueeze(1)
        tensors[f"{prefix}.mixer.conv1d.bias"] = _vector(MAMBA_CONV, seed + 4)
        tensors[f"{prefix}.mixer.A_log"] = _vector(2, seed + 5)
        tensors[f"{prefix}.mixer.D"] = _vector(2, seed + 6)
        tensors[f"{prefix}.mixer.dt_bias"] = _vector(2, seed + 7)
        tensors[f"{prefix}.mixer.norm.weight"] = _vector(16, seed + 8)

    def attention_layer(prefix: str, seed: int) -> None:
        tensors[f"{prefix}.norm.weight"] = _vector(HIDDEN, seed)
        tensors[f"{prefix}.mixer.q_proj.weight"] = _matrix(HIDDEN, HIDDEN, seed + 1)
        tensors[f"{prefix}.mixer.k_proj.weight"] = _matrix(16, HIDDEN, seed + 2)
        tensors[f"{prefix}.mixer.v_proj.weight"] = _matrix(16, HIDDEN, seed + 3)
        tensors[f"{prefix}.mixer.o_proj.weight"] = _matrix(HIDDEN, HIDDEN, seed + 4)

    def moe_layer(prefix: str, seed: int) -> None:
        tensors[f"{prefix}.norm.weight"] = _vector(HIDDEN, seed)
        tensors[f"{prefix}.mixer.gate.weight"] = _matrix(EXPERTS, HIDDEN, seed + 1)
        tensors[f"{prefix}.mixer.gate.e_score_correction_bias"] = _vector(EXPERTS, seed + 2)
        tensors[f"{prefix}.mixer.fc1_latent_proj.weight"] = _matrix(LATENT, HIDDEN, seed + 3)
        tensors[f"{prefix}.mixer.fc2_latent_proj.weight"] = _matrix(HIDDEN, LATENT, seed + 4)
        tensors[f"{prefix}.mixer.shared_experts.up_proj.weight"] = _matrix(
            SHARED_HIDDEN, HIDDEN, seed + 5
        )
        tensors[f"{prefix}.mixer.shared_experts.down_proj.weight"] = _matrix(
            HIDDEN, SHARED_HIDDEN, seed + 6
        )
        for expert in range(EXPERTS):
            tensors[f"{prefix}.mixer.experts.{expert}.up_proj.weight"] = _matrix(
                EXPERT_HIDDEN, LATENT, seed + 10 + expert
            )
            tensors[f"{prefix}.mixer.experts.{expert}.down_proj.weight"] = _matrix(
                LATENT, EXPERT_HIDDEN, seed + 20 + expert
            )

    backbone = "language_model.backbone.layers"
    mamba_layer(f"{backbone}.0", 10)
    moe_layer(f"{backbone}.1", 30)
    attention_layer(f"{backbone}.2", 50)
    moe_layer(f"{backbone}.3", 70)

    mtp = "language_model.mtp.layers"
    tensors[f"{mtp}.0.eh_proj.weight"] = _matrix(HIDDEN, 2 * HIDDEN, 90)
    tensors[f"{mtp}.0.enorm.weight"] = _vector(HIDDEN, 91)
    tensors[f"{mtp}.0.hnorm.weight"] = _vector(HIDDEN, 92)
    tensors[f"{mtp}.0.final_layernorm.weight"] = _vector(HIDDEN, 93)
    attention_layer(f"{mtp}.0", 100)
    moe_layer(f"{mtp}.1", 130)

    radio = "vision_model.radio_model.model"
    tensors[f"{radio}.patch_generator.embedder.weight"] = _matrix(VISION_HIDDEN, 48, 200)
    tensors[f"{radio}.patch_generator.video_embedder.weight"] = _matrix(VISION_HIDDEN, 96, 201)
    tensors[f"{radio}.patch_generator.pos_embed"] = _matrix(4, VISION_HIDDEN, 202).unsqueeze(0)
    tensors[f"{radio}.patch_generator.cls_token.token"] = _matrix(2, VISION_HIDDEN, 203)
    tensors[f"{radio}.blocks.0.norm1.weight"] = _vector(VISION_HIDDEN, 204)
    tensors[f"{radio}.blocks.0.norm1.bias"] = _vector(VISION_HIDDEN, 205)
    tensors[f"{radio}.blocks.0.norm2.weight"] = _vector(VISION_HIDDEN, 206)
    tensors[f"{radio}.blocks.0.norm2.bias"] = _vector(VISION_HIDDEN, 207)
    tensors[f"{radio}.blocks.0.attn.qkv.weight"] = _matrix(3 * VISION_HIDDEN, VISION_HIDDEN, 208)
    tensors[f"{radio}.blocks.0.attn.qkv.bias"] = _vector(3 * VISION_HIDDEN, 209)
    tensors[f"{radio}.blocks.0.attn.proj.weight"] = _matrix(
        VISION_HIDDEN, VISION_HIDDEN, 210
    )
    tensors[f"{radio}.blocks.0.attn.proj.bias"] = _vector(VISION_HIDDEN, 211)
    tensors[f"{radio}.blocks.0.mlp.fc1.weight"] = _matrix(64, VISION_HIDDEN, 212)
    tensors[f"{radio}.blocks.0.mlp.fc1.bias"] = _vector(64, 213)
    tensors[f"{radio}.blocks.0.mlp.fc2.weight"] = _matrix(VISION_HIDDEN, 64, 214)
    tensors[f"{radio}.blocks.0.mlp.fc2.bias"] = _vector(VISION_HIDDEN, 215)
    tensors["vision_model.radio_model.input_conditioner.norm_mean"] = torch.zeros(
        3, 1, 1, dtype=torch.float32
    )
    tensors["vision_model.radio_model.input_conditioner.norm_std"] = torch.ones(
        3, 1, 1, dtype=torch.float32
    )
    tensors["vision_model.summary_idxs"] = torch.tensor([0], dtype=torch.int64)
    tensors["vision_projector.vision_final_layernorm.weight"] = _vector(VISION_HIDDEN, 216)
    tensors["vision_projector.vision_final_layernorm.bias"] = _vector(VISION_HIDDEN, 217)
    tensors["mlp1.0.weight"] = _vector(PROJECTOR_INPUT, 218)
    tensors["mlp1.1.weight"] = _matrix(PROJECTOR_INPUT, PROJECTOR_INPUT, 219)
    tensors["mlp1.3.weight"] = _matrix(HIDDEN, PROJECTOR_INPUT, 220)
    return tensors


def write_checkpoint(root: Path) -> Path:
    """Write one small synthetic checkpoint and return its directory."""

    root.mkdir(parents=True, exist_ok=True)
    save_file(synthetic_tensors(), root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(synthetic_config()), encoding="utf-8")
    return root


def convert(root: Path, output: Path, *extra: str) -> None:
    args = hf_to_mfq.build_parser().parse_args(
        ["--input", str(root), "--output", str(output), *extra]
    )
    hf_to_mfq.convert(args)


def test_bf16_conversion_keeps_the_fused_mamba_payload(tmp_path) -> None:
    root = write_checkpoint(tmp_path / "hf")
    output = tmp_path / "bf16.mfq"
    convert(root, output, "--bf16")

    header, store = load_mmap(output)
    try:
        assert header.model_arch == "nemotron_h_omni-hf-mfq-bf16"
        assert store.records["model.token_embedding.weight"].dtype == "BF16"
        assert store.records["model.output_norm.weight"].dtype == "F32"
        # One fused in_proj record; the runtime slices z/xBC/dt itself.
        assert store.records["model.block.0.linear_attention.in_projection.weight"].dtype == "BF16"
        assert store["model.block.0.linear_attention.in_projection.weight"].shape == (
            MAMBA_IN_PROJECTION,
            HIDDEN,
        )
        for role in ("a", "d", "dt_bias", "conv.weight", "conv.bias", "norm.weight"):
            name = f"model.block.0.linear_attention.{role}"
            assert store.records[name].dtype == "F32", name
        assert store.records["model.block.1.mlp.router.weight"].dtype == "F32"
        assert store.records["model.block.1.mlp.router.bias"].dtype == "F32"
        # Unquantized conversion keeps one record per expert.
        assert store.records["model.block.1.mlp.experts.0.up.weight"].dtype == "BF16"
        # MTP and vision tensors keep their canonical names.
        for name in (
            "predictor.fusion.weight",
            "predictor.embedding_norm.weight",
            "predictor.hidden_norm.weight",
            "predictor.output_norm.weight",
            "predictor.block.0.attention.query.weight",
            "predictor.block.1.mlp.router.weight",
            "vision.patch_embedding.weight",
            "vision.video_patch_embedding.weight",
            "vision.position_embedding.weight",
            "vision.cls_register_token.weight",
            "vision.block.0.attention.qkv.weight",
            "vision.projector.norm.bias",
            "vision.projector.mlp.up.weight",
            "vision.projector.mlp.down.weight",
            "vision.summary_indices",
        ):
            assert name in store.records, name
        assert store.records["vision.summary_indices"].dtype == "I64"
    finally:
        store.close()


def test_quantized_conversion_banks_ungated_experts(tmp_path) -> None:
    root = write_checkpoint(tmp_path / "hf")
    output = tmp_path / "nint.mfq"
    convert(root, output, "--quant-backend", "cpu", "--device", "cpu")

    header, store = load_mmap(output)
    try:
        for prefix in ("model.block.1.mlp.experts", "model.block.3.mlp.experts"):
            up = store.records[f"{prefix}.up.weight"]
            down = store.records[f"{prefix}.down.weight"]
            assert up.dtype == "MFE"
            assert down.dtype == "MFE"
            assert f"{prefix}.gate.weight" not in store.records
            assert inspect_mfe_header(store.read_blob(f"{prefix}.up.weight"))[0] == (
                EXPERTS,
                EXPERT_HIDDEN,
                LATENT,
            )
            assert inspect_mfe_header(store.read_blob(f"{prefix}.down.weight"))[0] == (
                EXPERTS,
                LATENT,
                EXPERT_HIDDEN,
            )
        # The per-expert source spelling never reaches the container.
        assert not [
            name for name in store.records if re.search(r"\.experts\.\d+\.", name)
        ]
        assert store.records["model.block.0.linear_attention.conv.weight"].dtype == "F32"
        assert store.records["model.block.0.linear_attention.dt_bias"].dtype == "F32"
        assert store.records["model.block.1.mlp.router.weight"].dtype == "F32"
        assert store.records["model.block.1.mlp.router.bias"].dtype == "F32"
        assert store.records["model.block.0.linear_attention.in_projection.weight"].dtype == "NINT"
        # MTP and vision stay at source precision without explicit opt-in,
        # including the predictor's own routed bank.
        assert store.records["predictor.block.1.mlp.experts.up.weight"].dtype == "BF16"
        assert store.records["vision.projector.mlp.up.weight"].dtype == "BF16"
        assert header.extra["mtp"]["included"] is True
        assert header.extra["mtp"]["hidden_layers"] == 1
        # The predictor body is the declared two-block list, not one stage.
        graph = json.loads(store.read_blob(MODEL_GRAPH_ASSET))
        assert graph["topology"] == {
            "text_layers": 4,
            "vision_layers": 1,
            "predictor_layers": 2,
        }
        assert graph["graph"]["backbone"] == "nemotron_h"
    finally:
        store.close()


def test_recipe_conversion_is_rejected_without_a_nemotron_recipe(tmp_path) -> None:
    root = write_checkpoint(tmp_path / "hf")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {"format": "mfq.gguf-recipe.v1", "tensor_types": {"blk.0.attn_q.weight": "Q4_K"}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tensor is absent from GGUF recipe"):
        convert(root, tmp_path / "recipe.mfq", "--recipe-gguf", str(recipe))


def test_checkpoint_geometry_mismatch_fails_before_quantization(tmp_path) -> None:
    root = write_checkpoint(tmp_path / "hf")
    tensors = synthetic_tensors()
    tensors["language_model.backbone.layers.0.mixer.in_proj.weight"] = _matrix(
        MAMBA_IN_PROJECTION - 2, HIDDEN, 999
    )
    save_file(tensors, root / "model.safetensors")
    with pytest.raises(ValueError, match="in_proj.weight"):
        convert(root, tmp_path / "broken.mfq", "--bf16")
