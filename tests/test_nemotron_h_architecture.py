"""Nemotron-H import contract: config, canonical names and graph spec."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import pytest

from mfq.architectures import NemotronHConfig, parse_nemotron_h_config
from mfq.architectures.tensor_schema import (
    TensorComponent,
    graph_spec_for_plan,
    map_source_tensor_name,
    tensor_schema_for_config,
    topology_from_config,
)
from mfq.server.runtime.capabilities import capabilities_for_architecture

# Released block schedule of the 120B-A12B omni checkpoint: 40 mamba,
# 40 moe and 8 attention blocks (M/E/A).
RELEASED_MIXER_PATTERN = (
    "MEMEMEMAEMEMEMEMAEMEMEMEMAEMEMEMEMEMAEMEMEME"
    "MEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMAEMEMEMEME"
)

# Point MFQ_NEMOTRON_HF_DIR at a released omni checkpoint to check the contract
# against its real config.json and model.safetensors.index.json.
RELEASED_CHECKPOINT_ENV = "MFQ_NEMOTRON_HF_DIR"


def released_checkpoint_file(name: str) -> Path:
    """Return one file of the released checkpoint, or skip when it is absent."""

    root = os.environ.get(RELEASED_CHECKPOINT_ENV)
    if not root:
        pytest.skip(f"{RELEASED_CHECKPOINT_ENV} is not set")
    path = Path(root) / name
    if not path.is_file():
        pytest.skip(f"released Nemotron-H {name} is not available")
    return path


def released_config() -> dict[str, object]:
    """Return the released omni geometry without any weight payload."""

    mixers = {
        "M": "mamba",
        "E": "moe",
        "A": "attention",
    }
    llm = {
        "architectures": ["NemotronHForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "chunk_size": 128,
        "conv_kernel": 4,
        "dtype": "bfloat16",
        "eos_token_id": 11,
        "expand": 2,
        "head_dim": 128,
        "hidden_dropout": 0.0,
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 2688,
        "layer_norm_epsilon": 1e-5,
        "layers_block_type": [mixers[value] for value in RELEASED_MIXER_PATTERN],
        "mamba_head_dim": 64,
        "mamba_hidden_act": "silu",
        "mamba_num_heads": 128,
        "mamba_proj_bias": False,
        "mamba_ssm_cache_dtype": "float32",
        "max_position_embeddings": 1048576,
        "mlp_bias": False,
        "mlp_hidden_act": "relu2",
        "model_type": "nemotron_h",
        "moe_intermediate_size": 2688,
        "moe_latent_size": 1024,
        "moe_shared_expert_intermediate_size": 5376,
        "moe_shared_expert_overlap": False,
        "mtp_layers_block_type": ["attention", "moe"],
        "n_group": 1,
        "n_groups": 8,
        "n_routed_experts": 512,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_experts_per_tok": 22,
        "num_key_value_heads": 2,
        "num_logits_to_keep": 1,
        "num_nextn_predict_layers": 1,
        "pad_token_id": 0,
        "rescale_prenorm_residual": True,
        "residual_in_fp32": False,
        "routed_scaling_factor": 5.0,
        "sliding_window": None,
        "ssm_state_size": 128,
        "tie_word_embeddings": False,
        "time_step_floor": 0.0001,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
        "time_step_max": 0.1,
        "time_step_min": 0.001,
        "topk_group": 1,
        "use_bias": False,
        "use_cache": True,
        "use_conv_bias": True,
        "use_mamba_kernels": True,
        "vocab_size": 131072,
    }
    return {
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "downsample_ratio": 0.5,
        "dtype": "bfloat16",
        "eos_token_id": 11,
        "force_image_size": 512,
        "image_tag_type": "internvl",
        "img_context_token": "<image>",
        "img_context_token_id": 18,
        "llm_config": llm,
        "model_type": "nemotron_h_omni",
        "norm_mean": [0.48145466, 0.4578275, 0.40821073],
        "norm_std": [0.26862954, 0.26130258, 0.27577711],
        "patch_size": 16,
        "projector_hidden_size": 20480,
        "ps_version": "v2",
        "sound_config": None,
        "tie_word_embeddings": False,
        "video_pruning_rate": 0.7,
        "video_temporal_patch_size": 2,
        "vision_config": {
            "attention_probs_dropout_prob": 0.0,
            "drop_path_rate": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 1280,
            "image_size": 224,
            "layer_norm_eps": 1e-6,
            "layerscale_value": 1.0,
            "max_img_size": 2048,
            "max_num_patches": 13312,
            "min_num_patches": 1024,
            "mlp_ratio": 4.0,
            "model_type": "radio",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_cls_tokens": 3,
            "num_hidden_layers": 32,
            "num_registers": 7,
            "patch_size": 16,
            "qkv_bias": True,
            "summary_idxs": [0, 1],
            "use_swiglu_ffn": False,
            "video_temporal_patch_size": 2,
        },
        "vit_hidden_size": 1280,
    }


def mutated_config(**overrides: object) -> dict[str, object]:
    """Return the released config with text-section fields overridden."""

    config = copy.deepcopy(released_config())
    text = config["llm_config"]
    assert isinstance(text, dict)
    for key, value in overrides.items():
        if value is None:
            text.pop(key, None)
        else:
            text[key] = value
    return config


def test_released_config_normalizes_the_declared_geometry() -> None:
    contract = parse_nemotron_h_config(released_config())

    assert contract.family == "nemotron_h"
    assert contract.model_type == "nemotron_h_omni"
    assert contract.text_model_type == "nemotron_h"
    assert contract.num_hidden_layers == 88
    assert contract.mixer_types.count("linear_attention") == 40
    assert contract.mixer_types.count("moe") == 40
    assert contract.mixer_types.count("full_attention") == 8
    assert contract.mamba_inner_size == 8192
    assert contract.mamba_conv_width == 10240
    assert contract.mamba_in_projection_width == 18560
    assert contract.attention_width == 4096
    assert contract.key_value_width == 256
    assert contract.num_experts == 512
    assert contract.num_experts_per_tok == 22
    assert contract.moe_latent_size == 1024
    assert contract.num_nextn_predict_layers == 1
    assert contract.mtp_mixer_types == ("full_attention", "moe")
    assert contract.rescale_prenorm_residual is True
    assert contract.residual_in_fp32 is False
    assert contract.vision is not None
    assert contract.vision.summary_tokens == 10
    assert contract.vision.position_side == 128
    assert contract.projector is not None
    assert contract.projector.patch_width == 5120


def test_infinite_time_step_limit_uses_the_serialized_float_form() -> None:
    contract = parse_nemotron_h_config(released_config())
    assert contract.time_step_limit == (0.0, float("inf"))
    assert contract.time_step_init_limit == (0.001, 0.1)
    assert contract.time_step_floor == 0.0001

    plain = mutated_config(time_step_limit=[0.0, float("inf")])
    assert parse_nemotron_h_config(plain).time_step_limit == (0.0, float("inf"))

    malformed = mutated_config(time_step_limit=[0.0, {"__float__": "wide"}])
    with pytest.raises(ValueError, match="time_step_limit must be a float"):
        parse_nemotron_h_config(malformed)


def test_released_checkpoint_parses_when_present() -> None:
    raw = json.loads(released_checkpoint_file("config.json").read_text(encoding="utf-8"))
    contract = NemotronHConfig.from_hf_config(raw)
    assert contract.num_hidden_layers == 88
    assert contract.mamba_in_projection_width == 18560


def test_text_only_checkpoint_uses_the_same_contract() -> None:
    config = copy.deepcopy(released_config())
    text = config.pop("llm_config")
    assert isinstance(text, dict)
    config.update(text)
    config["model_type"] = "nemotron_h"
    vision = config.pop("vision_config")
    assert vision is not None

    contract = parse_nemotron_h_config(config)
    assert contract.text_model_type == "nemotron_h"
    assert contract.vision is None
    assert contract.projector is None
    assert tensor_schema_for_config(config) is not None
    assert topology_from_config(config).text_layers == 88
    assert topology_from_config(config).predictor_layers == 2


RELEASED_TENSOR_PATTERNS: tuple[tuple[str, str, TensorComponent], ...] = (
    (
        "language_model.backbone.embeddings.weight",
        "model.token_embedding.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.norm_f.weight",
        "model.output_norm.weight",
        TensorComponent.MODEL,
    ),
    ("language_model.lm_head.weight", "model.output.weight", TensorComponent.MODEL),
    (
        "language_model.backbone.layers.0.norm.weight",
        "model.block.0.attention.norm.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.0.mixer.in_proj.weight",
        "model.block.0.linear_attention.in_projection.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.0.mixer.out_proj.weight",
        "model.block.0.linear_attention.output.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.0.mixer.conv1d.weight",
        "model.block.0.linear_attention.conv.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.0.mixer.conv1d.bias",
        "model.block.0.linear_attention.conv.bias",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.0.mixer.A_log",
        "model.block.0.linear_attention.a",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.0.mixer.D",
        "model.block.0.linear_attention.d",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.0.mixer.dt_bias",
        "model.block.0.linear_attention.dt_bias",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.0.mixer.norm.weight",
        "model.block.0.linear_attention.norm.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.1.mixer.gate.weight",
        "model.block.1.mlp.router.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias",
        "model.block.1.mlp.router.bias",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight",
        "model.block.1.mlp.latent.down.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.1.mixer.fc2_latent_proj.weight",
        "model.block.1.mlp.latent.up.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.1.mixer.experts.7.up_proj.weight",
        "model.block.1.mlp.experts.7.up.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.1.mixer.experts.7.down_proj.weight",
        "model.block.1.mlp.experts.7.down.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight",
        "model.block.1.mlp.shared_expert.up.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight",
        "model.block.1.mlp.shared_expert.down.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.7.mixer.q_proj.weight",
        "model.block.7.attention.query.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.7.mixer.k_proj.weight",
        "model.block.7.attention.key.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.7.mixer.v_proj.weight",
        "model.block.7.attention.value.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.backbone.layers.7.mixer.o_proj.weight",
        "model.block.7.attention.output.weight",
        TensorComponent.MODEL,
    ),
    (
        "language_model.mtp.layers.0.eh_proj.weight",
        "predictor.fusion.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.0.enorm.weight",
        "predictor.embedding_norm.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.0.hnorm.weight",
        "predictor.hidden_norm.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.0.final_layernorm.weight",
        "predictor.output_norm.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.0.norm.weight",
        "predictor.block.0.attention.norm.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.0.mixer.q_proj.weight",
        "predictor.block.0.attention.query.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.0.mixer.k_proj.weight",
        "predictor.block.0.attention.key.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.0.mixer.v_proj.weight",
        "predictor.block.0.attention.value.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.0.mixer.o_proj.weight",
        "predictor.block.0.attention.output.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.norm.weight",
        "predictor.block.1.attention.norm.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.mixer.gate.weight",
        "predictor.block.1.mlp.router.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.mixer.gate.e_score_correction_bias",
        "predictor.block.1.mlp.router.bias",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.mixer.fc1_latent_proj.weight",
        "predictor.block.1.mlp.latent.down.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.mixer.fc2_latent_proj.weight",
        "predictor.block.1.mlp.latent.up.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.mixer.experts.0.up_proj.weight",
        "predictor.block.1.mlp.experts.0.up.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.mixer.experts.0.down_proj.weight",
        "predictor.block.1.mlp.experts.0.down.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.mixer.shared_experts.up_proj.weight",
        "predictor.block.1.mlp.shared_expert.up.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "language_model.mtp.layers.1.mixer.shared_experts.down_proj.weight",
        "predictor.block.1.mlp.shared_expert.down.weight",
        TensorComponent.PREDICTOR,
    ),
    (
        "vision_model.radio_model.input_conditioner.norm_mean",
        "vision.input_conditioner.mean",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.input_conditioner.norm_std",
        "vision.input_conditioner.std",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.patch_generator.embedder.weight",
        "vision.patch_embedding.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.patch_generator.video_embedder.weight",
        "vision.video_patch_embedding.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.patch_generator.pos_embed",
        "vision.position_embedding.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.patch_generator.cls_token.token",
        "vision.cls_register_token.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.norm1.weight",
        "vision.block.0.norm1.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.norm1.bias",
        "vision.block.0.norm1.bias",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.norm2.weight",
        "vision.block.0.norm2.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.norm2.bias",
        "vision.block.0.norm2.bias",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight",
        "vision.block.0.attention.qkv.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.attn.qkv.bias",
        "vision.block.0.attention.qkv.bias",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.attn.proj.weight",
        "vision.block.0.attention.output.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.attn.proj.bias",
        "vision.block.0.attention.output.bias",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.mlp.fc1.weight",
        "vision.block.0.mlp.up.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.mlp.fc1.bias",
        "vision.block.0.mlp.up.bias",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.mlp.fc2.weight",
        "vision.block.0.mlp.down.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_model.radio_model.model.blocks.0.mlp.fc2.bias",
        "vision.block.0.mlp.down.bias",
        TensorComponent.VISION,
    ),
    (
        "vision_model.summary_idxs",
        "vision.summary_indices",
        TensorComponent.VISION,
    ),
    (
        "vision_projector.vision_final_layernorm.weight",
        "vision.projector.norm.weight",
        TensorComponent.VISION,
    ),
    (
        "vision_projector.vision_final_layernorm.bias",
        "vision.projector.norm.bias",
        TensorComponent.VISION,
    ),
    ("mlp1.0.weight", "vision.projector.mlp.norm.weight", TensorComponent.VISION),
    ("mlp1.1.weight", "vision.projector.mlp.up.weight", TensorComponent.VISION),
    ("mlp1.3.weight", "vision.projector.mlp.down.weight", TensorComponent.VISION),
)


@pytest.mark.parametrize("source,canonical,component", RELEASED_TENSOR_PATTERNS)
def test_every_released_tensor_pattern_has_a_canonical_name(
    source: str,
    canonical: str,
    component: TensorComponent,
) -> None:
    mapped = map_source_tensor_name(source, released_config(), require_registered=True)
    assert mapped is not None
    assert mapped.canonical_name == canonical
    assert mapped.component is component


def _normalized_pattern(name: str) -> str:
    """Collapse per-layer indices so a pattern table entry matches the index."""

    return re.sub(r"(?<=\.)\d+(?=\.)", "N", name)


def test_released_pattern_table_covers_the_whole_tensor_index() -> None:
    """Every pattern of the released index must appear in the mapping table."""

    index_path = released_checkpoint_file("model.safetensors.index.json")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    expected = {_normalized_pattern(name) for name in weight_map}
    covered = {_normalized_pattern(source) for source, _canonical, _ in RELEASED_TENSOR_PATTERNS}
    assert covered == expected


def test_registration_declares_the_import_policy() -> None:
    registration = tensor_schema_for_config(released_config())
    assert registration is not None
    assert registration.architecture == "nemotron_h_omni"
    assert registration.backbone == "nemotron_h"
    assert registration.expert_projections == ("up", "down")
    for role in (".linear_attention.a", ".linear_attention.conv.weight", ".mlp.router.bias"):
        assert role in registration.float32_weight_roles
    assert {spec.component for spec in registration.component_specs} == {
        TensorComponent.VISION,
        TensorComponent.PREDICTOR,
    }


def test_mixer_kind_comes_from_the_schedule_not_the_tensor_name() -> None:
    config = released_config()
    # Layer 1 is a MoE block; its block norm is not an attention-layer norm and
    # its mixer carries no attention projections.
    assert (
        map_source_tensor_name(
            "language_model.backbone.layers.1.norm.weight", config
        ).canonical_name
        == "model.block.1.attention.norm.weight"
    )
    assert (
        map_source_tensor_name(
            "language_model.backbone.layers.1.mixer.q_proj.weight", config
        )
        is None
    )
    assert (
        map_source_tensor_name(
            "language_model.backbone.layers.7.mixer.gate.weight", config
        )
        is None
    )
    assert (
        map_source_tensor_name(
            "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight", config
        )
        is None
    )


def test_dense_mlp_blocks_are_supported_when_the_schedule_uses_them() -> None:
    config = mutated_config(layers_block_type=["mamba", "mlp"])
    assert (
        map_source_tensor_name(
            "language_model.backbone.layers.1.mixer.up_proj.weight", config
        ).canonical_name
        == "model.block.1.mlp.up.weight"
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"n_group": 2}, "n_group and topk_group must both be 1"),
        ({"topk_group": 2}, "n_group and topk_group must both be 1"),
        ({"moe_latent_size": None}, "moe_latent_size"),
        ({"layers_block_type": ["mamba", "dense_attention"]}, "unsupported mixer types"),
        ({"num_hidden_layers": 40}, "disagrees with layers_block_type"),
        ({"head_dim": 96}, "attention width"),
        ({"mlp_hidden_act": "silu"}, "routed-expert activation"),
        ({"num_nextn_predict_layers": 2}, "multi-stage prediction heads"),
        ({"attention_bias": True}, "attention projections carry no bias"),
    ],
)
def test_inconsistent_configs_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_nemotron_h_config(mutated_config(**overrides))


def test_vision_and_projector_contract_must_match_the_checkpoint() -> None:
    config = released_config()
    vision = config["vision_config"]
    assert isinstance(vision, dict)
    vision["layerscale_value"] = 0.5
    with pytest.raises(ValueError, match="no LayerScale weights"):
        parse_nemotron_h_config(config)

    config = released_config()
    config["downsample_ratio"] = 0.25
    with pytest.raises(ValueError, match="downsample_ratio 0.5"):
        parse_nemotron_h_config(config)

    config = released_config()
    config["ps_version"] = "v1"
    with pytest.raises(ValueError, match="ps_version"):
        parse_nemotron_h_config(config)


def test_source_shape_validator_rejects_a_wrong_in_projection_width() -> None:
    config = released_config()
    registration = tensor_schema_for_config(config)
    assert registration is not None and registration.source_shape_validator is not None
    shapes = {
        "language_model.backbone.layers.0.mixer.in_proj.weight": (18432, 4096),
        "language_model.backbone.layers.0.mixer.conv1d.weight": (10240, 1, 4),
        "language_model.backbone.layers.7.mixer.k_proj.weight": (256, 4096),
        "language_model.mtp.layers.0.eh_proj.weight": (4096, 8192),
        "vision_model.radio_model.model.patch_generator.pos_embed": (1, 16384, 1280),
    }
    with pytest.raises(
        ValueError,
        match="shape mismatch for language_model.backbone.layers.0.mixer.in_proj.weight",
    ):
        registration.source_shape_validator(config, shapes)

    shapes["language_model.backbone.layers.0.mixer.in_proj.weight"] = (18560, 4096)
    registration.source_shape_validator(config, shapes)


def test_graph_spec_separates_radio_vision_from_the_text_backbone() -> None:
    graph = graph_spec_for_plan(
        released_config(),
        [
            "model.token_embedding.weight",
            "vision.patch_embedding.weight",
            "predictor.fusion.weight",
        ],
    )
    assert graph is not None
    payload = graph.as_dict()
    assert payload["graph"] == {"kind": "causal_lm", "backbone": "nemotron_h"}
    assert payload["topology"] == {
        "text_layers": 88,
        "vision_layers": 32,
        "predictor_layers": 2,
    }
    components = {item["kind"]: item for item in payload["components"]}
    assert components["vision"]["implementation"] == "radio"
    assert components["vision"]["input_contract"] == "radio_cpe.v1"
    assert components["predictor"]["implementation"] == "nemotron_h_mtp"
    encoded = json.dumps(payload, sort_keys=True)
    assert "language_model" not in encoded
    assert "vision_model" not in encoded
    assert "mtp." not in encoded


def test_capabilities_register_the_family_without_advertising_media_or_mtp() -> None:
    for model_type in ("nemotron_h", "nemotron_h_text", "nemotron_h_omni"):
        profile = capabilities_for_architecture(model_type)
        assert profile.architecture_family == "nemotron_h"
        assert profile.features.model_dump() == {
            "text": True,
            "image_input": False,
            "video_input": False,
            "audio_input": False,
            "audio_output": False,
            "full_duplex": False,
            "mtp": False,
        }
