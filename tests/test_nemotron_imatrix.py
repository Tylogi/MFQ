"""Nemotron-H layerwise calibration: forward replay and imatrix coverage."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from mfq.architectures.hf_config import load_hf_model_config
from mfq.architectures.nemotron_h import NemotronHConfig
from mfq.calibration.imatrix import (
    ActivationImatrixCollector,
    _backend,
    _targets,
)
from mfq.calibration.layerwise_nemotron_h import NemotronHLayerwiseBackend
from mfq.quantize.imatrix import ImportanceMatrix

HIDDEN = 16
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 8
MAMBA_HEADS = 2
MAMBA_HEAD_DIM = 8
GROUPS = 2
STATE_SIZE = 4
CONV_KERNEL = 4
EXPERTS = 3
EXPERT_HIDDEN = 12
LATENT = 4
SHARED_HIDDEN = 8
INTERMEDIATE = 12
VOCAB = 48
LAYERS = ["mamba", "moe", "attention", "mlp"]


def _text_config() -> dict[str, object]:
    return {
        "model_type": "nemotron_h",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 64,
        "attention_bias": False,
        "mlp_bias": False,
        "use_bias": False,
        "mamba_proj_bias": False,
        "layer_norm_epsilon": 1e-5,
        "layers_block_type": LAYERS,
        "mamba_num_heads": MAMBA_HEADS,
        "mamba_head_dim": MAMBA_HEAD_DIM,
        "ssm_state_size": STATE_SIZE,
        "n_groups": GROUPS,
        "conv_kernel": CONV_KERNEL,
        "use_conv_bias": True,
        "mamba_hidden_act": "silu",
        # One reference chunk keeps the Transformers torch path on its
        # single-chunk branch; the backend still uses four-token chunks.
        "chunk_size": 64,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
        "time_step_min": 0.001,
        "time_step_max": 0.1,
        "time_step_floor": 0.0001,
        "intermediate_size": INTERMEDIATE,
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
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": 0,
        "mamba_ssm_cache_dtype": "float32",
        "use_mamba_kernels": False,
    }


def _matrix(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def _tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    prefix = "language_model.backbone."
    tensors[prefix + "embeddings.weight"] = _matrix(VOCAB, HIDDEN, seed=1)
    tensors[prefix + "norm_f.weight"] = torch.ones(HIDDEN)
    tensors["language_model.lm_head.weight"] = _matrix(VOCAB, HIDDEN, seed=2)
    inner = MAMBA_HEADS * MAMBA_HEAD_DIM
    conv_width = inner + 2 * GROUPS * STATE_SIZE
    projection_width = 2 * inner + 2 * GROUPS * STATE_SIZE + MAMBA_HEADS
    seed = 10
    for layer, mixer in enumerate(LAYERS):
        block = prefix + f"layers.{layer}."
        tensors[block + "norm.weight"] = torch.ones(HIDDEN)
        if mixer == "mamba":
            tensors[block + "mixer.in_proj.weight"] = _matrix(
                projection_width, HIDDEN, seed=seed
            )
            seed += 1
            tensors[block + "mixer.conv1d.weight"] = _matrix(
                conv_width, 1, CONV_KERNEL, seed=seed
            )
            seed += 1
            tensors[block + "mixer.conv1d.bias"] = _matrix(conv_width, seed=seed)
            seed += 1
            tensors[block + "mixer.A_log"] = torch.log(
                torch.arange(1, MAMBA_HEADS + 1, dtype=torch.float32)
            )
            tensors[block + "mixer.D"] = torch.ones(MAMBA_HEADS)
            tensors[block + "mixer.dt_bias"] = torch.zeros(MAMBA_HEADS)
            tensors[block + "mixer.norm.weight"] = torch.ones(inner)
            tensors[block + "mixer.out_proj.weight"] = _matrix(
                HIDDEN, inner, seed=seed
            )
            seed += 1
        elif mixer == "attention":
            tensors[block + "mixer.q_proj.weight"] = _matrix(
                HIDDEN, HIDDEN, seed=seed
            )
            seed += 1
            tensors[block + "mixer.k_proj.weight"] = _matrix(
                KV_HEADS * HEAD_DIM, HIDDEN, seed=seed
            )
            seed += 1
            tensors[block + "mixer.v_proj.weight"] = _matrix(
                KV_HEADS * HEAD_DIM, HIDDEN, seed=seed
            )
            seed += 1
            tensors[block + "mixer.o_proj.weight"] = _matrix(
                HIDDEN, HIDDEN, seed=seed
            )
            seed += 1
        elif mixer == "mlp":
            tensors[block + "mixer.up_proj.weight"] = _matrix(
                INTERMEDIATE, HIDDEN, seed=seed
            )
            seed += 1
            tensors[block + "mixer.down_proj.weight"] = _matrix(
                HIDDEN, INTERMEDIATE, seed=seed
            )
            seed += 1
        elif mixer == "moe":
            tensors[block + "mixer.gate.weight"] = _matrix(
                EXPERTS, HIDDEN, seed=seed
            )
            seed += 1
            tensors[block + "mixer.gate.e_score_correction_bias"] = torch.zeros(EXPERTS)
            tensors[block + "mixer.fc1_latent_proj.weight"] = _matrix(
                LATENT, HIDDEN, seed=seed
            )
            seed += 1
            tensors[block + "mixer.fc2_latent_proj.weight"] = _matrix(
                HIDDEN, LATENT, seed=seed
            )
            seed += 1
            tensors[block + "mixer.shared_experts.up_proj.weight"] = _matrix(
                SHARED_HIDDEN, HIDDEN, seed=seed
            )
            seed += 1
            tensors[block + "mixer.shared_experts.down_proj.weight"] = _matrix(
                HIDDEN, SHARED_HIDDEN, seed=seed
            )
            seed += 1
            for expert in range(EXPERTS):
                tensors[block + f"mixer.experts.{expert}.up_proj.weight"] = _matrix(
                    EXPERT_HIDDEN, LATENT, seed=seed
                )
                seed += 1
                tensors[block + f"mixer.experts.{expert}.down_proj.weight"] = _matrix(
                    LATENT, EXPERT_HIDDEN, seed=seed
                )
                seed += 1
        else:
            raise AssertionError(mixer)
    return tensors


def _write_checkpoint(root: Path, *, omni: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    save_file(_tensors(), root / "model.safetensors")
    text = _text_config()
    config = (
        {"model_type": "nemotron_h_omni", "llm_config": text}
        if omni
        else text
    )
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def _reference_state(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if re.search(r"\.mixer\.experts\.\d+\.", name):
            continue
        if name.startswith("language_model.backbone."):
            state["model." + name[len("language_model.backbone.") :]] = value
        elif name.startswith("language_model.lm_head."):
            state[name[len("language_model.") :]] = value
    for layer, mixer in enumerate(LAYERS):
        if mixer != "moe":
            continue
        for projection in ("up", "down"):
            names = [
                f"language_model.backbone.layers.{layer}.mixer.experts."
                f"{expert}.{projection}_proj.weight"
                for expert in range(EXPERTS)
            ]
            state[f"model.layers.{layer}.mixer.experts.{projection}_proj"] = torch.stack(
                [tensors[name] for name in names]
            )
    return state


def _collect(root: Path, ids: torch.Tensor) -> tuple[
    NemotronHLayerwiseBackend,
    dict[str, object],
]:
    backend = NemotronHLayerwiseBackend(
        root,
        device="cpu",
        teacher_dtype=torch.float32,
        scan_chunk_size=4,
    )
    by_layer, targets = backend.imatrix_targets()
    collector = ActivationImatrixCollector(
        targets,
        torch.device("cpu"),
        accumulation_dtype=torch.float64,
        neural=True,
    )
    hidden = backend.initial_hidden(ids)
    for layer_index in range(backend.num_layers):
        with backend.layer(layer_index, quantized=False) as layer:
            collector.install_layer(layer, layer_index, by_layer[layer_index])
            try:
                hidden = backend.forward_layer(layer, layer_index, hidden)
            finally:
                collector.close()
    return backend, collector.entries()


def test_layerwise_forward_matches_transformers_reference(tmp_path) -> None:
    from transformers import NemotronHForCausalLM

    root = _write_checkpoint(tmp_path / "hf")
    tensors = _tensors()
    backend = NemotronHLayerwiseBackend(
        root,
        device="cpu",
        teacher_dtype=torch.float32,
        scan_chunk_size=4,
    )
    reference = NemotronHForCausalLM(backend.config).to(torch.float32).eval()
    missing, unexpected = reference.load_state_dict(
        _reference_state(tensors), strict=False
    )
    assert not missing
    assert not [name for name in unexpected if ".experts." not in name]

    ids = torch.arange(12).reshape(1, 12) % VOCAB
    hidden = backend.initial_hidden(ids)
    for layer_index in range(backend.num_layers):
        with backend.layer(layer_index, quantized=False) as layer:
            hidden = backend.forward_layer(layer, layer_index, hidden)
    with torch.no_grad():
        reference_hidden = reference.model.embeddings(ids)
        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
        )

        positions = torch.arange(12).unsqueeze(0)
        masks = {
            "full_attention": create_causal_mask(
                config=reference.config,
                inputs_embeds=reference_hidden,
                attention_mask=None,
                past_key_values=None,
                position_ids=positions,
            ),
            "linear_attention": create_recurrent_attention_mask(
                config=reference.config,
                inputs_embeds=reference_hidden,
                attention_mask=None,
                past_key_values=None,
            ),
        }
        for block in reference.model.layers:
            reference_hidden = block(
                reference_hidden,
                attention_mask=masks.get(block.block_type),
                position_ids=positions,
                use_cache=False,
            )
        weight = backend.index.tensor("language_model.backbone.norm_f.weight")
        variance = hidden.float().square().mean(-1, keepdim=True)
        normalized = weight * (
            hidden.float() * torch.rsqrt(variance + backend.contract.norm_epsilon)
        )
        logits = torch.nn.functional.linear(
            normalized, tensors["language_model.lm_head.weight"]
        )
        reference_logits = reference(ids, use_cache=False).logits

    torch.testing.assert_close(
        hidden, reference_hidden, atol=0.02, rtol=1e-5
    )
    torch.testing.assert_close(
        logits, reference_logits, atol=1e-4, rtol=1e-5
    )


def test_imatrix_covers_every_nemotron_linear(tmp_path) -> None:
    root = _write_checkpoint(tmp_path / "hf")
    ids = torch.arange(12).reshape(1, 12) % VOCAB
    backend, entries = _collect(root, ids)
    by_layer, targets = backend.imatrix_targets()
    names = {target.name for target in targets}
    assert set(entries) == names

    entries_by_name = entries
    assert entries_by_name[
        "language_model.backbone.layers.0.mixer.in_proj.weight"
    ].values.shape == (1, HIDDEN)
    assert entries_by_name[
        "language_model.backbone.layers.0.mixer.out_proj.weight"
    ].values.shape == (1, MAMBA_HEADS * MAMBA_HEAD_DIM)
    for projection in ("q_proj", "k_proj", "v_proj"):
        assert entries_by_name[
            f"language_model.backbone.layers.2.mixer.{projection}.weight"
        ].values.shape == (1, HIDDEN)
    assert entries_by_name[
        "language_model.backbone.layers.2.mixer.o_proj.weight"
    ].values.shape == (1, HIDDEN)
    assert entries_by_name[
        "language_model.backbone.layers.3.mixer.up_proj.weight"
    ].values.shape == (1, HIDDEN)
    assert entries_by_name[
        "language_model.backbone.layers.3.mixer.down_proj.weight"
    ].values.shape == (1, INTERMEDIATE)

    assert entries_by_name[
        "language_model.backbone.layers.1.mixer.gate.weight"
    ].counts.tolist() == [12]
    assert entries_by_name[
        "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight"
    ].values.shape == (1, HIDDEN)
    assert entries_by_name[
        "language_model.backbone.layers.1.mixer.fc2_latent_proj.weight"
    ].values.shape == (1, LATENT)
    assert entries_by_name[
        "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight"
    ].values.shape == (1, HIDDEN)
    assert entries_by_name[
        "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight"
    ].values.shape == (1, SHARED_HIDDEN)

    for projection, width, rows in (
        ("up", LATENT, EXPERT_HIDDEN),
        ("down", EXPERT_HIDDEN, LATENT),
    ):
        entry = entries_by_name[
            f"model.block.1.mlp.experts.{projection}.weight"
        ]
        assert entry.values.shape == (EXPERTS, width)
        assert entry.counts.shape == (EXPERTS,)
        assert np.all(entry.counts > 0)
        assert entry.row_importance is not None
        assert entry.row_importance.shape == (EXPERTS * rows,)
    assert ids.shape[1] * 2 == int(
        entries_by_name["model.block.1.mlp.experts.up.weight"].counts.sum()
    )


def test_imatrix_names_bind_to_hf_conversion_plans(tmp_path) -> None:
    from mfq.calibration.artifact import ExpertPrecision
    from mfq.formats.nint import NintSpec
    from mfq.tools.quantize_hf_to_mfq import TensorPlan, _bind_hf_imatrix

    root = _write_checkpoint(tmp_path / "hf")
    ids = torch.arange(12).reshape(1, 12) % VOCAB
    _backend_instance, entries = _collect(root, ids)
    imatrix = ImportanceMatrix(
        path=tmp_path / "imatrix.mfq",
        entries=entries,
        datasets=("synthetic",),
        chunk_count=1,
        chunk_size=12,
        legacy=False,
    )

    linear = TensorPlan(
        name="model.block.0.linear_attention.in_projection.weight",
        shard="model.safetensors",
        shape=(2 * MAMBA_HEADS * MAMBA_HEAD_DIM + 2 * GROUPS * STATE_SIZE + MAMBA_HEADS, HIDDEN),
        source_dtype="F32",
        target_dtype="NINT4",
        gguf_name=None,
        source_name="language_model.backbone.layers.0.mixer.in_proj.weight",
    )
    bank = TensorPlan(
        name="model.block.1.mlp.experts.up.weight",
        shard="model.safetensors",
        shape=(EXPERTS, EXPERT_HIDDEN, LATENT),
        source_dtype="F32",
        target_dtype="MFE",
        gguf_name=None,
        expert_shape=(EXPERTS, EXPERT_HIDDEN, LATENT),
        expert_precisions=(ExpertPrecision("NINT4", NintSpec(4, 8, 6)),) * EXPERTS,
    )

    _, linear_entry = imatrix.find([linear.name, linear.source_name])
    assert linear_entry is not None
    binding = _bind_hf_imatrix(imatrix, [linear, bank])
    assert binding[linear.name].entry_name == linear.source_name
    assert binding[bank.name].entry_name == bank.name
    assert binding[linear.name].rows(0, 1).shape == (1, HIDDEN)
    assert binding[bank.name].rows(0, EXPERT_HIDDEN).shape == (EXPERT_HIDDEN, LATENT)


def test_backend_dispatches_text_and_omni_nemotron_configs(tmp_path) -> None:
    for name, omni in (("text", False), ("omni", True)):
        root = _write_checkpoint(tmp_path / name, omni=omni)
        backend, model_type = _backend(root, torch.device("cpu"), "sdpa")
        assert isinstance(backend, NemotronHLayerwiseBackend)
        assert model_type == "nemotron_h"
        by_layer, targets = _targets(backend, model_type)
        assert set(by_layer) == set(range(len(LAYERS)))
        assert len(targets) == sum(len(values) for values in by_layer.values())
        contract = load_hf_model_config(root)
        assert NemotronHConfig.from_hf_config(contract).num_hidden_layers == len(LAYERS)
