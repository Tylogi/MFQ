"""Nemotron-H V-tier recipes: GGUF names, registry caps and derived IQ4_XS."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import mfq.tools.quantize_hf_to_mfq as hf_to_mfq
from mfq.architectures.tensor_schema import (
    canonical_recipe_name,
    map_source_tensor_name,
    nemotron_h_recipe_name,
    qwen4_recipe_name,
)
from mfq.calibration.ud_recipes import (
    UD_TIERS,
    automatic_recipe,
    registered_model,
)
from mfq.quantize.imatrix import ImportanceEntry, save_importance_matrix
from mfq.tools import derive_gguf_recipe as derive
from tests.test_quantize.test_nemotron_h_import import (
    EXPERT_HIDDEN,
    EXPERTS,
    LATENT,
    synthetic_config,
    synthetic_tensors,
    write_checkpoint,
)

RECIPE_DIR = Path(__file__).resolve().parents[1] / "mfq/calibration/recipes/nemotron_h"
MTP_BLOCK = 88


def released_text_config() -> dict[str, object]:
    """Return the released 120B-A12B text geometry without a weight payload."""

    pattern = ("MEMEMEMAEMEMEMEMAEMEMEMEMAEMEMEMEMEMAEMEMEME"
               "MEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMAEMEMEMEME")
    mixers = {"M": "mamba", "E": "moe", "A": "attention"}
    return {
        "model_type": "nemotron_h",
        "layers_block_type": [mixers[value] for value in pattern],
        "mtp_layers_block_type": ["attention", "moe"],
        "num_nextn_predict_layers": 1,
        "hidden_size": 4096,
        "n_routed_experts": 512,
        "num_experts_per_tok": 22,
        "moe_latent_size": 1024,
        "vocab_size": 131072,
    }


@pytest.mark.parametrize(
    "canonical,recipe",
    [
        ("model.token_embedding.weight", "token_embd.weight"),
        ("model.output_norm.weight", "output_norm.weight"),
        ("model.output.weight", "output.weight"),
        ("model.block.0.attention.norm.weight", "blk.0.attn_norm.weight"),
        ("model.block.0.linear_attention.a", "blk.0.ssm_a"),
        ("model.block.0.linear_attention.d", "blk.0.ssm_d"),
        ("model.block.0.linear_attention.in_projection.weight", "blk.0.ssm_in.weight"),
        ("model.block.0.linear_attention.conv.weight", "blk.0.ssm_conv1d.weight"),
        ("model.block.0.linear_attention.conv.bias", "blk.0.ssm_conv1d.bias"),
        ("model.block.0.linear_attention.dt_bias", "blk.0.ssm_dt.bias"),
        ("model.block.0.linear_attention.norm.weight", "blk.0.ssm_norm.weight"),
        ("model.block.0.linear_attention.output.weight", "blk.0.ssm_out.weight"),
        ("model.block.7.attention.query.weight", "blk.7.attn_q.weight"),
        ("model.block.7.attention.key.weight", "blk.7.attn_k.weight"),
        ("model.block.7.attention.value.weight", "blk.7.attn_v.weight"),
        ("model.block.7.attention.output.weight", "blk.7.attn_output.weight"),
        ("model.block.1.mlp.router.weight", "blk.1.ffn_gate_inp.weight"),
        ("model.block.1.mlp.router.bias", "blk.1.exp_probs_b.bias"),
        ("model.block.1.mlp.latent.down.weight", "blk.1.ffn_latent_down.weight"),
        ("model.block.1.mlp.latent.up.weight", "blk.1.ffn_latent_up.weight"),
        ("model.block.1.mlp.shared_expert.up.weight", "blk.1.ffn_up_shexp.weight"),
        ("model.block.1.mlp.shared_expert.down.weight", "blk.1.ffn_down_shexp.weight"),
        ("model.block.1.mlp.experts.up.weight", "blk.1.ffn_up_exps.weight"),
        ("model.block.1.mlp.experts.down.weight", "blk.1.ffn_down_exps.weight"),
        ("model.block.3.mlp.up.weight", "blk.3.ffn_up.weight"),
        ("model.block.3.mlp.down.weight", "blk.3.ffn_down.weight"),
        # Per-expert views are banked into ffn_*_exps, so only the bank alias exists.
        ("model.block.1.mlp.experts.5.up.weight", None),
        # The GGUF conversion folds the whole MTP head into blk.88 plus nextn.*
        # roots, so the predictor graph keeps its own precision policy.
        ("predictor.fusion.weight", None),
        ("predictor.embedding_norm.weight", None),
        ("predictor.block.0.attention.norm.weight", None),
        ("predictor.block.1.mlp.experts.up.weight", None),
        ("vision.patch_embedding.weight", None),
    ],
)
def test_nemotron_recipe_name_table(canonical: str, recipe: str | None) -> None:
    assert nemotron_h_recipe_name(canonical) == recipe


def test_registered_mapper_fills_recipe_names_at_the_import_boundary() -> None:
    config = {"model_type": "nemotron_h_omni", "llm_config": released_text_config()}
    mapped = map_source_tensor_name(
        "language_model.backbone.layers.0.mixer.in_proj.weight", config
    )
    assert mapped is not None
    assert mapped.canonical_name == "model.block.0.linear_attention.in_projection.weight"
    assert mapped.recipe_name == "blk.0.ssm_in.weight"
    assert canonical_recipe_name("model.block.1.mlp.experts.up.weight", config) == (
        "blk.1.ffn_up_exps.weight"
    )


def test_qwen4_recipe_names_are_unchanged_by_the_registration() -> None:
    config = {"model_type": "qwen4_exp_text", "num_hidden_layers": 48}
    for canonical in (
        "model.token_embedding.weight",
        "model.output.weight",
        "model.block.3.attention.query.weight",
        "model.block.3.mlp.experts.up.weight",
        "model.block.3.position_embedding.ngram.shard.0.weight",
    ):
        assert canonical_recipe_name(canonical, config) == qwen4_recipe_name(canonical)
    mapped = map_source_tensor_name(
        "model.language_model.layers.3.linear_attn.in_proj_qkv.weight", config
    )
    assert mapped is not None and mapped.recipe_name == "blk.3.attn_qkv.weight"


# GGUF spelling -> canonical leaf for every main text block role of the family.
_MAIN_TEXT_ROLES = {
    "attn_norm.weight": "attention.norm.weight",
    "ssm_a": "linear_attention.a",
    "ssm_d": "linear_attention.d",
    "ssm_in.weight": "linear_attention.in_projection.weight",
    "ssm_conv1d.weight": "linear_attention.conv.weight",
    "ssm_conv1d.bias": "linear_attention.conv.bias",
    "ssm_dt.bias": "linear_attention.dt_bias",
    "ssm_norm.weight": "linear_attention.norm.weight",
    "ssm_out.weight": "linear_attention.output.weight",
    "attn_q.weight": "attention.query.weight",
    "attn_k.weight": "attention.key.weight",
    "attn_v.weight": "attention.value.weight",
    "attn_output.weight": "attention.output.weight",
    "ffn_gate_inp.weight": "mlp.router.weight",
    "exp_probs_b.bias": "mlp.router.bias",
    "ffn_latent_down.weight": "mlp.latent.down.weight",
    "ffn_latent_up.weight": "mlp.latent.up.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down.weight",
    "ffn_up_exps.weight": "mlp.experts.up.weight",
    "ffn_down_exps.weight": "mlp.experts.down.weight",
    "ffn_up.weight": "mlp.up.weight",
    "ffn_down.weight": "mlp.down.weight",
}
_MTP_ONLY_ROLES = {"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                   "nextn.shared_head_norm.weight", "post_attention_norm.weight"}
_RECIPE_ROOTS = {
    "token_embd.weight": "model.token_embedding.weight",
    "output_norm.weight": "model.output_norm.weight",
    "output.weight": "model.output.weight",
}


@pytest.mark.parametrize("tier", ["IQ2_M", "IQ4_XS"])
def test_recipes_cover_every_main_text_role_and_keep_mtp_entries(tier: str) -> None:
    document = json.loads((RECIPE_DIR / f"{tier}.json").read_text(encoding="utf-8"))
    assert document["format"] == "mfq.gguf-recipe.v1"
    assert document["repo"] == "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF"
    assert document["revision"] == "75a5829055f69a0359085ae6bbd0a81b60bc05c6"
    tensor_types = document["tensor_types"]
    assert len(tensor_types) == 781
    covered = 0
    for name in tensor_types:
        if name in _RECIPE_ROOTS:
            assert nemotron_h_recipe_name(_RECIPE_ROOTS[name]) == name
            covered += 1
            continue
        layer, _, suffix = name.removeprefix("blk.").partition(".")
        if suffix in _MTP_ONLY_ROLES:
            assert layer == str(MTP_BLOCK)
            continue
        canonical = f"model.block.{layer}.{_MAIN_TEXT_ROLES[suffix]}"
        assert nemotron_h_recipe_name(canonical) == name
        covered += 1
    # Every entry but the folded MTP block's own five roles has an alias.
    assert covered == len(tensor_types) - len(_MTP_ONLY_ROLES)


def test_registered_tiers_pin_the_generated_recipes() -> None:
    config = {"model_type": "nemotron_h_omni", "llm_config": released_text_config()}
    model = registered_model(config)
    assert model is not None
    for tier in ("IQ2_M", "IQ4_XS"):
        path = automatic_recipe(config, tier=tier)
        assert path == RECIPE_DIR / f"{tier}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["derivation"]["ud_file_bytes"] == model["caps"][tier]
        assert document["derivation"]["ftype"] == ("source" if tier == "IQ2_M" else tier)
    assert model["parameters"] == json.loads(
        (RECIPE_DIR / "IQ2_M.json").read_text(encoding="utf-8")
    )["derivation"]["main_text_elements"]
    assert automatic_recipe(config, 3.7).stem == "IQ2_M"
    assert automatic_recipe(config, 4.4).stem == "IQ4_XS"
    # IQ1_S belongs to the Qwen registry; a Nemotron config must reject it.
    with pytest.raises(ValueError, match="UD tier IQ1_S is unavailable"):
        automatic_recipe(config, tier="IQ1_S")


def test_quantize_cli_accepts_the_nemotron_tiers() -> None:
    from mfq.cli import _build_parser
    from mfq.commands.quantize import _UD_TIER_CHOICES

    # The literal CLI choices must stay identical to the registered tiers.
    assert tuple(_UD_TIER_CHOICES) == tuple(UD_TIERS)
    parser = _build_parser()
    for tier in ("IQ2_M", "IQ4_XS"):
        args = parser.parse_args(["quantize", "in", "out", "--ud-tier", tier])
        assert args.ud_tier == tier


def test_iq4_xs_derivation_types_and_ceiling() -> None:
    assert derive.payload_bytes((1024, 2688, 512), "IQ4_XS") == 1024 * 2688 * 512 // 256 * 136
    assert derive.payload_bytes((64,), "Q2_0") == 18
    assert derive.aligned_bytes(136) == 160
    assert derive.payload_bytes((7,), "F32") == 28
    assert derive.allows_quantization("blk.0.ssm_conv1d.weight", (4, 10240)) is False
    assert derive.allows_quantization("blk.0.attn_norm.weight", (4096,)) is False
    assert derive.allows_quantization("blk.1.ffn_gate_inp.weight", (4096, 512)) is False
    assert derive.allows_quantization("blk.1.exp_probs_b.bias", (512,)) is False
    assert derive.allows_quantization("blk.1.ffn_up_exps.weight", (1024, 2688, 512)) is True

    def tensor_type(name: str, shape: tuple[int, ...]) -> str:
        return derive.derive_tensor_type(
            name,
            shape,
            "F32",
            ftype="IQ4_XS",
            has_imatrix=True,
            n_gqa=0,
            n_expert=512,
            n_layer_all=89,
            counters={"i_ffn_down": 0, "i_attention_wv": 0},
        )

    assert tensor_type("output.weight", (4096, 131072)) == "Q6_K"
    assert tensor_type("token_embd.weight", (4096, 131072)) == "IQ4_XS"
    # llama.cpp evaluates hparams.n_gqa() at block 0, which is recurrent here
    # (head_count_kv[0] == 0), so the IQ4_XS attention-V bump never fires.
    assert tensor_type("blk.7.attn_v.weight", (4096, 256)) == "IQ4_XS"
    assert tensor_type("blk.7.attn_q.weight", (4096, 4096)) == "IQ4_XS"
    assert tensor_type("blk.88.nextn.eh_proj.weight", (8192, 4096)) == "IQ4_XS"
    # 2688 is not a multiple of the 256-element IQ4_XS block, so llama.cpp
    # demotes the routed Down bank to the 32-element IQ4_NL block.
    assert tensor_type("blk.1.ffn_down_exps.weight", (2688, 1024, 512)) == "IQ4_NL"
    assert tensor_type("blk.1.ffn_down_shexp.weight", (5376, 4096)) == "IQ4_XS"
    assert tensor_type("blk.1.ffn_up_exps.weight", (1024, 2688, 512)) == "IQ4_XS"
    assert tensor_type("blk.0.ssm_conv1d.weight", (4, 10240)) == "F32"


def test_iq4_xs_ceiling_adds_the_iq2_m_header_overhead() -> None:
    records = [
        {"name": "token_embd.weight", "type": "Q5_K", "shape": [256, 8]},
        {"name": "output.weight", "type": "Q6_K", "shape": [256, 8]},
        {"name": "blk.0.attn_norm.weight", "type": "F32", "shape": [256]},
        {"name": "blk.0.ffn_down_exps.weight", "type": "42", "shape": [256, 8, 2]},
    ]
    reference = derive.tensor_payload_bytes(
        records, {record["name"]: derive.type_name(record["type"]) for record in records}
    )
    document = derive.build_recipe_document(
        records,
        ftype="IQ4_XS",
        has_imatrix=True,
        n_gqa=0,
        n_expert=2,
        block_count=1,
        mtp_blocks=0,
        reference_file_bytes=reference,
        repo="example/model",
        revision="0" * 40,
        architecture="nemotron_h_moe",
    )
    types = document["tensor_types"]
    assert types["output.weight"] == "Q6_K"
    assert types["blk.0.attn_norm.weight"] == "F32"
    assert types["blk.0.ffn_down_exps.weight"] == "IQ4_XS"
    derivation = document["derivation"]
    assert derivation["file_overhead_bytes"] == 0
    assert derivation["ud_file_bytes"] == derive.tensor_payload_bytes(records, types)
    assert derivation["main_text_elements"] == 256 * 8 + 256 * 8 + 256 + 256 * 8 * 2


def test_strict_recipe_loader_accepts_both_nemotron_recipes() -> None:
    from mfq.calibration.alphaq_workflow import recipe_types

    types = hf_to_mfq._load_gguf_recipe(RECIPE_DIR / "IQ2_M.json")
    assert len(types) == 781
    assert types["blk.1.ffn_down_exps.weight"] == "Q2_0"
    assert hf_to_mfq._dtype_for_recipe_type("Q2_0", "F32") == "NINT2"
    derived = hf_to_mfq._load_gguf_recipe(RECIPE_DIR / "IQ4_XS.json")
    assert derived["blk.1.ffn_down_exps.weight"] == "IQ4_NL"
    assert derived["blk.7.attn_v.weight"] == "IQ4_XS"
    # The AlphaQ path replaces routed-expert types before planning.
    allocated = recipe_types(RECIPE_DIR / "IQ2_M.json")
    assert allocated["blk.1.ffn_up_exps.weight"] == "Q8_0"
    assert allocated["blk.0.ssm_in.weight"] == "Q6_K"


def _synthetic_recipe() -> dict[str, str]:
    """Build a GGUF recipe for the synthetic checkpoint through the import map."""

    config = synthetic_config()
    text = config["llm_config"]
    types: dict[str, str] = {}
    for name, value in synthetic_tensors().items():
        mapped = map_source_tensor_name(name, config)
        if mapped is None or mapped.recipe_name is None:
            continue
        types[mapped.recipe_name] = "Q4_K" if value.ndim > 1 else "F32"
    # The GGUF conversion banks routed experts, so the recipe names the bank.
    for layer, mixer in enumerate(text["layers_block_type"]):
        if mixer != "moe":
            continue
        for projection in ("up", "down"):
            types[f"blk.{layer}.ffn_{projection}_exps.weight"] = "IQ3_S"
    return types


def test_plan_maps_nemotron_tensors_onto_the_recipe(tmp_path) -> None:
    root = write_checkpoint(tmp_path / "hf")
    recipe = _synthetic_recipe()
    plans = hf_to_mfq._plan(
        root,
        text_only=False,
        recipe_types=recipe,
        dense_dtype="F32",
        quantize_mtp=True,
    )
    by_name = {item.name: item for item in plans}
    assert by_name["model.block.2.attention.query.weight"].gguf_name == "blk.2.attn_q.weight"
    assert by_name["model.block.2.attention.query.weight"].target_dtype == "NINT4"
    assert by_name["model.block.0.linear_attention.in_projection.weight"].gguf_name == (
        "blk.0.ssm_in.weight"
    )
    assert by_name["model.block.1.mlp.shared_expert.down.weight"].gguf_name == (
        "blk.1.ffn_down_shexp.weight"
    )
    bank = by_name["model.block.1.mlp.experts.up.weight"]
    assert bank.gguf_name == "blk.1.ffn_up_exps.weight"
    assert bank.target_dtype == "MFE"
    assert {value.family for value in bank.expert_precisions or ()} == {"NVQ3J-L"}
    # The folded MTP head has no recipe alias; it keeps the family default
    # integer precision instead of being dropped or stored dense.
    predictor = by_name["predictor.block.0.attention.query.weight"]
    assert predictor.gguf_name is None
    assert predictor.target_dtype == "NINT4"


def test_plan_preserves_mtp_without_an_opt_in(tmp_path) -> None:
    root = write_checkpoint(tmp_path / "hf")
    plans = hf_to_mfq._plan(
        root,
        text_only=False,
        recipe_types=_synthetic_recipe(),
        dense_dtype="F32",
    )
    by_name = {item.name: item for item in plans}
    assert by_name["predictor.block.0.attention.query.weight"].target_dtype == "BF16"
    assert by_name["predictor.block.1.mlp.experts.up.weight"].target_dtype == "BF16"


def test_recipe_conversion_applies_the_nemotron_gguf_recipe(tmp_path) -> None:
    from mfq.formats.io import load_mmap

    root = write_checkpoint(tmp_path / "hf")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"format": "mfq.gguf-recipe.v1", "tensor_types": _synthetic_recipe()}),
        encoding="utf-8",
    )
    output = tmp_path / "recipe.mfq"
    hf_to_mfq.convert(hf_to_mfq.build_parser().parse_args([
        "--input", str(root), "--output", str(output), "--recipe-gguf", str(recipe),
    ]))
    header, store = load_mmap(output)
    try:
        assert header.model_arch == "nemotron_h_omni-hf-mfq-nint-recipe"
        assert store.records["model.block.2.attention.query.weight"].dtype == "NINT"
        assert store.records["model.block.1.mlp.experts.up.weight"].dtype == "MFE"
        # Recurrent and router parameters keep their exact source precision.
        assert store.records["model.block.0.linear_attention.conv.weight"].dtype == "F32"
        assert store.records["model.block.1.mlp.router.weight"].dtype == "F32"
        # The recipe pins the main text, so MTP and vision keep source precision.
        assert store.records["predictor.block.0.attention.query.weight"].dtype == "BF16"
        assert store.records["vision.projector.mlp.up.weight"].dtype == "BF16"
    finally:
        store.close()


def test_ud_tier_dry_run_plans_a_synthetic_nemotron(tmp_path, monkeypatch, capsys) -> None:
    root = write_checkpoint(tmp_path / "hf")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"format": "mfq.gguf-recipe.v1", "tensor_types": _synthetic_recipe()}),
        encoding="utf-8",
    )
    imatrix = tmp_path / "imatrix.npz"
    entries = {}
    for layer in (1, 3):
        for projection, rows, width in (("up", EXPERT_HIDDEN, LATENT), ("down", LATENT, EXPERT_HIDDEN)):
            entries[f"model.block.{layer}.mlp.experts.{projection}.weight"] = ImportanceEntry(
                np.ones((EXPERTS, width), dtype=np.float32),
                np.ones(EXPERTS, dtype=np.int64),
                row_importance=np.ones(EXPERTS * rows, dtype=np.float32),
            )
    save_importance_matrix(imatrix, entries)
    cap = 5_000_000
    plans = hf_to_mfq._plan(
        root, text_only=False, recipe_types=None, dense_dtype="F32",
        quantize_ple=True, quantize_mtp=True,
    )
    parameters = sum(
        math.prod(item.shape)
        for item in plans
        if not item.name.startswith(("vision.", "predictor."))
    )
    monkeypatch.setattr(
        "mfq.calibration.alphaq_workflow.automatic_recipe",
        lambda *_args, **_kwargs: recipe,
    )
    monkeypatch.setattr(
        "mfq.calibration.alphaq_workflow.registered_model",
        lambda _config: {"parameters": parameters, "caps": {"IQ2_M": cap}},
    )
    from mfq.cli import main

    assert main([
        "quantize", str(root), str(tmp_path / "dry.mfq"),
        "--imatrix", str(imatrix), "--ud-tier", "IQ2_M", "--dry-run",
        "--alphaq-cache", str(tmp_path / "cache"), "--alphaq-mtp", "quantize",
    ]) == 0
    assert not (tmp_path / "dry.mfq").exists()
    report = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    budget = next(item for item in report if item.get("stage") == "alphaq_file_budget")
    assert budget["maximum_file_bytes"] == cap
    assert budget["dense_mode"] == "recipe"
    assert budget["parameters"] == parameters
    assert budget["mtp_bytes"] > 0


def test_alphaq_routing_reads_top_k_from_the_nested_llm_config() -> None:
    from types import SimpleNamespace

    from mfq.calibration.alphaq_workflow import routing

    # NVIDIA Omni wrappers nest the text model under llm_config, not text_config.
    config = {"llm_config": {"num_experts_per_tok": 22}, "vision_config": {}}
    scope = SimpleNamespace(experts=[SimpleNamespace(name="model.block.1.mlp.experts.up.weight",
                                                     shape=(512, 2688, 1024))])
    multipliers, exposure = routing(scope, None, config, "none", "", Path("."))
    assert multipliers is None
    assert exposure == {"model.block.1.mlp.experts.up.weight": 22 / 512}
