from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = (ROOT / "cpp_runtime/core/grid_vision.cpp").read_text()
QWEN_CONFIG = (
    ROOT / "cpp_runtime/core/models/qwen35.cpp"
).read_text()
COMMON_CONFIG = (
    ROOT / "cpp_runtime/core/models/model_config.cpp"
).read_text()
CUDA_ROOT = ROOT / "cpp_runtime/backends/cuda"
CUDA = (CUDA_ROOT / "runtime/grid_vision_runtime.h").read_text()
CUDA_APP = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
CUDA_PLAN_TEST = (ROOT / "cpp_runtime/backends/cuda/tests/cuda_model_plan_test.cpp").read_text()
METAL = (ROOT / "cpp_runtime/backends/metal/runtime/mlx_grid_vision.cpp").read_text()
METAL_MM = (ROOT / "cpp_runtime/backends/metal/runtime/mlx_multimodal.cpp").read_text()
PLAN = (ROOT / "cpp_runtime/backends/cuda/include/cuda_model_plan.h").read_text()
PREPARED = (ROOT / "cpp_runtime/backends/cuda/runtime/prepared_prompt.h").read_text()
CUDA_COMPONENTS = "\n".join(
    (CUDA_ROOT / "runtime" / name).read_text()
    for name in ("runtime_components.h", "runtime_components.cpp")
)


def test_grid_vision_policies_are_owned_by_core() -> None:
    assert "grid_vision_canonical_name" in CORE
    assert "make_grid_vision_layout" in CORE
    assert "make_learned_position_interpolation" in CORE
    assert "build_grid_mrope_positions" in CORE
    assert "GridVisionTensorSchema" not in (ROOT / "cpp_runtime/core/grid_vision.h").read_text()
    assert "spatial_positions(" not in METAL
    assert "::mfq::build_grid_mrope_positions(" in METAL_MM


def test_cuda_grid_vision_uses_existing_primitives_and_canonical_names() -> None:
    assert "load_quant_linear(model, weight_name)" in CUDA
    assert "mfq_tensor_backend::layer_norm(" in CUDA
    assert "attention_cuda(" in CUDA
    assert "Tensor gelu_tanh(" in CUDA
    assert "mfq_tensor_backend::tanh(" in CUDA
    assert 'merger_up_(merger_norm_(patches)' in CUDA
    assert '})), "none"))' in CUDA
    assert "grid_vision_canonical_name(" in CUDA
    assert "index_select(0, index.reshape({-1}))" in CUDA
    assert "model.visual" not in CUDA
    assert "qwen" not in CUDA.lower()
    assert "__global__" not in CUDA
    assert "class CudaGridVisionPromptComponent" in CUDA
    assert "input_contract_" in CUDA and "position_policy_" in CUDA
    assert "component->input_contract, component->position_policy" in CUDA_COMPONENTS


def test_prepared_prompt_separates_semantic_and_cache_positions() -> None:
    assert "struct CudaPreparedPrompt" in PREPARED
    assert "ids, prepared.embeddings, prepared.positions" in CUDA_APP
    assert "mfq_nullopt, nullptr, mfq_nullopt, true" in CUDA_APP
    assert "cache_positions + decode_position_delta" in CUDA_APP
    assert "rope.sections.numel() > 0" in CUDA_APP
    assert "context.positions=local_pos" in CUDA_APP
    assert "context.cache_positions=local_cache_positions" in CUDA_APP
    assert "cache_positions.has_value()" in CUDA_APP


def test_prepared_prompt_supports_mtp_and_safe_session_reuse() -> None:
    assert 'rope_parameters.value("mrope_interleaved", false)' in QWEN_CONFIG
    assert "interleaved_order" in CUDA_APP
    assert "hidden_forward_prepared_chunked" in CUDA_APP
    assert "prepared_offset + offset" in CUDA_APP
    assert "model.last_logits_prepared(*prepared)" not in CUDA_APP
    assert "mtp.step_positioned(" in CUDA_APP
    assert "cache_pos, cache_pos + tokens, pos.options()" in CUDA_APP
    assert "!transformed_prompt && mtp != nullptr" not in CUDA_APP
    assert "transformed_prompt ? 0" not in CUDA_APP
    assert "state.input_key = input_key" in CUDA_APP
    assert "!transformed_prompt && graph_enabled" in CUDA_APP
    assert "runtime_components.grid_vision->prepare(" in CUDA_APP
    assert "runtime_components.mtp.get()" in CUDA_APP
    assert "continuous_batcher->submit(" in CUDA_APP


def test_batched_text_positions_do_not_select_grid_mrope_sections() -> None:
    assert "bool grid_mrope_positions = false" in CUDA_APP
    assert "grid_mrope_positions ? sections : empty_sections" in CUDA_APP
    assert "const bool grid_mrope_positions = B == 1" in CUDA_APP
    assert "cache_positions.has_value() && pos.dim() == 2" in CUDA_APP
    assert "pos.size(0) == 3" in CUDA_APP
    assert "!grid_mrope_positions && positions.dim() == 2" in CUDA_APP
    assert "empty_sections = sections" in CUDA_APP


def test_multimodal_model_type_prefers_outer_root() -> None:
    assert "config.model_type = root.value(" in COMMON_CONFIG
    assert '"model_type", text.value("model_type"' in COMMON_CONFIG


def test_cuda_registration_is_exact_and_video_is_not_advertised() -> None:
    for value in (
        'graph.backbone == "qwen3_5"',
        'vision->implementation == "grid_vit"',
        "kMfqGridVisionInputContract",
        "kMfqGridMropePositionPolicy",
    ):
        assert value in PLAN
    assert "!runtime_components.grid_vision.has_value()" in CUDA_APP
    assert "accepts exactly one image and no video" in CUDA
    for mutation in (
        "wrong_backbone",
        "wrong_root",
        "wrong_implementation",
        "wrong_input",
        "wrong_position",
    ):
        assert mutation in CUDA_PLAN_TEST
