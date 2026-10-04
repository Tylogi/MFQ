from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = (ROOT / "csrc/core/grid_vision.cpp").read_text()
QWEN_CONFIG = (
    ROOT / "csrc/models/qwen35/config.cpp"
).read_text()
COMMON_CONFIG = (
    ROOT / "csrc/models/common/model_config.cpp"
).read_text()
CUDA_ROOT = ROOT / "csrc/backends/cuda"
CUDA = (CUDA_ROOT / "models/common/grid_vision_component.h").read_text()
CUDA_APP = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
CUDA_PLAN_TEST = (ROOT / "csrc/backends/cuda/tests/cuda_model_plan_test.cpp").read_text()
METAL = (ROOT / "csrc/backends/metal/runtime/mlx_grid_vision.cpp").read_text()
METAL_MM = (ROOT / "csrc/backends/metal/runtime/mlx_multimodal.cpp").read_text()
PLAN = (ROOT / "csrc/backends/cuda/models/cuda_model_plan.h").read_text()
PREPARED = (ROOT / "csrc/backends/cuda/ops/cuda_execution.h").read_text()
CUDA_COMPONENTS = "\n".join(
    path.read_text()
    for path in (
        CUDA_ROOT / "storage" / "model_loader.h",
        CUDA_ROOT / "storage" / "model_loader.cpp",
    )
)


SHARED_ENGINE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "csrc" / "engine" / "include").glob("*.h")
)
SHARED_MODELS = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "csrc" / "models").rglob("*.h")
)

CUDA_APP += SHARED_ENGINE + SHARED_MODELS

def test_grid_vision_policies_are_owned_by_core() -> None:
    assert "grid_vision_canonical_name" in CORE
    assert "make_grid_vision_layout" in CORE
    assert "make_learned_position_interpolation" in CORE
    assert "build_grid_mrope_positions" in CORE
    assert "GridVisionTensorSchema" not in (ROOT / "csrc/core/grid_vision.h").read_text()
    assert "spatial_positions(" not in METAL
    assert "::mfq::build_grid_mrope_positions(" in METAL_MM


def test_cuda_grid_vision_uses_existing_primitives_and_canonical_names() -> None:
    assert "load_quant_linear(execution, model, weight_name)" in CUDA
    assert "mfq_tensor_backend::layer_norm(" in CUDA
    assert "attention_cuda(" in CUDA
    assert "Tensor gelu_tanh(" in CUDA
    assert "mfq_tensor_backend::tanh(" in CUDA
    assert "mfq::models::grid_vision::merge(" in CUDA
    assert "mfq::models::grid_vision::prepare(" in CUDA
    assert "mfq_tensor_backend::gelu(value, \"none\")" in CUDA
    assert "return mlp(std::move(grouped), up, gelu, down)" in SHARED_MODELS
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
    assert "prepared->embeddings.narrow(1, chunk.offset, chunk.count)" in CUDA_APP
    assert "prepared->positions.narrow(-1, chunk.offset, chunk.count)" in CUDA_APP
    assert "mfq_nullopt,nullptr,mfq_nullopt,true" in "".join(CUDA_APP.split())
    assert "ops.offset_positions(ops.cache_positions, model.decode_position_delta)" in CUDA_APP
    assert "rope.sections.numel() > 0" in CUDA_APP
    assert "context.positions=local_pos" in "".join(CUDA_APP.split())
    assert "context.cache_positions=local_cache_positions" in "".join(CUDA_APP.split())
    assert "cache_positions.has_value()" in CUDA_APP


def test_prepared_prompt_supports_mtp_and_safe_session_reuse() -> None:
    assert 'rope_parameters.value("mrope_interleaved", false)' in QWEN_CONFIG
    assert "interleaved_order" in CUDA_APP
    assert "prepared->embeddings.narrow(1, chunk.offset, chunk.count)" in CUDA_APP
    assert "co_yield PrefillProgress" in CUDA_APP
    assert "model.last_logits_prepared(*prepared)" not in CUDA_APP
    assert "mtp.step_positioned(" in CUDA_APP
    assert "ops.cache_positions(pos, model.cache_pos, tokens)" in CUDA_APP
    assert "arange(start, start + tokens, positions.options())" in CUDA_APP
    assert "!transformed_prompt && mtp != nullptr" not in CUDA_APP
    assert "transformed_prompt ? 0" not in CUDA_APP
    assert "state.input_key = input_key" in CUDA_APP
    assert "(!prepared || !prepared->transformed()) && !constraint" in CUDA_APP
    assert "components.grid_vision->prepare(" in CUDA_APP
    assert "components.mtp.get()" in CUDA_APP
    assert "return batching->generate(request_id, input, output)" in CUDA_APP
    assert "advance_preparation" in CUDA_APP


def test_batched_text_positions_do_not_select_grid_mrope_sections() -> None:
    assert "bool grid_mrope_positions = false" in CUDA_APP
    assert "grid_mrope_positions ? sections : empty_sections" in CUDA_APP
    assert "constboolgrid_mrope_positions=B==1" in "".join(CUDA_APP.split())
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
    assert (
        "result.video_input = plan.vision == CudaVisionAdapter::minicpmo45;"
        in PLAN
    )
    assert "accepts exactly one image and no video" in CUDA
    for mutation in (
        "wrong_backbone",
        "wrong_root",
        "wrong_implementation",
        "wrong_input",
        "wrong_position",
    ):
        assert mutation in CUDA_PLAN_TEST
