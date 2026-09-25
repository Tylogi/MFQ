"""检查 Python 能力注册与 C++ 架构声明；前端能力交互由 Vitest 覆盖。"""
from pathlib import Path

from mfq.server.protocol.output_protocols import output_protocol_for_architecture
from mfq.server.runtime.capabilities import capabilities_for_architecture

ROOT = Path(__file__).resolve().parents[1]
SERVER = (ROOT / "cpp_runtime" / "server" / "src" / "server.cpp").read_text(encoding="utf-8")
CUDA_PLAN = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "include" / "cuda_model_plan.h"
).read_text(encoding="utf-8")
CUDA_ROOT = ROOT / "cpp_runtime" / "backends" / "cuda"
CUDA_DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
CUDA_COMPONENTS = "\n".join(
    (CUDA_ROOT / "runtime" / name).read_text(encoding="utf-8")
    for name in ("server_components.h", "server_components.cpp")
)


def test_minicpmo_family_registers_every_supported_modality() -> None:
    profile = capabilities_for_architecture("minicpmo")
    assert profile.architecture_family == "minicpmo"
    assert profile.features.model_dump() == {
        "text": True,
        "image_input": True,
        "video_input": True,
        "audio_input": True,
        "audio_output": True,
        "full_duplex": True,
        "mtp": False,
    }


def test_text_architecture_families_do_not_advertise_media() -> None:
    for model_type, family in (
        ("deepseek_v4", "deepseek_v4"),
        ("glm_moe_dsa", "glm_dsa"),
        ("gemma4_text", "gemma4"),
        ("qwen3_5_text", "qwen3_5"),
    ):
        profile = capabilities_for_architecture(model_type)
        assert profile.architecture_family == family
        assert profile.features.text is True
        assert not any(
            (
                profile.features.image_input,
                profile.features.video_input,
                profile.features.audio_input,
                profile.features.audio_output,
                profile.features.full_duplex,
            )
        )


def test_deepseek_v4_vision_alias_advertises_only_image_input() -> None:
    profile = capabilities_for_architecture("deepseek_v4_vision")
    assert profile.architecture_family == "deepseek_v4"
    assert profile.features.model_dump() == {
        "text": True,
        "image_input": True,
        "video_input": False,
        "audio_input": False,
        "audio_output": False,
        "full_duplex": False,
        "mtp": True,
    }


def test_deepseek_v41_is_not_folded_into_the_v4_family() -> None:
    profile = capabilities_for_architecture("deepseek_v41_vision")
    assert profile.architecture_family == "deepseek_v41"
    assert profile.features.model_dump() == {
        "text": True,
        "image_input": True,
        "video_input": False,
        "audio_input": False,
        "audio_output": False,
        "full_duplex": False,
        "mtp": True,
    }


def test_every_mtp_runtime_family_registers_architecture_support() -> None:
    for model_type in (
        "deepseek_v41",
        "deepseek_v41_vision",
        "deepseek_v4",
        "deepseek_v4_vision",
        "qwen3_5",
        "qwen3_5_text",
        "qwen4_exp",
        "glm5_next",
    ):
        assert capabilities_for_architecture(model_type).features.mtp


def test_unknown_architecture_keeps_text_and_a_stable_family_key() -> None:
    profile = capabilities_for_architecture("Future Model/2")
    assert profile.architecture_family == "future_model_2"
    assert profile.features.text is True
    assert profile.source == "architecture-registry:future_model_2"


def test_generated_output_protocol_is_resolved_by_the_registry() -> None:
    for model_type in (
        "deepseek_v4",
        "DeepSeek-V4-Flash",
    ):
        protocol = output_protocol_for_architecture(model_type)
        assert protocol.reasoning_format == "none"
        assert protocol.create_reasoning_parser() is not None
        assert protocol.create_tool_call_parser({}) is not None
        assert protocol.tool_call_protocol_name == "dsml"

    protocol = output_protocol_for_architecture("deepseek_v41_vision")
    assert protocol.reasoning_format == "none"
    assert protocol.create_reasoning_parser() is not None
    assert protocol.create_tool_call_parser({}) is not None
    assert protocol.tool_call_protocol_name == "dsml_v41"

    for model_type in (
        "qwen3_5",
        "Qwen3.6-27B",
        "Qwen3.8-27B",
        "qwen4_exp",
    ):
        protocol = output_protocol_for_architecture(model_type)
        assert protocol.reasoning_format == "auto"
        assert protocol.create_reasoning_parser() is not None
        assert protocol.create_tool_call_parser({}) is not None
        assert protocol.tool_call_protocol_name == "qwen_xml"

    for model_type in ("glm5_next", "GLM-5.3-Flash"):
        protocol = output_protocol_for_architecture(model_type)
        assert protocol.reasoning_format == "auto"
        assert protocol.create_reasoning_parser() is not None
        assert protocol.create_tool_call_parser({}) is not None
        assert protocol.tool_call_protocol_name == "glm_xml"


def test_cpp_server_publishes_the_same_architecture_capability_contract() -> None:
    assert "architecture_capability_profile(" in SERVER
    assert "kModelCapabilityRegistry" in SERVER
    for model_type in (
        "minicpmo",
        "deepseek_v41",
        "deepseek_v41_vision",
        "deepseek_v4",
        "deepseek_v4_vision",
        "glm_moe_dsa",
        "qwen3_5",
        "qwen4_exp",
        "glm5_next",
    ):
        assert f'"{model_type}"' in SERVER
    for feature in (
        "text",
        "image_input",
        "video_input",
        "audio_input",
        "audio_output",
        "full_duplex",
        "mtp",
    ):
        assert f'{{"{feature}", profile.{feature}}}' in SERVER
    assert '{"model_capabilities", model_capabilities}' in SERVER
    for state in (
        "vision_supported",
        "vision_available",
        "video_available",
        "mtp_supported",
        "mtp_available",
        "vision_enabled_default",
        "mtp_enabled_default",
    ):
        assert f'"{state}"' in SERVER


def test_cpp_server_keeps_health_metrics_out_of_response_performance() -> None:
    assert SERVER.count("add_request_runtime_metrics(performance)") == 2
    assert "add_runtime_metrics(performance)" not in SERVER
    for metric in (
        "mtp_available",
        "mtp_used",
        "mtp_cycles",
        "mtp_drafted_tokens",
        "mtp_accepted_tokens",
        "mtp_acceptance_rate",
        "mtp_target_ms",
        "mtp_head_ms",
        "mtp_rollback_ms",
    ):
        assert f'"{metric}"' in SERVER


def test_cuda_uses_one_architecture_and_optional_component_registry() -> None:
    assert "CudaModelPlan" in CUDA_PLAN
    assert "CudaComponentState" in CUDA_PLAN
    assert "cuda_model_plan(" in CUDA_PLAN
    assert "cuda_component_state(" in CUDA_PLAN
    for implementation in (
        "deepseek_v4",
        "qwen3_5",
        "gemma4",
        "glm_dsa",
        "minicpmo45",
    ):
        assert f'"{implementation}"' in CUDA_PLAN

    assert "load_runtime_components(" in CUDA_DECODE
    assert "auto server_components" in CUDA_DECODE
    assert "switch (result.plan.vision)" in CUDA_COMPONENTS
    assert "server_minicpmo_runtime" not in CUDA_DECODE
