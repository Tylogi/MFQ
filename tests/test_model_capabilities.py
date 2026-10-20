"""检查 Python 能力注册与 C++ 架构声明；前端能力交互由 Vitest 覆盖。"""
from mfq.server.protocol.output_protocols import output_protocol_for_architecture
from mfq.server.runtime.capabilities import capabilities_for_architecture


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
