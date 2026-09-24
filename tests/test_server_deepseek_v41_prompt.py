from __future__ import annotations

from mfq.server.protocol.deepseek_v41_prompt import render_deepseek_v41_prompt
from mfq.server.protocol.input_protocols import render_preformatted_prompt
from mfq.server.protocol.models import ToolDefinition
from mfq.server.protocol.processor_prompt_protocols import (
    DEEPSEEK_V4_CHAT_TEMPLATE,
    DEEPSEEK_V41_CHAT_TEMPLATE,
    processor_prompt_protocol_for_architecture,
)


def _weather_tool() -> ToolDefinition:
    return ToolDefinition.model_validate(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    )


def test_basic_prompt_matches_released_v41_encoding() -> None:
    prompt = render_deepseek_v41_prompt(
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ],
        enable_thinking=True,
        reasoning_effort="high",
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 "
        "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        "Be helpful.<｜User｜>Hello<｜Assistant｜><think>"
    )


def test_vision_blocks_use_the_released_image_placeholder() -> None:
    prompt = render_deepseek_v41_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "text", "text": "last"},
                ],
            }
        ],
        enable_thinking=False,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>first\n\n"
        "<｜deepseek_image｜>\n\nlast<｜Assistant｜></think>"
    )


def test_tools_and_history_use_v41_spaced_dsml_tags() -> None:
    prompt = render_deepseek_v41_prompt(
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "reasoning_content": "I should check.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        tools=[_weather_tool()],
        enable_thinking=True,
    )

    assert "### Available Tool Schemas" in prompt
    assert "I should check.</think>" in prompt
    assert '<｜DSML｜ invoke name="get_weather">' in prompt
    assert (
        '<｜DSML｜ parameter name="city" string="true">Paris'
        '</｜DSML｜ parameter>'
    ) in prompt
    assert "<｜User｜><tool_result>sunny</tool_result>" in prompt
    assert prompt.endswith("<｜Assistant｜><think>")


def test_input_protocol_registry_separates_v4_and_v41() -> None:
    v41 = render_preformatted_prompt(
        "deepseek_v41_vision",
        [{"role": "user", "content": "hello"}],
        enable_thinking=True,
    )
    v4 = render_preformatted_prompt(
        "DeepSeek-V4-Flash",
        [{"role": "user", "content": "hello"}],
        enable_thinking=True,
    )

    assert v41 is not None and "<｜System｜>Reasoning Effort: 75" in v41
    assert v4 is not None and "<｜System｜>Reasoning Effort: 75" not in v4
    assert render_preformatted_prompt(
        "qwen3_5", [{"role": "user", "content": "hello"}]
    ) is None


def test_processor_prompt_registry_covers_source_and_container_identities() -> None:
    for identity in (
        "deepseek_v4",
        "deepseek_v4_text",
        "deepseek_v4_vision",
        "DeepseekV4ForCausalLM",
    ):
        protocol = processor_prompt_protocol_for_architecture(identity)
        assert protocol is not None
        assert protocol.name == "deepseek_v4"
        assert protocol.bootstrap_chat_template == DEEPSEEK_V4_CHAT_TEMPLATE

    for identity in (
        "deepseek_v41",
        "deepseek_v41_text",
        "deepseek_v41_vision",
        "DeepseekV41ForConditionalGeneration",
        "deepseek-v41-mfq",
    ):
        protocol = processor_prompt_protocol_for_architecture(identity)
        assert protocol is not None
        assert protocol.name == "deepseek_v41"
        assert protocol.bootstrap_chat_template == DEEPSEEK_V41_CHAT_TEMPLATE

    assert processor_prompt_protocol_for_architecture("qwen3_5") is None
