from __future__ import annotations

from mfq.server.protocol.deepseek_v4_prompt import render_deepseek_v4_prompt
from mfq.server.protocol.models import ToolDefinition


def _weather_tool() -> ToolDefinition:
    return ToolDefinition.model_validate(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    )


def test_tool_schema_uses_released_deepseek_v4_prompt_protocol() -> None:
    prompt = render_deepseek_v4_prompt(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Weather in Beijing?"},
        ],
        tools=[_weather_tool()],
        tool_choice="required",
        enable_thinking=True,
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>You are a helpful assistant.\n\n"
        "You MUST call at least one available tool before giving your response."
    )
    assert "### Available Tool Schemas\n\n" in prompt
    assert '"name": "get_weather"' in prompt
    assert "<｜User｜>Weather in Beijing?<｜Assistant｜><think>" in prompt


def test_tool_history_is_encoded_as_dsml_and_tool_result_user_turn() -> None:
    prompt = render_deepseek_v4_prompt(
        [
            {"role": "user", "content": "Weather in Beijing?"},
            {
                "role": "assistant",
                "reasoning_content": "I should query it.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Beijing"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temperature":22}',
            },
        ],
        tools=[_weather_tool()],
        enable_thinking=True,
    )

    assert "I should query it.</think>" in prompt
    assert '<｜DSML｜invoke name="get_weather">' in prompt
    assert (
        '<｜DSML｜parameter name="location" string="true">Beijing'
        '</｜DSML｜parameter>'
    ) in prompt
    assert '<｜User｜><tool_result>{"temperature":22}</tool_result>' in prompt
    assert prompt.endswith("<｜Assistant｜><think>")


def test_text_blocks_follow_mlx_vlm_deepseek_v4_spacing() -> None:
    prompt = render_deepseek_v4_prompt(
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
