"""Canonical DeepSeek-V4 prompt preparation for the managed native worker.

DeepSeek-V4 Flash checkpoints intentionally ship their message encoder beside
the weights instead of declaring a Hugging Face ``chat_template``.  Match the
processor-first design used by mlx-vlm: normalize OpenAI messages here, render
the released DeepSeek encoding once, and pass the resulting prompt to the
native runtime without asking a cached tokenizer template to reconstruct it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from mfq.server.deepseek_v4_encoding import encode_messages
from mfq.server.models import (
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    NamedToolChoice,
    ResponseFormat,
    ToolChoice,
    ToolDefinition,
)

_REQUIRED_TOOL_INSTRUCTION = (
    "You MUST call at least one available tool before giving your response."
)
_NAMED_TOOL_INSTRUCTION = (
    'You MUST call the function "{name}" before giving your response.'
)
_SERIAL_TOOL_INSTRUCTION = "You MUST produce at most one tool call."


def _text_content(content: Any) -> Any:
    """Normalize OpenAI text blocks after the vision processor has run."""

    if not isinstance(content, list):
        return content
    pieces: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            pieces.append(str(block))
            continue
        block_type = block.get("type")
        if block_type in {"text", "input_text"}:
            pieces.append(str(block.get("text", block.get("content", ""))))
            continue
        if block_type in {"image", "image_url", "input_image"}:
            # Actual media is replaced by DeepseekV4VisionProcessor before this
            # function.  Keeping this guard makes direct unit use deterministic.
            pieces.append("<｜deepseek_image｜>")
            continue
        pieces.append(str(block.get("content", "")))
    return "\n\n".join(piece for piece in pieces if piece)


def _response_schema(response_format: ResponseFormat | None) -> dict[str, Any] | None:
    if response_format is None or response_format.type == "text":
        return None
    if isinstance(response_format, JsonObjectResponseFormat):
        return {"type": "object"}
    if isinstance(response_format, JsonSchemaResponseFormat):
        return response_format.json_schema.schema_
    raise TypeError(f"unsupported response format: {type(response_format).__name__}")


def _selected_tools(
    tools: Sequence[ToolDefinition],
    tool_choice: ToolChoice,
    *,
    parallel_tool_calls: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    rendered = [tool.model_dump(mode="json", by_alias=True) for tool in tools]
    instructions: list[str] = []
    if tool_choice == "none":
        return [], instructions
    if isinstance(tool_choice, NamedToolChoice):
        selected_name = tool_choice.function.name
        rendered = [
            tool
            for tool in rendered
            if tool.get("function", {}).get("name") == selected_name
        ]
        if not rendered:
            raise ValueError(f"unknown DeepSeek-V4 tool {selected_name!r}")
        instructions.append(
            _NAMED_TOOL_INSTRUCTION.format(name=selected_name)
        )
    elif tool_choice == "required":
        if not rendered:
            raise ValueError("tool_choice='required' requires at least one tool")
        instructions.append(_REQUIRED_TOOL_INSTRUCTION)
    elif tool_choice != "auto":
        raise ValueError(f"unsupported DeepSeek-V4 tool choice {tool_choice!r}")
    if rendered and not parallel_tool_calls:
        instructions.append(_SERIAL_TOOL_INSTRUCTION)
    return rendered, instructions


def render_deepseek_v4_prompt(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[ToolDefinition] = (),
    tool_choice: ToolChoice = "auto",
    response_format: ResponseFormat | None = None,
    enable_thinking: bool = True,
    reasoning_effort: str | None = None,
    parallel_tool_calls: bool = True,
) -> str:
    """Render the released DeepSeek-V4 OpenAI-message encoding exactly once."""

    normalized = copy.deepcopy(list(messages))
    for message in normalized:
        if "content" in message:
            message["content"] = _text_content(message.get("content"))

    prompt_tools, instructions = _selected_tools(
        tools,
        tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    response_schema = _response_schema(response_format)
    if prompt_tools or instructions or response_schema is not None:
        if not normalized or normalized[0].get("role") != "system":
            normalized.insert(0, {"role": "system", "content": ""})
        system = normalized[0]
        if instructions:
            content = str(system.get("content") or "")
            suffix = "\n".join(instructions)
            system["content"] = f"{content}\n\n{suffix}" if content else suffix
        if prompt_tools:
            system["tools"] = prompt_tools
        if response_schema is not None:
            system["response_format"] = response_schema

    effort = reasoning_effort
    if effort in {"low", "medium"}:
        effort = None
    elif effort in {"xhigh", "maximum"}:
        effort = "max"
    if effort not in {None, "high", "max"}:
        raise ValueError(
            "DeepSeek-V4 reasoning_effort must be low, high, or max"
        )

    prompt = encode_messages(
        normalized,
        thinking_mode="thinking" if enable_thinking else "chat",
        reasoning_effort=effort,
    )
    if not prompt:
        raise ValueError("DeepSeek-V4 prompt encoder returned an empty prompt")
    return prompt


__all__ = ["render_deepseek_v4_prompt"]
