"""Released DeepSeek-V4.1 text/vision prompt protocol."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from mfq.server.protocol.models import (
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    NamedToolChoice,
    ResponseFormat,
    ToolChoice,
    ToolDefinition,
)

_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_SYSTEM = "<｜System｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_THINK_START = "<think>"
_THINK_END = "</think>"
_IMAGE = "<｜deepseek_image｜>"
_DSML = "｜DSML｜"
_REQUIRED_TOOL_INSTRUCTION = (
    "You MUST call at least one available tool before giving your response."
)
_NAMED_TOOL_INSTRUCTION = (
    'You MUST call the function "{name}" before giving your response.'
)
_SERIAL_TOOL_INSTRUCTION = "You MUST produce at most one tool call."


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    pieces: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            pieces.append(str(block))
            continue
        block_type = block.get("type")
        if block_type in {"text", "input_text"}:
            pieces.append(str(block.get("text", block.get("content", ""))))
        elif block_type in {"image", "image_url", "input_image"}:
            pieces.append(_IMAGE)
        else:
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
        name = tool_choice.function.name
        rendered = [
            tool
            for tool in rendered
            if tool.get("function", {}).get("name") == name
        ]
        if not rendered:
            raise ValueError(f"unknown DeepSeek-V4.1 tool {name!r}")
        instructions.append(_NAMED_TOOL_INSTRUCTION.format(name=name))
    elif tool_choice == "required":
        if not rendered:
            raise ValueError("tool_choice='required' requires at least one tool")
        instructions.append(_REQUIRED_TOOL_INSTRUCTION)
    elif tool_choice != "auto":
        raise ValueError(f"unsupported DeepSeek-V4.1 tool choice {tool_choice!r}")
    if rendered and not parallel_tool_calls:
        instructions.append(_SERIAL_TOOL_INSTRUCTION)
    return rendered, instructions


def _reasoning_budget(value: str | None) -> int:
    if value is None:
        return 75
    mapping = {
        "low": 50,
        "medium": 75,
        "high": 75,
        "xhigh": 100,
        "max": 100,
        "maximum": 100,
    }
    try:
        return mapping[value.lower()]
    except KeyError as error:
        raise ValueError(
            "DeepSeek-V4.1 reasoning_effort must be low, medium, high, or max"
        ) from error


def _render_tools(tools: Sequence[Mapping[str, Any]]) -> str:
    schemas = "\n".join(
        json.dumps(tool["function"], ensure_ascii=False) for tool in tools
    )
    return f"""## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _tool_result(content: Any) -> str:
    return f"<tool_result>{_text_content(content)}</tool_result>"


def _prepare_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for source in copy.deepcopy(list(messages)):
        role = source.get("role")
        if role == "developer":
            role = "system"
            source["role"] = role
        if role == "tool":
            content = _tool_result(source.get("content"))
            if prepared and prepared[-1].get("role") == "user":
                previous = _text_content(prepared[-1].get("content"))
                prepared[-1]["content"] = "\n\n".join(
                    piece for piece in (previous, content) if piece
                )
            else:
                prepared.append({"role": "user", "content": content})
            continue
        if "content" in source:
            source["content"] = _text_content(source.get("content"))
        if role == "user" and prepared and prepared[-1].get("role") == "user":
            previous = _text_content(prepared[-1].get("content"))
            prepared[-1]["content"] = "\n\n".join(
                piece for piece in (previous, source.get("content", "")) if piece
            )
            continue
        prepared.append(source)
    return prepared


def _render_arguments(arguments: Any) -> str:
    for _ in range(2):
        if not isinstance(arguments, str):
            break
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            break
    if not isinstance(arguments, Mapping):
        arguments = {"arguments": arguments}
    values: list[str] = []
    for name, value in arguments.items():
        is_string = isinstance(value, str)
        encoded = value if is_string else json.dumps(value, ensure_ascii=False)
        values.append(
            f'<｜DSML｜ parameter name="{name}" '
            f'string="{str(is_string).lower()}">{encoded}'
            f'</｜DSML｜ parameter>'
        )
    return "\n".join(values)


def _render_tool_calls(calls: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for call in calls:
        function = call.get("function")
        if not isinstance(function, Mapping) or not function.get("name"):
            raise ValueError("DeepSeek-V4.1 assistant tool call is malformed")
        rendered.append(
            f'<｜DSML｜ invoke name="{function["name"]}">\n'
            f'{_render_arguments(function.get("arguments", {}))}\n'
            f'</｜DSML｜ invoke>'
        )
    return "\n\n<｜DSML｜ calls>\n" + "\n".join(rendered) + "\n</｜DSML｜ calls>"


def render_deepseek_v41_prompt(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[ToolDefinition] = (),
    tool_choice: ToolChoice = "auto",
    response_format: ResponseFormat | None = None,
    enable_thinking: bool = True,
    reasoning_effort: str | None = None,
    parallel_tool_calls: bool = True,
) -> str:
    """Render OpenAI messages with DeepSeek's released V4.1 protocol."""

    normalized = copy.deepcopy(list(messages))
    prompt_tools, instructions = _selected_tools(
        tools,
        tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    schema = _response_schema(response_format)
    if prompt_tools or instructions or schema is not None:
        if not normalized or normalized[0].get("role") != "system":
            normalized.insert(0, {"role": "system", "content": ""})
        system = normalized[0]
        if instructions:
            content = _text_content(system.get("content"))
            suffix = "\n".join(instructions)
            system["content"] = f"{content}\n\n{suffix}" if content else suffix
        if prompt_tools:
            system["tools"] = prompt_tools
        if schema is not None:
            system["response_format"] = schema

    normalized = _prepare_messages(normalized)
    if not normalized:
        raise ValueError("DeepSeek-V4.1 prompt requires at least one message")
    last_user = max(
        (
            index
            for index, message in enumerate(normalized)
            if message.get("role") == "user"
            or (message.get("role") == "system" and index > 0)
        ),
        default=-1,
    )
    keep_thinking = any(message.get("tools") for message in normalized)
    prompt = _BOS
    budget = _reasoning_budget(reasoning_effort)

    for index, message in enumerate(normalized):
        role = message.get("role")
        if index == 0 and (enable_thinking or role == "system"):
            prompt += _SYSTEM
        if index == 0 and enable_thinking:
            prompt += (
                f"Reasoning Effort: {budget} (range 1-100, the higher the "
                "value, the more thorough the reasoning)\n\n"
            )

        content = _text_content(message.get("content"))
        if role == "system":
            if index > 0:
                prompt += _SYSTEM
            prompt += content
            if message.get("tools"):
                prompt += "\n\n" + _render_tools(message["tools"])
            if message.get("response_format") is not None:
                prompt += (
                    "\n\n## Response Format:\n\nYou MUST strictly adhere to the "
                    "following schema to reply:\n"
                    + json.dumps(message["response_format"], ensure_ascii=False)
                )
        elif role == "user":
            prompt += _USER + content
        elif role == "latest_reminder":
            prompt += "<｜latest_reminder｜>" + content
        elif role == "assistant":
            reasoning = message.get("reasoning_content")
            if (
                enable_thinking
                and isinstance(reasoning, str)
                and reasoning
                and (keep_thinking or index > last_user)
            ):
                prompt += reasoning + _THINK_END
            prompt += content
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                prompt += _render_tool_calls(calls)
            if message.get("wo_eos") is not True:
                prompt += _EOS
        else:
            raise ValueError(f"unsupported DeepSeek-V4.1 message role {role!r}")

        next_role = (
            normalized[index + 1].get("role")
            if index + 1 < len(normalized)
            else None
        )
        if next_role is not None and next_role not in {"assistant", "latest_reminder"}:
            continue
        if role == "user" or (role == "system" and index > 0):
            prompt += _ASSISTANT
            prompt += _THINK_START if enable_thinking else _THINK_END

    return prompt


__all__ = ["render_deepseek_v41_prompt"]
