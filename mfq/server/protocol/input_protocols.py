"""Architecture-registered prompt encoders at the managed API boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mfq.server.protocol.models import ResponseFormat, ToolChoice, ToolDefinition
from mfq.server.protocol.processor_prompt_protocols import (
    processor_prompt_protocol_for_architecture,
)


def render_preformatted_prompt(
    architecture: str,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[ToolDefinition] = (),
    tool_choice: ToolChoice = "auto",
    response_format: ResponseFormat | None = None,
    enable_thinking: bool = True,
    reasoning_effort: str | None = None,
    parallel_tool_calls: bool = True,
) -> str | None:
    """Render processor-owned protocols; return ``None`` for Jinja models."""

    protocol = processor_prompt_protocol_for_architecture(architecture)
    if protocol is None:
        return None
    return protocol.renderer(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        parallel_tool_calls=parallel_tool_calls,
    )


__all__ = ["render_preformatted_prompt"]
