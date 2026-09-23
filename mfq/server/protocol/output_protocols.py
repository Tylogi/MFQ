"""Registry for model-family output protocols at the OpenAI API boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from mfq.server.protocol.dsml import DSMLStreamParser
from mfq.server.protocol.reasoning import TaggedReasoningParser
from mfq.server.protocol.xml_tool_calls import XMLToolCallStreamParser
from mfq.server.runtime.capabilities import (
    capabilities_for_architecture,
    normalize_architecture,
)


class ParsedToolCall(Protocol):
    name: str
    arguments: str


class IncrementalReasoningParser(Protocol):
    def feed(self, text: str) -> tuple[str, str]: ...
    def finish(self) -> tuple[str, str]: ...


class IncrementalToolCallParser(Protocol):
    def feed(self, text: str) -> tuple[str, tuple[ParsedToolCall, ...]]: ...
    def finish(self) -> tuple[str, tuple[ParsedToolCall, ...]]: ...


ReasoningParserFactory = Callable[[], IncrementalReasoningParser]
ToolCallParserFactory = Callable[
    [Mapping[str, Mapping[str, Any]]],
    IncrementalToolCallParser,
]


@dataclass(frozen=True)
class ModelOutputProtocol:
    """All presentation-layer behavior needed by the generic backend."""

    reasoning_format: str = "auto"
    reasoning_parser_factory: ReasoningParserFactory | None = None
    tool_call_parser_factory: ToolCallParserFactory | None = None
    tool_call_protocol_name: str = "native"

    def create_reasoning_parser(self) -> IncrementalReasoningParser | None:
        return (
            self.reasoning_parser_factory()
            if self.reasoning_parser_factory is not None
            else None
        )

    def create_tool_call_parser(
        self,
        schemas: Mapping[str, Mapping[str, Any]],
    ) -> IncrementalToolCallParser | None:
        return (
            self.tool_call_parser_factory(schemas)
            if self.tool_call_parser_factory is not None
            else None
        )


@dataclass(frozen=True)
class _OutputProtocolRegistration:
    families: frozenset[str]
    aliases: frozenset[str]
    prefixes: tuple[str, ...]
    protocol: ModelOutputProtocol


_REASONING_PARSER = partial(TaggedReasoningParser, start_in_reasoning=True)

_DEEPSEEK_V4_PROTOCOL = ModelOutputProtocol(
    reasoning_format="none",
    reasoning_parser_factory=_REASONING_PARSER,
    tool_call_parser_factory=DSMLStreamParser,
    tool_call_protocol_name="dsml",
)

_DEEPSEEK_V41_PROTOCOL = ModelOutputProtocol(
    reasoning_format="none",
    reasoning_parser_factory=_REASONING_PARSER,
    tool_call_parser_factory=partial(DSMLStreamParser, spaced_tags=True),
    tool_call_protocol_name="dsml_v41",
)

_QWEN_PROTOCOL = ModelOutputProtocol(
    reasoning_parser_factory=_REASONING_PARSER,
    tool_call_parser_factory=partial(XMLToolCallStreamParser, dialect="qwen"),
    tool_call_protocol_name="qwen_xml",
)

_GLM_PROTOCOL = ModelOutputProtocol(
    reasoning_parser_factory=_REASONING_PARSER,
    tool_call_parser_factory=partial(XMLToolCallStreamParser, dialect="glm"),
    tool_call_protocol_name="glm_xml",
)

_REGISTRY = (
    _OutputProtocolRegistration(
        families=frozenset({"deepseek_v41"}),
        aliases=frozenset(
            {"deepseek_v41", "deepseek_v41_text", "deepseek_v41_vision"}
        ),
        prefixes=("deepseek_v41",),
        protocol=_DEEPSEEK_V41_PROTOCOL,
    ),
    _OutputProtocolRegistration(
        families=frozenset({"deepseek_v4"}),
        aliases=frozenset(
            {
                "deepseek_v4",
                "deepseek_v4_vision",
            }
        ),
        prefixes=("deepseek_v4",),
        protocol=_DEEPSEEK_V4_PROTOCOL,
    ),
    _OutputProtocolRegistration(
        families=frozenset({"qwen3_5", "qwen4_exp"}),
        aliases=frozenset(
            {
                "qwen3_5",
                "qwen3_5_text",
                "qwen35",
                "qwen3_6",
                "qwen3_6_text",
                "qwen36",
                "qwen3_8",
                "qwen3_8_text",
                "qwen38",
                "qwen4_exp",
                "qwen4_exp_text",
            }
        ),
        prefixes=(
            "qwen3_5",
            "qwen35",
            "qwen3_6",
            "qwen36",
            "qwen3_8",
            "qwen38",
            "qwen4_exp",
        ),
        protocol=_QWEN_PROTOCOL,
    ),
    _OutputProtocolRegistration(
        families=frozenset({"glm5_next"}),
        aliases=frozenset(
            {
                "glm5_next",
                "glm5_next_text",
                "glm5_3",
                "glm5_3_text",
                "glm53",
                "glm_5_3",
                "glm_5_3_text",
                "glm_53",
            }
        ),
        prefixes=("glm5_next", "glm5_3", "glm53", "glm_5_3", "glm_53"),
        protocol=_GLM_PROTOCOL,
    ),
)

_DEFAULT_PROTOCOL = ModelOutputProtocol()


def output_protocol_for_architecture(model_type: str) -> ModelOutputProtocol:
    """Resolve a protocol profile from a runtime type or public model name."""

    identity = normalize_architecture(model_type)
    family = capabilities_for_architecture(model_type).architecture_family
    for registration in _REGISTRY:
        if (
            family in registration.families
            or identity in registration.aliases
            or identity.startswith(registration.prefixes)
        ):
            return registration.protocol
    return _DEFAULT_PROTOCOL


__all__ = [
    "IncrementalReasoningParser",
    "IncrementalToolCallParser",
    "ModelOutputProtocol",
    "ParsedToolCall",
    "output_protocol_for_architecture",
]
