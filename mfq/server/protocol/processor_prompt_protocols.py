"""Registry for model families whose prompt protocol is processor-owned."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mfq.server.protocol.deepseek_v4_prompt import render_deepseek_v4_prompt
from mfq.server.protocol.deepseek_v41_prompt import render_deepseek_v41_prompt
from mfq.server.runtime.capabilities import capabilities_for_architecture, normalize_architecture

DEEPSEEK_V4_CHAT_TEMPLATE = r"""{{- bos_token -}}{%- for message in messages -%}{%- if message['role'] == 'system' and loop.first -%}{{- message['content'] -}}{%- elif message['role'] == 'user' -%}{{- '<｜User｜>' + message['content'] -}}{%- elif message['role'] == 'tool' -%}{{- '<｜User｜><tool_result>' + message['content'] + '</tool_result>' -}}{%- elif message['role'] == 'assistant' -%}{{- '<｜Assistant｜>' -}}{%- if enable_thinking and message['reasoning_content'] is defined and message['reasoning_content'] -%}{{- message['reasoning_content'] + '</think>' -}}{%- endif -%}{{- message['content'] + eos_token -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<｜Assistant｜>' -}}{%- if enable_thinking -%}{{- '<think>' -}}{%- else -%}{{- '</think>' -}}{%- endif -%}{%- endif -%}"""


DEEPSEEK_V41_CHAT_TEMPLATE = r"""{{- bos_token -}}{%- set thinking = enable_thinking | default(true) -%}{%- set effort = reasoning_effort | default('medium') -%}{%- if effort == 'low' -%}{%- set budget = '50' -%}{%- elif effort in ['xhigh', 'max', 'maximum'] -%}{%- set budget = '100' -%}{%- else -%}{%- set budget = '75' -%}{%- endif -%}{%- if thinking -%}{{- '<｜System｜>Reasoning Effort: ' + budget + ' (range 1-100, the higher the value, the more thorough the reasoning)\n\n' -}}{%- endif -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{%- if not loop.first or not thinking -%}{{- '<｜System｜>' -}}{%- endif -%}{{- message['content'] or '' -}}{%- elif message['role'] == 'user' -%}{{- '<｜User｜>' + (message['content'] or '') -}}{%- elif message['role'] == 'tool' -%}{{- '<｜User｜><tool_result>' + (message['content'] or '') + '</tool_result>' -}}{%- elif message['role'] == 'assistant' -%}{{- '<｜Assistant｜>' -}}{%- if thinking and message['reasoning_content'] is defined and message['reasoning_content'] -%}{{- message['reasoning_content'] + '</think>' -}}{%- endif -%}{{- (message['content'] or '') + eos_token -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<｜Assistant｜>' -}}{%- if thinking -%}{{- '<think>' -}}{%- else -%}{{- '</think>' -}}{%- endif -%}{%- endif -%}"""

PromptRenderer = Callable[..., str]


@dataclass(frozen=True)
class ProcessorPromptProtocol:
    """Prompt ownership plus the native-tokenizer bootstrap contract."""

    name: str
    families: frozenset[str]
    aliases: frozenset[str]
    prefixes: tuple[str, ...]
    bootstrap_chat_template: str
    renderer: PromptRenderer


_REGISTRY = (
    ProcessorPromptProtocol(
        name="deepseek_v41",
        families=frozenset({"deepseek_v41"}),
        aliases=frozenset(
            {
                "deepseek_v41",
                "deepseek_v41_text",
                "deepseek_v41_vision",
                "deepseekv41",
            }
        ),
        prefixes=("deepseek_v41", "deepseekv41"),
        bootstrap_chat_template=DEEPSEEK_V41_CHAT_TEMPLATE,
        renderer=render_deepseek_v41_prompt,
    ),
    ProcessorPromptProtocol(
        name="deepseek_v4",
        families=frozenset({"deepseek_v4"}),
        aliases=frozenset(
            {
                "deepseek_v4",
                "deepseek_v4_text",
                "deepseek_v4_vision",
                "deepseekv4",
            }
        ),
        prefixes=("deepseek_v4", "deepseekv4"),
        bootstrap_chat_template=DEEPSEEK_V4_CHAT_TEMPLATE,
        renderer=render_deepseek_v4_prompt,
    ),
)


def processor_prompt_protocol_for_architecture(
    architecture: str,
) -> ProcessorPromptProtocol | None:
    """Resolve processor prompt ownership from canonical or source identities."""

    identity = normalize_architecture(architecture)
    family = capabilities_for_architecture(architecture).architecture_family
    for registration in _REGISTRY:
        if (
            family in registration.families
            or identity in registration.aliases
            or identity.startswith(registration.prefixes)
        ):
            return registration
    return None


def processor_prompt_protocols() -> tuple[ProcessorPromptProtocol, ...]:
    """Return registered protocols for cache identity and contract tests."""

    return _REGISTRY


__all__ = [
    "DEEPSEEK_V41_CHAT_TEMPLATE",
    "DEEPSEEK_V4_CHAT_TEMPLATE",
    "ProcessorPromptProtocol",
    "processor_prompt_protocol_for_architecture",
    "processor_prompt_protocols",
]
