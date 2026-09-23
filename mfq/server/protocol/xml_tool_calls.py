"""Incremental parsing for the XML tool-call dialects used by Qwen and GLM."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


class XMLToolCallParseError(ValueError):
    """Raised when generated XML tool-call output is incomplete or malformed."""


@dataclass(frozen=True)
class XMLToolCall:
    name: str
    arguments: str


class XMLToolCallStreamParser:
    """Extract Qwen- or GLM-style calls from arbitrarily fragmented text."""

    START = "<tool_call>"
    END = "</tool_call>"

    def __init__(
        self,
        tool_schemas: Mapping[str, Mapping[str, Any]],
        *,
        dialect: Literal["qwen", "glm"],
    ) -> None:
        self._tool_schemas = dict(tool_schemas)
        self._dialect = dialect
        self._buffer = ""
        self._in_call = False

    @staticmethod
    def _protected_prefix_width(value: str, marker: str) -> int:
        return max(
            (
                width
                for width in range(1, min(len(value), len(marker) - 1) + 1)
                if value.endswith(marker[:width])
            ),
            default=0,
        )

    @staticmethod
    def _skip_whitespace(value: str, position: int) -> int:
        while position < len(value) and value[position].isspace():
            position += 1
        return position

    def _schema_for(self, name: str) -> Mapping[str, Any]:
        schema = self._tool_schemas.get(name)
        if schema is None:
            raise XMLToolCallParseError(
                f"{self._dialect} XML invokes unavailable tool {name!r}"
            )
        return schema

    @staticmethod
    def _parameter_allows_string(schema: Mapping[str, Any], name: str) -> bool:
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return False
        definition = properties.get(name)
        if not isinstance(definition, Mapping):
            return False
        value_type = definition.get("type")
        return value_type == "string" or (
            isinstance(value_type, list) and "string" in value_type
        )

    @classmethod
    def _decode_value(
        cls,
        value: str,
        *,
        schema: Mapping[str, Any],
        name: str,
    ) -> Any:
        if cls._parameter_allows_string(schema, name):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            properties = schema.get("properties")
            declared = (
                properties.get(name)
                if isinstance(properties, Mapping)
                else None
            )
            if declared is None:
                return value
            raise XMLToolCallParseError(
                f"parameter {name!r} is not valid JSON"
            ) from error

    @staticmethod
    def _value_end(
        value: str,
        position: int,
        *,
        close: str,
        next_open: str,
    ) -> int:
        search = position
        while True:
            close_at = value.find(close, search)
            if close_at < 0:
                raise XMLToolCallParseError(f"missing {close!r}")
            after = XMLToolCallStreamParser._skip_whitespace(
                value,
                close_at + len(close),
            )
            if after == len(value) or value.startswith(next_open, after):
                return close_at
            search = close_at + len(close)

    def _build_call(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> XMLToolCall:
        if not name:
            raise XMLToolCallParseError("tool name is empty")
        self._schema_for(name)
        return XMLToolCall(
            name=name,
            arguments=json.dumps(
                dict(arguments),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    def _parse_qwen(self, body: str) -> XMLToolCall:
        body = body.strip()
        prefix = "<function="
        if not body.startswith(prefix):
            raise XMLToolCallParseError("expected Qwen <function=...> block")
        name_end = body.find(">", len(prefix))
        if name_end < 0:
            raise XMLToolCallParseError("unterminated Qwen function name")
        name = body[len(prefix) : name_end].strip()
        close_function = "</function>"
        function_end = body.rfind(close_function)
        if function_end < name_end:
            raise XMLToolCallParseError("missing Qwen </function> tag")
        if body[function_end + len(close_function) :].strip():
            raise XMLToolCallParseError("unexpected text after Qwen function block")

        schema = self._schema_for(name)
        parameters = body[name_end + 1 : function_end]
        arguments: dict[str, Any] = {}
        position = 0
        parameter_prefix = "<parameter="
        close_parameter = "</parameter>"
        while True:
            position = self._skip_whitespace(parameters, position)
            if position == len(parameters):
                break
            if not parameters.startswith(parameter_prefix, position):
                raise XMLToolCallParseError("expected Qwen <parameter=...> block")
            key_end = parameters.find(">", position + len(parameter_prefix))
            if key_end < 0:
                raise XMLToolCallParseError("unterminated Qwen parameter name")
            key = parameters[position + len(parameter_prefix) : key_end].strip()
            if not key:
                raise XMLToolCallParseError("Qwen parameter name is empty")
            if key in arguments:
                raise XMLToolCallParseError(f"duplicate parameter {key!r}")
            value_start = key_end + 1
            value_end = self._value_end(
                parameters,
                value_start,
                close=close_parameter,
                next_open=parameter_prefix,
            )
            raw_value = parameters[value_start:value_end].strip()
            arguments[key] = self._decode_value(
                raw_value,
                schema=schema,
                name=key,
            )
            position = value_end + len(close_parameter)
        return self._build_call(name, arguments)

    def _parse_glm(self, body: str) -> XMLToolCall:
        body = body.strip()
        key_open = "<arg_key>"
        key_close = "</arg_key>"
        value_open = "<arg_value>"
        value_close = "</arg_value>"
        first_key = body.find(key_open)
        name = (body if first_key < 0 else body[:first_key]).strip()
        schema = self._schema_for(name)
        arguments: dict[str, Any] = {}
        position = len(body) if first_key < 0 else first_key

        while position < len(body):
            position = self._skip_whitespace(body, position)
            if position == len(body):
                break
            if not body.startswith(key_open, position):
                raise XMLToolCallParseError("expected GLM <arg_key> block")
            key_end = body.find(key_close, position + len(key_open))
            if key_end < 0:
                raise XMLToolCallParseError("missing GLM </arg_key> tag")
            key = body[position + len(key_open) : key_end].strip()
            if not key:
                raise XMLToolCallParseError("GLM argument key is empty")
            if key in arguments:
                raise XMLToolCallParseError(f"duplicate argument {key!r}")
            position = self._skip_whitespace(body, key_end + len(key_close))
            if not body.startswith(value_open, position):
                raise XMLToolCallParseError("expected GLM <arg_value> block")
            value_start = position + len(value_open)
            value_end = self._value_end(
                body,
                value_start,
                close=value_close,
                next_open=key_open,
            )
            raw_value = body[value_start:value_end].strip()
            arguments[key] = self._decode_value(
                raw_value,
                schema=schema,
                name=key,
            )
            position = value_end + len(value_close)
        return self._build_call(name, arguments)

    def _parse_call(self, body: str) -> XMLToolCall:
        if self._dialect == "qwen":
            return self._parse_qwen(body)
        return self._parse_glm(body)

    def feed(self, text: str) -> tuple[str, tuple[XMLToolCall, ...]]:
        """Consume one text delta and return visible text plus complete calls."""

        self._buffer += text
        visible: list[str] = []
        calls: list[XMLToolCall] = []

        while self._buffer:
            if self._in_call:
                search = 0
                while True:
                    close_at = self._buffer.find(self.END, search)
                    if close_at < 0:
                        break
                    try:
                        call = self._parse_call(self._buffer[:close_at])
                    except XMLToolCallParseError:
                        search = close_at + len(self.END)
                        continue
                    self._buffer = self._buffer[close_at + len(self.END) :]
                    self._in_call = False
                    calls.append(call)
                    break
                if self._in_call:
                    break
                continue

            marker_at = self._buffer.find(self.START)
            if marker_at >= 0:
                visible.append(self._buffer[:marker_at])
                self._buffer = self._buffer[marker_at + len(self.START) :]
                self._in_call = True
                continue

            protected = self._protected_prefix_width(self._buffer, self.START)
            if protected:
                visible.append(self._buffer[:-protected])
                self._buffer = self._buffer[-protected:]
            else:
                visible.append(self._buffer)
                self._buffer = ""
            break

        return "".join(visible), tuple(calls)

    def finish(self) -> tuple[str, tuple[XMLToolCall, ...]]:
        """Flush visible text, rejecting truncated XML protocol output."""

        if self._in_call:
            close_at = self._buffer.rfind(self.END)
            if close_at >= 0:
                call = self._parse_call(self._buffer[:close_at])
                if self._buffer[close_at + len(self.END) :].strip():
                    raise XMLToolCallParseError(
                        "unexpected text after XML tool-call block"
                    )
                self._buffer = ""
                self._in_call = False
                return "", (call,)
            raise XMLToolCallParseError(
                f"incomplete {self._dialect} XML tool-call block"
            )

        trailing, self._buffer = self._buffer, ""
        if trailing and self.START.startswith(trailing):
            raise XMLToolCallParseError("incomplete XML tool-call marker")
        return trailing, ()


__all__ = [
    "XMLToolCall",
    "XMLToolCallParseError",
    "XMLToolCallStreamParser",
]
