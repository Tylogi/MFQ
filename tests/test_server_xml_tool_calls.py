from __future__ import annotations

import json

import pytest

from mfq.server.protocol.xml_tool_calls import (
    XMLToolCallParseError,
    XMLToolCallStreamParser,
)

TOOLS = {
    "write": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "count": {"type": "integer"},
            "options": {"type": "object"},
        },
    },
    "lookup": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    },
}


def test_qwen_xml_parser_withholds_fragmented_markers_and_decodes_call() -> None:
    parser = XMLToolCallStreamParser(TOOLS, dialect="qwen")
    chunks = (
        "I'll write it.\n\n<tool_",
        "call>\n<function=write>\n<parameter=content>\nhel",
        "lo\n</parameter>\n<parameter=count>2</parameter>\n",
        "</function>\n</tool_call>",
    )

    visible: list[str] = []
    calls = []
    for chunk in chunks:
        content, parsed = parser.feed(chunk)
        visible.append(content)
        calls.extend(parsed)
        assert "tool_call" not in content
    trailing, parsed = parser.finish()
    visible.append(trailing)
    calls.extend(parsed)

    assert "".join(visible) == "I'll write it.\n\n"
    assert len(calls) == 1
    assert calls[0].name == "write"
    assert json.loads(calls[0].arguments) == {"content": "hello", "count": 2}


def test_qwen_xml_parser_preserves_string_types_and_parallel_calls() -> None:
    parser = XMLToolCallStreamParser(TOOLS, dialect="qwen")
    content, calls = parser.feed(
        "<tool_call><function=write>"
        "<parameter=content>42</parameter>"
        '<parameter=options>{"append":true}</parameter>'
        "</function></tool_call>"
        "<tool_call><function=lookup>"
        "<parameter=query>mfq</parameter>"
        "</function></tool_call>"
    )

    assert content == ""
    assert [call.name for call in calls] == ["write", "lookup"]
    assert json.loads(calls[0].arguments) == {
        "content": "42",
        "options": {"append": True},
    }
    assert json.loads(calls[1].arguments) == {"query": "mfq"}
    assert parser.finish() == ("", ())


def test_qwen_xml_parser_tolerates_close_marker_inside_string_value() -> None:
    parser = XMLToolCallStreamParser(TOOLS, dialect="qwen")
    _content, calls = parser.feed(
        "<tool_call><function=write>"
        "<parameter=content>literal </tool_call> marker</parameter>"
        "</function></tool_call>"
    )

    assert json.loads(calls[0].arguments) == {
        "content": "literal </tool_call> marker"
    }


def test_glm_xml_parser_decodes_fragmented_call() -> None:
    parser = XMLToolCallStreamParser(TOOLS, dialect="glm")
    visible, calls = parser.feed(
        "checking\n<tool_call>write<arg_key>content</arg_key>"
        "<arg_value>hello"
    )
    assert visible == "checking\n"
    assert calls == ()

    visible, calls = parser.feed(
        " world</arg_value><arg_key>count</arg_key>"
        "<arg_value>3</arg_value></tool_call>"
    )
    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "write"
    assert json.loads(calls[0].arguments) == {
        "content": "hello world",
        "count": 3,
    }
    assert parser.finish() == ("", ())


@pytest.mark.parametrize("dialect", ["qwen", "glm"])
def test_xml_parser_rejects_unknown_and_incomplete_calls(
    dialect: str,
) -> None:
    parser = XMLToolCallStreamParser(TOOLS, dialect=dialect)  # type: ignore[arg-type]
    generated = (
        "<tool_call><function=missing></function></tool_call>"
        if dialect == "qwen"
        else "<tool_call>missing</tool_call>"
    )
    parser.feed(generated)
    with pytest.raises(XMLToolCallParseError, match="unavailable tool"):
        parser.finish()

    parser = XMLToolCallStreamParser(TOOLS, dialect=dialect)  # type: ignore[arg-type]
    parser.feed("answer<tool_call>")
    with pytest.raises(XMLToolCallParseError, match="incomplete"):
        parser.finish()
