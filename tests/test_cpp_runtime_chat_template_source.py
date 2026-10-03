"""校验原生对话模板、推理能力发布及上下文重载契约。"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_SRC = ROOT / "csrc" / "transport"
SERVER = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(TRANSPORT_SRC.rglob("*"))
    if path.suffix in {".cpp", ".h"}
)
ENGINE_SRC = ROOT / "csrc" / "engine"
ENGINE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(ENGINE_SRC.rglob("*"))
    if path.suffix in {".cpp", ".h"}
)
SCHEDULER = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted((ROOT / "csrc" / "scheduler").rglob("*"))
    if path.suffix in {".cpp", ".h"}
)
CUDA_RUNTIME = ROOT / "csrc" / "backends" / "cuda" / "engine"
DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (
        CUDA_RUNTIME / "generation.cpp",
        CUDA_RUNTIME.parent / "ops" / "include" / "cuda_sampling.h",
        CUDA_RUNTIME / "mtp.cpp",
        ROOT / "csrc" / "backends" / "cuda" / "commands" / "runtime.cpp",
    )
)
CMAKE = (ROOT / "csrc" / "CMakeLists.txt").read_text(
    encoding="utf-8"
)
TOKENIZER_CMAKE = (
    ROOT / "csrc" / "components" / "tokenizer" / "CMakeLists.txt"
).read_text(encoding="utf-8")
SERVER_CMAKE = (
    ROOT / "csrc" / "transport" / "CMakeLists.txt"
).read_text(encoding="utf-8")
ENGINE_CMAKE = (
    ROOT / "csrc" / "engine" / "CMakeLists.txt"
).read_text(encoding="utf-8")
CUDA_ENGINE = (CUDA_RUNTIME / "cuda_engine.cpp").read_text(encoding="utf-8")
METAL_CMAKE = (
    ROOT / "csrc" / "backends" / "metal" / "CMakeLists.txt"
).read_text(encoding="utf-8")
METAL_DECODE = (
    ROOT / "csrc" / "backends" / "metal" / "apps" / "mfq_decode_mlx.cpp"
).read_text(encoding="utf-8")
METAL_DSV4 = (
    ROOT / "csrc" / "backends" / "metal" / "models/deepseek_v4" / "mlx_deepseek_v4_causal_lm.cpp"
).read_text(encoding="utf-8")
TEXT_CHAT = (
    ROOT / "csrc" / "components" / "tokenizer" / "chat" / "chat.cpp"
).read_text(encoding="utf-8")
TEXT_CHAT_H = (
    ROOT / "csrc" / "components" / "tokenizer" / "chat" / "chat.h"
).read_text(encoding="utf-8")
def _section(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


SHARED_ENGINE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "csrc" / "engine" / "include").glob("*.h")
)
SHARED_MODELS = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "csrc" / "models").rglob("*.h")
)

DECODE += SHARED_ENGINE + SHARED_MODELS

def test_cpp_runtime_dependencies_are_integrated() -> None:
    assert not (ROOT / "third_party").exists()
    assert not (ROOT / "csrc" / "llama").exists()
    active_roots = [
        ROOT / "csrc" / "components" / "tokenizer",
        ROOT / "mfq" / "kernels" / "cuda",
    ]
    assert not [
        path
        for active_root in active_roots
        for path in active_root.rglob("*llama*")
    ]
    active_source = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for active_root in active_roots
        for path in active_root.rglob("*")
        if path.suffix in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    )
    assert re.search(r"\b(?:llama_|LLAMA_|MFQ_LLAMA)", active_source) is None
    assert (ROOT / "csrc" / "components" / "tokenizer" / "include" / "mfq_text.h").is_file()
    assert not (ROOT / "csrc" / "components" / "tokenizer" / "include" / "llama.h").exists()
    assert (ROOT / "csrc" / "components" / "tokenizer" / "CMakeLists.txt").is_file()
    assert (ROOT / "csrc" / "components" / "http" / "httplib.cpp").is_file()
    assert (ROOT / "csrc" / "components" / "json" / "nlohmann" / "json.hpp").is_file()
    assert (ROOT / "NOTICE").is_file()
    assert "third_party" not in CMAKE


def test_engine_owns_native_gguf_jinja_template_and_common_parser() -> None:
    assert "common_chat_templates_apply" in ENGINE
    assert "common_chat_templates_apply" not in SERVER
    assert "common_chat_msgs_parse_oaicompat" in SERVER
    assert "common_chat_parse" in ENGINE
    assert "common_chat_msg_diff::compute_diffs" in ENGINE
    assert "common_chat_parse" not in SERVER
    assert "MfqTokenizer" not in SERVER
    assert ".tokenize(" not in SERVER
    assert "TextProcessor::load(" in CUDA_ENGINE
    assert "std::make_unique<TextProcessor>" in ENGINE
    assert "format_gemma4_chat_prompt" not in SERVER
    assert "format_dsv4_chat_prompt" not in SERVER


def test_processor_owned_prompts_bypass_cached_jinja_templates() -> None:
    prepare = _section(
        ENGINE,
        "InferenceRequest TextProcessor::prepare",
        "const MfqTokenizer& TextProcessor::tokenizer",
    )

    assert "if (chat.preformatted_prompt)" in prepare
    assert "prompt = *chat.preformatted_prompt;" in prepare
    assert "work.chat_parser.parse_tool_calls = false;" in prepare
    assert "json_schema_to_grammar(" in prepare
    assert "make_chat_token_constraint(" in prepare
    assert prepare.index("if (chat.preformatted_prompt)") < prepare.index(
        "common_chat_templates_apply("
    )


def test_server_enforces_complete_chat_template_tool_calls() -> None:
    assert "class GrammarConstraint" in ENGINE
    assert "class GrammarConstraint" not in SERVER
    assert "make_chat_token_constraint(" in ENGINE
    assert "make_chat_token_constraint(" not in SERVER
    assert "work.token_constraint" in ENGINE
    assert "if (partial)" in ENGINE
    assert "parsed.tool_calls.clear()" in ENGINE
    assert "parsed.tool_calls.clear()" not in SERVER
    assert 'uses_tool_calls ? "tool_calls" : "function_calls"' in TEXT_CHAT
    assert 'src.find("tool_calls") != std::string::npos' in TEXT_CHAT
    assert "token_constraint->apply" in METAL_DSV4
    assert "token_constraint->accept" in METAL_DSV4
    assert "token_constraint," in METAL_DECODE
    assert "CUDA constrained sampler returned an invalid token" in DECODE
    assert "masked.to(logits.device())" in DECODE
    assert "constraint->clone()" in DECODE
    assert "restored.tokens, restored.mtp_last_target_hidden" in DECODE
    assert "if(constraint_cursor)constraint_cursor->accept(pending);" in "".join(DECODE.split())
    assert "constraint_cursor->accept(result.next_token);" in DECODE


def test_native_server_cancels_active_session_between_steps() -> None:
    assert 'R"(/runtime/sessions/([A-Za-z0-9._:-]{1,128})/cancel)"' in SERVER
    assert "scheduler.cancel_session(session_id)" in SERVER
    assert "engine_.cancel(request.input.id)" in SCHEDULER
    assert 'result.finish_reason = "cancelled"' in ENGINE
    assert "else if (!result.tool_calls.empty())" in ENGINE
    assert "cancel_requested" not in SERVER
    assert "work.cache_plan.stable_prefix_tokens = 0;" not in SERVER
    assert "engine_.step(eligible)" in SCHEDULER
    assert "request.cache_plan = {};" not in SCHEDULER


def test_server_links_integrated_text_runtime() -> None:
    assert "add_subdirectory(components/tokenizer)" in CMAKE
    assert "add_library(mfq-tokenizer STATIC" in TOKENIZER_CMAKE
    assert "add_library(mfq-text-runtime ALIAS mfq-tokenizer)" in TOKENIZER_CMAKE
    assert "mfq-tokenizer" in ENGINE_CMAKE
    assert "mfq-tokenizer" not in SERVER_CMAKE
    assert "BUILD_WITH_INSTALL_RPATH ON" in METAL_CMAKE


def test_cpp_runtime_transport_has_no_public_openai_routes() -> None:
    assert '"/v1/' not in SERVER
    assert '"/api/' not in SERVER
    assert 'server.Post("/runtime/generate"' in SERVER


def test_cuda_runtime_accepts_an_external_tokenizer_only() -> None:
    assert "model runtime does not accept an external model config" in DECODE
    assert "model runtime requires model config and tokenizer GGUF" in DECODE
    assert "options.tokenizer_model" in CUDA_ENGINE
    assert "transport_config.tokenizer_model" not in DECODE


def test_server_publishes_template_gated_reasoning_effort() -> None:
    assert "chat_template_capabilities_json" in SERVER
    assert '{"chat_template_capabilities", chat_template_capabilities}' in SERVER
    assert 'source.find("enable_thinking")' in ENGINE
    assert 'chat_template.find("enable_thinking")' not in SERVER


def test_native_server_does_not_bundle_or_mount_a_webui() -> None:
    assert "mfq-web-assets" not in CMAKE
    assert "Copying MFQ WebUI assets" not in METAL_CMAKE
    assert "--web-root" not in DECODE
    assert "--web-root" not in METAL_DECODE
    assert 'server.Get("/admin"' not in SERVER
    assert "set_mount_point" not in SERVER


def test_dsv4_server_uses_exact_stable_prefix_kv_reuse() -> None:
    assert "MfqPromptCachePlan" in ENGINE
    assert 'normalized_identity(impl_->model_type).rfind(' in ENGINE
    assert 'work.sampling.enable_thinking\n            ? "<think>" : "</think>"' in ENGINE
    assert "stable_prefix_tokens" in ENGINE
    assert '{"prefill_tokens", values.prefill_tokens}' in SERVER


def test_server_validates_context_on_model_reload() -> None:
    assert 'server.Post("/runtime/reload"' in SERVER
    assert "context_size must be within the model context capacity" in SERVER


def test_server_supports_structured_output_and_named_tool_choice() -> None:
    assert "static std::string request_json_schema(const json & body)" in SERVER
    assert 'body.contains("response_format")' in SERVER
    assert 'type == "json_object"' in SERVER
    assert 'type == "json_schema"' in SERVER
    assert "inputs.json_schema = request_json_schema(body);" in SERVER
    assert "tool_choice.is_object()" in SERVER
    assert "named tool_choice must select a function name" in SERVER
    assert "named tool_choice does not match any supplied tool" in SERVER
    assert "inputs.tools = {*selected};" in SERVER
    assert "inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;" in SERVER


def test_dsv4_template_preserves_message_extensions() -> None:
    assert "std::map<std::string, std::string>        extra_fields;" in TEXT_CHAT_H
    assert 'for (const char * key : {"task", "tools", "response_format"})' in TEXT_CHAT
    assert "msg.extra_fields[key] = message.at(key).dump();" in TEXT_CHAT
    assert "for (const auto & [key, value] : extra_fields)" in TEXT_CHAT
    assert "jmsg[key] = json::parse(value);" in TEXT_CHAT
    assert "has_message_scoped_tools" in TEXT_CHAT


def test_dsv4_template_uses_message_scoped_tools_and_schema() -> None:
    dsv4 = _section(
        TEXT_CHAT,
        "static common_chat_params common_chat_params_init_deepseek_v3_2",
        "static common_chat_params common_chat_params_init_cohere2moe",
    )
    gpt_oss = _section(
        TEXT_CHAT,
        "static common_chat_params common_chat_params_init_gpt_oss",
        "static common_chat_params common_chat_params_init_gemma4",
    )

    assert "json available_tools = inputs.tools.is_array()" in dsv4
    assert "std::set<std::string> available_tool_names;" in dsv4
    assert 'message.contains("tools")' in dsv4
    assert "available_tools.push_back(tool);" in dsv4
    assert 'message.contains("response_format")' in dsv4
    assert "normalize_response_format(message.at(\"response_format\"))" in dsv4
    assert 'type == "text"' in dsv4
    assert 'type == "json_object"' in dsv4
    assert 'type == "json_schema"' in dsv4
    assert 'response_format = response_format.at("schema");' in dsv4
    assert "foreach_function(available_tools" in dsv4
    assert 'p.schema(p.json(), "response-format-schema", response_schema)' in dsv4
    assert "auto schema = response_schema;" in dsv4
    assert "json available_tools = inputs.tools.is_array()" not in gpt_oss


def test_dsv4_disabled_thinking_consumes_close_marker() -> None:
    dsv4 = _section(
        TEXT_CHAT,
        "static common_chat_params common_chat_params_init_deepseek_v3_2",
        "static common_chat_params common_chat_params_init_cohere2moe",
    )
    assert "p.optional(p.literal(THINK_END)) +" in dsv4
    assert "p.literal(THINK_START) +" in dsv4
    assert "p.until(THINK_END) +" in dsv4
    assert "p.literal(THINK_END));" in dsv4
