import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "cpp_runtime"


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_raw_cli_loads_eos_without_changing_tokenized_or_fixed_input_paths(backend: str) -> None:
    source = (ROOT / "backends" / backend / "commands/diagnostics.cpp").read_text()
    assert "if (!ignore_eos && tokenizer_model.empty() && decode_inputs.empty())" in source
    assert "mfq::engine::model_stop_token_ids(" in source
    assert "mfq::cuda::load_model_config_json(*model.source, config_path), model.vocab_size()" in source
    assert re.search(
        r"execution,\s*model,\s*ids,\s*gen,\s*profile,\s*t0,\s*t1,\s*decode_inputs,\s*stop_ids"
        r"(?:,\s*!router_lookahead_audit.empty\(\))?\s*\)",
        source,
    )
    assert (
        'option == "--ignore-eos"' in source
        or re.search(r'\{"--ignore-eos",\s*&DiagnosticsCommandOptions::ignore_eos\}', source)
    )
    assert "warning: no model EOS token IDs" in source
    engine = (ROOT / "backends" / backend / "engine/cuda_engine.cpp").read_text()
    assert "load_model_config_json(*state.language.source, options.config_path)" in engine


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_raw_cli_checks_first_eager_and_graph_tokens_and_trims_unused_capacity(backend: str) -> None:
    source = (ROOT / "backends" / backend / "diagnostics/token_generation.h").read_text()
    assert "if (use_cuda_graph && !hit_eos)" in source
    assert "else if (!hit_eos)" in source
    assert source.count("stop_token_ids.contains(next.template item<int64_t>())") == 2
    assert "stop_token_ids.contains(static_next.template item<int64_t>())" in source
    assert "gen = produced;" in source
    assert "generated_cuda.narrow(0, 0, produced)" in source
    assert '(hit_eos ? "stop" : "length")' in source


def test_shared_output_only_uses_metadata_eos_when_tokenizer_is_absent() -> None:
    source = (ROOT / "engine/src/inference.cpp").read_text()
    assert "tokenizer_ ? tokenizer_->is_eog(token) : request_.stop_token_ids.contains(token)" in source
    append = source[source.index("TokenOutput InferenceOutput::append"):]
    assert append.index("request_.stop_token_ids.contains(token)") < append.index("if (!tokenizer_) continue;")
    admission = (ROOT / "engine/include/request_executor.h").read_text()
    assert "input.stop_token_ids = info.stop_token_ids;" in admission
