"""保留原生预填充计时契约；富文本与指标展示已迁入 Vitest。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_HEADER = (ROOT / "cpp_runtime" / "server" / "include" / "mfq" / "server.h").read_text(
    encoding="utf-8"
)
SERVER = (ROOT / "cpp_runtime" / "server" / "src" / "server.cpp").read_text(encoding="utf-8")
RUNTIME = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "runtime" / "cuda_decode_runtime.cpp"
).read_text(encoding="utf-8")
SAMPLING = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "runtime" / "cuda_sampling.h"
).read_text(encoding="utf-8")


def test_prefill_speed_uses_cuda_events_around_only_the_first_model_eval() -> None:
    assert "using MfqPrefillCallback" in SERVER_HEADER
    assert "const MfqPrefillCallback & on_prefill" in RUNTIME
    assert "class ServerPrefillCudaTimer" in RUNTIME
    assert "cudaEventRecord(started_, stream_)" in RUNTIME
    assert "cudaEvent_t prefill_finished = nullptr" in RUNTIME
    assert RUNTIME.count(
        "prefill_finished, mfq_get_current_cuda_stream()"
    ) == 2
    assert "prefill_timer.finished_event()" in RUNTIME
    assert "const int64_t token = next.template item<int64_t>();" in RUNTIME
    assert "const double prefill_ms = prefill_timer.elapsed_ms();" in RUNTIME
    assert "on_prefill(MfqPrefillTiming{" in RUNTIME
    assert "prompt.size() - reused_tokens,\n                prefill_ms,\n                0.0,\n                prefill_ms" in RUNTIME

    sample = RUNTIME.split(
        "static mfq_tensor_backend::Tensor sample_server_token(", 1
    )[1]
    sample = sample.split("class ServerPrefillCudaTimer", 1)[0]
    logits = sample.index("auto logits = model.last_logits(ids)")
    finished = sample.index("cudaEventRecord(", logits)
    sampling = sample.index("mfq::cuda::sample_logits(", logits)
    assert logits < finished < sampling
    assert "sample_apply_penalties_cuda(" in SAMPLING

    first = RUNTIME.split("auto sample_first_token = [&]()", 1)[1]
    first = first.split("const char * reprefill_env", 1)[0]
    assert first.index("ServerPrefillCudaTimer prefill_timer") < first.index(
        "mfq_tensor_backend::Tensor next;"
    )
    assert first.index(
        "const int64_t token = next.template item<int64_t>();"
    ) < first.index("prefill_timer.elapsed_ms()")
    assert "1000.0 * metrics.prefill_tokens / metrics.prefill_ms" in SERVER
    assert '{"prefill_tps", values.prefill_tps}' in SERVER
    assert '{"prefill_ms", values.prefill_ms}' in SERVER
