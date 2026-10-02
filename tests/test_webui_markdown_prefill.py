"""保留原生预填充计时契约；富文本与指标展示已迁入 Vitest。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_HEADER = (ROOT / "cpp_runtime" / "core" / "include" / "mfq" / "runtime.h").read_text(
    encoding="utf-8"
)
TRANSPORT_SRC = ROOT / "cpp_runtime" / "transport"
SERVER = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(TRANSPORT_SRC.rglob("*"))
    if path.suffix in {".cpp", ".h"}
)
CUDA_ROOT = ROOT / "cpp_runtime" / "backends" / "cuda"
RUNTIME = "\n".join(
    (CUDA_ROOT / name).read_text(encoding="utf-8")
    for name in (
        "ops/include/cuda_execution.h",
        "ops/cuda_execution.cpp",
        "engine/generation.h",
        "engine/generation.cpp",
    )
)
SAMPLING = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "ops" / "include" / "cuda_sampling.h"
).read_text(encoding="utf-8")


def test_prefill_speed_uses_cuda_events_around_only_the_first_model_eval() -> None:
    assert "using MfqPrefillCallback" in SERVER_HEADER
    assert "const MfqPrefillCallback& on_prefill" in RUNTIME
    assert "class PrefillCudaTimer" in RUNTIME
    assert "cudaEventRecord(started_, stream_)" in RUNTIME
    assert "cudaEvent_t prefill_finished = nullptr" in RUNTIME
    assert RUNTIME.count(
        "prefill_finished, mfq_get_current_cuda_stream()"
    ) == 2

    sample = RUNTIME.split(
        "static mfq_tensor_backend::Tensor sample_token(", 1
    )[1]
    sample = sample.split(
        "template <typename Model>\nstatic mfq_tensor_backend::Tensor prefill_tail(",
        1,
    )[0]
    logits = sample.index("auto logits = model.last_logits(ids)")
    finished = sample.index("cudaEventRecord(", logits)
    sampling = sample.index("mfq::cuda::sample_logits(", logits)
    assert logits < finished < sampling
    assert "inline SamplingOps::Tensor sample_logits(" in SAMPLING
    assert "sample_apply_penalties_cuda(" in SAMPLING

    prefill = RUNTIME.split("mfq::engine::PrefillResult prefill(", 1)[1]
    prefill = prefill.split("bool graph_eligible()", 1)[0]
    assert prefill.index("PrefillCudaTimer timer") < prefill.index("auto ids =")
    assert "timer.finished_event()" in prefill
    assert prefill.index(
        "const auto token = pending.template item<int64_t>();"
    ) < prefill.index("timer.elapsed_ms()")
    assert "prompt.size() - reused, prefill_ms, multimodal_ms,\n                        prefill_ms + multimodal_ms" in prefill
    assert "1000.0 * metrics.prefill_tokens / metrics.prefill_ms" in SERVER
    assert '{"prefill_tps", values.prefill_tps}' in SERVER
    assert '{"prefill_ms", values.prefill_ms}' in SERVER
