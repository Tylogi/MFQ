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


def test_prefill_speed_uses_cuda_events_per_bounded_model_chunk() -> None:
    assert "MfqPrefillCallback" not in SERVER_HEADER + RUNTIME
    assert "class PrefillCudaTimer" in RUNTIME
    assert "cudaEventRecord(started_, stream_)" in RUNTIME
    prefill = RUNTIME.split("while (offset < full_ids.size(1)", 1)[1]
    assert prefill.index("PrefillCudaTimer timer") < prefill.index("auto ids =")
    logits = prefill.index("auto logits = model.logits_from_hidden")
    finished = prefill.index("cudaEventRecord(timer.finished_event()", logits)
    sampling = prefill.index("mfq::cuda::sample_logits(", logits)
    assert logits < finished < sampling
    assert "elapsed += timer.elapsed_ms()" in prefill
    assert "co_yield PrefillProgress" in prefill
    assert "inline SamplingOps::Tensor sample_logits(" in SAMPLING
    assert "sample_apply_penalties_cuda(" in SAMPLING
    assert "1000.0 * metrics.prefill_tokens / metrics.prefill_ms" in SERVER
    assert '{"prefill_tps", values.prefill_tps}' in SERVER
    assert '{"prefill_ms", values.prefill_ms}' in SERVER
