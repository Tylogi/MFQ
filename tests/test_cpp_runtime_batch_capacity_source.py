"""Backend parity for reported multi-request capacity.

The control plane sizes its admission limit from the worker handshake, so every
backend must publish the rows it can actually batch. These source checks keep
CUDA, HIP and Metal on one contract instead of rediscovering the gap per
release.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "cpp_runtime"


def _read(*parts: str) -> str:
    return (RUNTIME.joinpath(*parts)).read_text(encoding="utf-8")


def test_shared_engine_reports_batch_capacity() -> None:
    engine = _read("engine", "include", "engine.h")
    assert "std::size_t batch_capacity = 1;" in engine
    assert "Concurrent rows the loaded model can actually batch" in engine


def test_cuda_and_hip_report_loaded_capacity() -> None:
    for backend in ("cuda", "hip"):
        source = _read("backends", backend, "engine", "cuda_engine.cpp")
        assert "info.batch_capacity = state.batch_capacity();" in source, backend
        assert "info.max_requests = info.batch_capacity;" in source, backend
        assert "std::size_t batch_capacity() const {" in source, backend
        # An unsupported model must keep the single-request path so a serving
        # concurrency hint can never fail a load.
        assert "continuous batching is unavailable for this model" not in source
        assert "continuous_batching unavailable model=" in source, backend


def test_metal_reports_loaded_capacity() -> None:
    engine = _read("backends", "metal", "runtime", "mlx_engine.cpp")
    assert "info_.batch_capacity = info_.max_requests;" in engine


def test_transports_publish_batch_capacity() -> None:
    for transport in ("stdio.cpp", "http.cpp"):
        source = _read("transport", "src", transport)
        assert "batch_capacity" in source, transport
        assert "scheduler.info().batch_capacity" in source, transport


def test_diagnostics_report_capacity() -> None:
    for backend in ("cuda", "hip"):
        source = _read("backends", backend, "diagnostics", "runtime_checks.cpp")
        assert "info.batch_capacity = info.max_requests;" in source, backend


def test_server_side_policy_matches_native_limits() -> None:
    native = (ROOT / "mfq" / "server" / "runtime" / "native.py").read_text(encoding="utf-8")
    pool = (ROOT / "mfq" / "server" / "runtime" / "runtime_pool.py").read_text(
        encoding="utf-8"
    )
    backend = (ROOT / "mfq" / "server" / "runtime" / "backend.py").read_text(
        encoding="utf-8"
    )
    models = (ROOT / "mfq" / "server" / "protocol" / "models.py").read_text(
        encoding="utf-8"
    )
    assert "MAX_RUNTIME_CONCURRENCY = 32" in native
    assert 'frozenset({"cuda", "hip", "metal"})' in native
    assert "batch_capacity" in backend
    assert "batch_capacity: int = Field(default=1, ge=1)" in models
    assert "def _apply_batch_capacity(" in pool
    assert "reported >= instance.request_capacity" in pool
