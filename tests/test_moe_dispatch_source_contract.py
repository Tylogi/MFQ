"""Ownership gates; numerical coverage is in the native dispatch/cache tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_dispatch_policy_is_backend_neutral():
    policy = (ROOT / "cpp_runtime/core/include/mfq/moe_dispatch_plan.h").read_text()
    assert "cuda_runtime" not in policy and "mfq_tensor_backend" not in policy


def test_hybrid_dispatch_is_owned_by_shared_storage():
    shared = (ROOT / "cpp_runtime/backends/cuda/storage/moe_cached_source_internal.h").read_text()
    assert '#include "mfq/moe_dispatch_plan.h"' in shared
    for path in (ROOT / "cpp_runtime/backends/cuda/models").rglob("*"):
        if path.suffix in {".cpp", ".h"}:
            assert "MFQ_MOE_HYBRID_CPU" not in path.read_text(), str(path)
