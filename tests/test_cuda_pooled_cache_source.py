"""Keep completed-group state and pooling operations in the shared CUDA layer."""

from pathlib import Path


CUDA = Path(__file__).resolve().parents[1] / "cpp_runtime/backends/cuda"


def test_qsa_adapts_the_shared_pooled_cache() -> None:
    model = (CUDA / "models/qwen4_exp/model.h").read_text(encoding="utf-8")
    qsa = model.split("class Qsa {", 1)[1]
    assert "PooledCache pooled_;" in qsa
    assert "pooled_.update_graph(" in qsa
    assert "pooled_.update_range(" in qsa
    assert "pooled_.truncate(keep,c_.pool)" in qsa
    assert "state.pooled=pooled_.capture(count,c_.pool)" in qsa
    assert "pooled_.restore(state.pooled,count,c_.pool)" in qsa
    assert ".mean(2)" not in qsa
    assert "live_index_pool(" not in qsa


def test_shared_pooled_cache_owns_watermark_and_rotary_dispatch() -> None:
    cache = (CUDA / "core/pooled_cache.cpp").read_text(encoding="utf-8")
    assert "completed_=std::min(completed_,tokens/pool)" in cache
    assert "rotary.pool_groups_into(" in cache
    assert "positions_.index_copy_(" in cache
    assert "models/" not in cache
