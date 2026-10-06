"""Only build dependency contracts belong here.

Execution, cancellation, batching, sessions, failures and protocol behavior are
exercised by the C++ Interface/fake-Adapter tests and real-weight integration
checks, not by matching implementation text.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "csrc"


def cmake(module: str) -> str:
    return (ROOT / module / "CMakeLists.txt").read_text(encoding="utf-8")


def test_transport_and_scheduler_only_link_their_downward_contracts() -> None:
    scheduler = cmake("scheduler")
    transport = cmake("transport")
    assert "PUBLIC mfq-engine" in scheduler
    assert "mfq-runtime-scheduler" in transport
    for source in (scheduler, transport):
        assert not re.search(r"mfq-(?:cuda|metal|models|tokenizer|paged-prefix-cache)", source)
        assert "/engine/internal" not in source


def test_shared_execution_has_no_backend_build_dependencies() -> None:
    for module in ("engine", "models"):
        source = cmake(module)
        assert "backends/" not in source
        assert not re.search(r"mfq-(?:cuda|metal|runtime-communication)", source)
    assert "PUBLIC ${PROJECT_SOURCE_DIR}" not in cmake("models")
    engine = cmake("engine")
    assert re.search(r"PRIVATE\s+\$\{CMAKE_CURRENT_SOURCE_DIR\}/internal", engine)
    assert "PRIVATE mfq-tokenizer" in engine


def test_cuda_ops_do_not_link_shared_execution_or_serving() -> None:
    source = cmake("backends/cuda")
    backend = source.split("function(mfq_configure_cuda_backend_target target)", 1)[1].split("endfunction()", 1)[0]
    for dependency in ("mfq-engine", "mfq-models", "mfq-runtime-communication"):
        assert dependency not in backend
    runtime = source.split("function(mfq_configure_cuda_runtime_target target)", 1)[1].split("endfunction()", 1)[0]
    assert "mfq-runtime-communication" not in runtime
    assert "mfq_configure_cuda_backend_target(mfq-cuda-ops)" in source
    loading = source.split("add_library(mfq-cuda-loading OBJECT", 1)[1].split("set(MFQ_CUDA_INSTANTIATION_SOURCES", 1)[0]
    assert "mfq_configure_cuda_backend_target(mfq-cuda-loading)" in loading
    for dependency in ("mfq-engine", "mfq-tokenizer", "mfq_configure_cuda_runtime_target"):
        assert dependency not in loading
    model_config = source.split("function(mfq_configure_cuda_model_target target)", 1)[1].split("endfunction()", 1)[0]
    assert "${MFQ_CUDA_ROOT}/engine" not in model_config
    assert "mfq-tokenizer" not in model_config
    assert "add_library(mfq-cuda-models OBJECT ${MFQ_CUDA_MODEL_SOURCES})" in source
    assert "$<TARGET_OBJECTS:mfq-cuda-models>" in source
    assert "MFQ_BUILD_RUNTIME_COMMUNICATION" not in cmake("core")


def test_cuda_engine_is_the_only_cross_family_composition_root() -> None:
    cuda = ROOT / "backends" / "cuda"
    models = cuda / "models"
    common = models / "common"
    model_sources = [
        path for path in models.rglob("*") if path.suffix in {".h", ".cpp", ".tpp"}
    ]
    for path in model_sources:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r'#include\s+["<](?:\.\./)*engine/', source), path
        assert "load_runtime_components(" not in source, path
        assert "MFQ_BIND_CUDA_MODEL" not in source, path
        assert "MFQ_INSTANTIATE_CUDA_MODEL" not in source, path
    family_include = re.compile(
        r'#include\s+["<](?:models/)?(?:deepseek|gemma|glm|minicpm|qwen)[^">]*[">]'
    )
    for path in common.rglob("*"):
        if path.suffix in {".h", ".cpp", ".tpp"}:
            assert not family_include.search(path.read_text(encoding="utf-8")), path
    assert not (models / "qwen35" / "batching.cpp").exists()
    assert not (cuda / "engine" / "qwen_batching.cpp").exists()
    assert not (cuda / "engine" / "qwen_batching.h").exists()
    assert (cuda / "engine" / "model_components.cpp").is_file()
    assert (cuda / "engine" / "model_bindings.cpp").is_file()
