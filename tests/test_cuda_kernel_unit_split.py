"""G2 build isolation for the inference/tool kernel split.

Each mixed CUDA unit is split by actual consumers: the inference dequant,
online activation quantization and forward kernels stay in
``mfq-cuda-native-kernels``; the tool-only backward units are compiled only by
the Python calibration extension and, for MXFP4-SQ, by the native numeric test.
The manifest checks run anywhere; the build-graph check runs a real
``cmake -S csrc -B <tmp>`` configure and reads the generated Ninja build graph,
so it proves the compile list rather than only the source list.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mfq.commands.build import _cuda_compiler
from mfq.kernels.cuda._ext import _SOURCES

ROOT = Path(__file__).parents[1]
CSRC = ROOT / "csrc"
CUDA_BACKEND = CSRC / "backends" / "cuda"
KERNELS_OBJECT_DIR = "mfq-cuda-native-kernels.dir"

# Inference-side units the native kernel library must compile.
INFERENCE_UNITS = (
    "nvq/dequant.cu",
    "quant/mx_matmul.cu",
    "quant/fp8_sq.cu",
    "quant/mxfp4_sq.cu",
    "quant/nepq.cu",
    "quant/nepq_residual.cu",
    "quant/nint_matmul.cu",
)

# Tool-only backward units that must stay out of the native kernel library.
TOOL_ONLY_UNITS = (
    "nvq/backward.cu",
    "quant/mx_matmul_backward.cu",
    "quant/fp8_sq_backward.cu",
    "quant/mxfp4_sq_backward.cu",
    "quant/nepq_backward.cu",
    "quant/nepq_residual_backward.cu",
    "quant/nint_matmul_backward.cu",
)


def _kernel_sources_block() -> str:
    text = (CUDA_BACKEND / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"set\(MFQ_CUDA_KERNEL_SOURCES(.*?)\n\)", text, re.S)
    assert match, "MFQ_CUDA_KERNEL_SOURCES is missing from the CUDA backend CMakeLists"
    return match.group(1)


def _generated_build_graph(build_dir: Path, run) -> str:
    """Configure the real project and return the generated Ninja build graph."""
    result = run(
        [shutil.which("cmake"), "-S", str(CSRC), "-B", str(build_dir), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return (build_dir / "build.ninja").read_text(encoding="utf-8")


def test_native_kernel_manifest_excludes_tool_backward_units() -> None:
    block = _kernel_sources_block()
    for unit in INFERENCE_UNITS:
        assert unit in block, f"{unit} left the native kernel manifest"
    for unit in TOOL_ONLY_UNITS:
        assert unit not in block, f"{unit} entered the native kernel manifest"


def test_python_extension_manifest_keeps_both_sides() -> None:
    named = {Path(path).as_posix() for path in _SOURCES}
    for unit in INFERENCE_UNITS + TOOL_ONLY_UNITS:
        assert any(path.endswith(unit) for path in named), (
            f"{unit} is missing from the extension manifest")


def test_generated_build_graph_excludes_backward_from_inference_library(
    tmp_path: Path,
) -> None:
    if platform.system() == "Darwin":
        pytest.skip("the CUDA backend is not built on Apple platforms")
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        pytest.skip("the build-graph check needs cmake and ninja")
    if _cuda_compiler() is None:
        pytest.skip("configuring the CUDA backend needs nvcc")

    graph = _generated_build_graph(tmp_path / "cuda-graph", subprocess.run)
    for unit in INFERENCE_UNITS:
        assert f"{KERNELS_OBJECT_DIR}/kernels/{unit}.o" in graph, unit
    for unit in TOOL_ONLY_UNITS:
        assert f"{KERNELS_OBJECT_DIR}/kernels/{unit}.o" not in graph, unit
    # The MXFP4-SQ backward is still compiled, just not for inference.
    assert "mxfp4_sq_backward.cu.o" in graph


def test_a_failed_configure_is_a_failure_not_a_skip(tmp_path: Path, monkeypatch) -> None:
    """A CMake regression must fail the check instead of hiding behind a skip."""

    class Failed:
        returncode = 1
        stdout = "CMake Error at backends/cuda/CMakeLists.txt:6 (enable_language)"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Failed())

    with pytest.raises(AssertionError, match="enable_language"):
        _generated_build_graph(tmp_path / "cuda-graph", run=subprocess.run)
