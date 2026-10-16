"""Backend switches in csrc/CMakeLists.txt, checked by configuring the real project.

Each check runs `cmake -S csrc -B <tmp>` and reads the generated target graph.
Nothing here compiles native code, loads a model or touches a GPU.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mfq.commands.build import _cuda_compiler

ROOT = Path(__file__).parents[1]
CSRC = ROOT / "csrc"

# A host with no CUDA toolkit. Only a build that includes the CUDA backend can
# notice either of these.
NO_CUDA_TOOLKIT = (
    "-DCMAKE_CUDA_COMPILER=/nonexistent/nvcc",
    "-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON",
)

# Targets that only exist inside backends/cuda.
CUDA_ONLY_TARGETS = (
    "mfq-runtime",
    "mfq-cuda-runtime",
    "mfq-cuda-native-kernels",
    "mfq-diagnostics",
    "mfq-eval",
)

NOT_APPLE = platform.system() != "Darwin"


def _configure_command(cmake: str, build_dir: Path, *options: str) -> list[str]:
    command = [
        cmake,
        "-S",
        str(CSRC),
        "-B",
        str(build_dir),
        f"--graphviz={build_dir / 'targets.dot'}",
    ]
    # Same convention as mfq build: Ninja only when it is installed.
    if shutil.which("ninja"):
        command += ["-G", "Ninja"]
    return command + list(options)


def _configure(tmp_path: Path, name: str, *options: str) -> tuple[int, str, set[str]]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")
    build_dir = tmp_path / name
    result = subprocess.run(
        _configure_command(cmake, build_dir, *options),
        capture_output=True,
        text=True,
    )
    graph_path = build_dir / "targets.dot"
    graph = graph_path.read_text(encoding="utf-8") if graph_path.is_file() else ""
    # Node labels are "name\n(mfq::alias)"; keep the target name.
    targets = {label.split("\\n")[0] for label in re.findall(r'"([^"]+)"', graph)}
    return result.returncode, result.stdout + result.stderr, targets


def test_common_engine_configures_without_a_cuda_toolkit(tmp_path) -> None:
    code, output, targets = _configure(
        tmp_path,
        "no-cuda",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMFQ_BUILD_CUDA_RUNTIME=OFF",
        "-DMFQ_BUILD_METAL_RUNTIME=OFF",
        *NO_CUDA_TOOLKIT,
    )

    assert code == 0, output
    for target in CUDA_ONLY_TARGETS:
        assert target not in targets
    assert "mfq-engine" in targets
    assert "mfq-engine-contract-test" in targets
    # The CUDA-named CPU tests need no device SDK and stay registered.
    assert "mfq-cuda-tensor-view-test" in targets
    assert "mfq-cuda-model-plan-test" in targets


def test_communication_off_keeps_the_common_engine_and_its_tests(tmp_path) -> None:
    code, output, targets = _configure(
        tmp_path,
        "no-cuda-no-communication",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMFQ_BUILD_CUDA_RUNTIME=OFF",
        "-DMFQ_BUILD_METAL_RUNTIME=OFF",
        "-DMFQ_BUILD_RUNTIME_COMMUNICATION=OFF",
        *NO_CUDA_TOOLKIT,
    )

    assert code == 0, output
    assert "mfq-engine" in targets
    assert "mfq-engine-contract-test" in targets
    assert "mfq-runtime-communication" not in targets


def test_default_configuration_enables_the_cuda_backend(tmp_path) -> None:
    if not NOT_APPLE:
        pytest.skip("the CUDA backend default is off on Apple platforms")
    if _cuda_compiler() is None:
        pytest.skip("the CUDA default configuration needs nvcc")

    code, output, targets = _configure(tmp_path, "cuda-default", "-DCMAKE_BUILD_TYPE=Release")

    assert code == 0, output
    assert {"mfq-runtime", "mfq-cuda-runtime", "mfq-cuda-native-kernels"} <= targets


def test_enabled_cuda_backend_still_requires_the_toolkit(tmp_path) -> None:
    if not NOT_APPLE:
        pytest.skip("the CUDA backend is not supported on Apple platforms")

    code, output, _ = _configure(
        tmp_path,
        "cuda-without-toolkit",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMFQ_BUILD_CUDA_RUNTIME=ON",
        "-DMFQ_BUILD_METAL_RUNTIME=OFF",
        *NO_CUDA_TOOLKIT,
    )

    assert code != 0
    assert "backends/cuda" in output


def test_a_cache_configured_off_gets_the_runtime_back(tmp_path) -> None:
    """`mfq build --backend cuda` passes the switch explicitly for this case."""
    if not NOT_APPLE or _cuda_compiler() is None:
        pytest.skip("re-enabling the CUDA backend needs nvcc")

    code, output, targets = _configure(
        tmp_path,
        "reused-cache",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMFQ_BUILD_CUDA_RUNTIME=OFF",
    )
    assert code == 0, output
    assert "mfq-runtime" not in targets

    code, output, targets = _configure(tmp_path, "reused-cache", "-DMFQ_BUILD_CUDA_RUNTIME=ON")

    assert code == 0, output
    assert "mfq-runtime" in targets


def test_configure_selects_ninja_only_when_it_is_installed(tmp_path: Path, monkeypatch) -> None:
    installed = {"cmake": "/usr/bin/cmake", "ninja": "/usr/bin/ninja"}
    monkeypatch.setattr(shutil, "which", lambda name: installed.get(name))

    with_ninja = _configure_command("cmake", tmp_path / "build", "-DMFQ_BUILD_CUDA_RUNTIME=OFF")
    installed.pop("ninja")
    without_ninja = _configure_command("cmake", tmp_path / "build", "-DMFQ_BUILD_CUDA_RUNTIME=OFF")

    assert with_ninja[with_ninja.index("-G") + 1] == "Ninja"
    assert "-G" not in without_ninja
    assert without_ninja[-1] == "-DMFQ_BUILD_CUDA_RUNTIME=OFF"


def test_metal_runtime_still_rejects_a_non_apple_host(tmp_path) -> None:
    if not NOT_APPLE:
        pytest.skip("the Metal runtime is the Apple platform default")

    code, output, _ = _configure(tmp_path, "metal-on-linux", "-DMFQ_BUILD_METAL_RUNTIME=ON")

    assert code != 0
    assert "MFQ_BUILD_METAL_RUNTIME requires macOS" in output
