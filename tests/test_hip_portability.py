from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mfq.kernels.hip._build import stage_sources

ROOT = Path(__file__).resolve().parents[1]


def cmake_path():
    path = shutil.which("cmake") or str(Path(sys.executable).with_name("cmake"))
    if not Path(path).is_file():
        pytest.skip("CMake is unavailable")
    return path


@pytest.mark.parametrize("cuda,hip,flags,expected", [
    (True, True, [], "ON;OFF"),
    (True, False, [], "ON;OFF"),
    (False, True, [], "OFF;ON"),
    (False, False, [], "OFF;OFF"),
    (True, True, ["-DMFQ_BUILD_HIP_RUNTIME=ON"], "OFF;ON"),
    (True, True, ["-DMFQ_BUILD_CUDA_RUNTIME=OFF"], "OFF;ON"),
    (True, True, ["-DMFQ_BUILD_CUDA_RUNTIME=ON", "-DMFQ_BUILD_HIP_RUNTIME=OFF"], "ON;OFF"),
    (True, True, ["-DMFQ_BUILD_CUDA_RUNTIME=ON", "-DMFQ_BUILD_HIP_RUNTIME=ON"], None),
])
def test_cmake_accelerator_selection(tmp_path, cuda, hip, flags, expected):
    source = (ROOT / "cpp_runtime/CMakeLists.txt").read_text()
    options = source.split("include(CheckLanguage)\n", 1)[1].split("\nif(APPLE)\n", 1)[0]
    options = options.replace("check_language(CUDA)", "").replace("check_language(HIP)", "")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\nproject(backend_options NONE)\nset(APPLE FALSE)\n"
        f"set(CMAKE_CUDA_COMPILER {'mock' if cuda else 'NOTFOUND'})\n"
        f"set(CMAKE_HIP_COMPILER {'mock' if hip else 'NOTFOUND'})\n" + options +
        '\nfile(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${MFQ_BUILD_CUDA_RUNTIME};${MFQ_BUILD_HIP_RUNTIME}")\n'
    )
    output = tmp_path / "build"
    command = [cmake_path(), "-S", str(tmp_path), "-B", str(output)]
    result = subprocess.run([*command, *flags], capture_output=True, text=True)
    if expected is None:
        assert result.returncode != 0
        assert "Select either MFQ_BUILD_CUDA_RUNTIME or MFQ_BUILD_HIP_RUNTIME" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (output / "selected.txt").read_text() == expected
        if flags == ["-DMFQ_BUILD_HIP_RUNTIME=ON"]:
            result = subprocess.run([*command, "-DMFQ_BUILD_CUDA_RUNTIME=ON", "-DMFQ_BUILD_HIP_RUNTIME=OFF"],
                                    capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
            assert (output / "selected.txt").read_text() == "ON;OFF"


def test_wave_intrinsics_preserve_subwarps_and_both_half_waves(tmp_path):
    compiler = shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++ compiler is unavailable")
    include = tmp_path / "hip"
    include.mkdir()
    (include / "hip_bf16.h").write_text("")
    (include / "hip_runtime.h").write_text(r"""
#pragma once
#define __device__
#define __forceinline__ inline
struct Dim { unsigned x, y, z; };
inline Dim threadIdx{}, blockDim{8, 8, 4};
inline unsigned warpSize = 64, current_lane;
inline unsigned long long votes;
template<typename T> T __shfl(T v, int s, int w) { return v - current_lane % w + s; }
template<typename T> T __shfl_down(T v, unsigned d, int w) { return current_lane % w + d < unsigned(w) ? v + d : v; }
template<typename T> T __shfl_up(T v, unsigned d, int w) { return current_lane % w >= d ? v - d : v; }
template<typename T> T __shfl_xor(T v, int s, int w) { return v - current_lane % w + ((current_lane % w) ^ s); }
inline unsigned long long __ballot(bool) { return votes; }
""")
    source = tmp_path / "wave.cpp"
    source.write_text(r"""
#include <cassert>
#define __HIP_PLATFORM_AMD__ 1
#include "mfq_hip_compat.h"
int main() {
    assert(__dp4a(int(0x80808080u), int(0x80808080u), 0) == 65536);
    assert(__dp4a(int(0xfffe0180u), int(0x027f80ffu), 50) == -206);
    assert(__dp4a(0x7f7f7f7f, 0x7f7f7f7f, 2147483647) == -2147419133);
    assert(__dp4a(int(0x80808080u), 0x7f7f7f7f, -2147483647 - 1) == 2147418624);
    for (unsigned wave : {32u, 64u}) {
        warpSize = wave;
        for (unsigned thread = 0; thread < 256; ++thread) {
            threadIdx = {thread % 8, thread / 8 % 8, thread / 64};
            current_lane = thread % wave;
            const int v = int(thread);
            assert(__shfl_sync(~0u, v, 1) == int(thread / 32 * 32 + 1));
            assert(__shfl_sync(~0u, float(v), 3, 16) == float(thread / 16 * 16 + 3));
            for (int width : {8, 16, 32}) {
                assert(__shfl_sync(~0u, v, 1, width) == int(thread / width * width + 1));
                assert(__shfl_down_sync(~0u, v, 1, width) == (thread % width + 1 < unsigned(width) ? v + 1 : v));
                assert(__shfl_up_sync(~0u, v, 1, width) == (thread % width ? v - 1 : v));
                assert(__shfl_xor_sync(~0u, v, 1, width) == (v ^ 1));
            }
            for (int pattern = 0; pattern < 4; ++pattern) {
                votes = 0;
                unsigned expected = 0;
                for (unsigned lane = 0; lane < wave; ++lane) {
                    unsigned id = thread / wave * wave + lane;
                    bool p = pattern == 0 ? id % 64 < 48 : pattern == 1 ? id % 64 >= 32 :
                        pattern == 2 ? id % 3 == 0 : id % 64 < 17;
                    if (p) votes |= 1ull << lane;
                    if (p && id / 32 == thread / 32) expected |= 1u << (id % 32);
                }
                assert(__ballot_sync(~0u, true) == expected);
                assert(__ballot_sync(0x55555555u, true) == (expected & 0x55555555u));
            }
        }
    }
}
""")
    executable = tmp_path / "wave"
    result = subprocess.run([compiler, "-std=c++17", "-include", "initializer_list",
                             "-I", str(tmp_path), "-I", str(ROOT / "mfq/kernels/hip"),
                             str(source), "-o", str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True)


def test_host_translation_units_do_not_require_device_builtins(tmp_path):
    compiler = shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++ compiler is unavailable")
    include = tmp_path / "hip"
    include.mkdir()
    (include / "hip_bf16.h").write_text("")
    (include / "hip_runtime.h").write_text("#define __device__\n#define __forceinline__ inline\n")
    source = tmp_path / "host.cpp"
    source.write_text('#define __HIP_PLATFORM_AMD__ 1\n#include "mfq_hip_compat.h"\n'
                      'int main() { return 0; }\n')
    executable = tmp_path / "host"
    result = subprocess.run([compiler, "-std=c++17", "-I", str(tmp_path),
                             "-I", str(ROOT / "mfq/kernels/hip"), str(source), "-o", str(executable)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True)


def hashes(root):
    return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_staging_is_content_addressed_and_excludes_cuda(tmp_path):
    root = tmp_path / "repository"
    kernel = root / "mfq/kernels/hip"
    kernel.mkdir(parents=True)
    (kernel / "fixed.cuh").write_text("#pragma once\n#define WIDTH 8\n")
    cuda = root / "mfq/kernels/cuda"
    cuda.mkdir()
    (cuda / "fixed.cuh").write_text("#pragma once\n#define WIDTH 32\n")
    original = hashes(root)
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=4) as executor:
        stages = list(executor.map(lambda _: stage_sources(root, cache), range(4)))
    assert len(set(stages)) == 1
    stage = stages[0]
    assert not (stage / "mfq/kernels/cuda").exists()
    target = stage / "mfq/kernels/hip/fixed.cuh"
    stamp = target.stat().st_mtime_ns
    assert stage_sources(root, cache) == stage
    assert target.stat().st_mtime_ns == stamp
    assert hashes(root) == original
    (kernel / "fixed.cuh").write_text("#pragma once\n#define WIDTH 16\n")
    updated = stage_sources(root, cache)
    assert updated != stage
    assert (updated / "mfq/kernels/hip/fixed.cuh").read_text().endswith("WIDTH 16\n")
    assert target.read_text().endswith("WIDTH 8\n")


def test_real_hipify_does_not_replace_staged_or_repository_hip_headers(tmp_path):
    pytest.importorskip("torch")
    from torch.utils.hipify import hipify_python
    from mfq.kernels.hip._ext import _SOURCES

    stage = stage_sources(ROOT, tmp_path / "cache")
    original = hashes(ROOT / "mfq/kernels/hip")
    original_cuda = hashes(ROOT / "mfq/kernels/cuda")
    staged = hashes(stage / "mfq/kernels/hip")
    build = tmp_path / "build"
    build.mkdir()
    hipify_python.hipify(project_directory=str(build), output_directory=str(build),
                        header_include_dirs=[str(stage)],
                        extra_files=[str(stage / "mfq/kernels/hip" / Path(source).name) for source in _SOURCES],
                        show_progress=False, is_pytorch_extension=True)
    after = hashes(stage / "mfq/kernels/hip")
    assert all(after[path] == digest for path, digest in staged.items())
    assert hashes(ROOT / "mfq/kernels/hip") == original
    assert hashes(ROOT / "mfq/kernels/cuda") == original_cuda


def test_loader_only_passes_staged_inputs_to_torch(tmp_path, monkeypatch):
    from mfq.kernels.hip import _ext

    captured = {}
    def load(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(hip="test")))
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", SimpleNamespace(
        load=load, _get_build_directory=lambda *args, **kwargs: str(tmp_path / "cache")))
    monkeypatch.setattr(_ext, "_module", None)
    module = _ext.ext()
    assert _ext.ext() is module
    for path in captured["sources"] + captured["extra_include_paths"]:
        value = Path(path)
        assert value.exists() and value.is_relative_to(tmp_path)
    assert captured["build_directory"] == str(tmp_path / "cache")
