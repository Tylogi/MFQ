#!/usr/bin/env python3
"""G3 source isolation: a shared CUDA kernel unit must not reach a Tensor backend.

``kernels/activation.cu`` and ``kernels/embedding.cu`` hold the device bodies and
the launchers only (``TensorView`` + ``cudaStream_t`` for activation, raw
pointers + ``cudaStream_t`` for embedding).  The allocating, validating wrappers
live in ``ops/activation.cpp`` and ``ops/embedding.cpp``, the one unit of each
pair that may include ``native/tensor/backend.h``.  This check walks the include closure of every
registered kernel unit and fails when it reaches that dual-mode backend header or
a Torch/ATen/c10/pybind header, so the kernel unit compiles with either backend
mode or with neither.

This is source isolation only.  The same unit is still compiled by nvcc without
``--use_fast_math`` for the native target and with it by
``mfq/kernels/cuda/_ext.py``; removing an include does not unify those flags.

Run: python3 cuda/tests/test_cuda_kernel_torch_free.py
     pytest cuda/tests/test_cuda_kernel_torch_free.py
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from include_graph import include_roots

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(REPO, "csrc")
BACKEND = os.path.join(CSRC, "backends", "cuda")

# Kernel units that must stay free of any Tensor backend, under kernels/.
KERNEL_DIR_UNITS = ("activation.cu", "embedding.cu")
# Adapter units that own the validation, allocation, view and stream work, and
# are compiled by both the native ops target and the Python extension manifest.
ADAPTER_UNITS = ("ops/activation.cpp", "ops/embedding.cpp")
KERNEL_UNITS = tuple(f"kernels/{unit}" for unit in KERNEL_DIR_UNITS)

BACKEND_HEADER = "native/tensor/backend.h"
BACKEND_TOKEN = re.compile(r"^(torch|ATen|c10|pybind11)/")
# Both include forms are valid C++ and both can name a backend header, so one
# pattern covers them: #include "x" and #include <x>, with or without a space
# after "include".
INCLUDE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]')


def include_paths(csrc):
    """The include directories the CUDA targets put on the search path."""
    backend = os.path.join(csrc, "backends", "cuda")
    return [os.path.dirname(csrc), backend, os.path.join(csrc, "core")] + include_roots(csrc)


def resolve(raw, including_file, paths):
    for candidate in [os.path.join(os.path.dirname(including_file), raw)] + \
                     [os.path.join(root, raw) for root in paths]:
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate):
            return candidate
    return None


def closure(unit, csrc):
    """Every first-party file a unit reaches, plus its Torch-flavoured includes."""
    paths = include_paths(csrc)
    seen, tokens = set(), set()
    stack = [os.path.abspath(unit)]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        with open(path, errors="ignore") as handle:
            for lineno, line in enumerate(handle, 1):
                match = INCLUDE.match(line)
                if not match:
                    continue
                raw = match.group(1)
                if BACKEND_TOKEN.match(raw):
                    tokens.add(
                        f"{os.path.relpath(path, csrc).replace(os.sep, '/')}:{lineno}"
                        f": {raw}")
                target = resolve(raw, path, paths)
                if target:
                    stack.append(target)
    return seen, tokens


def check(csrc, units):
    """Fail a kernel unit whose closure reaches a Tensor backend or Torch headers."""
    backend = os.path.join(csrc, "backends", "cuda")
    failures = []
    for unit in units:
        files, tokens = closure(os.path.join(backend, unit), csrc)
        reached = [
            path for path in files
            if os.path.relpath(path, backend).replace(os.sep, "/") == BACKEND_HEADER
        ]
        if reached:
            failures.append(f"{unit} reaches the dual-mode Tensor backend {BACKEND_HEADER}")
        failures.extend(f"{unit} reaches a Torch header at {token}" for token in sorted(tokens))
    return failures


def _cmake_block(name):
    with open(os.path.join(BACKEND, "CMakeLists.txt"), encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(rf"set\({name}(.*?)\n\)", text, re.S)
    assert match, f"{name} is missing from the CUDA backend CMakeLists"
    return match.group(1)


def test_kernel_units_stay_free_of_a_tensor_backend():
    failures = check(CSRC, KERNEL_UNITS)
    assert not failures, "\n".join(failures)


def test_kernel_and_adapter_units_are_built_on_both_sides():
    from mfq.kernels.cuda._ext import _SOURCES

    named = {os.path.normpath(path).replace(os.sep, "/") for path in _SOURCES}
    kernel_block = _cmake_block("MFQ_CUDA_KERNEL_SOURCES")
    ops_block = _cmake_block("MFQ_CUDA_RUNTIME_OP_SOURCES")
    for unit in KERNEL_DIR_UNITS:
        assert f"${{MFQ_CUDA_KERNEL_ROOT}}/{unit}" in kernel_block, (
            f"kernels/{unit} left the native kernel manifest")
        assert any(path.endswith(f"/kernels/{unit}") for path in named), (
            f"kernels/{unit} left the extension manifest")
    for unit in ADAPTER_UNITS:
        assert f"${{MFQ_CUDA_ROOT}}/{unit}" in ops_block, f"{unit} left the native ops manifest"
        assert any(path.endswith("/" + unit) for path in named), f"{unit} left the extension"


# A rule with nothing to check it against is not a rule.
BASE_TREE = {
    "backends/cuda/kernels/glu.cuh": "#include <cuda_runtime.h>\n",
    "backends/cuda/kernels/kernels.h": '#include "native/tensor/view.h"\n',
    "backends/cuda/native/tensor/view.h": "#include <cstdint>\n",
    "backends/cuda/native/tensor/backend.h": "#include <ATen/Parallel.h>\n",
}
CLEAN_UNIT = '#include "glu.cuh"\n#include "kernels/kernels.h"\n'
CLEAN_FIXTURES = [
    ("a clean kernel unit passes", CLEAN_UNIT, {}, None),
]
# Each negative control is one of the bypasses the gate has to close: either
# include form, resolved or not, direct or transitive.
NEGATIVE_FIXTURES = [
    ("quoted include of the dual-mode backend header",
     CLEAN_UNIT + '#include "native/tensor/backend.h"\n', {},
     "reaches the dual-mode Tensor backend"),
    ("angle include of the dual-mode backend header",
     CLEAN_UNIT + "#include <native/tensor/backend.h>\n", {},
     "reaches the dual-mode Tensor backend"),
    ("quoted include of a Torch header",
     CLEAN_UNIT + '#include "torch/extension.h"\n', {},
     "reaches a Torch header"),
    ("angle include of a Torch header",
     CLEAN_UNIT + "#include <ATen/cuda/CUDAContext.h>\n", {},
     "reaches a Torch header"),
    ("backend header reached transitively by a quoted include",
     CLEAN_UNIT,
     {"backends/cuda/kernels/glu.cuh": '#include "native/tensor/backend.h"\n'},
     "reaches the dual-mode Tensor backend"),
    ("Torch header reached transitively by an angle include",
     CLEAN_UNIT,
     {"backends/cuda/kernels/glu.cuh": "#include <c10/cuda/CUDAGuard.h>\n"},
     "reaches a Torch header"),
    ("quoted Torch header with no space after include",
     CLEAN_UNIT + '#include"torch/extension.h"\n', {},
     "reaches a Torch header"),
    ("angle backend header with no space after include",
     CLEAN_UNIT + "#include<native/tensor/backend.h>\n", {},
     "reaches the dual-mode Tensor backend"),
]
FIXTURES = CLEAN_FIXTURES + NEGATIVE_FIXTURES


def run_fixture(body, extra_tree, expected):
    with tempfile.TemporaryDirectory() as tmp:
        csrc = os.path.join(tmp, "csrc")
        tree = {**BASE_TREE, **extra_tree,
                "backends/cuda/kernels/activation.cu": body}
        for path, text in tree.items():
            full = os.path.join(csrc, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as handle:
                handle.write(text)
        found = check(csrc, ("kernels/activation.cu",))
        if expected is None:
            return [f"unexpected {found}"] if found else []
        return [] if any(expected in failure for failure in found) else ["rule did not fire"]


def _parametrize(argnames, argvalues, ids=None):
    def decorate(func):
        return func
    return decorate


try:
    import pytest
    parametrize = pytest.mark.parametrize
except ImportError:
    parametrize = _parametrize


@parametrize("body,extra_tree,expected",
             [(body, extra, expected) for _, body, extra, expected in FIXTURES],
             ids=[name for name, _, _, _ in FIXTURES])
def test_rule(body, extra_tree, expected):
    assert run_fixture(body, extra_tree, expected) == []


def main() -> int:
    failures = check(CSRC, KERNEL_UNITS)
    fixture_failures = [f"{name}: {why}" for name, body, extra, expected in FIXTURES
                        for why in run_fixture(body, extra, expected)]
    for failure in failures + fixture_failures:
        print(failure)
    if failures or fixture_failures:
        print(f"cuda kernel torch-free: {len(failures) + len(fixture_failures)} problem(s)")
        return 1
    print(f"cuda kernel torch-free: OK ({len(KERNEL_UNITS)} kernel unit, "
          f"{len(CLEAN_FIXTURES)} clean fixture, "
          f"{len(NEGATIVE_FIXTURES)} negative controls fire)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
