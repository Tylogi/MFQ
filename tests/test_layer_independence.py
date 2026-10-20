#!/usr/bin/env python3
"""Shared layers must not name a backend, and generic engine code must not name a family.

The rules are about responsibility, not about directory names:

  1. core, scheduler, transport, engine, models and file I/O never name a backend
     (mfq::cuda, mfq::metal, mfq_tensor_backend, backends/ includes).
  2. A model family may be named in engine/model/<family>/ and in models/, where
     the family lives.  The generic request table, the Planner, both Stores, the
     text layer and the execution flow may not branch on a family.
  3. The shared model layer may use the resource interfaces it is handed --
     engine/state and engine/weights -- but never the Executor, the request
     table, the text layer or the Engine composition.
  4. Include rules are judged on the file the include resolves to, not on how
     the include is spelled, using the same resolution as the engine boundary
     check.

Rename-stable by construction: every check forbids a token or a resolved target
instead of pinning one name.  Behavior lives in the C++ Interface tests and
real-weight regressions.

Run: python3 tests/test_layer_independence.py
     pytest cuda/tests/test_layer_independence.py
"""
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from include_graph import include_roots, includes, resolve, sources

try:
    import pytest
except ImportError:
    pytest = None

REPO = Path(__file__).resolve().parents[1]
CSRC = REPO / "csrc"
SHARED = ("core", "scheduler", "transport", "engine", "models", "components/io")
BACKEND_TOKEN = re.compile(r"mfq::(cuda|metal)\b|mfq_tensor_backend\b|backends/")
# Family stems anywhere in an identifier: Qwen35, qwen3_5 and is_qwen35 all hit.
FAMILY_TOKEN = re.compile(
    r"(qwen|deepseek|gemma|glm|minicpm|dspark|engram|flash[_-]?next)", re.IGNORECASE)
# The model layer may use the state and weight interfaces; the Executor and the
# shared request/text/composition code are not resources a family composes.
MODEL_FORBIDDEN = ("executor/", "requests/", "batch/", "text/", "src/")
# Shared files that still name a family, listed one by one.  Delete an entry
# when its consumer moves behind a capability.
FAMILY_EXCEPTIONS = (
    # ReasoningFormat values are wire-visible protocol names, not a model chain.
    "engine/include/engine.h",
    # Media preprocessing and chat-placeholder geometry still dispatch per
    # family; that belongs to the family, and moving it is the open item.
    "engine/src/processor.cpp",
    "engine/src/media.cpp",
)


def code_lines(path):
    for lineno, line in enumerate(Path(path).read_text(errors="ignore").splitlines(), 1):
        if not line.lstrip().startswith(("//", "*", "/*")):
            yield lineno, line


def check_tree(csrc):
    csrc = str(csrc)
    roots = include_roots(csrc)
    failures = []
    for path in sources(csrc):
        relative = os.path.relpath(path, csrc).replace(os.sep, "/")
        module = relative.split("/")[0]
        if not relative.startswith(tuple(owner + "/" for owner in SHARED)):
            continue
        under_model = relative.startswith("engine/model/")
        under_io = relative.startswith("components/io/")
        exempt = relative in FAMILY_EXCEPTIONS
        for lineno, line in code_lines(path):
            if BACKEND_TOKEN.search(line):
                failures.append(f"{relative}:{lineno}: shared layer names a backend: {line.strip()}")
            if module == "engine" and not under_model and not exempt and FAMILY_TOKEN.search(line):
                failures.append(f"{relative}:{lineno}: generic engine code names a model family: {line.strip()}")
        if not (under_model or under_io):
            continue
        for lineno, raw in includes(path):
            hits = resolve(raw, path, roots)
            if not hits:
                continue
            target = os.path.relpath(hits[0], csrc).replace(os.sep, "/")
            if under_io:
                if (target.startswith(("core/", "models/", "model_source/", "engine/", "formats/")) or
                        target.startswith("backends/") and not target.startswith("backends/cpu/native/")):
                    failures.append(f"{relative}:{lineno}: file I/O reaches a higher layer: {raw}")
                continue
            if not target.startswith("engine/"):
                if target.startswith("backends/"):
                    failures.append(f"{relative}:{lineno}: shared model layer includes a backend header: {raw}")
                continue
            name = target.split("/", 1)[1]
            if name.startswith(MODEL_FORBIDDEN):
                failures.append(f"{relative}:{lineno}: shared model layer may not reach {name}")
    return failures


def test_layer_independence():
    failures = check_tree(CSRC)
    assert not failures, "\n".join(failures)


# A rule with nothing to check it against is not a rule.
FIXTURES = [
    ("file I/O may use host thread primitives", {
        "components/io/read.cpp": '#include "mfq/host_parallel.h"\n',
        "backends/cpu/native/include/mfq/host_parallel.h": "struct Pool {};\n",
    }, None),
    ("file I/O may not name a backend", {
        "components/io/read.cpp": "void f() { mfq::cuda::run(); }\n",
    }, "shared layer names a backend"),
    ("file I/O may not reach a model source", {
        "components/io/read.cpp": '#include "mfq/model_source/model_source.h"\n',
        "core/include/mfq/model_source/model_source.h": "struct Source {};\n",
    }, "file I/O reaches a higher layer"),
    ("file I/O may not reach a backend tensor by a bare include", {
        "components/io/read.cpp": '#include "tensor.h"\n',
        "backends/cuda/tensor.h": "struct Tensor {};\n",
    }, "file I/O reaches a higher layer"),
    ("model layer may use the state and weight interfaces", {
        "engine/model/qwen35/forward.h": '#include "pages.h"\n#include "slots.h"\n',
        "engine/state/pages.h": "struct Pages {};\n",
        "engine/weights/slots.h": "struct Slots {};\n",
    }, None),
    ("model layer may not reach the Executor", {
        "engine/model/qwen35/forward.h": '#include "driver.h"\n',
        "engine/executor/driver.h": "struct D {};\n",
    }, "shared model layer may not reach executor/driver.h"),
    ("model layer may not reach the request table", {
        "engine/model/qwen35/forward.h": '#include "table.h"\n',
        "engine/requests/table.h": "struct Row {};\n",
    }, "shared model layer may not reach requests/table.h"),
    ("model layer may not include a backend header", {
        "engine/model/qwen35/forward.h": '#include "tensor.h"\n',
        "backends/cuda/tensor.h": "struct T {};\n",
    }, "shared model layer includes a backend header"),
    ("generic request code may not branch on a family", {
        "engine/requests/table.h": "bool is_qwen35() { return true; }\n",
    }, "generic engine code names a model family"),
    ("a Store rule may not name a family", {
        "engine/weights/ranker.h": "int qwen35_slots() { return 1; }\n",
    }, "generic engine code names a model family"),
    ("shared code may not name a backend", {
        "scheduler/side.cpp": "void f() { mfq::cuda::run(); }\n",
    }, "shared layer names a backend"),
    ("a family may be named where the family lives", {
        "engine/model/qwen35/config.h": "struct Qwen35Config {};\n",
    }, None),
]


def run_fixture(files, expected):
    with tempfile.TemporaryDirectory() as tmp:
        for path, body in files.items():
            full = os.path.join(tmp, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as handle:
                handle.write(body)
        found = check_tree(tmp)
        if expected is None:
            return [f"unexpected {found}"] if found else []
        return [] if any(expected in f for f in found) else ["rule did not fire"]


def _parametrize(argnames, argvalues, ids=None):
    def decorate(func):
        return func
    return decorate


parametrize = pytest.mark.parametrize if pytest else _parametrize


@parametrize("files,expected", [(files, expected) for _, files, expected in FIXTURES],
             ids=[name for name, _, _ in FIXTURES])
def test_rule(files, expected):
    assert run_fixture(files, expected) == []


def main() -> int:
    failures = check_tree(CSRC)
    for failure in failures:
        print(failure)
    fixture_failures = [f"{name}: {why}" for name, files, expected in FIXTURES
                        for why in run_fixture(files, expected)]
    for failure in fixture_failures:
        print(failure)
    if failures or fixture_failures:
        print(f"layer independence: {len(failures) + len(fixture_failures)} problem(s)")
        return 1
    print(f"layer independence: OK ({len(SHARED)} shared modules, "
          f"{len(FIXTURES)} rules fire)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
