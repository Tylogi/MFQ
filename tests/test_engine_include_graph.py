#!/usr/bin/env python3
"""Shared inference-engine include-boundary check.

engine/<module> headers are PRIVATE to mfq-engine, but the CUDA and Metal
backends, the test suite and the benchmarks add those directories to their
include path and include the headers bare.  That is the only way a backend can
drive the shared generation loop, so the arrangement is allowed -- but only for
them.

Rules:
  1. engine/** never includes backends/**.
  2. engine private module headers are included only from engine/, backends/,
     tests/ and bench/.  Every scanned file is judged, including core/,
     scheduler/ and transport/, whichever way the include is spelled.
  3. Inside engine, modules only point one way (FORBIDDEN below): the request
     table, the Planner, the Store rules and the text layer never reach the
     Executor driver.
  4. A backend's device implementation (ops, kernels, native primitives,
     execution) never reaches the shared execution driver or either Store;
     backend storage is the Store's physical implementation, so it may use the
     shared state and weight rules.  Listed exceptions carry open debt.
  5. The engine header graph -- public, private and model headers alike -- is a
     DAG, and every private header has a consumer.
  6. A bare include never names two different files, which is what makes the
     bare-include convention safe.

Run: python3 cuda/tests/test_engine_include_graph.py [--verbose]
     pytest cuda/tests/test_engine_include_graph.py
"""
import collections
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from include_graph import (engine_modules, find_cycles, include_roots, includes,
                           resolve, sources)

try:
    import pytest
except ImportError:
    pytest = None

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CSRC = os.path.join(REPO, "csrc")
BENCH = os.path.join(REPO, "bench")
ALLOWED_CONSUMERS = ("engine", "backends", "tests", "bench")

# "<includer module>": engine headers it must never pull in.  The Executor
# driver and the shared request flow own execution; a Store, the Planner, the
# request table or the text layer that depended on them could not be reused by
# a backend that executes differently.
FORBIDDEN = {
    "requests": ("executor/driver.h", "executor/flow.h"),
    "batch": ("executor/driver.h", "executor/flow.h"),
    "text": ("executor/driver.h", "executor/flow.h", "state/"),
    "state": ("executor/", "requests/", "text/", "src/"),
    "weights": ("executor/", "requests/", "text/", "src/", "state/", "batch/"),
    "executor": ("src/",),
}
# Backend device code may use shared value/policy headers (sampling rules,
# request facts) but not the execution driver, either Store or the Engine
# composition.  storage/ is the Store's physical implementation, so the shared
# state and weight rules are part of its contract.
DEVICE_FORBIDDEN = ("executor/", "state/", "weights/", "src/",
                    "requests/table.h", "requests/inference.h")
STORAGE_FORBIDDEN = ("executor/", "requests/table.h", "requests/inference.h")
DEVICE_DIRS = ("ops", "kernels", "native", "device", "execution")
# (backend file, engine header) still open; delete the entry when the consumer moves.
BACKEND_EXCEPTIONS = {
    ("backends/cuda/storage/session/text_cache.cpp", "src/settings.h"),
}


def rel(path, base=CSRC):
    return os.path.relpath(path, base)


class Graph:
    """Include graph over one csrc-shaped tree plus the top-level benchmarks."""

    def __init__(self, csrc, bench=None):
        self.csrc = csrc
        self.bench = bench if bench and os.path.isdir(bench) else None
        self.engine = os.path.join(csrc, "engine")
        self.modules = engine_modules(csrc)
        self.private_headers = sorted(
            p for d in self.modules
            for p in sources(os.path.join(self.engine, d)) if p.endswith(".h"))
        self.engine_headers = {p for p in sources(self.engine) if p.endswith(".h")}
        self.roots = include_roots(csrc)

    def module(self, path):
        for d in self.modules:
            if path.startswith(os.path.join(self.engine, d) + os.sep):
                return d
        return None

    def area(self, path):
        if self.bench and path.startswith(self.bench + os.sep):
            return "bench"
        return rel(path, self.csrc).split(os.sep)[0]

    def backend_rule(self, src):
        """Forbidden engine headers for a backend file, by which directory it sits in."""
        parts = rel(src, self.csrc).split(os.sep)
        if len(parts) < 3 or parts[0] != "backends":
            return ()
        kind = parts[2]
        if kind in DEVICE_DIRS:
            return DEVICE_FORBIDDEN
        if kind == "storage":
            return STORAGE_FORBIDDEN
        return ()

    def check(self):
        edges = collections.defaultdict(set)
        includers = collections.defaultdict(set)
        violations = []
        for src in sources(self.csrc, *( (self.bench,) if self.bench else () )):
            from_engine = rel(src, self.csrc).startswith("engine" + os.sep)
            forbidden = self.backend_rule(src)
            for lineno, raw in includes(src):
                hits = resolve(raw, src, self.roots)
                if len(hits) > 1 and not raw.startswith(("mfq/", "core/")):
                    violations.append((src, lineno, raw,
                        "bare include is ambiguous: " +
                        ", ".join(rel(h, self.csrc) for h in hits)))
                    continue
                if not hits:
                    continue
                target = hits[0]
                if not target.endswith(".h"):
                    continue
                engine_relative = rel(target, self.csrc)
                if not engine_relative.startswith("engine" + os.sep):
                    if from_engine and engine_relative.startswith("backends" + os.sep):
                        violations.append((src, lineno, raw,
                            "engine may not include a backend header"))
                    continue
                if target in self.private_headers:
                    includers[target].add(src)
                    if self.area(src) not in ALLOWED_CONSUMERS:
                        violations.append((src, lineno, raw,
                            "engine private headers may not be included here"))
                if from_engine and src in self.engine_headers:
                    edges[src].add(target)
                name = engine_relative.split(os.sep, 1)[1]
                if from_engine:
                    owner = self.module(src)
                    if owner:
                        for bad in FORBIDDEN.get(owner, ()):
                            if name.startswith(bad.replace("/", os.sep)):
                                violations.append((src, lineno, raw,
                                    f"engine/{owner} may not include {bad}"))
                elif forbidden:
                    for bad in forbidden:
                        if name.startswith(bad.replace("/", os.sep)):
                            pair = (rel(src, self.csrc), name)
                            if pair not in BACKEND_EXCEPTIONS:
                                violations.append((src, lineno, raw,
                                    "backend device implementation may not include " + bad))
        orphans = [p for p in self.private_headers if not includers[p]]
        return edges, includers, violations, orphans


def check_tree(csrc, bench=None):
    edges, includers, violations, orphans = Graph(csrc, bench).check()
    return edges, includers, violations, orphans, find_cycles(edges)


# Small positive and negative trees: a rule that never fires is not a rule.
FIXTURES = [
    ("backend assembly may use the shared templates", {
        "backends/cuda/engine.cpp": '#include "instance.h"\n',
        "engine/src/instance.h": '#include "table.h"\n',
        "engine/requests/table.h": "struct Row {};\n",
    }, None),
    ("shared template may not include a backend header", {
        "engine/executor/driver.h": '#include "tensor.h"\n',
        "backends/cuda/tensor.h": "struct T {};\n",
    }, "engine may not include a backend header"),
    ("backend ops may not include the shared execution driver", {
        "backends/cuda/ops/linear.cpp": '#include "driver.h"\n',
        "engine/executor/driver.h": "struct D {};\n",
    }, "backend device implementation may not include executor/"),
    ("backend ops may not include the weight residency rule", {
        "backends/cuda/ops/probe.cpp": '#include "slots.h"\n',
        "engine/weights/slots.h": "struct S {};\n",
    }, "backend device implementation may not include weights/"),
    ("Store rule may not depend on the Executor", {
        "engine/state/sessions.h": '#include "driver.h"\n',
        "engine/executor/driver.h": "struct D {};\n",
    }, "engine/state may not include executor/"),
    ("weight residency rule may not depend on the Executor", {
        "engine/weights/slots.h": '#include "driver.h"\n',
        "engine/executor/driver.h": "struct D {};\n",
    }, "engine/weights may not include executor/"),
    ("Planner may not depend on the Executor driver", {
        "engine/batch/plan.h": '#include "driver.h"\n',
        "engine/executor/driver.h": "struct D {};\n",
    }, "engine/batch may not include executor/driver.h"),
    ("core may not include an engine private header", {
        "core/side.cpp": '#include "table.h"\n',
        "engine/requests/table.h": "struct Row {};\n",
    }, "engine private headers may not be included here"),
    ("file I/O may not include an engine private header", {
        "components/io/read.cpp": '#include "table.h"\n',
        "engine/requests/table.h": "struct Row {};\n",
    }, "engine private headers may not be included here"),
    ("scheduler may not reach a private header by relative path", {
        "scheduler/side.cpp": '#include "../engine/requests/table.h"\n',
        "engine/requests/table.h": "struct Row {};\n",
    }, "engine private headers may not be included here"),
    ("transport may not reach the shared execution driver", {
        "transport/side.cpp": '#include "flow.h"\n',
        "engine/executor/flow.h": "struct F {};\n",
    }, "engine private headers may not be included here"),
    ("a public engine header cycle must be found", {
        "engine/include/engine.h": '#include "table.h"\n',
        "engine/requests/table.h": '#include "engine.h"\n',
    }, "cycle"),
    ("a real file cycle must fail", {
        "engine/requests/step.h": '#include "budget.h"\n',
        "engine/executor/budget.h": '#include "step.h"\n',
    }, "cycle"),
    ("two bare-includable headers with one name must fail", {
        "engine/text/emitter.h": "struct E {};\n",
        "backends/cuda/emitter.h": "struct E {};\n",
        "backends/cuda/engine.cpp": '#include "emitter.h"\n',
    }, "bare include is ambiguous"),
]


def run_fixture(files, expected):
    with tempfile.TemporaryDirectory() as tmp:
        for path, body in files.items():
            full = os.path.join(tmp, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as handle:
                handle.write(body)
        _, _, violations, orphans, cycles = check_tree(tmp)
        found = [why for _, _, _, why in violations] + \
                (["cycle"] if cycles else []) + \
                [f"orphan {os.path.basename(p)}" for p in orphans]
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


def test_engine_boundary():
    graph = Graph(CSRC, BENCH)
    _, _, violations, orphans, cycles = check_tree(CSRC, BENCH)
    problems = [f"{rel(src)}:{lineno}: {raw}  [{why}]"
                for src, lineno, raw, why in violations]
    problems += ["cycle: " + " -> ".join(rel(p) for p in cycle) for cycle in cycles]
    problems += [f"no consumer: {rel(p)}" for p in orphans]
    assert not problems, "\n".join(problems)


def main():
    verbose = "--verbose" in sys.argv
    graph = Graph(CSRC, BENCH)
    edges, includers, violations, orphans, cycles = check_tree(CSRC, BENCH)
    if verbose:
        for src in sorted(edges):
            print(f"{rel(src):34s} -> {', '.join(sorted(rel(t) for t in edges[src]))}")
        print()
        for header in graph.private_headers:
            consumers = sorted({graph.area(p) for p in includers[header]})
            print(f"{os.path.basename(header):16s} <- {', '.join(consumers) or 'nobody'}")
    ok = True
    if violations:
        ok = False
        print(f"boundary violations ({len(violations)}):")
        for src, lineno, raw, why in violations:
            print(f"  {rel(src)}:{lineno}: {raw}  [{why}]")
    if cycles:
        ok = False
        print(f"engine header cycles ({len(cycles)}):")
        for cycle in cycles:
            print("  " + " -> ".join(rel(p) for p in cycle))
    if orphans:
        ok = False
        print(f"engine headers nobody includes ({len(orphans)}):")
        for p in orphans:
            print(f"  {rel(p)}")
    fixture_failures = [f"{name}: {why}" for name, files, expected in FIXTURES
                        for why in run_fixture(files, expected)]
    if fixture_failures:
        ok = False
        print("rule self-test failed:")
        for failure in fixture_failures:
            print(f"  {failure}")
    if ok:
        print(f"engine include boundary: OK ({len(graph.private_headers)} private headers, "
              "engine is backend agnostic, module rules hold, DAG, rules fire)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
