#!/usr/bin/env python3
"""Shared inference-engine include-boundary check.

engine/<module> headers are PRIVATE to mfq-engine, but the CUDA and Metal
backends and the test suite add those directories to their include path and
include the headers bare.  That is the only way a backend can drive the shared
generation loop, so the arrangement is allowed -- but only for them.

Rules:
  1. engine/** never includes backends/**.
  2. engine private module headers are included only from engine/, backends/
     and tests/.
  3. Inside engine, modules only point one way (FORBIDDEN below): the request
     table, the Planner, the Store rules and the text layer never reach the
     Executor driver.
  4. A backend's device implementation (ops, kernels, device primitives,
     execution) never reaches the shared execution driver or the Store rules;
     backend storage is the Store's physical implementation, so it may use the
     shared state rules.  Listed exceptions carry the debt that is still open.
  5. The engine header graph is a DAG, and every private header has a consumer.
  6. A bare include never names two different files, which is what makes the
     bare-include convention safe.

Run: python3 cuda/tests/test_engine_include_graph.py [--verbose]
"""
import collections
import os
import re
import sys
import tempfile

CSRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "csrc"))
ENGINE = os.path.join(CSRC, "engine")
MODULES = ("requests", "batch", "state", "executor", "text", "src")
SOURCE_EXT = (".cpp", ".cu", ".c", ".h")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
ALLOWED_CONSUMERS = ("engine", "backends", "tests")

# "<includer module>": engine headers it must never pull in.  The Executor
# driver and the shared request flow own execution; a Store, the Planner, the
# request table or the text layer that depended on them could not be reused by
# a backend that executes differently.
FORBIDDEN = {
    "requests": ("executor/driver.h", "executor/flow.h"),
    "batch": ("executor/driver.h", "executor/flow.h"),
    "text": ("executor/driver.h", "executor/flow.h", "state/"),
    "state": ("executor/", "requests/", "text/", "src/"),
    "executor": ("src/",),
}
# Backend device code may use shared value/policy headers (sampling rules,
# request facts) but not the execution driver, the Stores or the Engine
# composition.  storage/ is the Store's physical implementation, so the shared
# state rules are part of its contract.
DEVICE_FORBIDDEN = ("executor/", "state/", "src/", "requests/table.h", "requests/inference.h")
STORAGE_FORBIDDEN = ("executor/", "requests/table.h", "requests/inference.h")
DEVICE_DIRS = ("ops", "kernels", "native", "device", "execution")
# (backend file, engine header) still open; delete the entry when the consumer moves.
BACKEND_EXCEPTIONS = {
    ("backends/cuda/storage/session/text_cache.cpp", "src/settings.h"),
}


def sources(root):
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(SOURCE_EXT):
                yield os.path.join(dirpath, name)


def includes(path):
    with open(path, errors="ignore") as handle:
        for lineno, line in enumerate(handle, 1):
            match = INCLUDE_RE.match(line)
            if match:
                yield lineno, match.group(1)


def rel(path, base=CSRC):
    return os.path.relpath(path, base)


def include_roots(csrc):
    """Directories a bare include can resolve against, per the CMake targets."""
    roots = [csrc, os.path.join(csrc, "core", "include"),
             os.path.join(csrc, "engine", "include")]
    roots += [os.path.join(csrc, "engine", d) for d in MODULES]
    for backend in ("cuda", "metal", "cpu"):
        roots.append(os.path.join(csrc, "backends", backend))
        roots.append(os.path.join(csrc, "backends", backend, "include"))
    roots += [os.path.join(csrc, "components", "tokenizer"),
              os.path.join(csrc, "components", "tokenizer", "include"),
              os.path.join(csrc, "models")]
    return [r for r in roots if os.path.isdir(r)]


class Graph:
    """Include graph over one csrc-shaped tree."""

    def __init__(self, csrc):
        self.csrc = csrc
        self.engine = os.path.join(csrc, "engine")
        self.private = [os.path.join(self.engine, d) for d in MODULES]
        self.private_headers = sorted(
            p for d in self.private for p in sources(d) if p.endswith(".h"))
        self.engine_headers = {p for p in sources(self.engine) if p.endswith(".h")}
        self.roots = include_roots(csrc)

    def resolve(self, raw, including_file):
        candidates = [os.path.join(os.path.dirname(including_file), raw)]
        candidates += [os.path.join(r, raw) for r in self.roots]
        found = []
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.isfile(c) and c not in found:
                found.append(c)
        return found

    def module(self, path):
        for d in MODULES:
            if path.startswith(os.path.join(self.engine, d) + os.sep):
                return d
        return None

    def backend_rule(self, src):
        """Forbidden engine headers for a backend file, by which directory it sits in."""
        parts = rel(src, self.csrc).split(os.sep)
        if len(parts) < 3 or parts[0] != "backends":
            return None
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
        for src in sources(self.csrc):
            from_engine = rel(src, self.csrc).startswith("engine" + os.sep)
            forbidden = self.backend_rule(src)
            if not from_engine and forbidden is None:
                continue
            for lineno, raw in includes(src):
                hits = self.resolve(raw, src)
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
                    if from_engine and rel(target, self.csrc).startswith("backends" + os.sep):
                        violations.append((src, lineno, raw,
                            "engine may not include a backend header"))
                    continue
                if target in self.private_headers:
                    includers[target].add(src)
                    consumer = rel(src, self.csrc).split(os.sep)[0]
                    if consumer not in ALLOWED_CONSUMERS:
                        violations.append((src, lineno, raw,
                            "engine private headers may not be included here"))
                name = engine_relative.split(os.sep, 1)[1]
                if from_engine:
                    owner = self.module(src)
                    if owner:
                        edges[src].add(target)
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


def find_cycles(edges):
    cycles, color, stack = [], {}, []

    def visit(node):
        color[node] = 1
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            if color.get(nxt, 0) == 1:
                cycles.append(stack[stack.index(nxt):] + [nxt])
            elif color.get(nxt, 0) == 0:
                visit(nxt)
        stack.pop()
        color[node] = 2

    for node in sorted(edges):
        if color.get(node, 0) == 0:
            visit(node)
    return cycles


def check_tree(csrc):
    edges, includers, violations, orphans = Graph(csrc).check()
    return edges, includers, violations, orphans, find_cycles(edges)


def fixtures():
    """Small positive and negative trees; a rule that never fires is not a rule."""
    cases = [
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
        ("Store rule may not depend on the Executor", {
            "engine/state/sessions.h": '#include "driver.h"\n',
            "engine/executor/driver.h": "struct D {};\n",
        }, "engine/state may not include executor/"),
        ("Planner may not depend on the Executor driver", {
            "engine/batch/plan.h": '#include "driver.h"\n',
            "engine/executor/driver.h": "struct D {};\n",
        }, "engine/batch may not include executor/driver.h"),
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
    failures = []
    for name, files, expected in cases:
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
            if expected is None and found:
                failures.append(f"{name}: unexpected {found}")
            if expected and not any(expected in f for f in found):
                failures.append(f"{name}: rule did not fire")
    return failures


def main():
    verbose = "--verbose" in sys.argv
    graph = Graph(CSRC)
    edges, includers, violations, orphans, cycles = check_tree(CSRC)
    if verbose:
        for src in sorted(edges):
            print(f"{rel(src):34s} -> {', '.join(sorted(rel(t) for t in edges[src]))}")
        print()
        for header in graph.private_headers:
            consumers = sorted({rel(p).split(os.sep)[0] for p in includers[header]})
            print(f"{os.path.basename(header):16s} <- {', '.join(consumers) or 'nobody'}")
    fixture_failures = fixtures()
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
