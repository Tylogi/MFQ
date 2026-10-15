#!/usr/bin/env python3
"""Shared inference-engine include-boundary check.

engine/internal is PRIVATE in engine/CMakeLists.txt, but the CUDA and Metal
backends and the test suite add that directory to their include path and
include its headers bare.  That is the only way a backend can drive the
shared generation loop, so the arrangement is allowed -- but only for them.

Rules:
  1. engine/** never includes backends/** (the engine is backend agnostic).
  2. engine private module headers (internal/, request_table/, batch_planner/) are
     included only from engine/, backends/ and tests/.
  3. The engine/internal header graph is a DAG.
  4. Every engine/internal/*.h has at least one includer.

Run: python3 cuda/tests/test_engine_include_graph.py [--verbose]
"""
import collections
import os
import re
import sys

CSRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "csrc"))
ENGINE = os.path.join(CSRC, "engine")
INTERNAL = os.path.join(ENGINE, "internal")
# Private engine module directories. Backends and tests add all of them to the
# include path and include their headers bare, so they share one rule set.
INTERNAL_DIRS = [os.path.join(ENGINE, d)
                 for d in ("internal", "request_table", "batch_planner")]
SOURCE_EXT = (".cpp", ".cu", ".c", ".h")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
ALLOWED_CONSUMERS = ("engine", "backends", "tests")


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


def rel(path):
    return os.path.relpath(path, CSRC)


def resolve(raw, including_file):
    """Resolve a quoted include the way the compiler does for these targets."""
    candidates = [os.path.join(os.path.dirname(including_file), raw)]
    for base in (CSRC, ENGINE, *INTERNAL_DIRS, os.path.join(ENGINE, "include")):
        candidates.append(os.path.join(base, raw))
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            return c
    return None


def collect():
    internal_headers = sorted(
        p for d in INTERNAL_DIRS for p in sources(d) if p.endswith(".h"))
    by_name = {os.path.basename(p): p for p in internal_headers}
    edges = collections.defaultdict(set)
    includers = collections.defaultdict(set)
    violations = []
    for src in sources(CSRC):
        for lineno, raw in includes(src):
            target = resolve(raw, src)
            if target is None:
                continue
            if target in by_name:
                target = by_name[target]
            if target not in internal_headers:
                continue
            includers[target].add(src)
            consumer = rel(src).split(os.sep)[0]
            if consumer not in ALLOWED_CONSUMERS:
                violations.append(
                    (src, lineno, raw, "engine private headers may not be included here"))
            if os.path.dirname(src) in INTERNAL_DIRS and src != target:
                edges[src].add(target)
    for src in sources(os.path.join(ENGINE)):
        for lineno, raw in includes(src):
            target = resolve(raw, src)
            if target and rel(target).startswith("backends" + os.sep):
                violations.append((src, lineno, raw, "engine may not include a backend header"))
    return internal_headers, edges, includers, violations


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


def test_engine_include_boundary():
    """engine is backend agnostic, engine/internal is for backends and tests only."""
    internal_headers, edges, includers, violations = collect()
    assert not violations, "boundary violations:\n" + "\n".join(
        f"{rel(src)}:{lineno}: {raw} [{why}]" for src, lineno, raw, why in violations)
    cycles = find_cycles(edges)
    assert not cycles, "internal header cycles: " + "; ".join(
        " -> ".join(rel(p) for p in c) for c in cycles)
    orphans = [rel(p) for p in internal_headers if not includers[p]]
    assert not orphans, f"engine/internal headers nobody includes: {orphans}"


def main():
    verbose = "--verbose" in sys.argv
    internal_headers, edges, includers, violations = collect()
    if verbose:
        for src in sorted(edges):
            print(f"{rel(src):40s} -> {', '.join(sorted(rel(t) for t in edges[src]))}")
        print()
        for header in internal_headers:
            consumers = sorted({rel(p).split(os.sep)[0] for p in includers[header]})
            print(f"{os.path.basename(header):32s} <- {', '.join(consumers) or 'nobody'}")
    ok = True
    if violations:
        ok = False
        print(f"boundary violations ({len(violations)}):")
        for src, lineno, raw, why in violations:
            print(f"  {rel(src)}:{lineno}: {raw}  [{why}]")
    cycles = find_cycles(edges)
    if cycles:
        ok = False
        print(f"internal header cycles ({len(cycles)}):")
        for cycle in cycles:
            print("  " + " -> ".join(rel(p) for p in cycle))
    orphans = [p for p in internal_headers if not includers[p]]
    if orphans:
        ok = False
        print(f"engine/internal headers nobody includes ({len(orphans)}):")
        for p in orphans:
            print(f"  {rel(p)}")
    if ok:
        print(f"engine include boundary: OK ({len(internal_headers)} internal headers, "
              "engine is backend agnostic, DAG)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
