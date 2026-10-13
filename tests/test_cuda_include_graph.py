#!/usr/bin/env python3
"""CUDA backend include-graph layering check.

Every cross-directory include inside csrc/backends/cuda must be written as
"<dir>/<name>", resolved against the backend root.  The directory graph must
stay a DAG in the documented order:

    native -> kernels -> ops -> storage -> models -> engine

Run: python3 cuda/tests/cuda_include_graph.py [--verbose]
"""
import os
import re
import sys
import collections

BACKEND_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "csrc", "backends/cuda"))
# Directories that hold the layered library.  Everything else under the
# backend root (apps/, commands/, diagnostics/, eval/, tests/) is a top-level
# consumer and may include anything below it.
DIRS = ["native", "kernels", "ops", "storage", "models", "engine"]
LAYERS = DIRS
RANK = {d: i for i, d in enumerate(LAYERS)}
SOURCE_EXT = (".cpp", ".cu", ".cuh", ".c", ".h", ".tpp")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


def layer_of(path):
    rel = os.path.relpath(path, BACKEND_ROOT)
    if rel.startswith(".."):
        return None
    head = rel.split(os.sep)[0]
    return head if head in LAYERS else None


def dir_of(path):
    rel = os.path.relpath(path, BACKEND_ROOT)
    return None if rel.startswith("..") else rel.split(os.sep)[0]


REPO_ROOT = os.path.normpath(os.path.join(BACKEND_ROOT, "..", "..", ".."))
CSRC_ROOT = os.path.normpath(os.path.join(BACKEND_ROOT, "..", ".."))

# Every directory the CUDA targets put on the include path.  Kept here so a
# first-party include that no longer resolves is reported instead of skipped.
INCLUDE_ROOTS = [
    REPO_ROOT,
    CSRC_ROOT,
    BACKEND_ROOT,
    os.path.join(CSRC_ROOT, "core"),
    os.path.join(CSRC_ROOT, "core", "include"),
    os.path.join(CSRC_ROOT, "models"),
    os.path.join(CSRC_ROOT, "engine", "internal"),
]

# Basenames of every header under the backend root.  An include of one of these
# that does not resolve is a broken path, not a system header.
FIRST_PARTY = {
    name
    for layer in DIRS
    for dirpath, _, files in os.walk(os.path.join(BACKEND_ROOT, layer))
    for name in files
    if name.endswith(SOURCE_EXT)
}


def resolve(raw, including_file):
    """Resolve a quoted include the way the compiler does for this backend."""
    candidates = [os.path.normpath(os.path.join(os.path.dirname(including_file), raw))]
    candidates.extend(os.path.normpath(os.path.join(root, raw)) for root in INCLUDE_ROOTS)
    for d in DIRS:
        candidates.append(os.path.normpath(os.path.join(BACKEND_ROOT, d, raw)))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def collect():
    edges = collections.defaultdict(set)
    violations = []
    for layer in DIRS:
        for dirpath, _, files in os.walk(os.path.join(BACKEND_ROOT, layer)):
            for name in files:
                if not name.endswith(SOURCE_EXT):
                    continue
                src = os.path.join(dirpath, name)
                src_layer = layer_of(src)
                with open(src, errors="ignore") as handle:
                    for lineno, line in enumerate(handle, 1):
                        match = INCLUDE_RE.match(line)
                        if not match:
                            continue
                        raw = match.group(1)
                        if raw.startswith("/"):
                            violations.append(
                                (src, lineno, raw, "absolute include"))
                            continue
                        relative = raw.startswith("../")
                        if relative:
                            violations.append(
                                (src, lineno, raw, "relative include; write \"<dir>/<name>\""))
                        target = resolve(raw, src)
                        if target is None:
                            if os.path.basename(raw) in FIRST_PARTY:
                                violations.append(
                                    (src, lineno, raw, "include does not resolve"))
                            continue  # system / third-party / generated header
                        dst_dir = dir_of(target)
                        if dst_dir is not None and dst_dir != dir_of(src) \
                                and "/" not in raw and not relative:
                            violations.append(
                                (src, lineno, raw,
                                 f"bare cross-directory include of {dst_dir}/{raw}"))
                        dst_layer = layer_of(target)
                        if dst_layer is None or dst_layer == src_layer:
                            continue
                        edges[src_layer].add(dst_layer)
    return edges, violations


def find_cycles(edges):
    cycles = []
    color = {}
    stack = []

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


def test_cuda_include_graph_is_a_dag():
    """Every cross-directory CUDA include is qualified and the layer graph is a DAG."""
    edges, violations = collect()
    assert not violations, "unqualified includes:\n" + "\n".join(
        f"{os.path.relpath(src, BACKEND_ROOT)}:{lineno}: {raw} [{why}]"
        for src, lineno, raw, why in violations)
    upward = sorted((s, d) for s in edges for d in edges[s] if RANK[s] < RANK[d])
    assert not upward, f"upward layer edges: {upward}"
    cycles = find_cycles(edges)
    assert not cycles, "layer cycles: " + "; ".join(" -> ".join(c) for c in cycles)


def main():
    verbose = "--verbose" in sys.argv
    edges, violations = collect()
    if verbose:
        for src in sorted(edges):
            print(f"{src} -> {', '.join(sorted(edges[src]))}")
    ok = True
    if violations:
        ok = False
        print(f"include style violations ({len(violations)}):")
        for src, lineno, raw, why in violations:
            print(f"  {os.path.relpath(src, BACKEND_ROOT)}:{lineno}: {raw}  [{why}]")
    cycles = find_cycles(edges)
    upward = [(s, d) for s in edges for d in edges[s] if RANK[s] < RANK[d]]
    if upward:
        ok = False
        print(f"upward layer edges ({len(upward)}):")
        for s, d in sorted(upward):
            print(f"  {s} -> {d}")
    if cycles:
        ok = False
        print(f"layer cycles ({len(cycles)}):")
        for cycle in cycles:
            print("  " + " -> ".join(cycle))
    if ok:
        print("cuda include graph: OK (DAG, all cross-directory includes qualified)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
