#!/usr/bin/env python3
"""One place that knows how an include line turns into a real file.

The dependency checks (engine boundary, cuda include graph, layer independence)
all need the same two things: the include directories the CMake targets actually
add, and resolution of a bare include to the file it names.  Keeping that in one
module is what stops the checks from drifting apart, and engine modules are
discovered from the tree so a new module cannot be forgotten by a stale list.
"""
import os
import re

SOURCE_EXT = (".cpp", ".cu", ".c", ".cuh", ".h", ".hpp")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


def engine_modules(csrc):
    """Private engine module directories, read from the tree."""
    engine = os.path.join(csrc, "engine")
    if not os.path.isdir(engine):
        return []
    return sorted(d for d in os.listdir(engine)
                  if os.path.isdir(os.path.join(engine, d)) and d != "include")


def include_roots(csrc):
    """Directories a bare include can resolve against, per the CMake targets."""
    engine = os.path.join(csrc, "engine")
    roots = [csrc, os.path.join(engine, "include"), os.path.join(csrc, "core", "include")]
    roots += [os.path.join(engine, d) for d in engine_modules(csrc)]
    for backend in ("cuda", "metal", "cpu"):
        roots += [os.path.join(csrc, "backends", backend),
                  os.path.join(csrc, "backends", backend, "include")]
    roots += [os.path.join(csrc, "backends", "cpu", layer, "include")
              for layer in ("native", "execution")]
    roots += [os.path.join(csrc, "components", "tokenizer"),
              os.path.join(csrc, "components", "tokenizer", "include"),
              os.path.join(csrc, "models")]
    return [r for r in roots if os.path.isdir(r)]


def sources(*roots):
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.endswith(SOURCE_EXT):
                    yield os.path.join(dirpath, name)


def includes(path):
    try:
        with open(path, errors="ignore") as handle:
            for lineno, line in enumerate(handle, 1):
                match = INCLUDE_RE.match(line)
                if match:
                    yield lineno, match.group(1)
    except OSError:
        return


def resolve(raw, including_file, roots):
    """Every existing file a include line names; more than one means ambiguity."""
    found = []
    for candidate in [os.path.join(os.path.dirname(including_file), raw)] + \
                     [os.path.join(r, raw) for r in roots]:
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate) and candidate not in found:
            found.append(candidate)
    return found


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
