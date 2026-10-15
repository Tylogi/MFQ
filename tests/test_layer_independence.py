#!/usr/bin/env python3
"""Shared layers must not name a backend or a model family.

Rename-stable by construction: every check forbids a token instead of pinning
one.  Behavior lives in the C++ Interface tests and real-weight regressions.

Run: python3 tests/test_layer_independence.py
"""
import re
from pathlib import Path

CSRC = Path(__file__).resolve().parents[1] / "csrc"
SHARED = ["core", "scheduler", "transport", "engine", "models"]
BACKEND_TOKEN = re.compile(r"mfq::(cuda|metal)\b|mfq_tensor_backend\b|backends/")
FAMILY_TOKEN = re.compile(
    r"\b(qwen|deepseek|gemma|glm|minicpm|dspark|engram|flash_next)\b", re.IGNORECASE)
SOURCE_SUFFIXES = {".h", ".hpp", ".cpp", ".cu", ".cuh", ".tpp"}
# engine/<module> holds mfq-engine's private headers; the shared model layer
# composes families through the public engine interface instead.
ENGINE_MODULES = {"requests", "batch", "state", "executor", "text", "src"}
# Shared files that still name a family, listed one by one.  Delete an entry
# when its consumer moves behind a capability.
FAMILY_EXCEPTIONS = {
    # ReasoningFormat values are wire-visible protocol names, not a model chain.
    CSRC / "engine/include/engine.h",
    # Media preprocessing and chat-placeholder geometry still dispatch per
    # family; that belongs to the family, and moving it is the open item.
    CSRC / "engine/src/processor.cpp",
    CSRC / "engine/src/media.cpp",
}


def sources(module):
    for path in sorted((CSRC / module).rglob("*")):
        if path.suffix in SOURCE_SUFFIXES:
            yield path


def code_lines(path):
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith(("//", "*", "/*")):
            yield lineno, line


def main() -> int:
    failures = []
    for module in SHARED:
        for path in sources(module):
            for lineno, line in code_lines(path):
                hit = BACKEND_TOKEN.search(line)
                if hit:
                    failures.append(f"{path}:{lineno}: shared layer names a backend: {line.strip()}")
                if module == "engine" and path not in FAMILY_EXCEPTIONS:
                    hit = FAMILY_TOKEN.search(line)
                    if hit:
                        failures.append(f"{path}:{lineno}: shared engine names a model family: {line.strip()}")
    model_dir = CSRC / "engine/model"
    if model_dir.is_dir():
        include = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
        for path in sources("engine/model"):
            for lineno, line in code_lines(path):
                match = include.match(line)
                if match and match.group(1).split("/")[0] in ENGINE_MODULES:
                    failures.append(f"{path}:{lineno}: shared model layer reaches an engine private header: {line.strip()}")
    for failure in failures:
        print(failure)
    if failures:
        print(f"layer independence: {len(failures)} violation(s)")
        return 1
    print(f"layer independence: OK ({len(SHARED)} shared modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
