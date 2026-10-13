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


def sources(module):
    for path in sorted((CSRC / module).rglob("*")):
        if path.suffix in SOURCE_SUFFIXES:
            yield path


def main() -> int:
    failures = []
    for module in SHARED:
        for path in sources(module):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith(("//", "*", "/*")):
                    continue
                hit = BACKEND_TOKEN.search(line)
                if hit:
                    failures.append(f"{path}:{lineno}: shared layer names a backend: {line.strip()}")
        # engine/src still validates per-family media and chat-placeholder
        # geometry; that belongs to the family, and moving it is the open item.
        if module == "engine":
            for path in sources(module):
                if path.parent.name != "internal":
                    continue
                for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if line.lstrip().startswith(("//", "*", "/*")):
                        continue
                    hit = FAMILY_TOKEN.search(line)
                    if hit:
                        failures.append(f"{path}:{lineno}: shared engine names a model family: {line.strip()}")
    for failure in failures:
        print(failure)
    if failures:
        print(f"layer independence: {len(failures)} violation(s)")
        return 1
    print(f"layer independence: OK ({len(SHARED)} shared modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
