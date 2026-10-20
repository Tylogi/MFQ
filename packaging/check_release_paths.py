"""Reject local build paths in release files and frozen Python code."""

from __future__ import annotations

import argparse
import marshal
from pathlib import Path
from types import CodeType
import zipfile


class PathChecker:
    def __init__(self, prefixes: list[str]):
        self.needles: set[bytes] = set()
        for prefix in prefixes:
            normalized = prefix.rstrip("/\\").replace("\\", "/")
            for spelling in (normalized, normalized.replace("/", "\\")):
                for escaped in (spelling, spelling.replace("\\", "\\\\")):
                    for encoding in ("utf-8", "utf-16le"):
                        self.needles.add(escaped.lower().encode(encoding))
        self.overlap = max(map(len, self.needles), default=1) - 1
        self.failures: set[str] = set()
        self.files = 0
        self.python_modules = 0

    def check_bytes(self, data: bytes, label: str) -> None:
        lowered = data.lower()
        if any(needle in lowered for needle in self.needles):
            self.failures.add(label)

    def check_code(self, code: CodeType, label: str) -> None:
        # Marshal is uncompressed and includes filenames and nested constants,
        # including tuples and strings shared by multiple nested code objects.
        self.check_bytes(marshal.dumps(code), label)

    def check_file(self, path: Path) -> None:
        self.files += 1
        label = str(path)
        previous = b""
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                data = previous + block
                self.check_bytes(data, label)
                previous = data[-self.overlap:] if self.overlap else b""
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    data = archive.read(name)
                    member = f"{label}!{name}"
                    self.check_bytes(data, member)
                    if name.endswith(".pyc"):
                        code = marshal.loads(data[16:])
                        if isinstance(code, CodeType):
                            self.check_code(code, member)

    def check_python_cli(self, path: Path) -> None:
        from PyInstaller.archive.readers import CArchiveReader

        archive = CArchiveReader(str(path))
        for name, entry in archive.toc.items():
            if entry[-1] == "z":
                pyz = archive.open_embedded_archive(name)
                for module in pyz.toc:
                    code = pyz.extract(module)
                    self.python_modules += 1
                    if isinstance(code, CodeType):
                        self.check_code(code, f"{path.name}:{module}")
                    elif isinstance(code, bytes):
                        self.check_bytes(code, f"{path.name}:{module}")
            elif entry[-1] in ("s", "m", "M"):
                data = archive.extract(name)
                code = marshal.loads(data)
                if isinstance(code, CodeType):
                    self.check_code(code, f"{path.name}:{name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--private-prefix", action="append", required=True)
    parser.add_argument("--python-cli", type=Path)
    args = parser.parse_args()
    checker = PathChecker(args.private_prefix)
    for artifact in args.artifacts:
        if not artifact.exists():
            parser.error(f"missing release artifact: {artifact}")
        files = artifact.rglob("*") if artifact.is_dir() else (artifact,)
        for path in files:
            if path.is_file():
                checker.check_file(path)
    if args.python_cli:
        checker.check_python_cli(args.python_cli)
    if checker.failures:
        for label in sorted(checker.failures):
            print(f"Private build path in {label}")
        print(f"FAIL: {len(checker.failures)} affected files or Python modules; checked {checker.files} files and {checker.python_modules} frozen modules")
        return 1
    print(f"PASS: {checker.files} files and {checker.python_modules} frozen Python modules contain no private build prefixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
