from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def fingerprint(source: bytes, mode: int, language: int) -> str:
    return hashlib.sha256(source + b"\0" + str(mode).encode() + b"\0" + str(language).encode()).hexdigest()


def worker_count(jobs: int) -> int:
    return max(1, min(jobs, os.cpu_count() or 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metal", required=True)
    parser.add_argument("--metallib", required=True)
    parser.add_argument("--sdk", required=True)
    args = parser.parse_args()
    sources = sorted(args.sources.glob("*.metal"))
    if not sources:
        parser.error("no exported Metal sources")
    args.output.mkdir(parents=True, exist_ok=True)

    def compile_one(job: tuple[Path, str, int, str]) -> dict:
        source, standard, language, minimum = job
        key = fingerprint(source.read_bytes(), 2, language)
        target = args.output / (key + ".metallib")
        with tempfile.TemporaryDirectory(prefix="mfq-metal-") as temporary:
            air = Path(temporary) / "kernel.air"
            library = Path(temporary) / "kernel.metallib"
            subprocess.run([args.metal, "-std=" + standard, "-ffast-math",
                "-isysroot", args.sdk, "-mmacosx-version-min=" + minimum,
                "-c", str(source), "-o", str(air)], check=True)
            subprocess.run([args.metallib, str(air), "-o", str(library)], check=True)
            target.write_bytes(library.read_bytes())
        return {"name": source.stem, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "math_mode": 2, "language": language, "file": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    jobs = [(source, standard, language, minimum) for source in sources
        for standard, language, minimum in [("metal3.2", (3 << 16) + 2, "15.0"),
            ("metal4.0", 4 << 16, "26.0")]]
    with ThreadPoolExecutor(max_workers=worker_count(len(jobs))) as executor:
        libraries = list(executor.map(compile_one, jobs))
    (args.output / "manifest.json").write_text(json.dumps({"libraries": libraries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
