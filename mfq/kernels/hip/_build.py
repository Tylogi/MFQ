from __future__ import annotations

import errno
import hashlib
from pathlib import Path
import shutil
import tempfile


def stage_sources(repository: Path, build_directory: Path) -> Path:
    trees = (
        "mfq/kernels/hip",
        "csrc/backends/hip/include",
        "csrc/formats",
        "csrc/components/ggml/include",
        "csrc/components/ggml/src/ggml-hip",
    )
    suffixes = {".h", ".cuh", ".hpp", ".cpp", ".hip"}
    files = {
        path.relative_to(repository)
        for tree in trees
        for path in (repository / tree).rglob("*")
        if path.is_file() and path.suffix in suffixes
    }
    files.update(
        path.relative_to(repository)
        for path in (repository / "csrc/components/ggml/src").glob("*")
        if path.is_file() and path.suffix in {".h", ".cuh", ".hpp"}
    )
    files = sorted(files)
    if not files:
        raise RuntimeError("HIP compiler sources are missing")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.as_posix().encode() + b"\0")
        digest.update(hashlib.sha256((repository / path).read_bytes()).digest())
    cache = build_directory / "mfq-sources"
    destination = cache / digest.hexdigest()
    if destination.is_dir():
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="stage-", dir=cache))
    try:
        for path in files:
            target = temporary / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / path, target)
        try:
            temporary.rename(destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not destination.is_dir():
                raise
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
