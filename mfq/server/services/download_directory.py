from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


class DownloadDirectorySettings:
    def __init__(self, workspace: Path, default_directory: Path) -> None:
        self.path = workspace / "download-directory.json"
        self.default_directory = default_directory.resolve()
        if self.default_directory.is_relative_to(workspace):
            self.default_directory.mkdir(parents=True, exist_ok=True)
        self.selected: Path | None = None
        self.evict_prefix_cache = False
        self.roots = {self.identifier(self.default_directory): self.default_directory}
        self._lock = asyncio.Lock()
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.evict_prefix_cache = bool(data.get("evict_prefix_cache", False))
            for value in data.get("directories", []):
                directory = Path(value)
                if not directory.is_absolute():
                    raise ValueError("saved download directories must be absolute")
                directory = directory.resolve()
                self.roots[self.identifier(directory)] = directory
            if data.get("selected") is not None:
                self.selected = Path(data["selected"]).resolve()
                if self.selected not in self.roots.values():
                    raise ValueError("saved download directory is not registered")

    @staticmethod
    def identifier(path: Path) -> str:
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:32]

    def status(self) -> dict[str, str | int | bool | None]:
        actual = self.selected or self.default_directory
        try:
            disk = shutil.disk_usage(actual)
            total, free = disk.total, disk.free
        except OSError:
            total = free = None
        return {"directory": str(self.selected) if self.selected else None,
            "actual_directory": str(actual),
            "default_directory": str(self.default_directory),
            "disk_total_bytes": total, "disk_free_bytes": free,
            "evict_prefix_cache": self.evict_prefix_cache}

    def _save(self, selected: Path | None, roots: dict[str, Path], evict_prefix_cache: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self.path.parent, prefix=".download-directory-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"selected": str(selected) if selected else None, "directories": [str(path) for path in roots.values()],
                    "evict_prefix_cache": evict_prefix_cache}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    async def configure(self, path: Path | None, *, evict_prefix_cache: bool | None = None) -> dict[str, str | int | bool | None]:
        selected = path.resolve(strict=True) if path is not None else None
        if selected is not None and (not selected.is_dir() or not os.access(selected, os.W_OK)):
            raise ValueError("download directory must be a writable folder")
        async with self._lock:
            roots = dict(self.roots)
            if selected is not None:
                roots[self.identifier(selected)] = selected
            reclaim = self.evict_prefix_cache if evict_prefix_cache is None else evict_prefix_cache
            await asyncio.to_thread(self._save, selected, roots, reclaim)
            self.roots = roots
            self.selected = selected
            self.evict_prefix_cache = reclaim
            return await asyncio.to_thread(self.status)

    def directory(self, path: str) -> Path:
        directory = Path(path).resolve()
        if directory not in self.roots.values():
            raise ValueError("download directory has not been selected on this server")
        if directory == self.default_directory:
            directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise ValueError("download directory is unavailable")
        return directory
