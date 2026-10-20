from __future__ import annotations

import os
import hashlib
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def volume(path: Path) -> int:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path.stat().st_dev


def _lock(stream, blocking: bool = True) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return True
    except (BlockingIOError, PermissionError, OSError):
        if blocking:
            raise
        return False


class DiskReservation:
    def __init__(self, budget: DiskBudget, path: Path, stream, device: int, size: int, directory: Path, target=None):
        self.budget, self.path, self.stream = budget, path, stream
        self.device, self.size = device, size
        self.target = target
        self.directory = directory

    def update(self, remaining: int) -> None:
        with self.budget.locked():
            if self.stream.closed:
                return
            remaining = max(0, remaining)
            if remaining > self.size and remaining - self.size > shutil.disk_usage(self.directory).free - self.budget._reserved(self.device):
                raise OSError('insufficient disk space to renew download reservation')
            self.size = remaining
            self.stream.seek(1)
            self.stream.write(f"{self.device} {self.size}\n".encode())
            self.stream.truncate()
            self.stream.flush()

    def close(self) -> None:
        with self.budget.locked():
            if not self.stream.closed:
                self.stream.close()
                self.path.unlink(missing_ok=True)
                if self.target is not None:
                    self.target.close()


class DiskBudget:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(os.environ.get("MFQ_DISK_BUDGET_DIR", "") or
            str(Path(tempfile.gettempdir()) / f"mfq-disk-budget-{os.getuid() if hasattr(os, 'getuid') else os.environ.get('USERNAME', 'user')}"))
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def locked(self):
        with (self.root / "lock").open("a+b") as stream:
            _lock(stream)
            yield

    def _reserved(self, device: int) -> int:
        total = 0
        for path in self.root.glob("*.lease"):
            with path.open("r+b") as stream:
                if _lock(stream, False):
                    stale = True
                else:
                    stale = False
                    stream.seek(1)
                    saved_device, size = map(int, stream.read(128).split())
                    if saved_device == device:
                        total += size
            if stale:
                path.unlink(missing_ok=True)
        return total

    def reserved(self, directory: Path) -> int:
        with self.locked():
            return self._reserved(volume(directory))

    def reserve(self, directory: Path, size: int, *, reclaimable: int = 0, exclusive: bool = False) -> DiskReservation:
        with self.locked():
            device = volume(directory)
            available = shutil.disk_usage(directory).free - self._reserved(device)
            if size > available + reclaimable:
                raise OSError("insufficient unreserved disk space")
            target = None
            if exclusive:
                key = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()
                target = (self.root / (key + '.target')).open('a+b')
                if not _lock(target, False):
                    target.close()
                    raise ValueError('this download destination is already in use')
            path = self.root / f"{uuid4().hex}.lease"
            stream = path.open("x+b")
            try:
                stream.write(b"\0" + f"{device} {size}\n".encode())
                stream.flush()
                _lock(stream)
                return DiskReservation(self, path, stream, device, size, directory, target)
            except BaseException:
                stream.close()
                path.unlink(missing_ok=True)
                if target is not None:
                    target.close()
                raise
