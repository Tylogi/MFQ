from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

from mfq._version import __version__

STUDIO_PROTOCOL_VERSION = "1.0"


class ServerAlreadyRunningError(RuntimeError):
    pass


def server_identity(data_dir: Path | None = None, *, instance_id: str | None = None,
    studio_owned: bool = False) -> dict[str, str | int]:
    workspace = os.path.normcase(str(data_dir.expanduser().resolve())) if data_dir is not None else ""
    return {
        "service": "mfq-server",
        "protocol_version": "1.0",
        "studio_protocol_version": STUDIO_PROTOCOL_VERSION,
        "version": __version__,
        "instance_id": str(UUID(instance_id)) if instance_id else str(uuid4()),
        "workspace_id": hashlib.sha256(workspace.encode()).hexdigest(),
        "pid": os.getpid(),
        "launch_mode": "studio" if studio_owned else "cli",
    }


class ServerInstanceLock:
    def __init__(self, data_dir: Path, *, instance_id: str | None = None, studio_owned: bool = False):
        self.identity = server_identity(data_dir, instance_id=instance_id, studio_owned=studio_owned)
        user = str(os.getuid()) if hasattr(os, "getuid") else hashlib.sha256(
            str(Path.home()).encode()).hexdigest()[:16]
        root = (Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "MFQ" / "run"
            if os.name == "nt" else Path("/tmp") / f"mfq-server-locks-{user}")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = root / f"{self.identity['workspace_id']}.lock"
        self.stream = None

    def __enter__(self) -> ServerInstanceLock:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        stream = os.fdopen(os.open(self.path, flags, 0o600), "r+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            if error.errno in {errno.EAGAIN, errno.EACCES, errno.EDEADLK}:
                raise ServerAlreadyRunningError("This MFQ data directory is already managed by another server. "
                    "Connect to that server or stop it before starting a new one.") from error
            raise
        try:
            stream.seek(1)
            stream.write(json.dumps(self.identity).encode())
            stream.truncate()
            stream.flush()
        except BaseException:
            stream.close()
            raise
        self.stream = stream
        return self

    def __exit__(self, *_args) -> None:
        if self.stream is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.stream.seek(0)
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.stream.close()
                self.stream = None
