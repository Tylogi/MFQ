from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class FileManagerUnavailableError(RuntimeError):
    pass


def directory_opener() -> str | None:
    if sys.platform == "darwin":
        return "/usr/bin/open"
    if sys.platform == "win32":
        return "explorer" if callable(getattr(os, "startfile", None)) else None
    if sys.platform.startswith("linux") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return shutil.which("xdg-open")
    return None


def open_directory(directory: Path) -> None:
    opener = directory_opener()
    if opener is None:
        raise FileManagerUnavailableError("no desktop file manager is available on this server")
    if sys.platform == "win32":
        os.startfile(str(directory), "explore")
        return
    command = ([opener, "-a", "Finder", str(directory)] if sys.platform == "darwin"
        else [opener, str(directory)])
    subprocess.run(command, check=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
