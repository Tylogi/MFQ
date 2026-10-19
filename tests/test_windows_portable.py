from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "windows_portable", Path(__file__).parents[1] / "packaging/build_portable_windows.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def project(tmp_path):
    root = tmp_path / "source tree"
    tauri = root / "MFQStudio/src-tauri"
    tauri.mkdir(parents=True)
    (tauri / "icons").mkdir()
    images = []
    entries = []
    offset = 6 + 16 * 3
    for size in (16, 32, 48):
        mask = b"\0" * (((size + 31) // 32) * 4 * size)
        pixels = b"\x20\x40\x60\xff" * size * size
        image = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0, len(pixels) + len(mask), 0, 0, 0, 0) + pixels + mask
        entries.append(struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(image), offset))
        images.append(image)
        offset += len(image)
    (tauri / "icons/icon.ico").write_bytes(struct.pack("<HHH", 0, 1, 3) + b"".join(entries + images))
    (root / "packaging").mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "packaging/windows_standalone.nsi", root / "packaging/windows_standalone.nsi")
    (tauri / "tauri.conf.json").write_text(json.dumps({"version": "0.4.0"}))
    (tauri / "tauri.release-windows.conf.json").write_text(json.dumps({"bundle": {
        "resources": {
            "../dist": "web-ui", "../../packaging/Resources/mfq-cli": "mfq-cli",
            "../../packaging/windows-runtime": "", "../../packaging/Resources/licenses": "licenses",
        },
        "externalBin": ["../../packaging/sidecars/mfq-runtime", "../../packaging/sidecars/mfq-diagnostics"],
    }}))
    for path in (
        "build/mfq-studio.exe", "MFQStudio/dist/index.html",
        "packaging/Resources/mfq-cli/mfq-cli.exe", "packaging/Resources/mfq-cli/_internal/python.dll",
        "packaging/windows-runtime/cudart64_12.dll", "packaging/Resources/licenses/NOTICE",
        "packaging/sidecars/mfq-runtime-x86_64-pc-windows-msvc.exe",
        "packaging/sidecars/mfq-diagnostics-x86_64-pc-windows-msvc.exe",
        "build/private-server.log", "packaging/.venv/private.txt",
    ):
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(file.name.encode())
    return root


def test_zip_contains_the_installer_resources_without_build_files(tmp_path):
    root = project(tmp_path)
    archive = module.build_archive(root, root / "build/mfq-studio.exe", tmp_path / "output")
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert set(package.namelist()) == {
            "MFQ Studio/mfq-studio.exe", "MFQ Studio/mfq-runtime.exe", "MFQ Studio/mfq-diagnostics.exe",
            "MFQ Studio/mfq-cli/mfq-cli.exe", "MFQ Studio/mfq-cli/_internal/python.dll",
            "MFQ Studio/cudart64_12.dll", "MFQ Studio/web-ui/index.html", "MFQ Studio/licenses/NOTICE",
        }
        assert package.read("MFQ Studio/mfq-runtime.exe") == b"mfq-runtime-x86_64-pc-windows-msvc.exe"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_suffix(".zip.sha256").read_text() == f"{digest}  {archive.name}\n"


def test_incomplete_payload_does_not_replace_a_previous_archive(tmp_path):
    root = project(tmp_path)
    (root / "MFQStudio/dist/index.html").unlink()
    output = tmp_path / "output"
    output.mkdir()
    archive = output / "MFQ Studio_0.4.0_x64-portable.zip"
    archive.write_bytes(b"previous package")
    with pytest.raises(FileNotFoundError, match="web-ui/index.html"):
        module.build_archive(root, root / "build/mfq-studio.exe", output)
    assert archive.read_bytes() == b"previous package"
    assert not archive.with_suffix(".zip.partial").exists()


@pytest.mark.parametrize("destination", ["../outside", "/outside", "C:/outside", "MFQ-STUDIO.EXE"])
def test_unsafe_or_conflicting_resource_destination_is_rejected(tmp_path, destination):
    root = project(tmp_path)
    config = root / "MFQStudio/src-tauri/tauri.release-windows.conf.json"
    value = json.loads(config.read_text())
    value["bundle"]["resources"] = {"../../build/mfq-studio.exe": destination}
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.payload(root, root / "build/mfq-studio.exe")


@pytest.mark.skipif(sys.platform != "win32" or not os.environ.get("MFQ_TEST_NSIS"), reason="Windows NSIS execution")
def test_single_executable_extracts_reuses_and_repairs_its_cache(tmp_path):
    root = project(tmp_path)
    executable = module.build_executable(root, root / "build/mfq-studio.exe", tmp_path / "output", Path(os.environ["MFQ_TEST_NSIS"]))
    with zipfile.ZipFile(executable.parent / "MFQ Studio_0.4.0_x64-portable.zip") as package:
        assert package.namelist() == ["MFQ Studio.exe"]
        assert package.read("MFQ Studio.exe") == executable.read_bytes()
    cache = tmp_path / "runtime cache"
    environment = dict(os.environ, MFQ_STUDIO_RUNTIME_CACHE=str(cache))
    arguments = [str(executable), "/EXTRACT_ONLY"]
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    directory, = cache.glob("0.4.0-*")
    native = directory / "mfq-runtime.exe"
    assert native.read_bytes() == b"mfq-runtime-x86_64-pc-windows-msvc.exe"
    assert (directory / "mfq-cli/_internal/python.dll").is_file()
    sentinel = directory / "cache-hit-check"
    sentinel.touch()
    timestamp = native.stat().st_mtime_ns
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    assert sentinel.exists() and native.stat().st_mtime_ns == timestamp
    (directory / "mfq-cli/mfq-cli.exe").unlink()
    subprocess.run(arguments, check=True, timeout=30, env=environment)
    assert (directory / "mfq-cli/mfq-cli.exe").is_file()
    assert not sentinel.exists()
    children = [subprocess.Popen(arguments, env=environment) for _ in range(3)]
    assert [child.wait(timeout=30) for child in children] == [0, 0, 0]


@pytest.mark.skipif(sys.platform != "win32" or not os.environ.get("MFQ_TEST_STANDALONE_APP"), reason="Windows launcher fixture")
def test_single_executable_launches_the_app_and_waits_for_an_update_parent(tmp_path):
    root = project(tmp_path)
    shutil.copyfile(os.environ["MFQ_TEST_STANDALONE_APP"], root / "build/mfq-studio.exe")
    executable = module.build_executable(root, root / "build/mfq-studio.exe", tmp_path / "output", Path(os.environ["MFQ_TEST_NSIS"]))
    cache, log = tmp_path / "runtime cache", tmp_path / "launched.txt"
    environment = dict(os.environ, MFQ_STUDIO_RUNTIME_CACHE=str(cache), MFQ_TEST_LAUNCH_LOG=str(log))
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    try:
        launcher = subprocess.Popen([str(executable), f"/WAITPID={parent.pid}"], env=environment)
        time.sleep(.1)
        assert not log.exists()
        parent.wait(timeout=10)
        assert launcher.wait(timeout=15) == 0
        deadline = time.monotonic() + 10
        while not log.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        original, actual, directory = log.read_text(encoding="utf-16le").splitlines()
        assert Path(original) == executable
        assert Path(actual).parent.parent == cache
        assert Path(directory) == Path(actual).parent
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
