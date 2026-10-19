"""Package the Windows release as a standalone executable or ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


def payload(project: Path, executable: Path) -> dict[str, Path]:
    tauri = project / "MFQStudio/src-tauri"
    bundle = json.loads((tauri / "tauri.release-windows.conf.json").read_text())["bundle"]
    files: dict[str, Path] = {}
    names: set[str] = set()

    def add(source: Path, destination: str) -> None:
        path = PurePosixPath(destination.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError(f"Unsafe archive path: {destination}")
        name = str(path)
        if name.casefold() in names:
            raise ValueError(f"Duplicate archive path: {name}")
        if not source.is_file():
            raise FileNotFoundError(source)
        files[name] = source
        names.add(name.casefold())

    add(executable, "mfq-studio.exe")
    for source, destination in bundle["resources"].items():
        root = (tauri / source).resolve()
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    add(path, str(PurePosixPath(destination) / path.relative_to(root).as_posix()))
        else:
            add(root, destination)
    for source in bundle["externalBin"]:
        path = (tauri / f"{source}-x86_64-pc-windows-msvc.exe").resolve()
        add(path, Path(source).name + ".exe")
    for required in ("mfq-runtime.exe", "mfq-diagnostics.exe", "mfq-cli/mfq-cli.exe", "web-ui/index.html"):
        if required not in files:
            raise FileNotFoundError(f"Missing portable resource: {required}")
    return files


def build_archive(project: Path, executable: Path, output: Path) -> Path:
    files = payload(project, executable)
    config = json.loads((project / "MFQStudio/src-tauri/tauri.conf.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"MFQ Studio_{config['version']}_x64-portable.zip"
    partial = archive.with_suffix(".zip.partial")
    total = sum(path.stat().st_size for path in files.values())
    completed = 0
    next_progress = 256 * 1024 * 1024
    print(f"Packing {len(files)} files ({total / 1024**3:.2f} GiB)", flush=True)
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
            for name, path in sorted(files.items()):
                package.write(path, "MFQ Studio/" + name)
                completed += path.stat().st_size
                if completed >= next_progress:
                    print(f"Packed {completed / 1024**3:.2f} / {total / 1024**3:.2f} GiB", flush=True)
                    next_progress = completed + 256 * 1024 * 1024
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Built {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return archive


def nsis_string(value: str) -> str:
    return value.replace("$", "$$").replace('"', '$\\"').replace("\r", "$\\r").replace("\n", "$\\n")


def build_executable(project: Path, executable: Path, output: Path, makensis: Path) -> Path:
    files = payload(project, executable)
    version = json.loads((project / "MFQStudio/src-tauri/tauri.conf.json").read_text())["version"]
    digest = hashlib.sha256()
    instructions = []
    directory = None
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
        target = PurePosixPath(name)
        parent = str(target.parent).replace("/", "\\")
        if parent != directory:
            instructions.append(f'  SetOutPath "$Stage\\{nsis_string(parent)}"')
            directory = parent
        instructions += [f'  File "/oname={nsis_string(target.name)}" "{nsis_string(str(path.resolve()))}"',
                         '  IfErrors failed']
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"MFQ Studio_{version}_x64.exe"
    partial = destination.with_suffix(".exe.partial")
    script = project / "packaging/build/standalone/launcher.nsi"
    script.parent.mkdir(parents=True, exist_ok=True)
    template = (project / "packaging/windows_standalone.nsi").read_text(encoding="utf-8")
    for key, value in {
        "OUTPUT": nsis_string(str(partial.resolve())), "VERSION": version,
        "ICON": nsis_string(str(project / "MFQStudio/src-tauri/icons/icon.ico")),
        "PAYLOAD_ID": f"{version}-{digest.hexdigest()[:16]}",
        "PAYLOAD_FILES": "\n".join(instructions),
    }.items():
        template = template.replace(f"@{key}@", value)
    script.write_text(template, encoding="utf-8")
    print(f"Packing {len(files)} files into a standalone executable", flush=True)
    try:
        subprocess.run([str(makensis), "-INPUTCHARSET", "UTF8", "-V2", str(script)], check=True)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    with destination.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    destination.with_suffix(".exe.sha256").write_text(f"{checksum}  {destination.name}\n", encoding="utf-8")
    print(f"Built {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    archive = output / f"MFQ Studio_{version}_x64-portable.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as package:
        package.write(destination, "MFQ Studio.exe")
    with archive.open("rb") as stream:
        zip_checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{zip_checksum}  {archive.name}\n", encoding="utf-8")
    print(f"Built {archive} (one executable)", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-executable", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--makensis", type=Path, help="Build a single-file app using this NSIS compiler")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.makensis:
        build_executable(project, args.studio_executable, args.output_directory, args.makensis)
    else:
        build_archive(project, args.studio_executable, args.output_directory)


if __name__ == "__main__":
    main()
