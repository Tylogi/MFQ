"""Release scanning must find paths even inside compressed Python payloads."""
import importlib.util
import json
import marshal
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "release_paths", Path(__file__).parents[1] / "packaging" / "check_release_paths.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("path", [
    r"C:\Users\Build User\source\crate.rs",
    "C:/Users/Build User/source/crate.rs",
    r"C:\\Users\\Build User\\source\\crate.rs",
])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16le"])
def test_private_windows_paths_are_detected(path, encoding):
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_bytes(path.encode(encoding), "program.exe")
    assert checker.failures == {"program.exe"}


def test_normalized_diagnostic_paths_are_allowed():
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_bytes(b"/mfq-build/home/.cargo/src/crate.rs", "program.exe")
    assert not checker.failures


def test_zip_python_filenames_are_inspected(tmp_path):
    code = compile("pass", r"C:\Users\Build User\stdlib\module.py", "exec")
    path = tmp_path / "base_library.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("module.pyc", b"\0" * 16 + marshal.dumps(code))
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_file(path)
    assert any("!module.pyc" in item for item in checker.failures)


def test_python_constant_tuples_are_inspected():
    code = compile("paths = (r'C:\\Users\\Build User\\source', 'other')", "module.py", "exec")
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_code(code, "module")
    assert checker.failures == {"module"}


def test_matches_across_scan_blocks_are_detected(tmp_path):
    path = tmp_path / "program.exe"
    path.write_bytes(b"\0" * (4 * 1024 * 1024 - 5) + br"C:\Users\Build User\source")
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_file(path)
    assert checker.failures == {str(path)}


@pytest.mark.parametrize("payload", ["native.dylib", "Resources/config.json"])
def test_private_posix_paths_are_detected_outside_the_main_executable(tmp_path, payload):
    path = tmp_path / payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"build root: /build-machine/private/source/file.cpp")
    checker = module.PathChecker(["/build-machine/private"])
    checker.check_file(path)
    assert checker.failures == {str(path)}


def test_sysconfig_hook_remaps_nested_metadata_only():
    hook = runpy.run_path(str(Path(__file__).parents[1] / "packaging/hooks/hook-sysconfig.py"))
    code = compile("paths = (('/build-machine/python/include',),)\nanswer = 42\ndef nested():\n return '/build-machine/python/lib'", "sysconfigdata.py", "exec")
    cleaned = hook["replace_build_prefix"](code, "/build-machine/python")
    assert b"/build-machine/python" not in marshal.dumps(cleaned)
    namespace = {}
    exec(cleaned, namespace)
    assert namespace["paths"] == (("/mfq-build/python/include",),)
    assert namespace["nested"]() == "/mfq-build/python/lib"
    assert namespace["answer"] == 42
    assert hook["replace_build_prefix"](frozenset({"/build-machine/python/lib"}), "/build-machine/python") == frozenset({"/mfq-build/python/lib"})


@pytest.mark.skipif(sys.platform == "win32", reason="macOS sysconfigdata hook")
def test_sysconfig_hook_updates_the_actual_module_graph_code():
    hook = runpy.run_path(str(Path(__file__).parents[1] / "packaging/hooks/hook-sysconfig.py"))
    node = SimpleNamespace(code=compile(f"build_prefix = {sys.base_prefix!r}", "sysconfigdata.py", "exec"))
    imports = []

    def import_hook(name, caller):
        imports.append((name, caller))
        return [node]

    caller = object()
    hook["hook"](SimpleNamespace(module=caller, module_graph=SimpleNamespace(import_hook=import_hook)))
    namespace = {}
    exec(node.code, namespace)
    assert namespace["build_prefix"] == "/mfq-build/python"
    assert imports == [(hook["sysconfig"]._get_sysconfigdata_name(), caller)]


def macos_path_environment(tmp_path, encoded=None):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is required for macOS build flag tests")
    project = tmp_path / "source tree's $literal"
    project.mkdir()
    venv = tmp_path / "release python"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    environment = dict(os.environ)
    environment.pop("CARGO_ENCODED_RUSTFLAGS", None)
    environment.update({
        "mfq_project_dir": str(project),
        "mfq_studio_dir": str(project / "MFQStudio"),
        "mfq_venv_dir": str(venv),
        "CARGO_HOME": str(tmp_path / "external cargo"),
        "RUSTUP_HOME": str(tmp_path / "external rustup"),
        "CARGO_TARGET_DIR": "relative target",
        "RUSTFLAGS": "-C opt-level=2",
        "CFLAGS": "-O2",
        "CXXFLAGS": "-O3",
        "OBJCXXFLAGS": "-fobjc-arc",
    })
    if encoded is not None:
        environment["CARGO_ENCODED_RUSTFLAGS"] = encoded
    helper = Path(__file__).parents[1] / "packaging/macos_release_paths.sh"
    result = subprocess.run([bash, "-euc", '''
source "$1"
mfq_configure_release_paths
"${mfq_venv_dir}/bin/python" -c 'import json, os, sys; keys = ("CARGO_ENCODED_RUSTFLAGS", "CFLAGS", "CXXFLAGS", "OBJCXXFLAGS", "CC_SHELL_ESCAPED_FLAGS"); print(json.dumps({"environment": {key: os.environ[key] for key in keys}, "prefixes": sys.argv[1:]}))' "${mfq_private_prefix_args[@]}"
''', "--", str(helper)], env=environment, text=True, capture_output=True, check=True)
    return project, json.loads(result.stdout)


@pytest.mark.skipif(sys.platform == "win32", reason="macOS build flags use POSIX paths")
@pytest.mark.parametrize("encoded", [None, "-C\x1flink-arg=a b", ""])
def test_macos_build_paths_cover_external_toolchains_and_preserve_flag_boundaries(tmp_path, encoded):
    project, configured = macos_path_environment(tmp_path, encoded)
    environment = configured["environment"]
    rust = environment["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    inherited = ["-C", "opt-level=2"] if encoded is None else (encoded.split("\x1f") if encoded else [])
    assert rust[:len(inherited)] == inherited
    assert f"--remap-path-prefix={project}=/mfq-src" in rust
    assert f"--remap-path-prefix={tmp_path / 'external cargo'}=/mfq-build/cargo" in rust
    assert f"--remap-path-prefix={project / 'MFQStudio/relative target'}=/mfq-build/target" in rust
    mapped = [flag.replace("--remap-path-prefix=", "-ffile-prefix-map=", 1) for flag in rust if flag.startswith("--remap-path-prefix=")]
    for name, inherited_native in [("CFLAGS", "-O2"), ("CXXFLAGS", "-O3"), ("OBJCXXFLAGS", "-fobjc-arc")]:
        assert shlex.split(environment[name]) == [inherited_native] + mapped
    assert configured["prefixes"][::2] == ["--private-prefix"] * len(mapped)
    assert str(project) in configured["prefixes"]
    assert environment["CC_SHELL_ESCAPED_FLAGS"] == "1"


@pytest.mark.skipif(sys.platform == "win32", reason="macOS build flags use POSIX paths")
def test_native_compiler_remaps_file_macro_with_spaces_and_shell_characters(tmp_path):
    compiler = shutil.which("clang")
    if not compiler:
        pytest.skip("clang is required for the native path remapping check")
    project, configured = macos_path_environment(tmp_path)
    source = project / "check.c"
    source.write_text('#include <stdio.h>\nint main(void) { puts(__FILE__); return 0; }\n')
    binary = tmp_path / "native-check"
    subprocess.run([compiler, *shlex.split(configured["environment"]["CFLAGS"]), str(source), "-o", str(binary)], check=True, capture_output=True)
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "/mfq-src/check.c"
    checker = module.PathChecker([str(project)])
    checker.check_file(binary)
    assert not checker.failures


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS frozen CLI integration")
def test_frozen_sysconfig_metadata_is_sanitized_before_packaging(tmp_path):
    pytest.importorskip("PyInstaller")
    source = tmp_path / "check.py"
    source.write_text('import importlib, json, sysconfig\nprint(json.dumps(importlib.import_module(sysconfig._get_sysconfigdata_name()).build_time_vars))\n')
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--name", "check", "--additional-hooks-dir", str(Path(__file__).parents[1] / "packaging/hooks"),
        "--distpath", str(tmp_path / "dist"), "--workpath", str(tmp_path / "work"),
        "--specpath", str(tmp_path), str(source)], check=True, capture_output=True)
    binary = tmp_path / "dist/check/check"
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
    assert sys.base_prefix not in result.stdout
    assert "/mfq-build/python" in result.stdout
    assert json.loads(result.stdout)["SOABI"]
    checker = module.PathChecker([sys.base_prefix, str(tmp_path)])
    for path in (tmp_path / "dist").rglob("*"):
        if path.is_file():
            checker.check_file(path)
    checker.check_python_cli(binary)
    assert checker.python_modules > 0
    assert not checker.failures
