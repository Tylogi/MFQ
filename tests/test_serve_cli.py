from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfq.cli import _build_parser
from mfq.commands.serve import (
    _prepare_web_root,
    _resolve_runtime_executable,
    _run,
    _select_backend,
    _server_storage_paths,
)
from mfq.server.runtime.native import (
    NativeRuntime,
    NativeRuntimeError,
    native_request_capacity,
    native_runtime_environment,
)


def test_server_runtime_control_plane_has_no_architecture_dispatch() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("mfq/server/runtime/native.py", "mfq/server/runtime/pool.py")
    )

    for forbidden in (
        "MODEL_GRAPH_ASSET",
        "resolve_runtime_route",
        "python_mlx_worker",
        'backbone="glm5_next"',
        'backbone="qwen3_5"',
        'backbone="qwen4_exp"',
    ):
        assert forbidden not in source


def test_server_imports_framework_without_loading_quantization_stack() -> None:
    script = """
import importlib
import sys

for name in (
    'mfq.server.api',
    'mfq.server.catalog',
    'mfq.commands.serve',
    'mfq.server.runtime.native',
    'mfq.server.runtime.pool',
    'mfq.server.service',
):
    importlib.import_module(name)

unexpected = sorted(
    name for name in sys.modules
    if name == 'torch'
    or name == 'mfq.quantize'
    or name.startswith('mfq.quantize.')
    or name == 'mfq.calibration'
    or name.startswith('mfq.calibration.')
)
if unexpected:
    raise SystemExit('eager imports: ' + ', '.join(unexpected))
if 'fastapi' not in sys.modules:
    raise SystemExit('FastAPI was not loaded by the server application')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_serve_exposes_public_host_and_port_options(tmp_path: Path) -> None:
    defaults = _build_parser().parse_args(["serve"])
    args = _build_parser().parse_args(
        [
            "serve",
            "--model",
            str(tmp_path / "model.mfq"),
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--max-queued-requests-per-runtime",
            "7",
            "--max-runtime-memory",
            "12G",
            "--runtime-idle-timeout",
            "300",
            "--transport",
            "HTTP",
        ]
    )

    assert defaults.host == "127.0.0.1"
    assert defaults.port == 8090
    assert defaults.model is None
    assert defaults.running_executable is None
    assert defaults.data_dir == Path(".mfq")
    assert defaults.db is None
    assert defaults.access_log is True
    assert defaults.max_queued_requests_per_runtime is None
    assert defaults.max_runtime_memory is None
    assert defaults.no_memory_guard is False
    assert defaults.moe_gpu_cache_gb is None
    assert defaults.runtime_idle_timeout is None
    assert defaults.transport == "stdio"
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.max_queued_requests_per_runtime == 7
    assert args.max_runtime_memory == 12 * 1024**3
    assert args.runtime_idle_timeout == 300
    assert args.transport == "http"


def test_serve_accepts_an_explicit_expert_cache_budget() -> None:
    args = _build_parser().parse_args(["serve", "--moe-gpu-cache-gb", "3.5"])

    assert args.moe_gpu_cache_gb == 3.5


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--port", "0"),
        ("--port", "65536"),
        ("--context-size", "-1"),
        ("--prefill-chunk-size", "0"),
        ("--moe-gpu-cache-gb", "nan"),
        ("--prefix-cache-disk-size", "invalid"),
        ("--prefix-cache-hot-size", "-1M"),
        ("--prefix-cache-block-tokens", "0"),
        ("--runtime-startup-timeout", "0"),
        ("--max-runtime-instances", "0"),
        ("--max-requests-per-runtime", "0"),
        ("--max-queued-requests-per-runtime", "-1"),
        ("--max-runtime-memory", "1XB"),
        ("--runtime-idle-timeout", "-1"),
    ],
)
def test_serve_rejects_invalid_numeric_options(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(["serve", option, value])

    assert error.value.code == 2


def test_serve_accepts_an_empty_initial_model_catalog() -> None:
    args = _build_parser().parse_args(["serve"])

    assert args.model is None


def test_serve_storage_paths_preserve_explicit_database_layout(tmp_path: Path) -> None:
    database = tmp_path / "custom.sqlite3"

    assert _server_storage_paths(tmp_path / "data", database) == (
        database.resolve(),
        database.resolve().with_name("custom.sqlite3.media"),
    )


def test_serve_storage_paths_reuse_legacy_studio_data(tmp_path: Path) -> None:
    legacy_database = tmp_path / "mfq-server.sqlite3"
    legacy_database.touch()

    assert _server_storage_paths(tmp_path, None) == (
        legacy_database.resolve(),
        tmp_path.resolve() / "mfq-server.sqlite3.media",
    )


def test_serve_storage_paths_prefer_current_layout(tmp_path: Path) -> None:
    (tmp_path / "mfq-server.sqlite3").touch()
    current_database = tmp_path / "server.sqlite3"
    current_database.touch()

    assert _server_storage_paths(tmp_path, None) == (
        current_database.resolve(),
        tmp_path.resolve() / "media",
    )


def test_serve_accepts_a_prebuilt_native_runtime(tmp_path: Path) -> None:
    executable = tmp_path / "mfq-decode-metal"
    executable.write_bytes(b"runtime")

    assert _resolve_runtime_executable("metal", executable) == executable.resolve()


def test_serve_uses_the_runtime_recorded_by_mfq_build(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "custom" / "metal" / "mfq-decode-metal"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    managed = SimpleNamespace(executable=executable)
    monkeypatch.setattr("mfq.commands.serve.load_managed_build", lambda _: managed)
    monkeypatch.setattr(
        "mfq.commands.serve.build_runtime",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )

    assert _resolve_runtime_executable("metal") == executable


def test_serve_uses_an_explicit_prebuilt_runtime_without_managed_lookup(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "release" / "mfq-decode-metal"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    monkeypatch.setattr(
        "mfq.commands.serve.load_managed_build",
        lambda _: (_ for _ in ()).throw(AssertionError("must not inspect managed builds")),
    )
    monkeypatch.setattr(
        "mfq.commands.serve.build_runtime",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    assert _resolve_runtime_executable("metal", executable) == executable.resolve()


def test_serve_rejects_a_missing_explicit_prebuilt_runtime(tmp_path: Path) -> None:
    missing = tmp_path / "mfq-decode-metal"

    with pytest.raises(FileNotFoundError, match="--running-executable"):
        _resolve_runtime_executable("metal", missing)


def test_prebuilt_cuda_runtime_does_not_require_a_local_compiler(monkeypatch) -> None:
    monkeypatch.setattr("mfq.commands.serve.platform.system", lambda: "Linux")
    monkeypatch.setattr("mfq.commands.serve.platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        "mfq.commands.serve.detect_backend",
        lambda _: (_ for _ in ()).throw(AssertionError("must not require nvcc")),
    )

    assert _select_backend("auto", Path("mfq-runtime")) == "cuda"


def test_serve_rebuilds_a_missing_runtime_from_its_recorded_recipe(
    tmp_path: Path, monkeypatch
) -> None:
    managed = SimpleNamespace(
        executable=tmp_path / "custom" / "metal" / "mfq-decode-metal",
        build_dir=tmp_path / "custom",
        build_type="RelWithDebInfo",
        generator="Ninja",
        cmake_args=("-DMFQ_EXPERIMENTAL_KERNEL=ON",),
    )
    captured: dict[str, object] = {}
    rebuilt = tmp_path / "rebuilt-runtime"
    monkeypatch.setattr("mfq.commands.serve.load_managed_build", lambda _: managed)
    monkeypatch.setattr(
        "mfq.commands.serve.build_runtime",
        lambda **options: captured.update(options) or rebuilt,
    )

    assert _resolve_runtime_executable("metal") == rebuilt
    assert captured == {
        "backend": "metal",
        "build_dir": tmp_path / "custom",
        "build_type": "RelWithDebInfo",
        "generator": "Ninja",
        "cmake_args": ("-DMFQ_EXPERIMENTAL_KERNEL=ON",),
    }


def test_native_cuda_worker_is_private_and_uses_a_loopback_port(tmp_path: Path) -> None:
    runtime = NativeRuntime(
        executable=tmp_path / "mfq-runtime",
        model=tmp_path / "model.mfq",
        model_name="model",
        backend="cuda",
        context_size=32768,
    )

    command = runtime.command(43123)

    assert command[:3] == [str(runtime.executable), "--model", str(runtime.model)]
    assert command[command.index("--transport") + 1] == "http"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "43123"
    assert command[command.index("--ctx-size") + 1] == "32768"
    assert "--prefill-chunk-size" not in command


def test_native_cuda_worker_forwards_explicit_continuous_batching(
    tmp_path: Path,
) -> None:
    runtime = NativeRuntime(
        executable=tmp_path / "mfq-runtime",
        model=tmp_path / "model.mfq",
        model_name="model",
        backend="cuda",
        continuous_batching=8,
    )

    command = runtime.command(43123)

    assert command[command.index("--continuous-batching") + 1] == "8"


def test_native_request_capacity_is_backend_and_storage_based() -> None:
    assert native_request_capacity(
        backend="cuda",
        routed_expert_bytes=0,
        requested=8,
    ) == 8
    assert native_request_capacity(
        backend="metal",
        routed_expert_bytes=0,
        requested=8,
    ) == 1
    assert native_request_capacity(
        backend="cuda",
        routed_expert_bytes=1,
        requested=8,
    ) == 1


def test_native_metal_worker_rejects_cuda_continuous_batching(
    tmp_path: Path,
) -> None:
    runtime = NativeRuntime(
        executable=tmp_path / "mfq-decode-metal",
        model=tmp_path / "model.mfq",
        model_name="model",
        backend="metal",
        continuous_batching=2,
    )

    with pytest.raises(NativeRuntimeError, match="requires CUDA"):
        runtime.command(43123)


def test_native_metal_worker_receives_prefill_chunk_size(tmp_path: Path) -> None:
    runtime = NativeRuntime(
        executable=tmp_path / "mfq-decode-metal",
        model=tmp_path / "model.mfq",
        model_name="model",
        backend="metal",
        prefill_chunk_size=4096,
    )

    command = runtime.command(43123)

    assert command[command.index("--prefill-chunk-size") + 1] == "4096"


def test_native_cuda_worker_receives_prefill_chunk_size(tmp_path: Path) -> None:
    runtime = NativeRuntime(
        executable=tmp_path / "mfq-runtime",
        model=tmp_path / "model.mfq",
        model_name="model",
        backend="cuda",
        prefill_chunk_size=4096,
    )

    command = runtime.command(43123)

    assert command[command.index("--prefill-chunk-size") + 1] == "4096"


def test_native_worker_receives_explicit_expert_cache_budget(tmp_path: Path) -> None:
    runtime = NativeRuntime(
        executable=tmp_path / "mfq-decode-metal",
        model=tmp_path / "model.mfq",
        model_name="model",
        backend="metal",
        moe_gpu_cache_gb=0.0,
    )

    command = runtime.command(43123)

    assert command[command.index("--moe-gpu-cache-gb") + 1] == "0.0"


def test_native_metal_worker_finds_release_resources(tmp_path: Path) -> None:
    executable = tmp_path / "sidecars" / "mfq-decode-metal"
    executable.parent.mkdir()
    executable.touch()
    resources = tmp_path / "Resources"
    resources.mkdir()
    metallib = resources / "mlx.metallib"
    metallib.touch()
    frameworks = tmp_path / "Frameworks"
    frameworks.mkdir()
    video_library = frameworks / "libmfq_avfoundation_video.dylib"
    video_library.touch()

    environment = native_runtime_environment(executable, "metal", {})

    assert environment["MFQ_MLX_METALLIB"] == str(metallib)
    assert environment["MFQ_AVFOUNDATION_VIDEO_LIBRARY"] == str(video_library)


def test_native_runtime_environment_preserves_explicit_resource_paths(tmp_path: Path) -> None:
    environment = native_runtime_environment(
        tmp_path / "mfq-decode-metal",
        "metal",
        {
            "MFQ_MLX_METALLIB": "/configured/mlx.metallib",
            "MFQ_AVFOUNDATION_VIDEO_LIBRARY": "/configured/video.dylib",
        },
    )

    assert environment == {
        "MFQ_MLX_METALLIB": "/configured/mlx.metallib",
        "MFQ_AVFOUNDATION_VIDEO_LIBRARY": "/configured/video.dylib",
    }


def test_serve_builds_web_ui_when_the_source_is_newer(tmp_path: Path, monkeypatch) -> None:
    web = tmp_path / "MFQStudio"
    source = web / "src" / "App.tsx"
    source.parent.mkdir(parents=True)
    source.write_text("export {};", encoding="utf-8")
    (web / "package.json").write_text("{}", encoding="utf-8")
    (web / "package-lock.json").write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "mfq.commands.serve.__file__", str(tmp_path / "mfq" / "commands" / "serve.py")
    )
    monkeypatch.setattr("mfq.commands.serve.shutil.which", lambda _: "/usr/bin/npm")

    def run(command, **_):
        calls.append(command)
        if command[-2:] == ["run", "build"]:
            (web / "dist").mkdir()
            (web / "dist" / "index.html").write_text("ready", encoding="utf-8")

    monkeypatch.setattr("mfq.commands.serve.subprocess.run", run)

    assert _prepare_web_root(None, disabled=False) == web / "dist"
    assert calls == [["/usr/bin/npm", "ci"], ["/usr/bin/npm", "run", "build"]]


def test_configured_web_root_must_contain_an_index(tmp_path: Path) -> None:
    web_root = tmp_path / "web"
    web_root.mkdir()

    with pytest.raises(FileNotFoundError, match="Web UI entry point does not exist"):
        _prepare_web_root(web_root, disabled=False)

    (web_root / "index.html").write_text("ready", encoding="utf-8")

    assert _prepare_web_root(web_root, disabled=False) == web_root


def test_environment_web_root_must_exist(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing-web"
    monkeypatch.setenv("MFQ_SERVER_WEB_ROOT", str(missing))

    with pytest.raises(FileNotFoundError, match="MFQ_SERVER_WEB_ROOT Web UI directory"):
        _prepare_web_root(None, disabled=False)


def test_serve_validates_web_ui_before_backend_build_or_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "model.mfq"
    model.write_bytes(b"model")
    missing_web_root = tmp_path / "missing-web"
    args = _build_parser().parse_args(
        [
            "serve",
            "--model",
            str(model),
            "--web-root",
            str(missing_web_root),
        ]
    )
    monkeypatch.setattr(
        "mfq.commands.serve.detect_backend",
        lambda _: (_ for _ in ()).throw(AssertionError("backend detection must not run")),
    )

    with pytest.raises(FileNotFoundError, match="--web-root Web UI directory"):
        _run(args)


def test_serve_can_disable_web_ui_build() -> None:
    assert _prepare_web_root(None, disabled=True) is None


def test_serve_starts_without_loading_an_initial_model(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    executable = tmp_path / "mfq-decode-metal"
    executable.write_bytes(b"runtime")
    data_dir = tmp_path / ".mfq"
    args = _build_parser().parse_args(
        [
            "serve",
            "--no-web-ui",
            "--backend",
            "auto",
            "--running-executable",
            str(executable),
            "--data-dir",
            str(data_dir),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "mfq.server.runtime.native.NativeRuntime.start",
        lambda _: (_ for _ in ()).throw(AssertionError("must not start a model runtime")),
    )

    def run(app, **options):
        captured["app"] = app
        captured.update(options)

    monkeypatch.setattr("uvicorn.run", run)

    assert _run(args) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8090
    assert captured["access_log"] is True
    assert "MFQ Server ready" not in capsys.readouterr().out
    assert (data_dir / "server.sqlite3").is_file()
    assert (data_dir / "media").is_dir()
    assert (data_dir / "models").is_dir()
