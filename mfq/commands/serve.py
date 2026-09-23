"""Implementation of the public ``mfq serve`` command."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from mfq.commands.build import BuildError, build_runtime, detect_backend, load_managed_build


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("port must be at most 65535")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _byte_size(value: str) -> int:
    text = value.strip().lower()
    multipliers = {
        "": 1,
        "k": 1 << 10,
        "kb": 1 << 10,
        "m": 1 << 20,
        "mb": 1 << 20,
        "g": 1 << 30,
        "gb": 1 << 30,
        "t": 1 << 40,
        "tb": 1 << 40,
    }
    suffix = ""
    while text and text[-1].isalpha():
        suffix = text[-1] + suffix
        text = text[:-1]
    try:
        number = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid byte size") from error
    if suffix not in multipliers or not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("invalid byte size")
    return int(number * multipliers[suffix])


def _validate_web_root(path: Path, *, source: str) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{source} Web UI directory does not exist: {root}")
    index = root / "index.html"
    if not index.is_file():
        raise FileNotFoundError(f"{source} Web UI entry point does not exist: {index}")
    return root


def _prepare_web_root(configured: Path | None, *, disabled: bool) -> Path | None:
    if disabled:
        return None
    if configured is not None:
        return _validate_web_root(configured, source="--web-root")
    environment_root = os.environ.get("MFQ_SERVER_WEB_ROOT")
    if environment_root:
        return _validate_web_root(
            Path(environment_root),
            source="MFQ_SERVER_WEB_ROOT",
        )
    web_dir = Path(__file__).resolve().parents[2] / "MFQStudio"
    package = web_dir / "package.json"
    output = web_dir / "dist"
    index = output / "index.html"
    if not package.is_file():
        return None

    inputs = [
        *(path for path in web_dir.iterdir() if path.is_file()),
        *(path for path in (web_dir / "src").rglob("*") if path.is_file()),
    ]
    newest_input = max(
        (path.stat().st_mtime_ns for path in inputs if path.is_file()),
        default=0,
    )
    if index.is_file() and index.stat().st_mtime_ns >= newest_input:
        return output

    npm = shutil.which("npm")
    if npm is None:
        print(
            "Web UI source was found but npm is unavailable; serving the API only",
            file=sys.stderr,
        )
        return None
    print("Building MFQ Web UI...", flush=True)
    try:
        subprocess.run([npm, "ci"], cwd=web_dir, check=True)
        subprocess.run([npm, "run", "build"], cwd=web_dir, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"MFQ Web UI build failed with exit code {error.returncode}") from error
    if not index.is_file():
        raise RuntimeError(f"Web UI build completed without producing {index}")
    return output


def _resolve_runtime_executable(
    backend: str,
    configured: Path | None = None,
) -> Path:
    if configured is not None:
        executable = configured.expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"--running-executable does not exist or is not a file: {executable}"
            )
        print(f"Using configured native runtime: {executable}", flush=True)
        return executable
    managed = load_managed_build(backend)
    if managed is not None and managed.executable.is_file():
        print(f"Using managed native runtime: {managed.executable}", flush=True)
        return managed.executable
    if managed is not None:
        print(
            f"Managed native runtime is missing; rebuilding in {managed.build_dir}",
            flush=True,
        )
        return build_runtime(
            backend=backend,
            build_dir=managed.build_dir,
            build_type=managed.build_type,
            generator=managed.generator,
            cmake_args=managed.cmake_args,
        )
    return build_runtime(backend=backend)


def _select_backend(requested: str, running_executable: Path | None) -> str:
    if running_executable is None:
        return detect_backend(requested)
    system = platform.system()
    machine = platform.machine().lower()
    if requested == "auto":
        if system == "Darwin" and machine in {"arm64", "aarch64"}:
            return "metal"
        if system in {"Linux", "Windows"}:
            return "cuda"
        raise BuildError(f"no prebuilt MFQ runtime backend is supported on {system} {machine}")
    if requested == "metal":
        if system != "Darwin" or machine not in {"arm64", "aarch64"}:
            raise BuildError("the Metal runtime requires Apple silicon and macOS")
        return "metal"
    if requested == "cuda":
        if system not in {"Linux", "Windows"}:
            raise BuildError("the CUDA runtime is supported on Linux and Windows")
        return "cuda"
    raise BuildError(f"unsupported inference backend: {requested}")


def _server_storage_paths(
    data_dir: Path,
    database: Path | None,
) -> tuple[Path, Path]:
    resolved_data_dir = data_dir.expanduser().resolve()
    if database is not None:
        database_path = database.expanduser().resolve()
        return database_path, database_path.with_name(f"{database_path.name}.media")

    database_path = resolved_data_dir / "server.sqlite3"
    legacy_database_path = resolved_data_dir / "mfq-server.sqlite3"
    if not database_path.exists() and legacy_database_path.exists():
        return (
            legacy_database_path,
            legacy_database_path.with_name(f"{legacy_database_path.name}.media"),
        )
    return database_path, resolved_data_dir / "media"


def _run(args: argparse.Namespace) -> int:
    from mfq.server.network import install_system_proxy_environment, network_auth_error

    client_api_key = os.environ.get(args.api_key_env, "")
    if error := network_auth_error(args.host, client_api_key):
        raise ValueError(error)
    install_system_proxy_environment()
    import uvicorn

    from mfq.server.api import create_app
    from mfq.server.auth import ApiKeyManager
    from mfq.server.catalog import ModelCatalog
    from mfq.server.cluster import ClusterBackend
    from mfq.server.components import VoiceOutputComponent
    from mfq.server.jobs import JobManager
    from mfq.server.models import ModelLoadRequest
    from mfq.server.runtime.pool import RuntimePool
    from mfq.server.service import ServerService
    from mfq.server.storage import SessionStore
    from mfq.server.tool_jobs import ToolJobHandlers, ToolJobPaths
    from mfq.server.vision import clear_image_decode_cache

    data_dir = args.data_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    model = args.model.expanduser().resolve() if args.model is not None else None
    if model is not None and not (model.is_file() or model.is_dir()):
        raise FileNotFoundError(model)
    web_root = _prepare_web_root(args.web_root, disabled=args.no_web_ui)
    selected_backend = _select_backend(args.backend, args.running_executable)
    executable = _resolve_runtime_executable(selected_backend, args.running_executable)
    runtime_environment: dict[str, str] = {}
    if args.no_prefix_cache:
        runtime_environment["MFQ_RUNTIME_DISABLE_PREFIX_CACHE"] = "1"
    if args.prefix_cache_dir is not None:
        runtime_environment["MFQ_RUNTIME_PREFIX_CACHE_DIR"] = str(
            args.prefix_cache_dir.expanduser().resolve()
        )
    if args.prefix_cache_disk_size is not None:
        runtime_environment["MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES"] = str(
            args.prefix_cache_disk_size
        )
    if args.prefix_cache_hot_size is not None:
        runtime_environment["MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES"] = str(
            args.prefix_cache_hot_size
        )
    if args.prefix_cache_block_tokens is not None:
        runtime_environment["MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS"] = str(
            args.prefix_cache_block_tokens
        )
    configured_roots = [path.expanduser().resolve() for path in args.model_dir]
    if not configured_roots:
        configured_roots = [
            Path(value)
            for value in os.environ.get("MFQ_SERVER_MODEL_DIRS", "").split(os.pathsep)
            if value
        ]
    if not configured_roots:
        configured_roots = [data_dir / "models"]
    configured_roots = [path.expanduser().resolve() for path in configured_roots]
    if model is not None:
        model_catalog_root = model if model.is_dir() else model.parent
        if model_catalog_root not in configured_roots:
            configured_roots.append(model_catalog_root)
    configured_roots[0].mkdir(parents=True, exist_ok=True)
    catalog = ModelCatalog(configured_roots)
    voice_component = VoiceOutputComponent(data_dir)
    startup_loads: list[ModelLoadRequest] = []
    if model is not None:
        initial_artifact = asyncio.run(catalog.resolve_path(model))
        startup_loads.append(
            ModelLoadRequest(
                model=initial_artifact.resource.name,
                artifact_uri=f"mfq://{initial_artifact.resource.id}",
                context_size=args.context_size or 32768,
                prefill_chunk_size=args.prefill_chunk_size,
                moe_gpu_cache_gb=args.moe_gpu_cache_gb,
                prefix_cache_enabled=not args.no_prefix_cache,
                prefix_cache_disk_bytes=args.prefix_cache_disk_size,
                prefix_cache_hot_bytes=args.prefix_cache_hot_size,
                prefix_cache_block_tokens=args.prefix_cache_block_tokens,
            )
        )
    runtime_manager = RuntimePool(
        catalog,
        executable,
        startup_timeout_seconds=args.runtime_startup_timeout,
        max_instances=args.max_runtime_instances,
        max_requests_per_instance=args.max_requests_per_runtime,
        max_queued_requests_per_instance=args.max_queued_requests_per_runtime,
        max_runtime_memory_bytes=args.max_runtime_memory or None,
        automatic_memory_budget=(
            not args.no_memory_guard and args.max_runtime_memory is None
        ),
        default_idle_ttl_seconds=args.runtime_idle_timeout,
        backend=selected_backend,
        voice_component=voice_component,
        runtime_environment=runtime_environment,
        startup_loads=startup_loads,
        shared_cache_reclaimer=clear_image_decode_cache,
        transport=args.transport,
    )
    database_path, media_root = _server_storage_paths(data_dir, args.db)
    store = SessionStore(database_path, media_root=media_root)
    backend = ClusterBackend(runtime_manager, store)
    binary_dir = Path(sys.executable).parent
    perplexity = executable.with_name("mfq-perplexity")
    handlers = ToolJobHandlers(
        catalog,
        ToolJobPaths(
            work_root=work_dir,
            python=Path(sys.executable),
            modelscope=(
                (binary_dir / "modelscope") if (binary_dir / "modelscope").is_file() else None
            ),
            huggingface=(binary_dir / "hf") if (binary_dir / "hf").is_file() else None,
            runtime=executable,
            perplexity=perplexity if perplexity.is_file() else None,
            standalone_cli=bool(getattr(sys, "frozen", False)),
            internal_modelscope=importlib.util.find_spec("modelscope_hub") is not None,
            internal_huggingface=importlib.util.find_spec("huggingface_hub") is not None,
        ),
        voice_component=voice_component,
        activate_voice_output=runtime_manager.enable_realtime,
    )
    jobs = JobManager(store, handlers.handlers())
    service = ServerService(
        store,
        backend,
        jobs=jobs,
        catalog=catalog,
        runtime_manager=runtime_manager,
        tool_handlers=handlers,
        cluster=backend,
        voice_component=voice_component,
    )
    api_keys = ApiKeyManager(store, client_api_key) if client_api_key else None
    if web_root is None:
        print("Web UI assets were not found; serving the API only")
    uvicorn.run(
        create_app(service, web_root=web_root, api_keys=api_keys),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.access_log,
    )
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="start the MFQ inference server and optionally load a model",
        description=(
            "Resolve the native runtime for this machine and expose the MFQ API and Web UI. "
            "An initial MFQ model is optional; additional models can be loaded through the API."
        ),
    )
    parser.add_argument("--model", type=Path, help="optional MFQ model to load at startup")
    parser.add_argument(
        "--running-executable",
        type=Path,
        help="prebuilt native runtime executable; skips managed runtime lookup and compilation",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "public API bind host (default: 127.0.0.1; non-loopback binds "
            "require an API key)"
        ),
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=8090,
        help="public API bind port (default: 8090)",
    )
    parser.add_argument("--context-size", type=_nonnegative_int, default=0)
    parser.add_argument("--prefill-chunk-size", type=_positive_int, default=2048)
    parser.add_argument(
        "--moe-gpu-cache-gb",
        type=_nonnegative_float,
        help=(
            "resident routed-expert cache in GiB; zero keeps experts resident, "
            "while native HF Metal models otherwise choose an automatic budget"
        ),
    )
    parser.add_argument(
        "--prefix-cache-dir",
        type=Path,
        help="persistent Session KV cache directory",
    )
    parser.add_argument(
        "--prefix-cache-disk-size",
        type=_byte_size,
        help="maximum persistent Session KV cache size, such as 100G",
    )
    parser.add_argument(
        "--prefix-cache-hot-size",
        type=_byte_size,
        help="maximum RAM hot-cache size, such as 2G",
    )
    parser.add_argument(
        "--prefix-cache-block-tokens",
        type=_positive_int,
        help="tokens per content-addressed Session KV block",
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="disable persistent Session KV caching",
    )
    parser.add_argument("--runtime-startup-timeout", type=_positive_float, default=1800.0)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".mfq"),
        help="managed server data directory (default: ./.mfq)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help=(
            "deprecated explicit session database path; overrides "
            "<data-dir>/server.sqlite3"
        ),
    )
    web = parser.add_mutually_exclusive_group()
    web.add_argument("--web-root", type=Path)
    web.add_argument("--no-web-ui", action="store_true")
    parser.add_argument("--model-dir", action="append", type=Path, default=[])
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--api-key-env",
        default="MFQ_SERVER_API_KEY",
        help=(
            "environment variable containing the root API key "
            "(required for non-loopback binds)"
        ),
    )
    parser.add_argument("--max-runtime-instances", type=_positive_int, default=2)
    parser.add_argument("--max-requests-per-runtime", type=_positive_int, default=1)
    parser.add_argument(
        "--max-queued-requests-per-runtime",
        type=_nonnegative_int,
        help="maximum waiting requests per model runtime (default: max(32, 4x concurrency))",
    )
    parser.add_argument(
        "--max-runtime-memory",
        type=_byte_size,
        help="aggregate resident-model admission budget, such as 96G",
    )
    parser.add_argument(
        "--no-memory-guard",
        action="store_true",
        help="disable the automatic Metal model-residency budget",
    )
    parser.add_argument(
        "--runtime-idle-timeout",
        type=_nonnegative_int,
        help="default seconds before an idle unpinned model is unloaded",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--access-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="emit one HTTP access-log line per request (default: enabled)",
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "metal"), default="auto")
    parser.add_argument(
        "--transport",
        type=str.lower,
        choices=("stdio", "http"),
        default="stdio",
        help="Python-to-native-runtime transport (default: stdio)",
    )
    parser.set_defaults(_impl=_run)
