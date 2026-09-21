"""Managed MFQ runtime processes and model-aware request routing."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from mfq.server.catalog import DiscoveredModel, ModelArtifactNotFoundError, ModelCatalog
from mfq.server.host_memory import host_memory_snapshot, total_physical_memory
from mfq.server.jobs import JobContext, JobExecutionError
from mfq.server.models import (
    ErrorDetail,
    ModelLoadRequest,
    ModelUnloadRequest,
    ResponseFormat,
    RuntimeCapabilitiesResource,
    RuntimeInstanceList,
    RuntimeInstanceResource,
    RuntimeInstanceState,
    RuntimeLogLevel,
    SamplingParams,
    ToolChoice,
    ToolDefinition,
    UpdateRuntimeInstanceRequest,
)
from mfq.server.runtime.backend import (
    BackendDelta,
    BackendError,
    ChatBackend,
    OpenAIChatBackend,
    closing_backend_stream,
    preflight_backend_request,
)
from mfq.server.runtime.client import HttpRuntimeClient, StdioRuntimeClient
from mfq.server.runtime.native import (
    append_native_prefill_chunk_override,
    find_native_runtime_resource,
    native_request_capacity,
    native_runtime_environment,
    native_tokenizer_arguments,
)


class RuntimeManagementError(RuntimeError):
    pass


class RuntimeInstanceNotFoundError(RuntimeManagementError):
    pass


class RuntimeConflictError(RuntimeManagementError):
    pass


_AUTOMATIC_MEMORY_SOFT_RATIO = 0.90
_AUTOMATIC_MEMORY_HARD_RATIO = 0.95
_AUTOMATIC_MEMORY_TARGET_RATIO = 0.85


def _job_error(code: str, message: str, *, retryable: bool = False) -> JobExecutionError:
    return JobExecutionError(ErrorDetail(code=code, message=message, retryable=retryable))


class _Runtime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID
    artifact: DiscoveredModel
    process: SkipValidation[asyncio.subprocess.Process | subprocess.Popen[bytes]]
    backend: SkipValidation[ChatBackend]
    port: int
    context_size: int
    sampling_defaults: SamplingParams | None = None
    idle_ttl_seconds: int | None = None
    pinned: bool = False
    state: RuntimeInstanceState = RuntimeInstanceState.LOADING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime | None = None
    active_requests: int = 0
    queued_requests: int = 0
    control_leases: int = 0
    request_slots: asyncio.Semaphore | None = None
    request_capacity: int = 1
    mtp_supported: bool = False
    mtp_available: bool = False
    error: ErrorDetail | None = None
    output_task: asyncio.Task[None] | None = None
    monitor_task: asyncio.Task[None] | None = None
    retirement_task: asyncio.Task[None] | None = None
    retirement_retry_task: asyncio.Task[None] | None = None
    retirement_failures: int = 0
    realtime_gateway: Any | None = None
    realtime_error: str | None = None
    reserved_bytes: int | None = None
    resident_bytes: int | None = None
    kv_bytes: int | None = None
    usage_refreshed_at: float = 0.0


class _RuntimeLoadContext:
    """Job-context subset used by request-driven model activation."""

    def __init__(self) -> None:
        self._cleanup_callbacks: list[Callable[[], Awaitable[None] | None]] = []

    def raise_if_cancelled(self) -> None:
        return None

    def add_cleanup(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        self._cleanup_callbacks.append(callback)

    async def progress(
        self,
        value: float,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        del value, message, data

    async def log(self, message: str, **options: Any) -> None:
        del message, options

    async def cleanup(self) -> None:
        for callback in reversed(self._cleanup_callbacks):
            with suppress(Exception):
                result = callback()
                if result is not None:
                    await result


class _LeasedRealtimeConnector:
    """Hold a managed runtime resident for one upstream WebSocket lifetime."""

    def __init__(
        self,
        pool: RuntimePool,
        instance: _Runtime,
        connector: Any,
    ) -> None:
        self.pool = pool
        self.instance = instance
        self.connector = connector
        self.leased = False

    async def __aenter__(self) -> Any:
        async with self.pool._lock:
            if (
                self.pool._instances.get(self.instance.id) is not self.instance
                or self.instance.state
                not in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
            ):
                raise BackendError(
                    "model_not_ready",
                    "realtime runtime is no longer available",
                    retryable=True,
                    status_code=409,
                )
            self.instance.control_leases += 1
            self.leased = True
        try:
            return await self.connector.__aenter__()
        except BaseException:
            await self._release()
            raise

    async def __aexit__(self, *error: object) -> bool | None:
        try:
            return await self.connector.__aexit__(*error)
        finally:
            await self._release()

    async def _release(self) -> None:
        if not self.leased:
            return
        self.leased = False
        await self.pool._release_control_lease(self.instance)


class _CachedLoadFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    detail: ErrorDetail
    failed_at: float


class RuntimePool:
    """Own local runtime processes while retaining an optional external fallback."""

    def __init__(
        self,
        catalog: ModelCatalog,
        executable: str | Path,
        *,
        fallback: ChatBackend | None = None,
        startup_timeout_seconds: float = 1800.0,
        max_instances: int = 2,
        max_requests_per_instance: int = 1,
        max_queued_requests_per_instance: int | None = None,
        max_runtime_memory_bytes: int | None = None,
        default_idle_ttl_seconds: int | None = None,
        load_failure_cooldown_seconds: float = 30.0,
        metric_interval_seconds: float = 2.0,
        backend: str = "metal",
        voice_component: Any | None = None,
        runtime_environment: dict[str, str] | None = None,
        startup_loads: Sequence[ModelLoadRequest] = (),
        automatic_memory_budget: bool = True,
        shared_cache_reclaimer: Callable[[], int] | None = None,
        transport: str = "http",
    ) -> None:
        if max_instances < 1:
            raise ValueError("max_instances must be positive")
        if max_requests_per_instance < 1:
            raise ValueError("max_requests_per_instance must be positive")
        if (
            max_queued_requests_per_instance is not None
            and max_queued_requests_per_instance < 0
        ):
            raise ValueError("max_queued_requests_per_instance must be non-negative")
        if max_runtime_memory_bytes is not None and max_runtime_memory_bytes < 1:
            raise ValueError("max_runtime_memory_bytes must be positive")
        if default_idle_ttl_seconds is not None and default_idle_ttl_seconds < 0:
            raise ValueError("default_idle_ttl_seconds must be non-negative")
        if load_failure_cooldown_seconds < 0:
            raise ValueError("load_failure_cooldown_seconds must be non-negative")
        self.catalog = catalog
        self.executable = Path(executable).expanduser().resolve()
        self.fallback = fallback
        self.startup_timeout_seconds = startup_timeout_seconds
        self.max_instances = max_instances
        self.max_requests_per_instance = max_requests_per_instance
        self.max_queued_requests_per_instance = (
            max(32, max_requests_per_instance * 4)
            if max_queued_requests_per_instance is None
            else max_queued_requests_per_instance
        )
        automatic_budget = (
            self._default_runtime_memory_budget(backend)
            if automatic_memory_budget and max_runtime_memory_bytes is None
            else None
        )
        self.max_runtime_memory_bytes = max_runtime_memory_bytes or automatic_budget
        self.automatic_memory_budget = automatic_budget is not None
        self.default_idle_ttl_seconds = default_idle_ttl_seconds
        self.load_failure_cooldown_seconds = load_failure_cooldown_seconds
        self.metric_interval_seconds = max(0.25, metric_interval_seconds)
        if backend not in {"cuda", "metal"}:
            raise ValueError(f"unsupported native backend: {backend}")
        if transport not in {"stdio", "http"}:
            raise ValueError(f"unsupported native runtime transport: {transport}")
        self.backend = backend
        self.transport = transport
        self.voice_component = voice_component
        self.runtime_environment = dict(runtime_environment or {})
        self._startup_loads = [request.model_copy(deep=True) for request in startup_loads]
        self.shared_cache_reclaimer = shared_cache_reclaimer
        self._shared_cache_reclaims = 0
        self._shared_cache_released_bytes = 0
        self._shared_cache_reclaim_failures = 0
        self._shared_cache_pressure_checked_at = 0.0
        self.store = None
        self._instances: dict[UUID, _Runtime] = {}
        self._loading_model_names: set[str] = set()
        self._load_events: dict[str, asyncio.Event] = {}
        self._load_errors: dict[str, ErrorDetail] = {}
        self._load_failures: dict[str, _CachedLoadFailure] = {}
        self._runtime_revival_at: dict[str, float] = {}
        self._loading_artifact_ids: dict[str, str] = {}
        self._load_requests: dict[str, ModelLoadRequest] = {}
        self._reserved_ports: set[int] = set()
        self._load_ports: dict[str, int] = {}
        self._load_bytes: dict[str, int] = {}
        self._session_routes: dict[UUID, UUID] = {}
        self._last_instance_id: UUID | None = None
        self._lock = asyncio.Lock()
        self._realtime_activation_lock = asyncio.Lock()
        self._idle_reaper_task: asyncio.Task[None] | None = None
        self._idle_reaper_wakeup = asyncio.Event()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._lease_release_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    @staticmethod
    def _default_runtime_memory_budget(backend: str) -> int | None:
        """Return a conservative automatic residency ceiling for Metal."""

        if backend != "metal":
            return None
        total_bytes = total_physical_memory()
        if total_bytes is None:
            return None
        if total_bytes <= 0:
            return None
        reserve = 4 << 30 if total_bytes < 24 << 30 else 6 << 30
        if total_bytes <= reserve:
            return max(1, total_bytes * 3 // 4)
        return total_bytes - reserve

    async def start(self) -> None:
        """Start lifecycle monitors and configured startup models."""

        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            for instance in self._instances.values():
                if instance.monitor_task is None:
                    instance.monitor_task = asyncio.create_task(
                        self._monitor(instance),
                        name=f"mfq-server-runtime-monitor-{instance.id}",
                    )
            if self._idle_reaper_task is None or self._idle_reaper_task.done():
                self._idle_reaper_wakeup.clear()
                self._idle_reaper_task = asyncio.create_task(
                    self._idle_reaper(),
                    name="mfq-server-runtime-idle-reaper",
                )
            if self.voice_component is not None and self.voice_component.ready():
                task = asyncio.create_task(
                    self.enable_realtime(),
                    name="mfq-server-enable-voice-output",
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_task_done)
            startup_loads = tuple(self._startup_loads)
            self._startup_loads.clear()

        for index, request in enumerate(startup_loads):
            context = _RuntimeLoadContext()
            try:
                await self.load(
                    context,  # type: ignore[arg-type]
                    request.model_dump(mode="python"),
                )
            except BaseException:
                async with self._lock:
                    self._startup_loads[:0] = startup_loads[index:]
                raise
            finally:
                await context.cleanup()

    def _background_task_done(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.discard(task)
        if not task.cancelled():
            # Retrieve unexpected failures so optional background work cannot
            # produce an unobserved-task warning during server shutdown.
            task.exception()

    def register_started(
        self,
        *,
        artifact: DiscoveredModel,
        process: subprocess.Popen[bytes],
        backend: ChatBackend,
        port: int,
        load_request: ModelLoadRequest,
    ) -> UUID:
        """Register a ready process started before the server event loop exists."""

        if self._closed:
            raise RuntimeManagementError("runtime pool is closed")
        status = process.poll()
        if status is not None:
            raise RuntimeManagementError(
                f"initial runtime process has already exited with status {status}"
            )
        if any(
            item.artifact.resource.name == artifact.resource.name
            for item in self._instances.values()
        ):
            raise RuntimeConflictError(f"model is already loaded: {artifact.resource.name}")
        active_count = sum(
            item.state != RuntimeInstanceState.FAILED for item in self._instances.values()
        )
        if active_count >= self.max_instances:
            raise RuntimeConflictError("managed runtime instance limit reached")
        request_capacity = native_request_capacity(
            backend=self.backend,
            routed_expert_bytes=artifact.routed_expert_bytes,
            requested=self.max_requests_per_instance,
        )
        canonical_request = load_request.model_copy(
            update={"model": artifact.resource.name}
        )
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=process,
            backend=backend,
            port=port,
            context_size=canonical_request.context_size,
            sampling_defaults=canonical_request.sampling_defaults,
            state=RuntimeInstanceState.READY,
            last_used_at=datetime.now(timezone.utc),
            idle_ttl_seconds=(
                canonical_request.idle_ttl_seconds
                if canonical_request.idle_ttl_seconds is not None
                else self.default_idle_ttl_seconds
            ),
            pinned=canonical_request.pin,
            request_slots=asyncio.Semaphore(request_capacity),
            request_capacity=request_capacity,
            reserved_bytes=self._estimated_load_bytes(
                artifact,
                canonical_request,
            ),
        )
        self._instances[instance.id] = instance
        self._load_requests[artifact.resource.name] = canonical_request
        self._last_instance_id = instance.id
        return instance.id

    async def load(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = ModelLoadRequest.model_validate(payload)
        if self.backend == "metal" and request.device_ids not in ([], ["metal"]):
            raise _job_error("unsupported_device", "the Metal runtime accepts device 'metal'")
        if self.backend == "cuda" and any(not value.isdecimal() for value in request.device_ids):
            raise _job_error("unsupported_device", "CUDA device IDs must be non-negative integers")
        try:
            artifact = await self.catalog.resolve(request.model, request.artifact_uri)
        except ModelArtifactNotFoundError as error:
            raise _job_error(
                "model_artifact_not_found", f"model artifact was not found: {error}"
            ) from error
        if not artifact.resource.loadable:
            raise _job_error(
                (
                    "model_conversion_required"
                    if artifact.resource.complete and artifact.resource.format == "hf"
                    else "model_artifact_incomplete"
                ),
                artifact.resource.error or "model artifact is incomplete",
            )
        # Automatic residency is recalculated whenever an idle model is restored.
        replay_request = request.model_copy(deep=True)
        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            residency_ceiling = self._effective_runtime_memory_budget_locked()
        request = self._apply_automatic_expert_residency(
            artifact,
            request,
            memory_ceiling=residency_ceiling,
        )
        request_capacity = native_request_capacity(
            backend=self.backend,
            routed_expert_bytes=artifact.routed_expert_bytes,
            requested=self.max_requests_per_instance,
        )
        model_name = artifact.resource.name
        incoming_bytes = self._estimated_load_bytes(
            artifact,
            request,
        )
        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            if (
                self.max_runtime_memory_bytes is not None
                and incoming_bytes > self.max_runtime_memory_bytes
            ):
                raise _job_error(
                    "runtime_model_too_large",
                    f"model has an estimated resident set of {incoming_bytes} bytes "
                    f"but the runtime memory budget is "
                    f"{self.max_runtime_memory_bytes} bytes",
                )
        await self._reclaim_shared_cache_for_budget(
            additional_bytes=incoming_bytes,
            prospective_model=model_name,
        )
        await self._trim_idle_prefix_caches_for_budget(
            additional_bytes=incoming_bytes,
            prospective_model=model_name,
        )
        evicted: list[_Runtime] = []
        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            existing = next(
                (
                    item
                    for item in self._instances.values()
                    if item.artifact.resource.name == model_name
                ),
                None,
            )
            if existing is not None and existing.state == RuntimeInstanceState.FAILED:
                self._detach_instance_locked(existing)
                existing = None
            revision_changed = existing is not None and not self._same_artifact_revision(
                existing.artifact,
                artifact,
            )
            replace_named_revision = (
                revision_changed
                and request.artifact_uri is None
                and request.model == model_name
            )
            if existing is not None and replace_named_revision:
                if (
                    existing.state != RuntimeInstanceState.READY
                    or existing.active_requests
                    or existing.queued_requests
                    or existing.control_leases
                ):
                    raise _job_error(
                        "runtime_revision_busy",
                        "the model artifact changed while its current runtime is busy",
                        retryable=True,
                    )
                evicted.append(existing)
                existing = None
            if existing is not None:
                raise _job_error(
                    "model_already_loaded",
                    f"model is already loaded by runtime instance {existing.id}",
                )
            if model_name in self._loading_model_names:
                raise _job_error(
                    "model_already_loading",
                    f"model is already loading: {model_name}",
                    retryable=True,
                )
            claimed_instance_ids = {item.id for item in evicted}
            resident_names = {
                item.artifact.resource.name
                for item in self._instances.values()
                if item.state != RuntimeInstanceState.FAILED
                and item.id not in claimed_instance_ids
            }
            active_count = len(resident_names) + sum(
                name not in resident_names for name in self._loading_model_names
            )
            committed_bytes = max(
                0,
                self._committed_pool_bytes_locked()
                - sum(self._committed_runtime_bytes(item) for item in evicted),
            )
            memory_ceiling = self._effective_runtime_memory_budget_locked()
            while (
                active_count >= self.max_instances
                or (
                    memory_ceiling is not None
                    and committed_bytes + incoming_bytes > memory_ceiling
                )
            ):
                victim = self._lru_instance_for_unload_locked(
                    excluded_ids=claimed_instance_ids,
                )
                if victim is None:
                    memory_limited = (
                        memory_ceiling is not None
                        and committed_bytes + incoming_bytes > memory_ceiling
                    )
                    raise _job_error(
                        "runtime_memory_limit" if memory_limited else "runtime_instance_limit",
                        (
                            "runtime memory budget reached; all remaining instances "
                            "are pinned or busy"
                            if memory_limited
                            else "managed runtime instance limit reached; all instances "
                            "are pinned or busy"
                        ),
                        retryable=True,
                    )
                evicted.append(victim)
                claimed_instance_ids.add(victim.id)
                active_count = max(0, active_count - 1)
                committed_bytes = max(
                    0,
                    committed_bytes - self._committed_runtime_bytes(victim),
                )
            port = self._reserve_free_port_locked() if self.transport == "http" else 0
            for victim in evicted:
                self._mark_instance_unloading_locked(victim)
            load_event = asyncio.Event()
            self._loading_model_names.add(model_name)
            self._load_events[model_name] = load_event
            self._load_errors.pop(model_name, None)
            self._load_failures.pop(model_name, None)
            self._loading_artifact_ids[model_name] = artifact.resource.id
            if port:
                self._load_ports[model_name] = port
            self._load_bytes[model_name] = incoming_bytes

        if evicted:
            try:
                for victim in evicted:
                    await context.log(
                        f"Evicting idle runtime {victim.artifact.resource.name} "
                        f"before loading {model_name}"
                    )
                    await self._retire_instance(victim)
            except BaseException as error:
                for victim in evicted:
                    with suppress(Exception):
                        await self._retire_instance(victim)
                async with self._lock:
                    self._finish_model_load_locked(
                        model_name,
                        load_event,
                        self._runtime_load_error(error),
                    )
                raise

        try:
            command, process_environment = self._launch_configuration(
                artifact,
                request,
                port=port,
            )
            await context.progress(0.02, message="Starting runtime process")
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=(
                    asyncio.subprocess.PIPE
                    if self.transport == "stdio"
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=(
                    asyncio.subprocess.PIPE
                    if self.transport == "stdio"
                    else asyncio.subprocess.STDOUT
                ),
                start_new_session=True,
                env=process_environment,
                limit=64 * 1024 * 1024,
            )
        except BaseException as error:
            async with self._lock:
                self._finish_model_load_locked(
                    model_name,
                    load_event,
                    self._runtime_load_error(error),
                )
            raise
        configured_video_library = process_environment.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY")
        avfoundation_video_library = (
            Path(configured_video_library).expanduser().resolve()
            if configured_video_library
            else find_native_runtime_resource(
                self.executable,
                "libmfq_avfoundation_video.dylib",
            )
        )
        runtime_client = (
            StdioRuntimeClient(process)
            if self.transport == "stdio"
            else HttpRuntimeClient(f"http://127.0.0.1:{port}")
        )
        backend = OpenAIChatBackend(
            runtime_client,
            local_tensor_files=True,
            model_type=artifact.resource.architecture,
            avfoundation_video_library=(
                avfoundation_video_library
                if avfoundation_video_library is not None
                and avfoundation_video_library.is_file()
                else None
            ),
        )
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=process,
            backend=backend,
            port=port,
            context_size=request.context_size,
            sampling_defaults=request.sampling_defaults,
            idle_ttl_seconds=(
                request.idle_ttl_seconds
                if request.idle_ttl_seconds is not None
                else self.default_idle_ttl_seconds
            ),
            pinned=request.pin,
            request_slots=asyncio.Semaphore(request_capacity),
            request_capacity=request_capacity,
            reserved_bytes=incoming_bytes,
            # Keep TTL and memory-pressure eviction out of the load
            # transaction until the ready instance has been published.
            control_leases=1,
        )
        try:
            async with self._lock:
                if self._closed:
                    raise RuntimeManagementError("runtime pool is closed")
                self._instances[instance.id] = instance
        except BaseException as error:
            try:
                await self._stop_process(instance)
            finally:
                async with self._lock:
                    self._finish_model_load_locked(
                        model_name,
                        load_event,
                        self._runtime_load_error(error),
                    )
            raise

        keep_process = False

        async def cleanup_failed_start() -> None:
            if keep_process:
                return
            try:
                await self._retire_instance(instance)
            finally:
                async with self._lock:
                    self._finish_model_load_locked(
                        model_name,
                        load_event,
                        ErrorDetail(
                            code="runtime_start_failed",
                            message=f"runtime failed while loading {model_name}",
                            retryable=True,
                        ),
                    )

        context.add_cleanup(cleanup_failed_start)
        instance.output_task = asyncio.create_task(
            self._pump_output(instance, context),
            name=f"mfq-server-runtime-output-{instance.id}",
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout_seconds
        next_progress = 0.05
        while loop.time() < deadline:
            context.raise_if_cancelled()
            status = process.returncode
            if status is not None:
                raise _job_error(
                    "runtime_start_failed",
                    f"runtime exited during startup with status {status}",
                    retryable=True,
                )
            try:
                request_timeout = min(1.0, max(0.1, deadline - loop.time()))
                capabilities = await asyncio.wait_for(
                    backend.capabilities(), timeout=request_timeout
                )
            except (BackendError, TimeoutError):
                await asyncio.sleep(0.25)
                if next_progress < 0.9:
                    next_progress = min(0.9, next_progress + 0.005)
                    await context.progress(next_progress, message="Loading model")
                continue
            if capabilities.model != artifact.resource.name:
                raise _job_error(
                    "runtime_identity_mismatch",
                    "runtime health returned an unexpected model identity",
                )
            instance.mtp_supported = capabilities.model_capabilities.features.mtp
            instance.mtp_available = capabilities.mtp_available
            break
        else:
            raise _job_error(
                "runtime_start_timeout",
                "runtime did not become ready before the startup timeout",
                retryable=True,
            )

        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            instance.resident_bytes = await asyncio.to_thread(
                self._process_resident_bytes,
                pid,
            )
        async with self._lock:
            instance.state = RuntimeInstanceState.READY
            instance.last_used_at = datetime.now(timezone.utc)
            self._last_instance_id = instance.id
        await self._refresh_instance_usage(instance)
        instance.monitor_task = asyncio.create_task(
            self._monitor(instance), name=f"mfq-server-runtime-monitor-{instance.id}"
        )
        if self.voice_component is not None and self.voice_component.ready():
            await context.progress(0.96, message="Enabling voice output")
            await self.enable_realtime(instance.id)
        await context.progress(1.0, message="Model ready")
        async with self._lock:
            if (
                self._instances.get(instance.id) is not instance
                or instance.state not in {
                    RuntimeInstanceState.READY,
                    RuntimeInstanceState.BUSY,
                }
            ):
                raise _job_error(
                    "runtime_start_failed",
                    "runtime exited before model activation completed",
                    retryable=True,
                )
            self._load_requests[model_name] = replay_request.model_copy(
                update={"model": model_name}
            )
            self._finish_model_load_locked(model_name, load_event, None)
            keep_process = True
        await self._release_control_lease(instance)
        return {
            "instance_id": str(instance.id),
            "model_id": artifact.resource.name,
            "model": artifact.resource.name,
            "artifact_id": artifact.resource.id,
            "context_size": request.context_size,
        }

    async def unload(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = ModelUnloadRequest.model_validate(payload)
        async with self._lock:
            instance = self._instances.get(request.instance_id)
            if instance is None:
                raise _job_error(
                    "runtime_instance_not_found",
                    f"runtime instance was not found: {request.instance_id}",
                )
            if (
                instance.active_requests
                or instance.queued_requests
                or instance.control_leases
            ) and not request.force:
                raise _job_error(
                    "runtime_busy",
                    "runtime has active requests, queued requests, or control operations",
                    retryable=True,
                )
            instance.state = RuntimeInstanceState.UNLOADING

        released = False

        async def cleanup_incomplete_unload() -> None:
            if released:
                return
            await self._retire_instance(instance)

        context.add_cleanup(cleanup_incomplete_unload)
        await context.progress(0.2, message="Stopping runtime")
        await self._retire_instance(instance)
        await context.progress(0.9, message="Releasing runtime")
        async with self._lock:
            released = True
        await context.progress(1.0, message="Model unloaded")
        return {"instance_id": str(instance.id), "unloaded": True}

    async def instances(self) -> RuntimeInstanceList:
        async with self._lock:
            values = list(self._instances.values())
            now = asyncio.get_running_loop().time()
            refresh = []
            for item in values:
                if (
                    item.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }
                    and now - item.usage_refreshed_at >= self.metric_interval_seconds
                ):
                    item.usage_refreshed_at = now
                    refresh.append(item)
            route_counts: dict[UUID, int] = {}
            for instance_id in self._session_routes.values():
                route_counts[instance_id] = route_counts.get(instance_id, 0) + 1
        if refresh:
            await asyncio.gather(
                *(self._refresh_instance_usage(item) for item in refresh)
            )
        return RuntimeInstanceList(
            data=[
                RuntimeInstanceResource(
                    id=item.id,
                    model=item.artifact.resource.name,
                    state=item.state,
                    devices=[self.backend],
                    active_sessions=route_counts.get(item.id, 0),
                    queued_requests=item.queued_requests,
                    resident_bytes=item.resident_bytes,
                    kv_bytes=item.kv_bytes,
                    context_size=item.context_size,
                    started_at=item.started_at,
                    last_used_at=item.last_used_at,
                    idle_ttl_seconds=item.idle_ttl_seconds,
                    pinned=item.pinned,
                    mtp_supported=item.mtp_supported,
                    mtp_available=item.mtp_available,
                    error=item.error,
                )
                for item in values
            ]
        )

    async def update_instance(
        self,
        instance_id: UUID,
        request: UpdateRuntimeInstanceRequest,
    ) -> RuntimeInstanceResource:
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise BackendError(
                    "runtime_instance_not_found",
                    f"runtime instance was not found: {instance_id}",
                    status_code=404,
                )
            if instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                raise BackendError(
                    "model_not_ready",
                    f"model runtime is {instance.state.value}",
                    retryable=True,
                    status_code=409,
                )
            updates: dict[str, Any] = {}
            if "pinned" in request.model_fields_set:
                instance.pinned = bool(request.pinned)
                updates["pin"] = instance.pinned
            if "idle_ttl_seconds" in request.model_fields_set:
                instance.idle_ttl_seconds = request.idle_ttl_seconds
                updates["idle_ttl_seconds"] = request.idle_ttl_seconds
            model_name = instance.artifact.resource.name
            load_request = self._load_requests.get(model_name)
            if load_request is not None and updates:
                self._load_requests[model_name] = load_request.model_copy(
                    update=updates
                )
            active_sessions = sum(
                routed_id == instance.id
                for routed_id in self._session_routes.values()
            )
            resource = RuntimeInstanceResource(
                id=instance.id,
                model=model_name,
                state=instance.state,
                devices=[self.backend],
                active_sessions=active_sessions,
                queued_requests=instance.queued_requests,
                resident_bytes=instance.resident_bytes,
                kv_bytes=instance.kv_bytes,
                context_size=instance.context_size,
                started_at=instance.started_at,
                last_used_at=instance.last_used_at,
                idle_ttl_seconds=instance.idle_ttl_seconds,
                pinned=instance.pinned,
                mtp_supported=instance.mtp_supported,
                mtp_available=instance.mtp_available,
                error=instance.error,
            )
            self._idle_reaper_wakeup.set()
        return resource

    async def preflight(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> None:
        instance = await self._select(model, session_id=session_id)
        if instance is not None and instance.state == RuntimeInstanceState.LOADING:
            await self._wait_for_model_ready(instance.artifact.resource.name)
            instance = await self._select(model, session_id=session_id)
        if instance is None:
            instance = await self._ensure_model_loaded_with_revival(model)
        if instance is None:
            if self.fallback is None:
                raise BackendError(
                    "model_not_loaded",
                    f"model is not loaded: {model}",
                    status_code=404,
                )
            await preflight_backend_request(
                self.fallback,
                model=model,
                messages=messages,
                sampling=sampling,
                session_id=session_id,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
            return
        async with self._lock:
            current = self._instances.get(instance.id) is instance
            ready = instance.state in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }
            process_exited = instance.process.returncode is not None
            queue_full = (
                instance.active_requests + instance.queued_requests
                >= instance.request_capacity
                + self.max_queued_requests_per_instance
            )
        if not current or not ready or process_exited:
            raise BackendError(
                "model_not_ready",
                f"model runtime is {instance.state.value}",
                retryable=True,
                status_code=503,
            )
        if queue_full:
            raise BackendError(
                "runtime_queue_full",
                f"model runtime queue is full: {model}",
                retryable=True,
                status_code=429,
            )

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> AsyncIterator[BackendDelta]:
        instance = await self._select(model, session_id=session_id)
        if instance is not None and instance.state == RuntimeInstanceState.LOADING:
            await self._wait_for_model_ready(instance.artifact.resource.name)
            instance = await self._select(model, session_id=session_id)
        if instance is None:
            instance = await self._ensure_model_loaded_with_revival(model)
        if instance is None:
            if self.fallback is None:
                raise BackendError("model_not_loaded", f"model is not loaded: {model}")
            async with closing_backend_stream(
                self.fallback.stream(
                    model=model,
                    messages=messages,
                    sampling=sampling,
                    session_id=session_id,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                )
            ) as fallback_stream:
                async for delta in fallback_stream:
                    yield delta
            return
        assert instance.request_slots is not None
        acquired = False
        async with self._lock:
            if self._instances.get(instance.id) is not instance or instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                raise BackendError("model_not_ready", f"model runtime is {instance.state.value}")
            if (
                instance.active_requests + instance.queued_requests
                >= instance.request_capacity
                + self.max_queued_requests_per_instance
            ):
                raise BackendError(
                    "runtime_queue_full",
                    f"model runtime queue is full: {model}",
                    retryable=True,
                    status_code=429,
                )
            instance.queued_requests += 1
        try:
            await instance.request_slots.acquire()
            acquired = True
        except BaseException:
            async with self._lock:
                instance.queued_requests = max(0, instance.queued_requests - 1)
            raise
        async with self._lock:
            instance.queued_requests = max(0, instance.queued_requests - 1)
            if self._instances.get(instance.id) is not instance or instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                instance.request_slots.release()
                raise BackendError("model_not_ready", f"model runtime is {instance.state.value}")
            instance.active_requests += 1
            instance.state = RuntimeInstanceState.BUSY
            instance.last_used_at = datetime.now(timezone.utc)
            if session_id is not None:
                self._session_routes[session_id] = instance.id
            self._last_instance_id = instance.id
        try:
            effective_sampling = self._sampling_for_instance(instance, sampling)
            async with closing_backend_stream(
                instance.backend.stream(
                    model=instance.artifact.resource.name,
                    messages=messages,
                    sampling=effective_sampling,
                    session_id=session_id,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                )
            ) as backend_stream:
                async for delta in backend_stream:
                    yield delta
        finally:
            async with self._lock:
                instance.active_requests = max(0, instance.active_requests - 1)
                if acquired:
                    instance.request_slots.release()
                if instance.state == RuntimeInstanceState.BUSY and instance.active_requests == 0:
                    instance.state = RuntimeInstanceState.READY
                instance.last_used_at = datetime.now(timezone.utc)
                # A memory-budget pass may have skipped this runtime while its
                # request was active. Re-evaluate as soon as the final request
                # drains instead of retaining an over-budget process until the
                # next periodic metrics tick.
                if instance.active_requests == 0:
                    self._idle_reaper_wakeup.set()

    @staticmethod
    def _sampling_for_instance(
        instance: _Runtime,
        requested: SamplingParams,
    ) -> SamplingParams:
        defaults = instance.sampling_defaults
        if defaults is None:
            return requested
        return defaults.model_copy(
            update={
                name: getattr(requested, name)
                for name in requested.model_fields_set
            }
        )

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        async with self._lock:
            instance_id = self._session_routes.get(source_session_id)
            instance = self._instances.get(instance_id) if instance_id is not None else None
            if instance is not None and instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                self._session_routes.pop(source_session_id, None)
                instance = None
            if instance is not None:
                instance.control_leases += 1
        if instance is None:
            return (
                await self.fallback.fork_session(source_session_id, target_session_id)
                if self.fallback
                else False
            )
        try:
            succeeded = await instance.backend.fork_session(
                source_session_id,
                target_session_id,
            )
            if succeeded:
                async with self._lock:
                    if self._instances.get(instance.id) is instance:
                        self._session_routes[target_session_id] = instance.id
            return succeeded
        finally:
            await self._release_control_lease(instance)

    async def close_session(self, session_id: UUID) -> bool:
        async with self._lock:
            instance_id = self._session_routes.pop(session_id, None)
            instance = self._instances.get(instance_id) if instance_id is not None else None
            if instance is not None and instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                instance = None
            if instance is not None:
                instance.control_leases += 1
        if instance is None:
            return await self.fallback.close_session(session_id) if self.fallback else False
        try:
            return await instance.backend.close_session(session_id)
        finally:
            await self._release_control_lease(instance)

    async def cancel_response(self, session_id: UUID) -> bool:
        async with self._lock:
            instance_id = self._session_routes.get(session_id)
            instance = self._instances.get(instance_id) if instance_id is not None else None
            if instance is not None and instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                self._session_routes.pop(session_id, None)
                instance = None
            if instance is not None:
                instance.control_leases += 1
        backend = instance.backend if instance is not None else self.fallback
        if backend is None:
            return False
        cancel = getattr(backend, "cancel_response", None)
        try:
            return bool(await cancel(session_id)) if callable(cancel) else False
        finally:
            if instance is not None:
                await self._release_control_lease(instance)

    async def capabilities(
        self,
        instance_id: UUID | None = None,
    ) -> RuntimeCapabilitiesResource:
        async with self._runtime_control_lease(instance_id) as (_instance, backend):
            if backend is None:
                raise BackendError("model_not_loaded", "no runtime is available")
            return await backend.capabilities()

    async def runtime_status(
        self,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            (
                memory_pressure_level,
                memory_pressure_ratio,
                effective_memory_budget,
                committed_memory,
            ) = self._runtime_memory_pressure_locked()
            memory_status = {
                "runtime_memory_budget_bytes": self.max_runtime_memory_bytes,
                "runtime_memory_effective_budget_bytes": effective_memory_budget,
                "runtime_memory_budget_mode": (
                    "automatic"
                    if self.automatic_memory_budget
                    else "explicit"
                    if self.max_runtime_memory_bytes is not None
                    else "disabled"
                ),
                "runtime_memory_committed_bytes": committed_memory,
                "runtime_memory_headroom_bytes": (
                    max(0, effective_memory_budget - committed_memory)
                    if effective_memory_budget is not None
                    else None
                ),
                "runtime_memory_pressure_level": memory_pressure_level,
                "runtime_memory_pressure_ratio": memory_pressure_ratio,
                "runtime_memory_shared_cache_reclaims": self._shared_cache_reclaims,
                "runtime_memory_shared_cache_released_bytes": (
                    self._shared_cache_released_bytes
                ),
                "runtime_memory_shared_cache_reclaim_failures": (
                    self._shared_cache_reclaim_failures
                ),
            }
        async with self._runtime_control_lease(
            instance_id,
            allow_unready=True,
        ) as (instance, backend):
            if instance is not None and instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                return {
                    **memory_status,
                    "instance_id": str(instance.id),
                    "runtime_state": instance.state.value,
                    "model": instance.artifact.resource.name,
                    "active_requests": instance.active_requests,
                    "queued_requests": instance.queued_requests,
                    "reloading": False,
                    "error": (
                        instance.error.model_dump(mode="json")
                        if instance.error is not None
                        else None
                    ),
                }
            if backend is None:
                return {
                    **memory_status,
                    "runtime_state": "idle",
                    "model": None,
                    "active_requests": 0,
                    "total_requests": 0,
                    "failed_requests": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "reloading": False,
                }
            status = dict(await backend.runtime_status())
            status.update(memory_status)
            if instance is not None:
                status["instance_id"] = str(instance.id)
                status["runtime_state"] = instance.state.value
                if instance.sampling_defaults is not None:
                    status["sampling_defaults"] = (
                        instance.sampling_defaults.model_dump(mode="json")
                    )
                pid = getattr(instance.process, "pid", None)
                resident_bytes = (
                    await asyncio.to_thread(self._process_resident_bytes, pid)
                    if isinstance(pid, int) and pid > 0
                    else None
                )
                status["process_resident_bytes"] = resident_bytes
                kv_value = status.get(
                    "prefix_cache_hot_bytes",
                    status.get("prefix_cache_bytes"),
                )
                observed_resident_bytes = self._observed_runtime_bytes(
                    resident_bytes,
                    status,
                )
                async with self._lock:
                    if self._instances.get(instance.id) is instance:
                        if observed_resident_bytes is not None:
                            instance.resident_bytes = observed_resident_bytes
                        if isinstance(kv_value, (int, float)) and kv_value >= 0:
                            instance.kv_bytes = int(kv_value)
            return status

    async def runtime_models(self) -> dict[str, Any]:
        async with self._lock:
            managed = [
                item
                for item in self._instances.values()
                if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
            ]
        data = []
        for instance in managed:
            data.append(
                {
                    "id": instance.artifact.resource.name,
                    "object": "model",
                    "model": instance.artifact.resource.name,
                    "state": instance.state.value,
                    "instance_id": str(instance.id),
                }
            )
        if not data and self.fallback is not None:
            return await self.fallback.runtime_models()
        return {"object": "list", "data": data}

    async def realtime_capabilities(self) -> dict[str, Any]:
        async with self._lock:
            instance = self._current_instance_locked()
            gateway = instance.realtime_gateway if instance is not None else None
            if gateway is not None:
                instance.control_leases += 1
        if instance is not None:
            if gateway is None:
                return {"available": False, "modes": []}
            try:
                return await gateway.capabilities()
            finally:
                await self._release_control_lease(instance)
        if self.fallback is None:
            return {"available": False, "modes": []}
        return await self.fallback.realtime_capabilities()

    async def voice_output_status(self) -> dict[str, Any]:
        async with self._lock:
            instance = self._current_instance_locked()
        architecture = instance.artifact.resource.architecture if instance is not None else ""
        supported = self._supports_voice_output(architecture)
        return {
            "active": bool(instance is not None and instance.realtime_gateway is not None),
            "supported_model_loaded": bool(instance is not None and supported),
            "model": instance.artifact.resource.name if instance is not None else None,
            "error": instance.realtime_error if instance is not None else None,
        }

    async def enable_realtime(self, instance_id: UUID | None = None) -> dict[str, Any]:
        async with self._realtime_activation_lock:
            async with self._lock:
                instance = (
                    self._instances.get(instance_id)
                    if instance_id is not None
                    else self._current_instance_locked()
                )
                if instance is not None and instance.state in {
                    RuntimeInstanceState.READY,
                    RuntimeInstanceState.BUSY,
                }:
                    instance.control_leases += 1
                else:
                    instance = None
            if instance is None:
                return {"active": False, "reason": "model_not_loaded"}
            try:
                return await self._activate_realtime_instance(instance)
            finally:
                await self._release_control_lease(instance)

    async def _activate_realtime_instance(
        self,
        instance: _Runtime,
    ) -> dict[str, Any]:
        if not self._supports_voice_output(instance.artifact.resource.architecture):
            return {"active": False, "reason": "model_not_supported"}
        if self.voice_component is None or not self.voice_component.ready():
            return {"active": False, "reason": "component_not_ready"}
        if instance.realtime_gateway is not None:
            return {"active": True, "model": instance.artifact.resource.name}
        base_url = getattr(instance.backend, "base_url", None)
        if not isinstance(base_url, str) or not base_url:
            return {"active": False, "reason": "backend_not_supported"}
        try:
            from mfq.runtime.minicpmo45_realtime import (
                RealtimeGateway,
                _backend_token2wav_steps,
            )

            def create_gateway() -> Any:
                steps = _backend_token2wav_steps(base_url, "", None)
                return RealtimeGateway(
                    base_url,
                    self.voice_component.root,
                    token2wav_steps=steps,
                )

            gateway = await asyncio.to_thread(create_gateway)
        except Exception as error:
            instance.realtime_error = str(error)
            return {"active": False, "reason": "activation_failed", "error": str(error)}
        async with self._lock:
            if self._instances.get(instance.id) is not instance:
                return {"active": False, "reason": "model_unloaded"}
            instance.realtime_gateway = gateway
            instance.realtime_error = None
        return {"active": True, "model": instance.artifact.resource.name}

    async def realtime_serve(self, client: Any, *, mode: str = "audio") -> bool:
        if mode != "audio":
            return False
        async with self._lock:
            instance = self._current_instance_locked()
            gateway = instance.realtime_gateway if instance is not None else None
            if gateway is not None:
                instance.control_leases += 1
        if gateway is None:
            return False
        try:
            await gateway.serve(client)
            return True
        finally:
            assert instance is not None
            await self._release_control_lease(instance)

    async def reload_runtime(
        self,
        context_size: int,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        async with self._runtime_control_lease(instance_id) as (instance, backend):
            if backend is None:
                raise BackendError("model_not_loaded", "no runtime is available")
            result = await backend.reload_runtime(context_size)
            if instance is not None:
                async with self._lock:
                    if self._instances.get(instance.id) is instance:
                        instance.context_size = context_size
            return result

    async def clear_runtime_cache(
        self,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        async with self._runtime_control_lease(instance_id) as (_instance, backend):
            if backend is None:
                raise BackendError("model_not_loaded", "no runtime is available")
            return await backend.clear_runtime_cache()

    async def trim_runtime_cache(
        self,
        target_bytes: int = 0,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        if target_bytes < 0:
            raise ValueError("target_bytes must be non-negative")
        async with self._runtime_control_lease(instance_id) as (instance, backend):
            if backend is None:
                raise BackendError("model_not_loaded", "no runtime is available")
            trim = getattr(backend, "trim_runtime_cache", None)
            if not callable(trim):
                raise BackendError(
                    "unsupported_operation",
                    "this runtime does not expose a tiered prefix cache",
                    status_code=501,
                )
            result = await trim(target_bytes)
            if instance is not None:
                await self._refresh_instance_usage(instance)
            return result

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        instance = self._instances.get(self._last_instance_id) if self._last_instance_id else None
        if instance is not None and instance.state in {
            RuntimeInstanceState.READY,
            RuntimeInstanceState.BUSY,
        }:
            connector = instance.backend.realtime_connect(mode=mode)
            return _LeasedRealtimeConnector(self, instance, connector)
        if self.fallback is not None:
            return self.fallback.realtime_connect(mode=mode)
        raise BackendError("model_not_loaded", "no runtime is available")

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            instances = list(self._instances.values())
            idle_reaper = self._idle_reaper_task
            self._idle_reaper_task = None
            background_tasks = tuple(self._background_tasks)
            self._background_tasks.clear()
            self._idle_reaper_wakeup.set()
            for model, event in self._load_events.items():
                self._load_errors[model] = ErrorDetail(
                    code="runtime_pool_closed",
                    message="runtime pool closed while the model was loading",
                    retryable=True,
                )
                event.set()
            self._load_events.clear()
            self._loading_model_names.clear()
            self._loading_artifact_ids.clear()
            self._load_failures.clear()
            self._load_ports.clear()
            self._load_bytes.clear()
            self._reserved_ports.clear()
        if idle_reaper is not None and idle_reaper is not asyncio.current_task():
            await idle_reaper
        for task in background_tasks:
            if task is not asyncio.current_task() and not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        await self._drain_control_lease_releases()
        first_error: Exception | None = None
        retirements = await asyncio.gather(
            *(self._retire_instance(instance) for instance in instances),
            return_exceptions=True,
        )
        first_error = next(
            (
                result
                for result in retirements
                if isinstance(result, Exception)
            ),
            None,
        )
        async with self._lock:
            self._instances.clear()
            self._session_routes.clear()
        if self.fallback is not None:
            try:
                await self.fallback.aclose()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    async def _select(self, model: str, *, session_id: UUID | None) -> _Runtime | None:
        stale_backend: ChatBackend | None = None
        async with self._lock:
            if session_id is not None:
                instance_id = self._session_routes.get(session_id)
                if instance_id is not None:
                    routed = self._instances.get(instance_id)
                    if (
                        routed is not None
                        and self._matches_model(routed, model)
                        and routed.state in {
                            RuntimeInstanceState.READY,
                            RuntimeInstanceState.BUSY,
                        }
                    ):
                        return routed
                    self._session_routes.pop(session_id, None)
                    if routed is not None and routed.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }:
                        stale_backend = routed.backend
            matches = [
                item
                for item in self._instances.values()
                if self._matches_model(item, model)
                and item.state in {
                    RuntimeInstanceState.LOADING,
                    RuntimeInstanceState.READY,
                    RuntimeInstanceState.BUSY,
                }
            ]
        if len(matches) > 1:
            raise BackendError("ambiguous_model", f"multiple loaded runtimes match {model}")
        if stale_backend is not None and session_id is not None:
            close = getattr(stale_backend, "close_session", None)
            if callable(close):
                with suppress(Exception):
                    await close(session_id)
        return matches[0] if matches else None

    @staticmethod
    def _matches_model(instance: _Runtime, model: str) -> bool:
        resource = instance.artifact.resource
        return model == resource.name or model == resource.id

    @staticmethod
    def _same_artifact_revision(
        left: DiscoveredModel,
        right: DiscoveredModel,
    ) -> bool:
        return (
            left.resource.id == right.resource.id
            and left.resource.modified_at == right.resource.modified_at
            and left.path == right.path
        )

    def _begin_runtime_revival(self, model: str) -> bool:
        """Allow one immediate reload after an unexpected runtime exit."""

        now = time.monotonic()
        last = self._runtime_revival_at.get(model)
        if (
            last is not None
            and self.load_failure_cooldown_seconds > 0
            and now - last < self.load_failure_cooldown_seconds
        ):
            return False
        self._runtime_revival_at[model] = now
        return True

    async def _ensure_model_loaded_with_revival(
        self,
        model: str,
    ) -> _Runtime | None:
        try:
            return await self._ensure_model_loaded(model)
        except BackendError as error:
            if (
                error.code != "runtime_exited"
                or not error.retryable
                or not self._begin_runtime_revival(model)
            ):
                raise
            async with self._lock:
                self._load_failures.pop(model, None)
                self._load_errors.pop(model, None)
            return await self._ensure_model_loaded(model)

    async def _ensure_model_loaded(self, model: str) -> _Runtime | None:
        try:
            artifact = await self.catalog.resolve(model)
        except ModelArtifactNotFoundError:
            return None
        model_name = artifact.resource.name
        artifact_id = artifact.resource.id
        async with self._lock:
            if self._closed:
                return None
            ready = next(
                (
                    item
                    for item in self._instances.values()
                    if self._same_artifact_revision(item.artifact, artifact)
                    and item.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }
                ),
                None,
            )
            event = self._load_events.get(model_name)
            request = self._load_requests.get(model_name)
            cached_failure = self._load_failures.get(model_name)
            if cached_failure is not None and (
                cached_failure.artifact_id != artifact_id
                or self.load_failure_cooldown_seconds == 0
                or time.monotonic() - cached_failure.failed_at
                >= self.load_failure_cooldown_seconds
            ):
                self._load_failures.pop(model_name, None)
                self._load_errors.pop(model_name, None)
                cached_failure = None
        if ready is not None:
            return ready
        if event is not None:
            await self._wait_for_model_ready(model_name)
            return await self._select_loaded_artifact(model_name, artifact)
        if cached_failure is not None:
            raise self._backend_load_error(cached_failure.detail)

        exact_request = (request or ModelLoadRequest(model=model_name)).model_copy(
            update={
                "model": model_name,
                "artifact_uri": f"mfq://{artifact_id}",
            }
        )
        context = _RuntimeLoadContext()
        try:
            await self.load(
                context,  # type: ignore[arg-type]
                exact_request.model_dump(mode="python"),
            )
        except JobExecutionError as error:
            if error.detail.code in {
                "model_already_loaded",
                "model_already_loading",
            }:
                await self._wait_for_model_ready(model_name)
                return await self._select_loaded_artifact(model_name, artifact)
            if error.detail.code == "model_artifact_not_found":
                return None
            if error.detail.code == "runtime_revision_busy":
                raise self._backend_load_error(error.detail) from error
            async with self._lock:
                self._load_errors[model_name] = error.detail
            raise self._backend_load_error(error.detail) from error
        except Exception as error:
            detail = self._runtime_load_error(error)
            async with self._lock:
                self._load_errors[model_name] = detail
            raise self._backend_load_error(detail) from error
        finally:
            await context.cleanup()
        return await self._select_loaded_artifact(model_name, artifact)

    async def _select_loaded_artifact(
        self,
        model_name: str,
        artifact: DiscoveredModel,
    ) -> _Runtime | None:
        selected = await self._select(model_name, session_id=None)
        if selected is None:
            return None
        if not self._same_artifact_revision(selected.artifact, artifact):
            raise BackendError(
                "model_name_conflict",
                f"loaded model name {model_name!r} refers to a different artifact",
                status_code=409,
            )
        return selected

    async def _wait_for_model_ready(self, model: str) -> bool:
        async with self._lock:
            ready = next(
                (
                    item
                    for item in self._instances.values()
                    if item.artifact.resource.name == model
                    and item.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }
                ),
                None,
            )
            event = self._load_events.get(model)
            load_error = self._load_errors.get(model)
        if ready is not None:
            return True
        if event is None:
            if load_error is not None:
                raise self._backend_load_error(load_error)
            return False
        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=max(1.0, self.startup_timeout_seconds),
            )
        except TimeoutError:
            raise BackendError(
                "runtime_start_timeout",
                f"timed out waiting for model runtime: {model}",
                retryable=True,
                status_code=503,
            ) from None
        selected = await self._select(model, session_id=None)
        if selected is not None and selected.state in {
            RuntimeInstanceState.READY,
            RuntimeInstanceState.BUSY,
        }:
            return True
        async with self._lock:
            error = self._load_errors.get(model)
        if error is not None:
            raise self._backend_load_error(error)
        return False

    @asynccontextmanager
    async def _runtime_control_lease(
        self,
        instance_id: UUID | None,
        *,
        allow_unready: bool = False,
    ) -> AsyncIterator[tuple[_Runtime | None, ChatBackend | None]]:
        async with self._lock:
            if instance_id is None:
                instance = self._current_instance_locked()
            else:
                instance = self._instances.get(instance_id)
                if instance is None:
                    raise BackendError(
                        "runtime_instance_not_found",
                        f"runtime instance was not found: {instance_id}",
                        status_code=404,
                    )
                if not allow_unready and instance.state not in {
                    RuntimeInstanceState.READY,
                    RuntimeInstanceState.BUSY,
                }:
                    raise BackendError(
                        "model_not_ready",
                        f"model runtime is {instance.state.value}",
                        retryable=True,
                        status_code=409,
                    )
            backend = instance.backend if instance is not None else self.fallback
            leased = instance is not None and instance.state in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }
            if leased:
                instance.control_leases += 1
        try:
            yield instance, backend
        finally:
            if leased and instance is not None:
                await self._release_control_lease(instance)

    async def _decrement_control_lease(self, instance: _Runtime) -> None:
        async with self._lock:
            instance.control_leases = max(0, instance.control_leases - 1)

    def _finish_control_lease_release(self, task: asyncio.Task[None]) -> None:
        self._lease_release_tasks.discard(task)
        with suppress(asyncio.CancelledError):
            task.result()

    async def _release_control_lease(self, instance: _Runtime) -> None:
        """Release a lease even if its caller is cancelled during cleanup."""

        task = asyncio.create_task(
            self._decrement_control_lease(instance),
            name=f"mfq-server-runtime-lease-release-{instance.id}",
        )
        self._lease_release_tasks.add(task)
        task.add_done_callback(self._finish_control_lease_release)
        await asyncio.shield(task)

    async def _drain_control_lease_releases(self) -> None:
        while self._lease_release_tasks:
            await asyncio.gather(
                *tuple(self._lease_release_tasks),
                return_exceptions=True,
            )

    def _current_instance_locked(self) -> _Runtime | None:
        instance = (
            self._instances.get(self._last_instance_id)
            if self._last_instance_id is not None
            else None
        )
        if instance is None or instance.state not in {
            RuntimeInstanceState.READY,
            RuntimeInstanceState.BUSY,
        }:
            instance = next(
                (
                    item
                    for item in self._instances.values()
                    if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
                ),
                None,
            )
        return instance

    def _lru_instance_for_unload_locked(
        self,
        *,
        excluded_ids: set[UUID] | None = None,
    ) -> _Runtime | None:
        excluded = excluded_ids or set()
        candidates = [
            item
            for item in self._instances.values()
            if item.id not in excluded
            and not item.pinned
            and item.state == RuntimeInstanceState.READY
            and item.active_requests == 0
            and item.queued_requests == 0
            and item.control_leases == 0
        ]
        if not candidates:
            return None
        victim = min(
            candidates,
            key=lambda item: (item.last_used_at or item.started_at, item.started_at),
        )
        return victim

    def _claim_lru_instance_for_unload_locked(self) -> _Runtime | None:
        victim = self._lru_instance_for_unload_locked()
        if victim is None:
            return None
        self._mark_instance_unloading_locked(victim)
        return victim

    def _claim_over_budget_instances_for_unload_locked(
        self,
        *,
        memory_ceiling: int | None = None,
        pending_releases: Sequence[_Runtime] = (),
    ) -> list[_Runtime]:
        if memory_ceiling is None:
            memory_ceiling = self._effective_runtime_memory_budget_locked()
        if memory_ceiling is None:
            return []
        committed = max(
            0,
            self._committed_pool_bytes_locked()
            - sum(self._committed_runtime_bytes(item) for item in pending_releases),
        )
        victims = []
        while committed > memory_ceiling:
            victim = self._claim_lru_instance_for_unload_locked()
            if victim is None:
                break
            victims.append(victim)
            committed = max(
                0,
                committed - self._committed_runtime_bytes(victim),
            )
        return victims

    async def _run_shared_cache_reclaimer(self) -> int:
        reclaimer = self.shared_cache_reclaimer
        if reclaimer is None:
            return 0
        try:
            released = max(0, int(await asyncio.to_thread(reclaimer)))
        except Exception:
            async with self._lock:
                self._shared_cache_reclaim_failures += 1
            return 0
        async with self._lock:
            if released > 0:
                self._shared_cache_reclaims += 1
                self._shared_cache_released_bytes += released
        return released

    async def _reclaim_shared_cache_for_budget(
        self,
        *,
        additional_bytes: int,
        prospective_model: str | None = None,
    ) -> int:
        """Release server-process caches before trimming or evicting models."""

        reclaimer = self.shared_cache_reclaimer
        if reclaimer is None or not self.automatic_memory_budget:
            return 0
        async with self._lock:
            if self._closed:
                return 0
            if prospective_model is not None and (
                prospective_model in self._loading_model_names
                or any(
                    item.artifact.resource.name == prospective_model
                    and item.state != RuntimeInstanceState.FAILED
                    for item in self._instances.values()
                )
            ):
                return 0
            memory_ceiling = self._effective_runtime_memory_budget_locked()
            if memory_ceiling is None:
                return 0
            committed = self._committed_pool_bytes_locked() + additional_bytes
            if committed <= memory_ceiling:
                return 0
        return await self._run_shared_cache_reclaimer()

    async def _reclaim_shared_cache_under_pressure(self) -> int:
        if self.shared_cache_reclaimer is None or not self.automatic_memory_budget:
            return 0
        async with self._lock:
            if self._closed:
                return 0
            level, _ratio, _ceiling, _committed = (
                self._runtime_memory_pressure_locked()
            )
            if level == "normal":
                return 0
            now = time.monotonic()
            if now - self._shared_cache_pressure_checked_at < max(
                1.0,
                self.metric_interval_seconds * 2,
            ):
                return 0
            self._shared_cache_pressure_checked_at = now
        return await self._run_shared_cache_reclaimer()

    async def _trim_idle_prefix_caches_for_budget(
        self,
        *,
        additional_bytes: int = 0,
        prospective_model: str | None = None,
        memory_target_bytes: int | None = None,
    ) -> None:
        if self.max_runtime_memory_bytes is None and memory_target_bytes is None:
            return
        async with self._lock:
            if self._closed:
                return
            memory_ceiling = self._effective_runtime_memory_budget_locked()
            if memory_ceiling is None:
                return
            if memory_target_bytes is not None:
                memory_ceiling = min(memory_ceiling, memory_target_bytes)
            if prospective_model is not None and (
                prospective_model in self._loading_model_names
                or any(
                    item.artifact.resource.name == prospective_model
                    and item.state != RuntimeInstanceState.FAILED
                    for item in self._instances.values()
                )
            ):
                return
            if prospective_model is not None:
                resident_names = {
                    item.artifact.resource.name
                    for item in self._instances.values()
                    if item.state != RuntimeInstanceState.FAILED
                }
                active_count = len(resident_names) + sum(
                    name not in resident_names for name in self._loading_model_names
                )
                if active_count >= self.max_instances:
                    return
            committed = self._committed_pool_bytes_locked() + additional_bytes
            if committed <= memory_ceiling:
                return
            candidates = sorted(
                (
                    item
                    for item in self._instances.values()
                    if item.state == RuntimeInstanceState.READY
                    and item.active_requests == 0
                    and item.queued_requests == 0
                    and item.control_leases == 0
                    and (item.kv_bytes or 0) > 0
                    and callable(getattr(item.backend, "trim_runtime_cache", None))
                ),
                key=lambda item: (item.last_used_at or item.started_at, item.started_at),
            )

        for instance in candidates:
            async with self._lock:
                if (
                    self._instances.get(instance.id) is not instance
                    or instance.state != RuntimeInstanceState.READY
                    or instance.active_requests != 0
                    or instance.queued_requests != 0
                    or instance.control_leases != 0
                ):
                    continue
                committed = self._committed_pool_bytes_locked() + additional_bytes
                excess = committed - memory_ceiling
                if excess <= 0:
                    return
                hot_bytes = instance.kv_bytes or 0
                target_bytes = max(0, hot_bytes - excess)
                trim = getattr(instance.backend, "trim_runtime_cache", None)
                if not callable(trim):
                    continue
                instance.control_leases += 1
            try:
                await asyncio.wait_for(trim(target_bytes), timeout=2.0)
            except Exception:
                pass
            else:
                await self._refresh_instance_usage(instance)
            finally:
                await self._release_control_lease(instance)

    def _finish_model_load_locked(
        self,
        model: str,
        event: asyncio.Event,
        error: ErrorDetail | None,
    ) -> None:
        owns_event = self._load_events.get(model) is event
        if not owns_event:
            event.set()
            return
        artifact_id = self._loading_artifact_ids.pop(model, None)
        if error is None:
            self._load_errors.pop(model, None)
            self._load_failures.pop(model, None)
        elif model not in self._load_errors:
            self._load_errors[model] = error
        if error is not None and artifact_id is not None:
            self._load_failures[model] = _CachedLoadFailure(
                artifact_id=artifact_id,
                detail=error,
                failed_at=time.monotonic(),
            )
        self._loading_model_names.discard(model)
        self._load_events.pop(model, None)
        port = self._load_ports.pop(model, None)
        if port is not None:
            self._reserved_ports.discard(port)
        self._load_bytes.pop(model, None)
        event.set()

    @staticmethod
    def _runtime_load_error(error: BaseException) -> ErrorDetail:
        if isinstance(error, JobExecutionError):
            return error.detail
        if isinstance(error, asyncio.CancelledError):
            return ErrorDetail(
                code="runtime_load_cancelled",
                message="runtime loading was cancelled",
                retryable=True,
            )
        return ErrorDetail(
            code="runtime_load_failed",
            message=str(error) or type(error).__name__,
            retryable=True,
        )

    @staticmethod
    def _backend_load_error(detail: ErrorDetail) -> BackendError:
        status_by_code = {
            "model_artifact_incomplete": 409,
            "model_conversion_required": 409,
            "runtime_model_too_large": 413,
            "runtime_revision_busy": 409,
            "unsupported_device": 422,
            "unsupported_runtime_option": 422,
            "runtime_identity_mismatch": 502,
        }
        return BackendError(
            detail.code,
            detail.message,
            retryable=detail.retryable,
            status_code=status_by_code.get(detail.code, 503),
        )

    @staticmethod
    def _committed_runtime_bytes(instance: _Runtime) -> int:
        estimates = [
            value
            for value in (instance.resident_bytes, instance.reserved_bytes)
            if value is not None and value > 0
        ]
        if estimates:
            # RSS can be temporarily smaller than the model's committed
            # capacity when mmap pages or an expert LRU have not been touched
            # yet. Conversely KV/runtime allocations can push live residency
            # beyond the load estimate. Admission must retain both ledgers.
            return max(estimates)
        return instance.artifact.resource.total_bytes

    def _committed_pool_bytes_locked(self) -> int:
        resident_names = {
            item.artifact.resource.name
            for item in self._instances.values()
            if item.state != RuntimeInstanceState.FAILED
        }
        return sum(
            self._committed_runtime_bytes(item)
            for item in self._instances.values()
            if item.state != RuntimeInstanceState.FAILED
        ) + sum(
            reserved_bytes
            for name, reserved_bytes in self._load_bytes.items()
            if name not in resident_names
        )

    def _effective_runtime_memory_budget_locked(self) -> int | None:
        """Cap automatic admission by memory the host can reclaim right now."""

        ceiling = self.max_runtime_memory_bytes
        if ceiling is None or not self.automatic_memory_budget:
            return ceiling
        snapshot = host_memory_snapshot()
        if snapshot is None:
            return ceiling
        dynamic_ceiling = (
            self._committed_pool_bytes_locked()
            + snapshot.reclaimable(active_ratio=0.5)
        )
        return max(1, min(ceiling, dynamic_ceiling))

    def _runtime_memory_pressure_locked(
        self,
    ) -> tuple[str, float | None, int | None, int]:
        committed = self._committed_pool_bytes_locked()
        effective = self._effective_runtime_memory_budget_locked()
        if not self.automatic_memory_budget or effective is None:
            return "disabled", None, effective, committed
        ratio = committed / max(1, effective)
        if ratio >= _AUTOMATIC_MEMORY_HARD_RATIO:
            level = "hard"
        elif ratio >= _AUTOMATIC_MEMORY_SOFT_RATIO:
            level = "soft"
        else:
            level = "normal"
        return level, ratio, effective, committed

    def _estimated_load_bytes(
        self,
        artifact: DiscoveredModel,
        request: ModelLoadRequest,
    ) -> int:
        total_bytes = artifact.resource.total_bytes
        cache_gb = request.moe_gpu_cache_gb
        streamed_bytes = artifact.routed_expert_bytes
        if streamed_bytes <= 0:
            return total_bytes
        if cache_gb is None:
            if self.backend != "metal" or artifact.resource.format != "hf":
                return total_bytes
            physical_bytes = total_physical_memory()
            if physical_bytes is None:
                # Conservatively reserve the complete model when the host
                # cannot reproduce the native Metal worker's automatic budget.
                return total_bytes
            cache_bytes = max(1 << 30, physical_bytes * 2 // 3)
        elif cache_gb <= 0:
            return total_bytes
        else:
            cache_bytes = int(cache_gb * (1 << 30))
        resident_expert_bytes = min(streamed_bytes, cache_bytes)
        return total_bytes - streamed_bytes + resident_expert_bytes

    def _apply_automatic_expert_residency(
        self,
        artifact: DiscoveredModel,
        request: ModelLoadRequest,
        *,
        memory_ceiling: int | None = None,
    ) -> ModelLoadRequest:
        """Choose native Metal expert residency within the runtime budget."""

        ceiling = (
            memory_ceiling
            if memory_ceiling is not None
            else self.max_runtime_memory_bytes
        )
        if (
            request.moe_gpu_cache_gb is not None
            or self.backend != "metal"
            or ceiling is None
            or artifact.resource.format not in {"hf", "mfq"}
            or artifact.routed_expert_bytes <= 0
        ):
            return request
        if artifact.resource.total_bytes <= ceiling:
            return (
                request.model_copy(update={"moe_gpu_cache_gb": 0.0})
                if artifact.resource.format == "hf"
                else request
            )
        dense_bytes = max(
            0,
            artifact.resource.total_bytes - artifact.routed_expert_bytes,
        )
        runtime_headroom = min(
            4 << 30,
            max(1 << 30, ceiling // 20),
        )
        resident_expert_bytes = min(
            artifact.routed_expert_bytes,
            max(0, ceiling - runtime_headroom - dense_bytes),
        )
        if resident_expert_bytes <= 0:
            return request
        return request.model_copy(
            update={
                "moe_gpu_cache_gb": resident_expert_bytes / float(1 << 30),
            }
        )

    @staticmethod
    def _observed_runtime_bytes(
        process_resident_bytes: int | None,
        status: dict[str, Any] | None,
    ) -> int | None:
        candidates = [
            process_resident_bytes
            if process_resident_bytes is not None and process_resident_bytes >= 0
            else 0
        ]
        if status is not None:
            mlx_active = status.get("mlx_active_bytes")
            mlx_cache = status.get("mlx_cache_bytes")
            if (
                isinstance(mlx_active, (int, float))
                and mlx_active >= 0
                and isinstance(mlx_cache, (int, float))
                and mlx_cache >= 0
            ):
                candidates.append(int(mlx_active + mlx_cache))
            for name in ("cuda_reserved_bytes", "cuda_allocated_bytes"):
                value = status.get(name)
                if isinstance(value, (int, float)) and value >= 0:
                    candidates.append(int(value))
        observed = max(candidates)
        return observed if observed > 0 else None

    def _unroute_instance_locked(self, instance: _Runtime) -> None:
        self._session_routes = {
            session_id: instance_id
            for session_id, instance_id in self._session_routes.items()
            if instance_id != instance.id
        }
        if self._last_instance_id == instance.id:
            self._last_instance_id = next(
                (
                    item.id
                    for item in self._instances.values()
                    if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
                ),
                None,
            )

    def _mark_instance_unloading_locked(self, instance: _Runtime) -> None:
        instance.state = RuntimeInstanceState.UNLOADING
        self._unroute_instance_locked(instance)

    def _detach_instance_locked(self, instance: _Runtime) -> None:
        self._instances.pop(instance.id, None)
        self._unroute_instance_locked(instance)

    async def _idle_reaper(self) -> None:
        while True:
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self._idle_reaper_wakeup.wait(),
                    timeout=self.metric_interval_seconds,
                )
            self._idle_reaper_wakeup.clear()
            async with self._lock:
                if self._closed:
                    return
                now_monotonic = asyncio.get_running_loop().time()
                refresh = [
                    item
                    for item in self._instances.values()
                    if item.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }
                    and now_monotonic - item.usage_refreshed_at
                    >= self.metric_interval_seconds
                ]
                for item in refresh:
                    item.usage_refreshed_at = now_monotonic
            if refresh:
                await asyncio.gather(
                    *(self._refresh_instance_usage(item) for item in refresh)
                )

            await self._reclaim_shared_cache_under_pressure()
            async with self._lock:
                pressure_level, _ratio, pressure_ceiling, _committed = (
                    self._runtime_memory_pressure_locked()
                )
                pressure_target = (
                    int(pressure_ceiling * _AUTOMATIC_MEMORY_TARGET_RATIO)
                    if pressure_ceiling is not None
                    and pressure_level in {"soft", "hard"}
                    else None
                )
            await self._trim_idle_prefix_caches_for_budget(
                memory_target_bytes=pressure_target,
            )

            victims: list[tuple[_Runtime, str]] = []
            now = datetime.now(timezone.utc)
            async with self._lock:
                if self._closed:
                    return
                pressure_level, _ratio, pressure_ceiling, _committed = (
                    self._runtime_memory_pressure_locked()
                )
                pressure_target = (
                    int(pressure_ceiling * _AUTOMATIC_MEMORY_TARGET_RATIO)
                    if pressure_ceiling is not None
                    and pressure_level == "hard"
                    else None
                )
                ttl_victims: list[_Runtime] = []
                for instance in list(self._instances.values()):
                    if (
                        instance.idle_ttl_seconds is None
                        or instance.pinned
                        or instance.state != RuntimeInstanceState.READY
                        or instance.active_requests != 0
                        or instance.queued_requests != 0
                        or instance.control_leases != 0
                        or instance.last_used_at is None
                    ):
                        continue
                    idle_seconds = (now - instance.last_used_at).total_seconds()
                    if idle_seconds < instance.idle_ttl_seconds:
                        continue
                    self._mark_instance_unloading_locked(instance)
                    ttl_victims.append(instance)
                    victims.append((instance, "idle_ttl"))
                victims.extend(
                    (instance, "memory_budget")
                    for instance in self._claim_over_budget_instances_for_unload_locked(
                        memory_ceiling=pressure_target,
                        pending_releases=ttl_victims,
                    )
                )
            async def retire_victim(
                instance: _Runtime,
                reason: str,
            ) -> None:
                try:
                    await self._retire_instance(instance)
                    if self.store is not None:
                        message = (
                            f"runtime unloaded after {instance.idle_ttl_seconds}s "
                            "of inactivity"
                            if reason == "idle_ttl"
                            else "runtime unloaded to enforce the aggregate memory budget"
                        )
                        await asyncio.to_thread(
                            self.store.append_runtime_log,
                            RuntimeLogLevel.INFO,
                            message,
                            instance_id=instance.id,
                            fields={"source": "runtime.lifecycle", "reason": reason},
                        )
                except Exception as error:
                    if self.store is not None:
                        await asyncio.to_thread(
                            self.store.append_runtime_log,
                            RuntimeLogLevel.ERROR,
                            f"runtime unload failed: {error}",
                            instance_id=instance.id,
                            fields={"source": "runtime.lifecycle", "reason": reason},
                        )

            if victims:
                await asyncio.gather(
                    *(
                        retire_victim(instance, reason)
                        for instance, reason in victims
                    )
                )

    async def _retire_instance(self, instance: _Runtime) -> None:
        """Stop one registered runtime once and detach it only after success."""

        async with self._lock:
            if self._instances.get(instance.id) is not instance:
                return
            self._mark_instance_unloading_locked(instance)
            task = instance.retirement_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._stop_and_detach_instance(instance),
                    name=f"mfq-server-runtime-retire-{instance.id}",
                )
                instance.retirement_task = task
                task.add_done_callback(self._retirement_task_done)
        await asyncio.shield(task)

    @staticmethod
    def _retirement_task_done(task: asyncio.Task[None]) -> None:
        with suppress(asyncio.CancelledError):
            task.exception()

    async def _stop_and_detach_instance(self, instance: _Runtime) -> None:
        try:
            await self._stop_process(instance)
        except BaseException as error:
            async with self._lock:
                if self._instances.get(instance.id) is instance:
                    instance.retirement_failures += 1
                    instance.error = ErrorDetail(
                        code="runtime_unload_failed",
                        message=str(error) or type(error).__name__,
                        retryable=True,
                    )
                    self._schedule_retirement_retry_locked(instance)
            raise
        async with self._lock:
            if self._instances.get(instance.id) is instance:
                self._detach_instance_locked(instance)

    def _schedule_retirement_retry_locked(self, instance: _Runtime) -> None:
        if self._closed:
            return
        retry = instance.retirement_retry_task
        if retry is not None and not retry.done():
            return
        retry = asyncio.create_task(
            self._retry_retirement(instance),
            name=f"mfq-server-runtime-retire-retry-{instance.id}",
        )
        instance.retirement_retry_task = retry
        self._background_tasks.add(retry)
        retry.add_done_callback(self._background_task_done)

    async def _retry_retirement(self, instance: _Runtime) -> None:
        """Retry a failed teardown until it succeeds or the pool closes."""

        while True:
            failures = max(1, instance.retirement_failures)
            delay = min(30.0, 0.25 * (2 ** min(failures - 1, 7)))
            await asyncio.sleep(delay)
            async with self._lock:
                if (
                    self._closed
                    or self._instances.get(instance.id) is not instance
                    or instance.state != RuntimeInstanceState.UNLOADING
                ):
                    return
            try:
                await self._retire_instance(instance)
            except Exception:
                continue
            return

    @staticmethod
    def _supports_voice_output(architecture: str) -> bool:
        return "minicpmo" in architecture.casefold()

    async def _pump_output(self, instance: _Runtime, context: JobContext) -> None:
        process = instance.process
        if isinstance(process, subprocess.Popen):
            return
        stream = process.stderr if self.transport == "stdio" else process.stdout
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            message = line.decode("utf-8", errors="replace").rstrip()
            if message and instance.state == RuntimeInstanceState.LOADING:
                await context.log(message[:4096])
            if message and self.store is not None:
                await asyncio.to_thread(
                    self.store.append_runtime_log,
                    (
                        RuntimeLogLevel.ERROR
                        if "error" in message.casefold()
                        else RuntimeLogLevel.INFO
                    ),
                    message[:4096],
                    instance_id=instance.id,
                    fields={
                        "source": (
                            "runtime.stderr"
                            if self.transport == "stdio"
                            else "runtime.stdout"
                        )
                    },
                )

    async def _monitor(self, instance: _Runtime) -> None:
        process = instance.process
        status = (
            await asyncio.to_thread(process.wait)
            if isinstance(process, subprocess.Popen)
            else await process.wait()
        )
        error = ErrorDetail(
            code="runtime_exited",
            message=f"runtime process exited with status {status}",
            retryable=True,
        )
        async with self._lock:
            if (
                instance.state == RuntimeInstanceState.UNLOADING
                or self._closed
                or self._instances.get(instance.id) is not instance
            ):
                return
            instance.state = RuntimeInstanceState.FAILED
            instance.error = error
            model_name = instance.artifact.resource.name
            self._load_errors[model_name] = error
            self._load_failures[model_name] = _CachedLoadFailure(
                artifact_id=instance.artifact.resource.id,
                detail=error,
                failed_at=time.monotonic(),
            )
            self._session_routes = {
                session_id: instance_id
                for session_id, instance_id in self._session_routes.items()
                if instance_id != instance.id
            }
            if self._last_instance_id == instance.id:
                self._last_instance_id = next(
                    (
                        item.id
                        for item in self._instances.values()
                        if item.state in {
                            RuntimeInstanceState.READY,
                            RuntimeInstanceState.BUSY,
                        }
                    ),
                    None,
                )
        if self.store is not None:
            await asyncio.to_thread(
                self.store.append_runtime_log,
                RuntimeLogLevel.ERROR,
                error.message,
                instance_id=instance.id,
                fields={"source": "runtime.lifecycle", "exit_status": status},
            )
        await instance.backend.aclose()

    async def _refresh_instance_usage(self, instance: _Runtime) -> None:
        pid = getattr(instance.process, "pid", None)
        resident_task = (
            asyncio.create_task(
                asyncio.to_thread(self._process_resident_bytes, pid)
            )
            if isinstance(pid, int) and pid > 0
            else None
        )
        runtime_status = getattr(instance.backend, "runtime_status", None)
        status: dict[str, Any] | None = None
        try:
            if callable(runtime_status):
                try:
                    value = await asyncio.wait_for(runtime_status(), timeout=1.0)
                    if isinstance(value, dict):
                        status = value
                except Exception:
                    pass
            resident = await resident_task if resident_task is not None else None
        except BaseException:
            # Cancelling the idle reaper or a load transaction must not leave
            # its concurrently scheduled process-memory probe behind.
            if resident_task is not None:
                if not resident_task.done():
                    resident_task.cancel()
                await asyncio.gather(resident_task, return_exceptions=True)
            raise
        observed_resident = self._observed_runtime_bytes(resident, status)
        kv_bytes: int | None = None
        if status is not None:
            value = status.get("prefix_cache_hot_bytes", status.get("prefix_cache_bytes"))
            if isinstance(value, (int, float)) and value >= 0:
                kv_bytes = int(value)
        async with self._lock:
            if (
                self._instances.get(instance.id) is not instance
                or instance.state
                not in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
            ):
                return
            if observed_resident is not None:
                instance.resident_bytes = observed_resident
            if kv_bytes is not None:
                instance.kv_bytes = kv_bytes

    async def _stop_process(self, instance: _Runtime) -> None:
        instance.state = RuntimeInstanceState.UNLOADING
        instance.realtime_gateway = None
        process = instance.process
        first_error: Exception | None = None
        try:
            if isinstance(process, subprocess.Popen):
                await asyncio.to_thread(self._stop_popen, process)
            elif process.returncode is None:
                stopped = False
                process_error: Exception | None = None
                try:
                    process.terminate()
                except ProcessLookupError:
                    stopped = True
                except Exception as error:
                    process_error = error
                if not stopped and process_error is None:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10.0)
                        stopped = True
                    except TimeoutError:
                        pass
                    except ProcessLookupError:
                        stopped = True
                    except Exception as error:
                        process_error = error
                if not stopped:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        stopped = True
                    except Exception as error:
                        if process_error is None:
                            process_error = error
                    else:
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                            stopped = True
                        except ProcessLookupError:
                            stopped = True
                        except Exception as error:
                            if process_error is None:
                                process_error = error
                if stopped:
                    process_error = None
                if process_error is not None:
                    raise process_error
        except ProcessLookupError:
            pass
        except Exception as error:
            first_error = error
        current = asyncio.current_task()
        for task in (instance.monitor_task, instance.output_task):
            if task is None or task is current or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as error:
                if first_error is None:
                    first_error = error
        try:
            await instance.backend.aclose()
        except Exception as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

    @staticmethod
    def _stop_popen(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        first_error: Exception | None = None
        try:
            process.terminate()
        except ProcessLookupError:
            return
        except Exception as error:
            first_error = error
        if first_error is None:
            try:
                process.wait(timeout=10.0)
                return
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                return
            except Exception as error:
                first_error = error
        try:
            process.kill()
        except ProcessLookupError:
            return
        except Exception as error:
            if first_error is None:
                first_error = error
        else:
            try:
                process.wait(timeout=5.0)
                return
            except ProcessLookupError:
                return
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _reserve_free_port_locked(self) -> int:
        for _ in range(128):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])
            if port not in self._reserved_ports:
                self._reserved_ports.add(port)
                return port
        raise RuntimeManagementError("unable to reserve a unique runtime port")

    def _launch_configuration(
        self,
        artifact: DiscoveredModel,
        request: ModelLoadRequest,
        *,
        port: int,
    ) -> tuple[list[str], dict[str, str]]:
        request_capacity = native_request_capacity(
            backend=self.backend,
            routed_expert_bytes=artifact.routed_expert_bytes,
            requested=self.max_requests_per_instance,
        )
        command = [
            str(self.executable),
            "--model",
            str(artifact.path),
            "--transport",
            self.transport,
        ]
        if self.transport == "http":
            command.extend(["--host", "127.0.0.1", "--port", str(port)])
        command.extend([
            "--ctx-size",
            str(request.context_size),
            "--model-name",
            artifact.resource.name,
        ])
        append_native_prefill_chunk_override(command, request.prefill_chunk_size)
        if self.backend == "cuda" and request_capacity > 1:
            command.extend(
                [
                    "--continuous-batching",
                    str(request_capacity),
                ]
            )
        command.extend(native_tokenizer_arguments(artifact.path, self.backend))
        if request.moe_gpu_cache_gb is not None:
            command.extend(["--moe-gpu-cache-gb", str(request.moe_gpu_cache_gb)])

        process_environment = native_runtime_environment(
            self.executable, self.backend, model=artifact.path
        )
        process_environment.update(self.runtime_environment)
        cache_environment = {
            "MFQ_RUNTIME_MAX_KV_SESSIONS": request.prefix_cache_max_sessions,
            "MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION": (
                request.prefix_cache_max_snapshots_per_session
            ),
            "MFQ_RUNTIME_KV_SESSION_BYTES": request.prefix_cache_max_bytes,
            "MFQ_RUNTIME_DISABLE_PREFIX_CACHE": (
                None if request.prefix_cache_enabled else 1
            ),
            "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES": request.prefix_cache_disk_bytes,
            "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES": (
                request.prefix_cache_hot_bytes
                if request.prefix_cache_hot_bytes is not None
                else request.prefix_cache_max_bytes
            ),
            "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS": request.prefix_cache_block_tokens,
            "MFQ_RUNTIME_PREFIX_CACHE_PENDING_BYTES": request.prefix_cache_pending_bytes,
        }
        for name, value in cache_environment.items():
            if value is not None:
                process_environment[name] = str(value)
        if self.backend == "cuda" and request.device_ids:
            process_environment["CUDA_VISIBLE_DEVICES"] = ",".join(request.device_ids)
        return command, process_environment

    @staticmethod
    def _process_resident_bytes(pid: int) -> int | None:
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)],
                text=True,
                timeout=1,
            ).strip()
            return int(output) * 1024 if output else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
