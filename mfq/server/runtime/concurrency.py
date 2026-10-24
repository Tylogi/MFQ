"""Shared validation for per-model concurrency and server admission ceilings."""

from mfq.server.protocol.models import (
    ModelLoadRequest,
    RuntimeConcurrencyPolicy,
    RuntimeConcurrencySettings,
    UpdateRuntimeConcurrencyRequest,
)
from mfq.server.runtime.backend import BackendError


def resolve_concurrency(
    policy: RuntimeConcurrencyPolicy,
    request: ModelLoadRequest | UpdateRuntimeConcurrencyRequest,
    current: RuntimeConcurrencySettings | None = None,
) -> RuntimeConcurrencySettings:
    """Inherit omitted settings and reject overrides beyond the server ceiling."""
    values = (current or policy.defaults).model_dump()
    for name in values:
        value = getattr(request, name)
        if value is None:
            continue
        limit = getattr(policy.limits, name)
        # A zero queue timeout means an unlimited wait, which exceeds any finite ceiling.
        unlimited_timeout = name == "queue_timeout_seconds" and value == 0
        exceeds = limit is not None and (value > limit or unlimited_timeout)
        if exceeds:
            raise BackendError(
                "runtime_concurrency_limit",
                f"{name} exceeds the server limit of {limit:g}",
                status_code=422,
            )
        values[name] = value
    return RuntimeConcurrencySettings(**values)
