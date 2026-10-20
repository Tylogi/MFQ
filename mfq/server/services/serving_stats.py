from __future__ import annotations

import math
from typing import Any


def request_values(values: dict[str, Any], instance: str | None, model: str | None, captured: float) -> tuple | None:
    request = values.get("last_request")
    if not isinstance(request, dict) or not isinstance(request.get("id"), str):
        return None
    if request.get("finish_reason") in {None, "cancelled", "error"}:
        return None

    def number(name: str, default: float = 0) -> float:
        value = request.get(name, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else default

    prompt = int(number("prompt_tokens"))
    prefill = int(number("prefill_tokens", prompt))
    completion = int(number("completion_tokens"))
    return (instance or model or "runtime", request["id"], model,
        number("completed_at", captured), prompt, prefill, completion,
        max(prompt - prefill, 0), number("prefill_ms"), number("decode_ms"),
        number("ttft_ms"), max(completion - 1, 0))


def summarize(row: dict[str, Any]) -> dict[str, Any]:
    def rate(tokens: str, duration: str) -> float | None:
        return 1000 * row[tokens] / row[duration] if row[duration] > 0 else None

    return {**row, "prefill_tps": rate("prefill_tokens", "prefill_ms"),
        "decode_tps": rate("decode_tokens", "decode_ms"),
        "ttft_ms": row["ttft_ms"] / row["completed_requests"] if row["completed_requests"] else None,
        "cache_hit_rate": 100 * row["cached_tokens"] / row["prompt_tokens"] if row["prompt_tokens"] else None}
