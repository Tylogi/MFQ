"""Validation for native inference benchmark results."""

import math


class BenchmarkValidationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_benchmark_result(result, prompt_tokens, output_tokens, mode):
    def fail(message, code="invalid_benchmark_metrics"):
        raise BenchmarkValidationError(code, message)

    if not isinstance(result, dict):
        fail("benchmark requires a native result")
    metrics, usage = result.get("metrics"), result.get("usage")
    if not isinstance(metrics, dict) or not isinstance(usage, dict):
        fail("benchmark requires native metrics and token usage")
    metrics = dict(metrics)

    def number(key, default=None, *, count=False):
        value = metrics.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            fail(f"invalid native {key}")
        if count and value != int(value):
            fail(f"invalid native {key}")
        return value

    prefill = number("prefill_tokens", count=True)
    for key in ("prefill_ms", "decode_ms", "ttft_ms"):
        number(key)
    for key in ("prefill_tps", "decode_tps", "generation_tps", "generation_ms"):
        if key in metrics:
            number(key)
    actual_prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(actual_prompt) is not int or actual_prompt != prompt_tokens or prefill != prompt_tokens:
        fail("native input sizing or cache bypass was not applied; refusing this timing", "benchmark_prefix_reused")
    if type(completion) is not int or not 1 <= completion <= output_tokens:
        fail("invalid completion token count")
    if metrics["prefill_ms"] <= 0 or (completion > 1 and metrics["decode_ms"] <= 0):
        fail("native timing interval is empty")
    if result.get("finish_reason") not in ("stop", "length"):
        fail("benchmark response did not finish normally")

    available = metrics.get("mtp_available")
    if "mtp_available" in metrics and (type(available) not in (bool, int, float) or available not in (0, 1)):
        fail("invalid native mtp_available")
    unavailable = available in (False, 0)
    used = metrics.get("mtp_used", False if unavailable else None)
    if type(used) not in (bool, int, float) or used not in (0, 1):
        fail("invalid native mtp_used")
    drafted = number("mtp_drafted_tokens", 0 if unavailable else None, count=True)
    accepted = number("mtp_accepted_tokens", 0 if unavailable else None, count=True)
    if accepted > drafted:
        fail("accepted MTP tokens exceed drafted tokens")
    if unavailable and (used or drafted or accepted):
        fail("unavailable MTP reported execution")
    if mode == "decode" and used:
        fail("ordinary decode unexpectedly used MTP", "benchmark_mode_mismatch")
    if mode == "mtp" and not used:
        fail("MTP was requested but did not execute; refusing to label ordinary decode as MTP", "mtp_not_used")
    if not used and (drafted or accepted):
        fail("ordinary decode reported MTP tokens")
    metrics.update(
        mtp_used=bool(used), mtp_drafted_tokens=drafted, mtp_accepted_tokens=accepted,
        prefill_tps=actual_prompt * 1000 / metrics["prefill_ms"],
        decode_tps=(completion - 1) * 1000 / metrics["decode_ms"] if completion > 1 else 0,
    )
    return {"mode": mode, "prompt_tokens": actual_prompt, "completion_tokens": completion,
            "decode_tokens": completion - 1, "finish_reason": result["finish_reason"],
            "metrics": metrics}
