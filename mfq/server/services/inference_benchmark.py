from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mfq.server.protocol.models import ErrorDetail
from mfq.server.services.jobs import JobContext, JobExecutionError


class InferenceBenchmarkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_id: UUID
    prompt: str = Field(min_length=1, max_length=262144)
    prompt_tokens: int = Field(default=4096, ge=1, le=1048576)
    prompt_token_lengths: list[Annotated[int, Field(strict=True, ge=1, le=1048576)]] | None = Field(default=None, min_length=1, max_length=8)
    output_tokens: int = Field(default=128, ge=2, le=8192)
    mode: Literal["decode", "mtp", "both"] = "both"
    repetitions: int = Field(default=3, ge=1, le=20)
    warmup_runs: int = Field(default=1, ge=0, le=3)
    cooldown_seconds: float = Field(default=0, ge=0, le=60)
    temperature: float = Field(default=0, ge=0, le=10)
    top_k: int = Field(default=20, ge=0, le=1024)
    top_p: float = Field(default=0.95, gt=0, le=1)
    seed: int = Field(default=42, ge=0)
    mtp_max_draft_tokens: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def validate_sampling(self) -> InferenceBenchmarkPayload:
        if self.prompt_token_lengths is not None:
            if len(set(self.prompt_token_lengths)) != len(self.prompt_token_lengths):
                raise ValueError("input lengths must be unique")
            self.prompt_token_lengths = sorted(self.prompt_token_lengths)
        if self.temperature > 0 and self.top_k == 0 and self.top_p < 1:
            raise ValueError("top_p below 1 requires top_k above 0")
        return self


class WikitextQualityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=255)
    dataset_id: UUID
    reference_logits: str = Field(min_length=1, max_length=1024)
    reference_manifest: str = Field(min_length=1, max_length=1024)
    context_size: int = Field(default=512, ge=32, le=1048576)
    chunks: int = Field(default=8, ge=1, le=10000)
    parallel: int = Field(default=1, ge=1, le=64)
    moe_gpu_cache_gb: float | None = Field(default=None, ge=0)


def _failure(code: str, message: str) -> JobExecutionError:
    return JobExecutionError(ErrorDetail(code=code, message=message))


def validate_benchmark_run(result: dict[str, Any], request: InferenceBenchmarkPayload, mode: str) -> dict[str, Any]:
    metrics, usage = result["metrics"], result["usage"]
    required = ("prefill_tokens", "prefill_ms", "decode_ms", "ttft_ms", "mtp_drafted_tokens", "mtp_accepted_tokens")
    values = {key: metrics.get(key, 0) for key in required}
    for key, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise _failure("invalid_benchmark_metrics", f"invalid native {key}")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens != request.prompt_tokens or values["prefill_tokens"] != request.prompt_tokens:
        raise _failure("benchmark_prefix_reused", "native input sizing or cache bypass was not applied; refusing this timing")
    if not isinstance(completion_tokens, int) or not 1 <= completion_tokens <= request.output_tokens:
        raise _failure("invalid_benchmark_metrics", "invalid completion token count")
    if values["prefill_ms"] <= 0 or (completion_tokens > 1 and values["decode_ms"] <= 0):
        raise _failure("invalid_benchmark_metrics", "native timing interval is empty")
    if values["mtp_accepted_tokens"] > values["mtp_drafted_tokens"]:
        raise _failure("invalid_benchmark_metrics", "accepted MTP tokens exceed drafted tokens")
    mtp_used = bool(metrics.get("mtp_used"))
    if mode == "decode" and mtp_used:
        raise _failure("benchmark_mode_mismatch", "ordinary decode unexpectedly used MTP")
    if mode == "mtp" and not mtp_used:
        raise _failure("mtp_not_used", "MTP was requested but did not execute; refusing to label ordinary decode as MTP")
    return {"mode": mode, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "decode_tokens": max(0, completion_tokens - 1), "finish_reason": result.get("finish_reason"),
        "metrics": metrics}


def aggregate_benchmark_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"runs": runs}
    for mode in ("decode", "mtp"):
        selected = [run for run in runs if run["mode"] == mode]
        if not selected:
            continue
        prefill_tokens = sum(run["prompt_tokens"] for run in selected)
        decode_tokens = sum(run["decode_tokens"] for run in selected)
        prefill_ms = sum(run["metrics"]["prefill_ms"] for run in selected)
        decode_ms = sum(run["metrics"]["decode_ms"] for run in selected)
        drafted = sum(run["metrics"].get("mtp_drafted_tokens", 0) for run in selected)
        accepted = sum(run["metrics"].get("mtp_accepted_tokens", 0) for run in selected)
        result.update({f"{mode}_prefill_tps": prefill_tokens * 1000 / prefill_ms,
            f"{mode}_decode_tps": decode_tokens * 1000 / decode_ms if decode_ms > 0 else None,
            f"{mode}_ttft_ms": sum(run["metrics"]["ttft_ms"] for run in selected) / len(selected),
            f"{mode}_completion_tokens": sum(run["completion_tokens"] for run in selected)})
        if mode == "mtp":
            result.update(mtp_drafted_tokens=drafted, mtp_accepted_tokens=accepted,
                mtp_acceptance_rate=accepted / drafted if drafted else None)
    return result


async def run_inference_benchmark(context: JobContext, payload: dict[str, Any], pool: Any) -> dict[str, Any]:
    request = InferenceBenchmarkPayload.model_validate(payload)
    lengths = request.prompt_token_lengths or [request.prompt_tokens]
    modes = ["decode", "mtp"] if request.mode == "both" else [request.mode]
    runs, warmups = [], []
    total = len(lengths) * len(modes) * (request.repetitions + request.warmup_runs)
    async with pool.benchmark_runtime(request.instance_id) as (instance, backend):
        if max(lengths) + request.output_tokens > instance.context_size:
            raise _failure("benchmark_context_exceeded", "input and output exceed the loaded model's context; reload it with a larger ctx first")
        if "mtp" in modes and not instance.mtp_available:
            raise _failure("mtp_unavailable", "the loaded model does not have usable MTP weights")
        identity = {"instance_id": str(instance.id), "model": instance.artifact.resource.name,
            "artifact_id": instance.artifact.resource.id, "context_size": instance.context_size,
            "runtime": pool.executable.name, "backend": pool.backend}
        load_request = getattr(pool, "_load_requests", {}).get(instance.artifact.resource.name)
        prefill_chunk_size = getattr(load_request, "prefill_chunk_size", None)
        identity["prefill_chunk_size"] = prefill_chunk_size
        for length in lengths:
            sized_request = request.model_copy(update={"prompt_tokens": length})
            for repeat in range(request.warmup_runs + request.repetitions):
                order = modes if repeat % 2 == 0 else list(reversed(modes))
                for mode in order:
                    context.raise_if_cancelled()
                    native = await backend.benchmark({"model": instance.artifact.resource.name,
                        "messages": [{"role": "user", "content": request.prompt}], "mfq_preformatted_prompt": request.prompt,
                        "mfq_benchmark_prompt_tokens": length,
                        "mfq_prefix_cache_enabled": False,
                        "max_tokens": request.output_tokens, "temperature": request.temperature,
                        "top_k": request.top_k, "top_p": request.top_p, "seed": request.seed,
                        "enable_mtp": mode == "mtp", "mtp_max_draft_tokens": request.mtp_max_draft_tokens,
                        "stream": False, "stream_options": {"include_usage": True}})
                    run = validate_benchmark_run(native, sized_request, mode)
                    run["repetition"] = repeat + 1
                    (warmups if repeat < request.warmup_runs else runs).append(run)
                    await context.progress((len(warmups) + len(runs)) / total * 0.99,
                        message=f"{length} / {mode}: {run['completion_tokens']} tokens", data={"run": run, "warmup": repeat < request.warmup_runs})
                    if request.cooldown_seconds:
                        await asyncio.sleep(request.cooldown_seconds)
        metrics = aggregate_benchmark_runs(runs)
        metrics["series"] = [{"prompt_tokens": length, **{key: value for key, value in
            aggregate_benchmark_runs([run for run in runs if run["prompt_tokens"] == length]).items() if key != "runs"}}
            for length in lengths]
        metrics["warmup_runs"] = warmups
        parameters = request.model_dump(mode="json")
        comparison = {key: value for key, value in parameters.items() if key not in {"instance_id", "prompt", "prompt_tokens", "prompt_token_lengths"}}
        comparison.update(prompt_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
            prompt_token_lengths=lengths,
            context_size=instance.context_size, prefill_chunk_size=prefill_chunk_size,
            prefix_cache=False, protocol="mfq-native-inference-benchmark-v2")
        evaluation = await context.evaluation(kind="inference_benchmark", model_id=instance.artifact.resource.name,
            metrics=metrics, parameters=parameters, comparison_parameters=comparison, runtime_identity=identity)
        return {"evaluation_id": str(evaluation.id), **metrics}
