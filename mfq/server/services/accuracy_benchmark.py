from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import pickle
import random
import time
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mfq.formats.assets import HF_GENERATION_CONFIG_ASSET, MODEL_CONFIG_ASSET
from mfq.formats.io import open_mmap
from mfq.formats.runtime_profile import (
    RUNTIME_SAMPLING_METADATA_KEY,
    architecture_profile,
    generation_config_profile,
    load_mfq_sidecar,
    model_profile,
    validate_runtime_profile,
)
from mfq.server.protocol.models import SamplingParams
from mfq.server.runtime.backend import closing_backend_stream, preflight_backend_request
from mfq.server.runtime.client import BackendError
from mfq.server.services.evaluation_datasets import (
    official_for_digest,
    parquet_rows,
    read_official_file,
)
from mfq.server.services.inference_benchmark import _failure
from mfq.server.services.jobs import JobContext
from mfq.server.services.official_benchmarks import (
    MMLU_DEV,
    MMLU_PRO_VALIDATION,
    OFFICIAL_BENCHMARKS,
    benchmark_directory,
    load_choice_scorer,
    pro_prompt,
    pro_scores,
    verified_artifact,
)
from mfq.server.services.probability_benchmark import mmlu_question, truthful_question
from mfq.server.services.scoring_sandbox import isolated_official_score


class AccuracyBenchmarkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_id: UUID
    dataset_id: UUID
    protocol: Literal["mcq", "math", "code", "likelihood"] = "mcq"
    official_defaults: bool = False
    sample_count: int = Field(default=100, ge=0, le=100000)
    seed: int | None = Field(default=42, ge=0)
    max_tokens: int | None = Field(default=512, ge=1, le=8192)
    enable_thinking: bool | None = False
    enable_mtp: bool = False
    temperature: float | None = Field(default=0., ge=0., le=10.)
    top_p: float | None = Field(default=1., gt=0., le=1.)
    top_k: int | None = Field(default=0, ge=0, le=1024)
    num_generations: int = Field(default=1, ge=1, le=20)


def resolve_benchmark_sampling(request: AccuracyBenchmarkPayload, artifact: Any) -> tuple[SamplingParams, dict[str, str]]:
    values = SamplingParams().model_dump()
    sources = {key: "runtime-default" for key in values}
    resource = artifact.resource
    identities = [getattr(resource, "architecture", None), resource.name]
    config, generation, embedded, sidecar = {}, {}, None, None
    path = getattr(artifact, "path", None)
    if path is not None:
        with open_mmap(path) as store:
            identities.append(store.header.model_arch)
            for name, target in ((MODEL_CONFIG_ASSET, config), (HF_GENERATION_CONFIG_ASSET, generation)):
                if name in store.records:
                    if store.records[name].nbytes > 1024 * 1024:
                        raise ValueError("model sampling metadata exceeds 1 MiB")
                    value = json.loads(store.read_blob(name))
                    if not isinstance(value, dict):
                        raise ValueError("model sampling metadata must be a JSON object")
                    target.update(value)
            embedded = store.header.extra.get(RUNTIME_SAMPLING_METADATA_KEY)
        sidecar = load_mfq_sidecar(path)
    identities.extend((config.get("_name_or_path"), config.get("model_type"), *(config.get("architectures") or ())))
    for profile in (architecture_profile(*identities), model_profile(*identities),
        generation_config_profile(generation), embedded, sidecar):
        if profile is not None:
            profile = validate_runtime_profile(profile)
            chat = profile.get("chat", {})
            values.update({key: value for key, value in chat.items() if key in values})
            source = profile.get("provenance", {}).get("source") or "model-runtime-profile"
            sources.update({key: source for key in chat if key in values})
    for key in ("max_tokens", "temperature", "top_p", "top_k", "seed", "enable_thinking", "enable_mtp"):
        value = getattr(request, key)
        if value is not None:
            values[key] = value
            sources[key] = "official-benchmark" if request.official_defaults else "custom"
    values["enable_vision"] = False
    sources["enable_vision"] = "benchmark"
    return SamplingParams.model_validate(values), sources


class Question(BaseModel):
    id: str | None = Field(default=None, max_length=128)
    question: str = Field(min_length=1, max_length=65536)
    choices: list[str] = Field(default_factory=list, max_length=26)
    answer: str
    category: str = Field(default="default", min_length=1, max_length=128)
    official_row: dict[str, Any] = Field(default_factory=dict)
    official_index: int | None = None
    protocol: Literal["mcq", "math", "code", "likelihood"] = "mcq"

    @model_validator(mode="after")
    def validate_answers(self) -> Question:
        if not self.answer.strip() or len(self.answer) > 4096:
            raise ValueError("answer must be a nonempty string of at most 4096 characters")
        if any(not value.strip() or len(value) > 16384 for value in self.choices):
            raise ValueError("choices must contain nonempty strings of at most 16384 characters")
        if self.protocol == "mcq" and len(self.choices) < 2:
            raise ValueError("MCQ requires at least two choices")
        return self


def read_questions(path: Path, dataset: Any, protocol: str, scorer: dict[str, Any], seed: int = 42) -> list[Question]:
    spec, raw = read_official_file(path, dataset)
    expected_protocol = {"mmlu-pro-test": "mcq", "gpqa-diamond": "mcq", "aime-2025": "math", "livecodebench-v6": "code", "mmlu-test": "likelihood", "truthfulqa-mc1": "likelihood"}
    if expected_protocol.get(spec.id) != protocol:
        raise _failure("wrong_dataset_protocol", "this collection's official runner is not available; approximate scoring is disabled")
    questions, ids = [], set()
    number = 1
    try:
        if spec.id == "mmlu-test":
            for number, row in enumerate(parquet_rows(raw), 1):
                if len(row["choices"]) != 4 or type(row["answer"]) is not int or not 0 <= row["answer"] < 4:
                    raise ValueError("MMLU requires four choices and an answer index")
                questions.append(Question(id=str(number - 1), question=row["question"], choices=row["choices"],
                    answer=chr(65 + row["answer"]), category=row["subject"], protocol="likelihood", official_row=row))
        elif spec.id == "truthfulqa-mc1":
            for number, row in enumerate(parquet_rows(raw), 1):
                mc1, mc2 = row["mc1_targets"], row["mc2_targets"]
                if sum(mc1["labels"]) != 1 or len(mc1["choices"]) != len(mc1["labels"]) or len(mc2["choices"]) != len(mc2["labels"]) or any(label not in {0, 1} for label in mc1["labels"] + mc2["labels"]):
                    raise ValueError("TruthfulQA MC1 labels are inconsistent")
                true = [choice for choice, label in zip(mc2["choices"], mc2["labels"], strict=True) if label]
                false = [choice for choice, label in zip(mc2["choices"], mc2["labels"], strict=True) if not label]
                best = mc1["choices"][mc1["labels"].index(1)]
                if best not in true or not false:
                    raise ValueError("TruthfulQA MC1 and MC2 references disagree")
                original = {"Question": row["question"], scorer["BEST_COL"]: best,
                    scorer["ANSWER_COL"]: "; ".join(true), scorer["INCORRECT_COL"]: "; ".join(false)}
                questions.append(Question(id=str(number - 1), question=row["question"], choices=row["mc1_targets"]["choices"],
                    answer=best, category="truthfulness", protocol="likelihood", official_row=original))
        if spec.id == "aime-2025":
            for row in parquet_rows(raw):
                number = len(questions) + 1
                questions.append(Question(id=str(row["problem_idx"]), question=row["problem"],
                    answer=str(row["answer"]), category=" · ".join(row["problem_type"]) or "math", protocol="math", official_row=row))
        elif spec.id == "livecodebench-v6":
            for line in raw.decode("utf-8").splitlines():
                number = len(questions) + 1
                row = json.loads(line)
                questions.append(Question(id=f'{row["platform"]}:{row["question_id"]}', question=row["question_content"],
                    answer="all-tests", category=row["difficulty"], protocol="code", official_row=row))
        if protocol in {"math", "code", "likelihood"}:
            if len(questions) != spec.rows or len({question.id for question in questions}) != spec.rows:
                raise ValueError("official row count or IDs are inconsistent")
            return questions
        if spec.id == "gpqa-diamond":
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                member = archive.getinfo("dataset/gpqa_diamond.csv")
                if member.file_size > 4 * 1024 * 1024:
                    raise ValueError("official CSV exceeds expected size")
                csv_bytes = archive.read(member, pwd=b"deserted-untie-orchid")
            examples = scorer["load_examples"](io.BytesIO(csv_bytes), seed)
            records = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"))))
            for number, (example, row) in enumerate(zip(examples, records, strict=True), 1):
                questions.append(Question(id=str(number - 1), question=example.question,
                    choices=list(example[1:5]), answer=chr(65 + example.correct_index),
                    category=row["High-level domain"], official_index=number - 1))
            if len(questions) != spec.rows or len(examples) != spec.rows or len(records) != spec.rows:
                raise ValueError("official question count is inconsistent")
            return questions
        for row in parquet_rows(raw):
            number = len(questions) + 1
            question = Question(id=str(row["question_id"]), question=row["question"], choices=row["options"],
                answer=row["answer"], category=row["category"], official_row=row)
            if not question.choices or not isinstance(question.answer, str) or question.answer not in [chr(65 + i) for i in range(len(question.choices))]:
                raise ValueError("MCQ answer must be an uppercase choice letter")
            if row["answer_index"] != ord(question.answer) - 65:
                raise ValueError("choice label and index disagree")
            if question.id in ids:
                raise ValueError("question IDs must be unique")
            ids.add(question.id)
            questions.append(question)
        scorer["preprocess"]([question.official_row for question in questions])
        for question in questions:
            question.choices = question.official_row["options"]
    except (ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        raise _failure("invalid_answer_dataset", f"invalid official question at row {number}: {error}") from error
    if len(questions) != spec.rows:
        raise _failure("invalid_answer_dataset", "official question count is inconsistent")
    return questions


def score_answer(content: str, question: Question, scorer: dict[str, Any]) -> tuple[str | None, bool]:
    if question.official_index is not None:
        predicted = scorer["AnswerPredictor"].parse_sampled_answer(content)
    else:
        predicted = scorer["extract_answer"](content.replace("**", ""))
    return predicted, predicted == question.answer


def question_messages(question: Question, scorer: dict[str, Any], examples: dict[str, Any]) -> list[dict[str, str]]:
    if question.protocol == "math":
        template = scorer["config"]["instruction"]
        if "{problem}" not in template:
            template += "\n\n{problem}"
        return [{"role": "user", "content": template.format(problem=question.question)}]
    if question.protocol == "code":
        return scorer["format_prompt_generation"](SimpleNamespace(**question.official_row), scorer["LMStyle"].OpenAIChat)
    if question.official_index is not None:
        example = scorer["Example"](question.question, *question.choices, ord(question.answer) - 65)
        prompt = scorer["zero_shot_prompt"](example)
        return [{"role": "system", "content": "You are a very intelligent assistant, who follows instructions directly."},
            {"role": "user", "content": prompt}]
    else:
        prompt = pro_prompt(scorer, question.official_row, examples)
    return [{"role": "user", "content": prompt}]


class RestrictedStringUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("private tests may not instantiate objects")


def code_tests(question: Question) -> dict[str, Any]:
    row = question.official_row
    try:
        public = json.loads(row["public_test_cases"])
        try:
            private = json.loads(row["private_test_cases"])
        except ValueError:
            compressed = base64.b64decode(row["private_test_cases"], validate=True)
            decompressor = zlib.decompressobj()
            decoded = decompressor.decompress(compressed, 64 * 1024 * 1024 + 1)
            if len(decoded) > 64 * 1024 * 1024 or decompressor.unconsumed_tail or not decompressor.eof:
                raise ValueError("private tests exceed the decoded limit") from None
            value = RestrictedStringUnpickler(io.BytesIO(decoded)).load()
            if not isinstance(value, str):
                raise ValueError("private tests must decode to a string") from None
            private = json.loads(value)
        tests = public + private
        if not tests or len(tests) > 10000:
            raise ValueError("official tests are missing or excessive")
        if any(not isinstance(case["input"], str) or not isinstance(case["output"], str) for case in tests):
            raise ValueError("official test cases require text input and output")
        return {"inputs": [case["input"] for case in tests], "outputs": [case["output"] for case in tests],
            "fn_name": json.loads(row["metadata"]).get("func_name")}
    except (ValueError, KeyError, TypeError, pickle.UnpicklingError, zlib.error) as error:
        raise _failure("invalid_code_dataset", f"invalid official tests for {question.id}: {error}") from error


async def run_accuracy_benchmark(context: JobContext, payload: dict[str, Any], pool: Any, path: Path, root: Path) -> dict[str, Any]:
    request = AccuracyBenchmarkPayload.model_validate(payload)
    dataset = await asyncio.to_thread(context.store.get_dataset, request.dataset_id)
    spec = official_for_digest(dataset.kind, dataset.sha256, dataset.byte_size)
    profile = OFFICIAL_BENCHMARKS.get(spec.id)
    if profile is None or not profile.supported:
        raise _failure("official_runner_pending", "this task's official runner is not integrated; approximate scoring is disabled")
    if request.official_defaults:
        defaults = profile.manifest()["defaults"]
        if not defaults["available"]:
            raise _failure("official_defaults_unavailable", "this benchmark's official defaults are not available")
        request = request.model_copy(update=defaults["parameters"])
    if request.num_generations != 1 and spec.id != "livecodebench-v6":
        raise _failure("unsupported_generation_count", "multiple generations are currently supported only for LiveCodeBench")
    scorer = await asyncio.to_thread(load_choice_scorer, root, spec.id)
    questions = await asyncio.to_thread(read_questions, path, dataset, request.protocol, scorer, request.seed or 0)
    examples = {}
    if spec.id == "mmlu-pro-test":
        validation = await asyncio.to_thread(verified_artifact, benchmark_directory(root, spec.id), "validation.parquet", MMLU_PRO_VALIDATION[1])
        examples = scorer["preprocess"](parquet_rows(validation))
    if spec.id == "mmlu-test":
        dev = await asyncio.to_thread(verified_artifact, benchmark_directory(root, spec.id), "dev.parquet", MMLU_DEV[1])
        for row in parquet_rows(dev):
            examples.setdefault(row["subject"], []).append(row)
        if len(examples) != 57 or any(len(rows) != 5 for rows in examples.values()):
            raise _failure("invalid_answer_dataset", "MMLU official dev set must contain five demonstrations for each of 57 subjects")
    indices = list(range(len(questions)))
    if request.sample_count and request.sample_count < len(indices):
        indices = sorted(random.Random(request.seed).sample(indices, request.sample_count))
    rows = []
    async with pool.benchmark_runtime(request.instance_id) as (instance, backend):
        try:
            sampling, sampling_sources = await asyncio.to_thread(resolve_benchmark_sampling, request, instance.artifact)
        except (OSError, ValueError, TypeError) as error:
            raise _failure("invalid_model_sampling_defaults", f"cannot resolve model/architecture defaults: {error}") from error
        request = request.model_copy(update={key: getattr(sampling, key) for key in (
            "max_tokens", "temperature", "top_p", "top_k", "seed", "enable_thinking", "enable_mtp")})
        if request.protocol == "likelihood":
            if request.enable_mtp or request.enable_thinking:
                raise _failure("invalid_probability_sampling", "probability benchmarks do not generate, sample, think or use MTP")
            if not callable(getattr(backend, "score", None)) or not await backend.probability_available():
                raise _failure("likelihood_unavailable", "the selected runtime does not expose native probabilities")
        if request.enable_mtp and not instance.mtp_available:
            raise _failure("mtp_unavailable", "the loaded model does not have usable MTP weights")
        if request.max_tokens >= instance.context_size:
            raise _failure("benchmark_context_exceeded", "output limit leaves no room for questions in the loaded ctx")
        model = instance.artifact.resource.name
        if request.protocol != "likelihood":
            first_question = next((questions[index] for index in indices if questions[index].official_index != 69), None)
            if first_question is not None:
                first = question_messages(first_question, scorer, examples)
                await preflight_backend_request(backend, model=model, messages=first, sampling=sampling)
        generation_indices = [(index, generation) for index in indices for generation in range(request.num_generations)]
        for index, generation in generation_indices:
            context.raise_if_cancelled()
            question = questions[index]
            if request.protocol == "likelihood":
                started = time.perf_counter()
                runner = mmlu_question if spec.id == "mmlu-test" else truthful_question
                try:
                    predicted, correct, grading = await runner(context, backend, question, scorer, examples)
                except BackendError as error:
                    raise _failure("accuracy_backend_error", f"probability scoring failed: {error}; no score was recorded") from error
                context.raise_if_cancelled()
                row = {"id": question.id, "category": question.category, "question": question.question,
                    "choices": question.choices, "expected": question.answer, "predicted": predicted, "correct": correct,
                    "status": "correct" if correct else "incorrect", "response": "", "reasoning": "", "finish_reason": None,
                    "latency_s": time.perf_counter() - started, "usage": None, "official_grading": grading}
                rows.append(row)
                await context.progress(len(rows) / len(indices) * .99, message=f"{len(rows)} / {len(indices)}: {row['status']}", data={"question": row})
                continue
            if question.official_index == 69:
                row = {"id": question.id, "category": question.category, "question": question.question,
                    "choices": question.choices, "expected": question.answer, "predicted": None,
                    "correct": False, "status": "official_skip", "response": "", "reasoning": "",
                    "finish_reason": None, "latency_s": 0, "usage": None}
                rows.append(row)
                await context.progress(len(rows) / len(indices) * .99, message=f"{len(rows)} / {len(indices)}: official_skip", data={"question": row})
                continue
            content, reasoning, finish_reason, usage = "", "", None, None
            started = time.perf_counter()
            try:
                generation_sampling = sampling.model_copy(update={"seed": sampling.seed + generation if sampling.seed is not None else None})
                async with closing_backend_stream(backend.stream(model=model, sampling=generation_sampling,
                    messages=question_messages(question, scorer, examples))) as stream:
                    async for delta in stream:
                        context.raise_if_cancelled()
                        content += delta.content_delta
                        reasoning += delta.reasoning_delta
                        finish_reason = delta.finish_reason or finish_reason
                        usage = delta.usage or usage
            except BackendError as error:
                raise _failure("accuracy_backend_error", f"question {question.id} failed: {error}; no score was recorded") from error
            if finish_reason not in {"stop", "length"}:
                raise _failure("accuracy_incomplete_response", f"question {question.id} did not finish normally; no score was recorded")
            grading = {}
            if question.protocol == "mcq":
                predicted, correct = score_answer(content, question, scorer)
            else:
                scoring_request = {"task": spec.id, "sources": scorer["sources"], "response": content,
                    "answer": question.answer, "output_tokens": usage.completion_tokens if usage else request.max_tokens}
                if question.protocol == "code":
                    scoring_request["tests"] = code_tests(question)
                    timeout = (len(scoring_request["tests"]["inputs"]) + 1) * 6 + 30
                else:
                    scoring_request["config"] = scorer["config"]
                    timeout = 30
                grading = await isolated_official_score(scoring_request, timeout)
                predicted, correct = grading["predicted"], grading["correct"]
            status = "truncated" if finish_reason == "length" else "no_answer" if not content.strip() else "parse_error" if predicted is None else "correct" if correct else "incorrect"
            row = {"id": question.id, "category": question.category, "question": question.question,
                "choices": question.choices, "expected": question.answer, "predicted": predicted,
                "correct": correct, "status": status, "response": content,
                "reasoning": reasoning, "finish_reason": finish_reason, "latency_s": time.perf_counter() - started,
                "usage": usage.model_dump(mode="json") if usage else None}
            if grading:
                row["official_grading"] = grading
            if request.protocol == "code":
                row["generation_index"] = generation
            rows.append(row)
            await context.progress(len(rows) / len(generation_indices) * .99, message=f"{len(rows)} / {len(generation_indices)}: {status}", data={"question": row})
        counts = Counter(row["status"] for row in rows)
        if spec.id == "mmlu-pro-test":
            official_rows = [{**questions[index].official_row, "pred": row["predicted"]} for index, row in zip(indices, rows, strict=True)]
            official_scores = await asyncio.to_thread(pro_scores, scorer, official_rows)
            for row, correct in zip(rows, official_scores, strict=True):
                row["correct"] = correct
                row["official_random_fallback"] = row["predicted"] is None
        categories = {}
        for category in sorted({row["category"] for row in rows}):
            group = [row for row in rows if row["category"] == category]
            categories[category] = {"total": len(group), "correct": sum(row["correct"] for row in group),
                "accuracy": sum(row["correct"] for row in group) / len(group)}
        metrics = {"accuracy": sum(row["correct"] for row in rows) / len(rows), "correct": sum(row["correct"] for row in rows),
            "sample_count": len(rows), "dataset_total": len(questions), "status_counts": dict(counts), "categories": categories, "questions": rows}
        if spec.id == "truthfulqa-mc1":
            metrics.update({name.lower(): sum(row["official_grading"][name] for row in rows) / len(rows) for name in ("MC1", "MC2", "MC3")})
        if request.protocol == "code":
            results = {questions[index].id: [row["official_grading"]["test_results"] for row in rows if row["id"] == questions[index].id] for index in indices}
            official_metrics = scorer["compute_metrics_from_results"](results, k_list=[1])
            metrics.update(accuracy=float(official_metrics["pass@1"]), pass_at_1=float(official_metrics["pass@1"]),
                question_count=len(indices), generations_per_question=request.num_generations)
        parameters = request.model_dump(mode="json")
        parameters.update(benchmark=spec.task, official_scoring=profile.manifest())
        parameters.update(resolved_sampling=sampling.model_dump(mode="json"), sampling_sources=sampling_sources)
        if spec.id == "gpqa-diamond":
            parameters["prompt_protocol"] = "gpqa-official-chat-v2"
        if request.protocol in {"math", "code"}:
            parameters["scoring_isolation"] = {"implementation": "macos-seatbelt", "rss_limit_bytes": 4 * 1024 ** 3,
                "output_limit_bytes": 1024 ** 2, "case_timeout_seconds": 6 if request.protocol == "code" else None}
        comparison = {key: value for key, value in parameters.items() if key not in {"instance_id", "dataset_id"}}
        comparison.update(dataset_sha256=dataset.sha256, selected_ids=[questions[i].id for i in indices],
            context_size=instance.context_size, scoring_version=profile.protocol)
        evaluation = await context.evaluation(kind="accuracy_benchmark", model_id=model, dataset_id=dataset.id,
            dataset_manifest={"sha256": dataset.sha256, "byte_size": dataset.byte_size, "artifact_uri": dataset.artifact_uri,
                "official_id": spec.id, "revision": spec.revision},
            metrics=metrics, parameters=parameters, comparison_parameters=comparison,
            runtime_identity={"instance_id": str(instance.id), "artifact_id": instance.artifact.resource.id,
                "runtime": pool.executable.name, "backend": pool.backend, "context_size": instance.context_size})
        return {"evaluation_id": str(evaluation.id), **metrics}
