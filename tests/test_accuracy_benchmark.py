import asyncio
import hashlib
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mfq.server.api import create_app
from mfq.server.protocol.models import CreateDatasetRequest
from mfq.server.runtime.backend import BackendDelta
from mfq.server.runtime.client import BackendError
from mfq.server.services.accuracy_benchmark import (
    Question,
    read_questions,
    run_accuracy_benchmark,
    score_answer,
)
from mfq.server.services.evaluation_datasets import OFFICIAL_DATASETS
from mfq.server.services.jobs import JobContext, JobExecutionError
from mfq.server.services.official_benchmarks import (
    OFFICIAL_BENCHMARKS,
    benchmark_readiness,
    official_definitions,
    verified_source,
)
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_service import FakeBackend


@pytest.mark.parametrize("text,predicted,correct", [
    ("B", "B", True), ("The answer is B.", "B", True), ("答案：B", "B", True),
    ("<think>Answer: A</think>\nB", "A", False), ("<think>Answer: B", "B", True),
    ("Maybe B or C", "C", False), ("C", "C", False), ("", None, False),
])
def test_choice_scoring_delegates_unchanged_to_official_extractor(text, predicted, correct):
    question = Question(question="question", choices=["1", "2", "3"], answer="B")
    calls = []
    def extract(value):
        calls.append(value)
        return predicted
    assert score_answer(text, question, {"extract_answer": extract}) == (predicted, correct)
    assert calls == [text]


def fixture_dataset(path, rows, monkeypatch):
    pq.write_table(pa.Table.from_pylist(rows), path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setitem(OFFICIAL_DATASETS, "mmlu-pro-test", replace(OFFICIAL_DATASETS["mmlu-pro-test"], sha256=digest, byte_size=len(raw), rows=len(rows)))
    return SimpleNamespace(kind="custom", sha256=digest, byte_size=len(raw))


@pytest.mark.parametrize("bad", ["hash", "duplicate", "answer", "empty", "malformed"])
def test_dataset_preflight_rejects_invalid_data(tmp_path, bad, monkeypatch):
    row = {"question_id": 1, "question": "question", "options": ["1", "2"], "answer": "X" if bad == "answer" else "B", "answer_index": 1, "category": "math"}
    path = tmp_path / "questions.parquet"
    dataset = fixture_dataset(path, [row, row] if bad == "duplicate" else [row], monkeypatch)
    if bad in {"hash", "empty", "malformed"}:
        path.write_bytes(b"" if bad == "empty" else b"not-parquet")
    with pytest.raises(JobExecutionError):
        read_questions(path, dataset, "mcq", {"preprocess": lambda rows: rows})


@pytest.mark.parametrize("case", ["normal", "backend_error", "cancel", "unfinished"])
def test_integrated_accuracy_records_details_and_never_scores_transport_errors(tmp_path, case, monkeypatch):
    scorer = {"preprocess": lambda rows: {"math": rows}, "extract_answer": lambda content: content}
    monkeypatch.setattr("mfq.server.services.accuracy_benchmark.load_choice_scorer", lambda *_: scorer)
    monkeypatch.setattr("mfq.server.services.accuracy_benchmark.verified_artifact", lambda *_: b"validation")
    monkeypatch.setattr("mfq.server.services.accuracy_benchmark.parquet_rows", lambda raw: pq.read_table(pa.BufferReader(raw)).to_pylist() if raw != b"validation" else [])
    monkeypatch.setattr("mfq.server.services.accuracy_benchmark.pro_prompt", lambda *args: "official prompt")
    monkeypatch.setattr("mfq.server.services.accuracy_benchmark.pro_scores", lambda scorer, rows: [row["pred"] == row["answer"] for row in rows])
    class Backend:
        def __init__(self):
            self.calls = []
            self.closed = False
        async def stream(self, **kwargs):
            self.calls.append(kwargs)
            try:
                if case == "backend_error":
                    raise BackendError("bad", "unavailable")
                if case == "cancel":
                    raise asyncio.CancelledError()
                yield BackendDelta(reasoning_delta="Answer: A", content_delta="B")
                if case != "unfinished":
                    yield BackendDelta(finish_reason="length" if len(self.calls) == 2 else "stop")
            finally:
                self.closed = True
    class Pool:
        executable = tmp_path / "runtime"
        backend = "metal"
        leased = False
        engine = Backend()
        @asynccontextmanager
        async def benchmark_runtime(self, instance_id):
            self.leased = True
            try:
                yield SimpleNamespace(id=instance_id, context_size=4096, mtp_available=True,
                    artifact=SimpleNamespace(resource=SimpleNamespace(name="model", id="asset"))), self.engine
            finally:
                self.leased = False
    async def run():
        path = tmp_path / "questions.parquet"
        rows = [{"question_id": i, "question": "1 + 1", "options": ["1", "2"], "answer": "B", "answer_index": 1, "category": "math"} for i in range(10)]
        fixture = fixture_dataset(path, rows, monkeypatch)
        store = SessionStore(tmp_path / "db.sqlite3")
        dataset = store.create_dataset(CreateDatasetRequest(name="questions", kind="custom", artifact_uri="workspace://questions.parquet"), sha256=fixture.sha256, byte_size=fixture.byte_size)
        job = store.create_job("evaluate.accuracy", {})
        store.claim_job(job.id)
        context = JobContext(store, job.id, asyncio.Event())
        pool = Pool()
        payload = {"instance_id": str(uuid4()), "dataset_id": str(dataset.id), "sample_count": 2}
        if case == "normal":
            result = await run_accuracy_benchmark(context, payload, pool, path, tmp_path)
            assert result["accuracy"] == 1
            assert result["status_counts"] == {"correct": 1, "truncated": 1}
            assert result["dataset_total"] == 10 and result["sample_count"] == 2
            assert result["questions"][0]["response"] == "B" and result["questions"][0]["reasoning"] == "Answer: A"
            assert all(not call["sampling"].enable_mtp and not call["sampling"].enable_thinking for call in pool.engine.calls)
            evaluation = store.list_evaluations()[0]
            assert evaluation.kind == "accuracy_benchmark" and evaluation.dataset_manifest["sha256"] == dataset.sha256
        else:
            with pytest.raises(asyncio.CancelledError if case == "cancel" else JobExecutionError):
                await run_accuracy_benchmark(context, payload, pool, path, tmp_path)
            assert not store.list_evaluations()
        assert pool.engine.closed and not pool.leased
    asyncio.run(run())


def test_tools_report_native_availability_and_remove_external_connector(tmp_path):
    async def run():
        service = ServerService(SessionStore(tmp_path / "db.sqlite3"), FakeBackend())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
            result = await client.get("/api/v1/evaluations/tools")
            assert result.json() == {"workspace_root": None, "api_base": "http://test/v1", "quality_available": False, "benchmark_available": False, "accuracy_available": False, "task_benchmarks": {}}
            assert (await client.get("/api/v1/evaluations/benchi")).status_code == 404
    asyncio.run(run())


def test_official_definitions_are_not_rewritten_and_require_hash(tmp_path):
    source = "def extract_answer(value):\n    return value.split(':')[0]\n"
    path = tmp_path / "score.py"
    path.write_text(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = official_definitions(verified_source(tmp_path, "score.py", digest), {"extract_answer"}, {})
    assert loaded["extract_answer"]("B:A") == "B"
    path.write_text(source.replace("[0]", "[-1]"))
    with pytest.raises(JobExecutionError, match="SHA256"):
        verified_source(tmp_path, "score.py", digest)


def test_all_task_profiles_pin_official_sources_and_fail_closed_without_installation(tmp_path):
    ready = benchmark_readiness(tmp_path)
    assert len(ready) == 6
    assert all(not item["available"] for item in ready.values())
    assert ready["mmlu-test"]["reason"] == "official_scorer_missing"
    for profile in OFFICIAL_BENCHMARKS.values():
        assert len(profile.revision) == 40
        assert all(len(digest) == 64 for digest in profile.sources.values())


@pytest.mark.parametrize("task,tokens,seed,generations", [
    ("mmlu-pro-test", 4000, None, 1), ("gpqa-diamond", 1000, 0, 1),
    ("livecodebench-v6", 2000, None, 10),
])
def test_official_defaults_override_client_parameters_and_run_all_questions(tmp_path, monkeypatch, task, tokens, seed, generations):
    from mfq.server.services import accuracy_benchmark as module
    saved = []
    scores = []
    protocol = "code" if task == "livecodebench-v6" else "mcq"
    questions = [Question(id=str(index), question="question", choices=["1", "2"], answer="B", category="math", protocol=protocol) for index in range(2)]
    if task == "gpqa-diamond":
        questions[0].official_index = 69
    monkeypatch.setattr(module, "official_for_digest", lambda *_: SimpleNamespace(id=task, task=task, revision="revision"))
    monkeypatch.setattr(module, "read_questions", lambda *_: questions)
    monkeypatch.setattr(module, "question_messages", lambda question, *_: [{"role": "user", "content": question.id}])
    preflights = []
    async def preflight(_backend, **kwargs):
        preflights.append(kwargs["messages"])
    monkeypatch.setattr(module, "preflight_backend_request", preflight)
    monkeypatch.setattr(module, "verified_artifact", lambda *_: b"validation")
    monkeypatch.setattr(module, "parquet_rows", lambda *_: [])
    monkeypatch.setattr(module, "pro_scores", lambda _, rows: [True] * len(rows))
    def compute(results, k_list):
        assert k_list == [1]
        scores.append(results)
        return {"pass@1": .5}
    monkeypatch.setattr(module, "load_choice_scorer", lambda *_: {"preprocess": lambda _: {}, "extract_answer": lambda _: "B", "sources": {}, "compute_metrics_from_results": compute})
    monkeypatch.setattr(module, "code_tests", lambda _: {"inputs": [""], "outputs": [""], "fn_name": None})
    async def grade(*_):
        return {"predicted": "code", "correct": True, "test_results": [1]}
    monkeypatch.setattr(module, "isolated_official_score", grade)
    calls = []
    class Backend:
        async def stream(self, **kwargs):
            calls.append(kwargs)
            yield BackendDelta(content_delta="B", finish_reason="stop")
    class Pool:
        executable = tmp_path / "runtime"
        backend = "metal"
        @asynccontextmanager
        async def benchmark_runtime(self, instance_id):
            yield SimpleNamespace(id=instance_id, context_size=32768, mtp_available=True,
                artifact=SimpleNamespace(resource=SimpleNamespace(name="model", id="asset"))), Backend()
    dataset = SimpleNamespace(kind="custom", sha256="hash", byte_size=1, id=uuid4(), artifact_uri="workspace://test")
    class Context:
        store = SimpleNamespace(get_dataset=lambda _: dataset)
        def raise_if_cancelled(self):
            pass
        async def progress(self, value, **_):
            assert 0 <= value <= 1
        async def evaluation(self, **kwargs):
            saved.append(kwargs)
            return SimpleNamespace(id=uuid4())
    async def run():
        result = await run_accuracy_benchmark(Context(), {"instance_id": str(uuid4()), "dataset_id": str(dataset.id),
            "official_defaults": True, "sample_count": 1, "max_tokens": 7, "seed": 42, "enable_thinking": True,
            "enable_mtp": True, "temperature": 9., "top_p": .01, "top_k": 99, "num_generations": 2}, Pool(), tmp_path / "test", tmp_path)
        assert len(calls) == (1 if task == "gpqa-diamond" else 2) * generations
        assert preflights == [[{"role": "user", "content": "1" if task == "gpqa-diamond" else "0"}]]
        assert all(call["sampling"].max_tokens == tokens and call["sampling"].seed == seed for call in calls)
        assert all(not call["sampling"].enable_thinking and not call["sampling"].enable_mtp and call["sampling"].top_k == 0 for call in calls)
        assert saved[0]["parameters"]["official_defaults"] is True
        assert saved[0]["parameters"]["sample_count"] == 0
        assert result["sample_count"] == 2 * generations
        if task == "gpqa-diamond":
            assert result["status_counts"]["official_skip"] == 1
            assert result["accuracy"] == .5
            assert saved[0]["parameters"]["prompt_protocol"] == "gpqa-official-chat-v2"
            assert saved[0]["comparison_parameters"]["prompt_protocol"] == "gpqa-official-chat-v2"
        if protocol == "code":
            assert all("stop" not in call and call["sampling"].temperature == .2 and call["sampling"].top_p == .95 for call in calls)
            assert result["accuracy"] == .5 and result["question_count"] == 2
            assert len(scores) == 1 and all(len(items) == 10 for items in scores[0].values())
        else:
            assert all(call["sampling"].temperature == 0 and call["sampling"].top_p == 1 for call in calls)
    asyncio.run(run())


def test_aime_resolves_missing_official_settings_and_records_actual_sampling(tmp_path, monkeypatch):
    from mfq.server.services import accuracy_benchmark as module
    monkeypatch.setattr(module, "official_for_digest", lambda *_: SimpleNamespace(id="aime-2025", task="AIME2025", revision="revision"))
    monkeypatch.setattr(module, "read_questions", lambda *_: [Question(id="1", question="problem", answer="123", protocol="math")])
    monkeypatch.setattr(module, "load_choice_scorer", lambda *_: {"config": {"instruction": "{problem}"}, "sources": {}})
    async def grade(*_):
        return {"predicted": "123", "correct": True}
    monkeypatch.setattr(module, "isolated_official_score", grade)
    calls, saved = [], []
    class Backend:
        async def stream(self, **kwargs):
            calls.append(kwargs)
            yield BackendDelta(content_delta="123", finish_reason="stop")
    class Pool:
        executable = tmp_path / "runtime"
        backend = "metal"
        @asynccontextmanager
        async def benchmark_runtime(self, instance_id):
            yield SimpleNamespace(id=instance_id, context_size=32768, mtp_available=True,
                artifact=SimpleNamespace(resource=SimpleNamespace(name="MiniCPM-o-4_5", id="asset", architecture="minicpmo"))), Backend()
    dataset = SimpleNamespace(kind="custom", sha256="hash", byte_size=1, id=uuid4(), artifact_uri="workspace://test")
    class Context:
        store = SimpleNamespace(get_dataset=lambda _: dataset)
        def raise_if_cancelled(self):
            pass
        async def progress(self, *_args, **_kwargs):
            pass
        async def evaluation(self, **kwargs):
            saved.append(kwargs)
            return SimpleNamespace(id=uuid4())
    async def run():
        result = await run_accuracy_benchmark(Context(), {"instance_id": str(uuid4()), "dataset_id": str(dataset.id),
            "official_defaults": True, "temperature": 9., "top_k": 1}, Pool(), tmp_path / "test", tmp_path)
        sampling = calls[0]["sampling"]
        assert (sampling.temperature, sampling.top_k, sampling.top_p, sampling.max_tokens) == (.7, 100, .8, 4096)
        assert sampling.enable_thinking is False and sampling.enable_mtp is False and sampling.seed is None
        assert saved[0]["parameters"]["temperature"] == .7
        assert saved[0]["parameters"]["sampling_sources"]["temperature"] == "model-registry:minicpm-o-4_5"
        assert saved[0]["parameters"]["resolved_sampling"] == sampling.model_dump(mode="json")
        assert result["correct"] == 1
    asyncio.run(run())


def test_model_sampling_precedence_and_explicit_zero_are_preserved(tmp_path):
    import json

    from mfq.formats.assets import HF_GENERATION_CONFIG_ASSET
    from mfq.formats.header import FileHeader
    from mfq.formats.io import save
    from mfq.formats.runtime_profile import RUNTIME_SAMPLING_METADATA_KEY
    from mfq.server.services.accuracy_benchmark import (
        AccuracyBenchmarkPayload,
        resolve_benchmark_sampling,
    )
    path = tmp_path / "model.mfq"
    save(path, FileHeader(version=2, model_arch="minicpmo", extra={RUNTIME_SAMPLING_METADATA_KEY:
        {"chat": {"temperature": .4}, "provenance": {"source": "embedded"}}}),
        {HF_GENERATION_CONFIG_ASSET: json.dumps({"temperature": .3, "top_p": .75, "max_new_tokens": 16000}).encode()})
    path.with_suffix(".runtime.json").write_text(json.dumps({"chat": {"temperature": .5}, "provenance": {"source": "sidecar"}}))
    artifact = SimpleNamespace(path=path, resource=SimpleNamespace(name="model", architecture="minicpmo"))
    request = AccuracyBenchmarkPayload(instance_id=uuid4(), dataset_id=uuid4(), max_tokens=None,
        temperature=None, top_p=None, top_k=None, enable_thinking=None)
    sampling, sources = resolve_benchmark_sampling(request, artifact)
    assert (sampling.temperature, sampling.top_p, sampling.top_k, sampling.max_tokens) == (.5, .75, 100, 16000)
    assert sources["temperature"] == "sidecar" and sources["top_p"] == "hf:generation_config.json"
    assert sampling.enable_thinking is False
    sampling, sources = resolve_benchmark_sampling(request.model_copy(update={"temperature": 0., "top_k": 0}), artifact)
    assert sampling.temperature == 0. and sampling.top_k == 0 and sources["temperature"] == "custom"


def test_gpqa_uses_the_official_chat_system_message_and_unmodified_question_prompt():
    from collections import namedtuple

    from mfq.server.services.accuracy_benchmark import Question, question_messages

    captured = []
    def prompt(example):
        captured.append(example)
        return "official question and choices"
    scorer = {"Example": namedtuple("Example", "question choice1 choice2 choice3 choice4 correct_index"),
        "zero_shot_prompt": prompt}
    messages = question_messages(Question(question="Question", choices=["a", "b", "c", "d"],
        answer="C", official_index=4), scorer, {})
    assert [item["role"] for item in messages] == ["system", "user"]
    assert "follows instructions directly" in messages[0]["content"]
    assert messages[1]["content"] == "official question and choices"
    assert captured[0].correct_index == 2
