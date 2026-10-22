import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from mfq.server.runtime.backend import OpenAIChatBackend
from mfq.server.runtime.client import BackendProtocolError, HttpRuntimeClient
from mfq.server.services.jobs import JobExecutionError
from mfq.server.services.probability_benchmark import mmlu_question
from tests.test_runtime_stdio_client import _client


def count_result(tokens=2100, context=2048):
    return {"prompt_tokens": tokens, "context_size": context, "tokenization": "raw-no-special-tokens"}


def test_token_count_http_and_stdio_preserve_raw_prompt():
    async def run():
        prompt = "raw <s> question\nAnswer:"
        seen = []

        def http(request):
            import json
            assert request.url.path == "/runtime/count_tokens"
            assert request.method == "POST"
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=count_result())

        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            backend = OpenAIChatBackend(HttpRuntimeClient("http://runtime", client=client))
            assert await backend.count_tokens(prompt) == count_result()

        def stdio(request):
            assert request["op"] == "count_tokens"
            seen.append(request["params"])
            return [{"v": 1, "id": request["id"], "type": "result", "data": count_result()}]

        native = _client(stdio)
        try:
            assert await OpenAIChatBackend(native).count_tokens(prompt) == count_result()
        finally:
            await native.aclose()
        assert seen == [{"prompt": prompt}, {"prompt": prompt}]

    asyncio.run(run())


@pytest.mark.parametrize("changes", [
    {"prompt_tokens": 0}, {"prompt_tokens": -1}, {"prompt_tokens": True},
    {"prompt_tokens": 2.5}, {"prompt_tokens": "3"}, {"prompt_tokens": None},
    {"context_size": 0}, {"context_size": False}, {"context_size": None},
    {"tokenization": "with-special-tokens"},
])
def test_token_count_rejects_malformed_responses(changes):
    class Runtime:
        async def count_tokens(self, _payload):
            return {**count_result(), **changes}

    async def run():
        with pytest.raises(BackendProtocolError):
            await OpenAIChatBackend(Runtime()).count_tokens("prompt")

    asyncio.run(run())


def scorer_for(question_size):
    def evaluate(args, subject, model, tokenize, dev, test):
        prompt = "d" * (400 * args.ntrain) + "q" * question_size
        assert tokenize(prompt).input_ids.shape == (1, len(prompt))
        scores = model().logits
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= probabilities.sum()
        return [True], 1.0, np.array([probabilities])

    return {
        "pd": SimpleNamespace(DataFrame=lambda rows: rows),
        "gen_prompt": lambda _dev, _subject, k: "d" * (400 * k),
        "format_example": lambda *_args, **_kwargs: "q" * question_size,
        "eval": evaluate,
    }


class SizedBackend:
    def __init__(self, context):
        self.context = context
        self.counted = []
        self.scored = []

    async def count_tokens(self, prompt):
        self.counted.append(len(prompt))
        return count_result(len(prompt), self.context)

    async def score(self, payload):
        size = len(payload["prompt"])
        assert size <= min(2048, self.context), "oversized prompt reached model scoring"
        self.scored.append(size)
        return {"prompt_tokens": size, "scores": [
            {"log_likelihood": value} for value in (-1.0, -2.0, -3.0, -4.0)
        ]}


def question():
    return SimpleNamespace(category="subject", official_row={"question": "q", "choices": list("abcd"), "answer": 0})


@pytest.mark.parametrize("context,shots,size", [(2048, 4, 1700), (1500, 3, 1300), (8192, 4, 1700)])
def test_mmlu_reduces_shots_before_scoring(context, shots, size):
    backend = SizedBackend(context)
    answer, correct, grading = asyncio.run(mmlu_question(
        SimpleNamespace(raise_if_cancelled=lambda: None), backend, question(), scorer_for(100), {"subject": []}
    ))
    assert answer == "A" and correct
    assert grading["shots"] == shots
    assert backend.counted == list(range(2100, size - 1, -400))
    assert backend.scored == [size]


def test_mmlu_reports_no_fit_without_scoring():
    backend = SizedBackend(2048)
    with pytest.raises(JobExecutionError, match="2048-token limit even without demonstrations"):
        asyncio.run(mmlu_question(
            SimpleNamespace(raise_if_cancelled=lambda: None), backend, question(), scorer_for(2100), {"subject": []}
        ))
    assert len(backend.counted) == 6
    assert backend.counted[-1] == 2100
    assert backend.scored == []


def test_mmlu_rejects_tokenization_change_after_sizing():
    class ChangedBackend(SizedBackend):
        async def score(self, payload):
            result = await super().score(payload)
            result["prompt_tokens"] += 1
            return result

    backend = ChangedBackend(2048)
    with pytest.raises(JobExecutionError, match="tokenization changed between sizing and scoring"):
        asyncio.run(mmlu_question(
            SimpleNamespace(raise_if_cancelled=lambda: None), backend, question(), scorer_for(100), {"subject": []}
        ))
