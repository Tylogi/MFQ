from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace
from typing import Any

import numpy as np

from mfq.server.services.inference_benchmark import _failure


class ScoringArray(np.ndarray):
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def to(self, device):
        if device not in (None, 'cpu'):
            raise ValueError('official scoring arrays are CPU-only')
        return self

    def squeeze(self, axis=None):
        if axis is not None and self.shape[axis] != 1:
            return self
        return super().squeeze(axis)


def _scoring_softmax(values, dim):
    values = np.asarray(values, dtype=np.float32)
    weights = np.exp(values - values.max(axis=dim, keepdims=True))
    return (weights / weights.sum(axis=dim, keepdims=True)).view(ScoringArray)


class ScoringArrayOps:
    nn = SimpleNamespace(functional=SimpleNamespace(softmax=_scoring_softmax))

    @staticmethod
    def tensor(values):
        return np.asarray(values, dtype=np.float32).view(ScoringArray)

    @staticmethod
    @contextlib.contextmanager
    def no_grad():
        yield


class NativeInput:
    def __init__(self, tokens: int):
        self.shape = (1, tokens)

    def cuda(self):
        return self


class NativeLogProbs:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def squeeze(self, _axis):
        return self

    def log_softmax(self, _axis):
        return self

    @property
    def shape(self):
        return (len(self.values), 1)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            rows, _tokens = index
            if isinstance(rows, slice):
                return NativeLogProbs(self.values[rows])
            return self.values[list(rows)]
        return NativeLogProbs(self.values[index])


class NativeEncoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


def mmlu_frame(scorer: dict[str, Any], rows: list[dict[str, Any]]):
    return scorer["pd"].DataFrame([[row["question"], *row["choices"], chr(65 + row["answer"])] for row in rows])


async def mmlu_question(context, backend, question, scorer, examples):
    subject = question.category
    test = mmlu_frame(scorer, [question.official_row])
    dev = mmlu_frame(scorer, examples[subject])
    k = 5
    while True:
        context.raise_if_cancelled()
        prompt = scorer["gen_prompt"](dev, subject, k) + scorer["format_example"](test, 0, include_answer=False)
        result = await backend.score({"prompt": prompt, "continuations": ["A", "B", "C", "D"], "mode": "next_token"})
        if result["prompt_tokens"] <= 2048:
            break
        if k == 0:
            raise _failure("benchmark_context_exceeded", "MMLU question exceeds the official 2048-token limit even without demonstrations")
        k -= 1
    def tokenize(text, **_kwargs):
        if text in {"A", "B", "C", "D"}:
            return SimpleNamespace(input_ids=[ord(text) - 65])
        if text and text != prompt:
            raise _failure("official_prompt_mismatch", "official MMLU prompt differs from the scored text")
        return SimpleNamespace(input_ids=NativeInput(result["prompt_tokens"] if text else 1))
    class Model:
        def _shift_right(self, ids):
            return ids
        def __call__(self, **_kwargs):
            return SimpleNamespace(logits=np.asarray([score["log_likelihood"] for score in result["scores"]], dtype=np.float32))
    with contextlib.redirect_stdout(io.StringIO()):
        correct, _accuracy, probabilities = scorer["eval"](SimpleNamespace(ntrain=k), subject, Model(), tokenize, dev, test)
    predicted = chr(65 + int(probabilities[0].argmax()))
    return predicted, bool(correct[0]), {"choice_probabilities": probabilities[0].tolist(),
        "native_scores": result, "shots": k, "model_adapter": "causal-next-token-letter-logits"}


async def truthful_question(context, backend, question, scorer, _examples):
    row = question.official_row
    frame = scorer["pd"].DataFrame([row])
    best = scorer["format_best"](row[scorer["BEST_COL"]])
    true = scorer["split_multi_answer"](row[scorer["ANSWER_COL"]])
    false = scorer["split_multi_answer"](row[scorer["INCORRECT_COL"]])
    if best not in true or not true or not false:
        raise _failure("invalid_answer_dataset", "TruthfulQA references are inconsistent")
    prompt = scorer["format_prompt"](frame.loc[0], "qa", format="general")
    full_prompts = [scorer["format_prompt_with_answer_strings"](row["Question"], answer, "qa", format="general") for answer in true + false]
    if any(not value.startswith(prompt) for value in full_prompts):
        raise _failure("official_prompt_mismatch", "official TruthfulQA answer prompt has a different prefix")
    native = []
    prompt_tokens = None
    for begin in range(0, len(full_prompts), 64):
        context.raise_if_cancelled()
        result = await backend.score({"prompt": prompt, "continuations": [value[len(prompt):] for value in full_prompts[begin:begin + 64]], "mode": "continuation"})
        if prompt_tokens is not None and prompt_tokens != result["prompt_tokens"]:
            raise _failure("probability_tokenization_changed", "prompt tokenization changed within a question")
        prompt_tokens = result["prompt_tokens"]
        native.extend(result["scores"])
    lookup = dict(zip(full_prompts, native, strict=True))
    def tokenize(text, **_kwargs):
        if text == prompt:
            count = prompt_tokens
        elif text in lookup:
            count = prompt_tokens + len(lookup[text]["token_ids"])
        else:
            raise _failure("official_prompt_mismatch", "official TruthfulQA prompt differs from the scored text")
        return NativeEncoding(input_ids=np.zeros((1, count), dtype=np.int64).view(ScoringArray))
    class Model:
        def __call__(self, ids):
            return [NativeLogProbs([0.] * (prompt_tokens - 1) + pending[0]["token_logprobs"] + [0.])]
    pending = []
    def tokenizer(text, **kwargs):
        if text in lookup:
            pending[:] = [lookup[text]]
        return tokenize(text, **kwargs)
    scorer["run_probs"](frame, "native", "native", "qa", model=Model(), tokenizer=tokenizer)
    grading = {name: float(frame.loc[0, f"native {name}"]) for name in ("MC1", "MC2", "MC3")}
    if any(not scorer["np"].isfinite(value) for value in grading.values()):
        raise _failure("official_probability_underflow", "official TruthfulQA scoring produced a non-finite result; no approximate normalization was substituted")
    grading.update(native_scores=native, preset="qa", answer_prefix_tokens_skipped=3)
    return "MC1", grading["MC1"] == 1, grading
