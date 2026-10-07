from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import random
import re
import ssl
import tempfile
import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from mfq.server.services.evaluation_datasets import dataset_error


@dataclass(frozen=True)
class OfficialBenchmark:
    repository: str
    revision: str
    protocol: str
    sources: dict[str, str]
    supported: bool = False

    def manifest(self) -> dict[str, Any]:
        return {"repository": self.repository, "revision": self.revision,
            "protocol": self.protocol, "scoring_files": self.sources,
            "defaults": OFFICIAL_DEFAULTS[self.protocol]}


OFFICIAL_BENCHMARKS = {
    "mmlu-test": OfficialBenchmark("hendrycks/test", "4450500f923c49f1fb1dd3d99108a0bd9717b660", "official-5shot-likelihood",
        {"evaluate.py": "8b7d84e5a2e9290df0d43d1c060aa68ecb638c4666415b3a58d709c18eace4ec",
         "evaluate_flan.py": "98178736d6afac0c1e48eb844151e71b50365ce17b5e79672015c6d289fc990f"}, True),
    "mmlu-pro-test": OfficialBenchmark("TIGER-AI-Lab/MMLU-Pro", "f418b116db00b065c2aea046518d8fcf74d39872", "official-api-cot",
        {"evaluate_from_api.py": "6e00c0d57ed307b9e707b712ea796fbd2286c6d21889cd63bb58090fa72c5a99"}, True),
    "gpqa-diamond": OfficialBenchmark("idavidrein/gpqa", "56686c06f5e19865c153de0fdb11be3890014df7", "official-zero-shot-sample",
        {"baselines/closed_book.py": "7815b8617dd6bac9f964a9b61520b6e07e9ccba0c20edf6003e8206486946ead",
         "baselines/utils.py": "63189efa76292b0f67d80ecf416f38d3db582b6570bdec433df8869868a136d4"}, True),
    "livecodebench-v6": OfficialBenchmark("LiveCodeBench/LiveCodeBench", "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24", "official-v6-python-pass1",
        {"lcb_runner/runner/parser.py": "e9641729a6030164e712b9de465dea03dae95c3fdf036aac87abbbfab9fd7f33",
         "lcb_runner/runner/oai_runner.py": "c7dc0b0dbd3ebf3c5bd20d0f79bc94d22b7335067a815bb00b40f20dcbafba34",
         "lcb_runner/evaluation/testing_util.py": "b7cb6a8a69807bb868150a61742e25d7bb5328bbe01d514471b6ec43c9fa9ed2",
         "lcb_runner/evaluation/pass_k_utils.py": "6af3ee55d40f96fa06fa511bd5e43d94b19604750e82454b4b3c1f45c5199010",
         "lcb_runner/utils/extraction_utils.py": "1267d1724c6d4ea6d237b907dd6b53dfc6f1787a04b1616b85c3bc8377446879",
         "lcb_runner/lm_styles.py": "57d645b09852d2030d389cf9a14ea818396b7b0510b3e2f3f22e3d3e6792b783",
         "lcb_runner/prompts/code_generation.py": "7f360dc01016c277e10c70959f46efe27b37c1d2c4ef72b077f93f348b3fd826"}, True),
    "aime-2025": OfficialBenchmark("eth-sri/matharena", "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b", "official-matharena-aime2025",
        {"src/matharena/parser.py": "e32c1a04dd6797c13a6e60281a3ec493cb68139d77b52930068ad3712890c44a",
         "src/matharena/parse_manual.py": "8858daf09f0b8fda2326853874db13fe32d03b506464edcd8f96215db59a3936",
         "configs/competitions/aime/aime_2025.yaml": "1994936c68f87d5d6710e70b804403ae21bcd14f98c512c8972c11c75e067a1f",
         "src/matharena/grader.py": "eb8cec29763930561a930ebb1b91d1b9430b0d54fcbe23f1dc9d652517c301fa",
         "src/matharena/utils.py": "88b234a9872dd246b09354ec4932338c66d618ede299e766d92e4bef341a9907"}, True),
    "truthfulqa-mc1": OfficialBenchmark("sylinrl/TruthfulQA", "d71c110897f5d31c5d7f309e7bc316c152f6f031", "official-mc-likelihood",
        {"truthfulqa/models.py": "b80120128f9f64f565a30be1af55a78134b7a958b4cfca3ba9b64543ebf465d2",
         "truthfulqa/utilities.py": "216dfa41328e00f97c5b7f5505eefe92ee8290c127f2b7457ac2922d617ca593",
         "truthfulqa/presets.py": "9585addd583dd56c8c0e9eb88624783162959da92001ea739580576ee71cae2b",
         "truthfulqa/configs.py": "620f649cf37d95c0acb8a1f072497c58c2c0d49c51c562d52d8758bdfb8101a4"}, True),
}
_PROBABILITY_DEFAULTS = {"protocol": "likelihood", "sample_count": 0, "seed": None,
    "max_tokens": 1, "enable_thinking": False, "enable_mtp": False,
    "temperature": 0., "top_p": 1., "top_k": 0, "num_generations": 1}
OFFICIAL_DEFAULTS = {
    "official-5shot-likelihood": {"available": True, "parameters": _PROBABILITY_DEFAULTS,
        "source": "evaluate_flan.py"},
    "official-mc-likelihood": {"available": True, "parameters": _PROBABILITY_DEFAULTS,
        "source": "truthfulqa/models.py"},
    "official-api-cot": {"available": True, "parameters": {**_PROBABILITY_DEFAULTS,
        "protocol": "mcq", "max_tokens": 4000}, "source": "evaluate_from_api.py:call_api (OpenAI-compatible)"},
    "official-zero-shot-sample": {"available": True, "parameters": {**_PROBABILITY_DEFAULTS,
        "protocol": "mcq", "max_tokens": 1000, "seed": 0}, "source": "baselines/closed_book.py + baselines/utils.py (chat)"},
    "official-v6-python-pass1": {"available": True, "parameters": {**_PROBABILITY_DEFAULTS,
        "protocol": "code", "max_tokens": 2000, "temperature": .2, "top_p": .95,
        "num_generations": 10}, "source": "lcb_runner/runner/parser.py + oai_runner.py (OpenAIChat)"},
    "official-matharena-aime2025": {"available": True, "parameters": {"protocol": "math", "sample_count": 0,
        "max_tokens": None, "temperature": None, "top_p": None, "top_k": None, "seed": None,
        "enable_thinking": None, "enable_mtp": False, "num_generations": 1},
        "source": "configs/competitions/aime/aime_2025.yaml; unspecified generation settings use model/architecture defaults"},
}
MMLU_PRO_VALIDATION = (
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/24ac2da5bb7c7b42ea1a984c6b535e35a73d30b3/data/validation-00000-of-00001.parquet",
    "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078",
)
MMLU_DEV = (
    "https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/all/dev-00000-of-00001.parquet",
    "2b19bde1ed8ca6b482fb283abc90e8e0d9d228947029c0b91795d64b28b3bc3f",
)


def benchmark_directory(root: Path, dataset_id: str) -> Path:
    profile = OFFICIAL_BENCHMARKS[dataset_id]
    directory = (root / "datasets" / "official-scoring" / dataset_id / profile.revision).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise dataset_error("path_outside_workspace", "official scoring destination is outside the workspace")
    return directory


def verified_artifact(directory: Path, filename: str, digest: str) -> bytes:
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise dataset_error("path_outside_workspace", "official scoring artifact is outside its managed directory")
    try:
        with path.open("rb") as file:
            raw = file.read(1024 * 1024 + 1)
    except OSError as error:
        raise dataset_error("official_scorer_missing", "download the collection to install its pinned official scoring code") from error
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
        raise dataset_error("official_scorer_changed", "official scoring code does not match its pinned SHA256; download the collection again")
    return raw


def verified_source(directory: Path, filename: str, digest: str) -> str:
    return verified_artifact(directory, filename, digest).decode("utf-8")


def benchmark_readiness(root: Path | None) -> dict[str, dict[str, Any]]:
    result = {}
    for dataset_id, profile in OFFICIAL_BENCHMARKS.items():
        reason = "official_runner_pending"
        if profile.supported and root is not None:
            try:
                directory = benchmark_directory(root, dataset_id)
                for filename, digest in profile.sources.items():
                    verified_source(directory, filename, digest)
                if dataset_id == "mmlu-pro-test":
                    verified_artifact(directory, "validation.parquet", MMLU_PRO_VALIDATION[1])
                if dataset_id == "mmlu-test":
                    verified_artifact(directory, "dev.parquet", MMLU_DEV[1])
                if (profile.protocol.endswith("likelihood") and importlib.util.find_spec("torch") is None) or (
                    dataset_id in {"gpqa-diamond", "mmlu-test", "truthfulqa-mc1"} and importlib.util.find_spec("pandas") is None):
                    reason = "official_dependencies_missing"
                elif dataset_id in {"aime-2025", "livecodebench-v6"}:
                    from mfq.server.services.scoring_sandbox import sandbox_available
                    if dataset_id == "aime-2025" and any(importlib.util.find_spec(name) is None for name in ("sympy", "regex", "loguru", "antlr4", "yaml")):
                        reason = "official_dependencies_missing"
                    else:
                        reason = "" if sandbox_available() else "scoring_sandbox_unavailable"
                else:
                    reason = ""
            except Exception:
                reason = "official_scorer_missing"
        result[dataset_id] = {**profile.manifest(), "available": not reason, "reason": reason}
    return result


async def install_official_scoring(context: Any, root: Path, dataset_id: str) -> None:
    from mfq.server.api.network import system_proxy_environment
    profile = OFFICIAL_BENCHMARKS.get(dataset_id)
    if profile is None:
        return
    directory = benchmark_directory(root, dataset_id)
    artifacts = [(name, f"https://raw.githubusercontent.com/{profile.repository}/{profile.revision}/{name}", sha)
        for name, sha in profile.sources.items()]
    if dataset_id == "mmlu-pro-test":
        artifacts.append(("validation.parquet", *MMLU_PRO_VALIDATION))
    if dataset_id == "mmlu-test":
        artifacts.append(("dev.parquet", *MMLU_DEV))
    for name, url, digest in artifacts:
        context.raise_if_cancelled()
        target = (directory / name).resolve()
        if not target.is_relative_to(directory):
            raise dataset_error("path_outside_workspace", "official scoring destination is outside its managed directory")
        try:
            verified_artifact(directory, name, digest)
            continue
        except Exception:
            pass
        environment = system_proxy_environment()
        proxy = environment.get("https_proxy") or environment.get("HTTPS_PROXY") or environment.get("all_proxy") or environment.get("ALL_PROXY")
        verify = ssl.create_default_context(cafile=environment.get("SSL_CERT_FILE"), capath=environment.get("SSL_CERT_DIR"))
        for selected_proxy in ([None, proxy] if proxy else [None]):
            raw = bytearray()
            try:
                async with httpx.AsyncClient(trust_env=False, proxy=selected_proxy, verify=verify, follow_redirects=True, timeout=httpx.Timeout(5 if selected_proxy is None else 30, connect=5)) as client, client.stream("GET", url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(65536):
                        context.raise_if_cancelled()
                        raw.extend(chunk)
                        if len(raw) > 1024 * 1024:
                            raise dataset_error("official_scorer_changed", "official scoring artifact exceeds the allowed size")
                break
            except httpx.HTTPError as error:
                if selected_proxy is not None or not proxy or isinstance(error, httpx.HTTPStatusError):
                    raise dataset_error("official_scorer_download_failed", f"official scoring download failed: {error}") from error
        if hashlib.sha256(raw).hexdigest() != digest:
            raise dataset_error("official_scorer_changed", "downloaded scoring code differs from its official SHA256")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".scoring-", delete=False) as file:
                temporary = Path(file.name)
                file.write(raw)
            context.raise_if_cancelled()
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def official_definitions(source: str, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise dataset_error("official_scorer_changed", "pinned official functions are missing")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<verified-official-scoring>", "exec"), namespace)
    return namespace


def load_choice_scorer(root: Path, dataset_id: str) -> dict[str, Any]:
    profile = OFFICIAL_BENCHMARKS[dataset_id]
    directory = benchmark_directory(root, dataset_id)
    sources = {name: verified_source(directory, name, sha) for name, sha in profile.sources.items()}
    namespace = {"re": re, "json": json, "os": os, "time": time, "random": random.Random(12345)}
    if dataset_id in {"mmlu-test", "truthfulqa-mc1"}:
        import warnings

        import numpy as np
        import pandas as pd
        import torch
        namespace.update(np=np, pd=pd, torch=torch, warnings=warnings)
        if dataset_id == "mmlu-test":
            namespace["choices"] = ["A", "B", "C", "D"]
            return official_definitions(sources["evaluate_flan.py"], {"format_subject", "format_example", "gen_prompt", "eval"}, namespace)
        exec(compile(sources["truthfulqa/presets.py"], "<verified-official-presets>", "exec"), namespace)
        exec(compile(sources["truthfulqa/configs.py"], "<verified-official-configs>", "exec"), namespace)
        official_definitions(sources["truthfulqa/utilities.py"], {"format_prompt", "format_prompt_with_answer_strings", "split_multi_answer", "format_best"}, namespace)
        return official_definitions(sources["truthfulqa/models.py"], {"set_columns", "MC_calcs", "run_probs"}, namespace)
    if dataset_id == "mmlu-pro-test":
        return official_definitions(sources["evaluate_from_api.py"], {
            "preprocess", "format_example", "extract_answer", "extract_again", "extract_final",
            "single_request", "update_result", "save_summary"}, namespace)
    if dataset_id == "gpqa-diamond":
        import pandas as pd
        namespace.update(pd=pd, List=list, Example=namedtuple("Example", ["question", "choice1", "choice2", "choice3", "choice4", "correct_index"]))
        official_definitions(sources["baselines/utils.py"], {"base_prompt", "zero_shot_prompt", "load_examples"}, namespace)
        return official_definitions(sources["baselines/closed_book.py"], {"AnswerPredictor"}, namespace)
    if dataset_id == "livecodebench-v6":
        from enum import Enum

        import numpy as np
        style = official_definitions(sources["lcb_runner/lm_styles.py"], {"LMStyle"}, {"Enum": Enum})["LMStyle"]
        scorer = official_definitions(sources["lcb_runner/prompts/code_generation.py"],
            {"PromptConstants", "get_generic_question_template_answer", "format_prompt_generation"},
            {"LMStyle": style, "CodeGenerationProblem": Any, "sources": sources})
        return official_definitions(sources["lcb_runner/evaluation/pass_k_utils.py"],
            {"estimate_pass_at_k", "compute_metrics_from_results"}, dict(scorer, np=np))
    if dataset_id == "aime-2025":
        import yaml
        return {"config": yaml.safe_load(sources["configs/competitions/aime/aime_2025.yaml"]), "sources": sources}
    raise dataset_error("official_runner_pending", "this task's official runner is not integrated; approximate scoring is disabled")


def pro_prompt(scorer: dict[str, Any], question: dict[str, Any], examples: dict[str, Any]) -> str:
    captured = []
    def capture(_client, instruction, inputs):
        captured.append(instruction + inputs)
        return "The answer is (A)"
    scorer["call_api"] = capture
    scorer["single_request"](None, question, examples, [])
    return captured[0]


def pro_scores(scorer: dict[str, Any], questions: list[dict[str, Any]]) -> list[bool]:
    scores = []
    with tempfile.TemporaryDirectory(prefix="mfq-official-score-") as directory:
        target = Path(directory) / "results.json"
        for question in questions:
            target.write_text(json.dumps([question]), encoding="utf-8")
            _, categories = scorer["update_result"](str(target))
            scores.append(bool(categories[question["category"]]["corr"]))
    return scores
