from __future__ import annotations

import ast
import contextlib
import io
import json
import resource
import sys
import types
from collections import defaultdict
from enum import Enum


def definitions(source, names, namespace):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("official definitions are missing")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<verified-official>", "exec"), namespace)
    return namespace


def run(request):
    sys.path.insert(0, request["site_packages"])
    sources = request["sources"]
    if request["task"] == "aime-2025":
        import regex
        import sympy
        from loguru import logger
        package = types.ModuleType("matharena")
        package.__path__ = []
        sys.modules["matharena"] = package
        manual = types.ModuleType("matharena.parse_manual")
        exec(compile(sources["src/matharena/parse_manual.py"], "<verified-official-manual>", "exec"), manual.__dict__)
        sys.modules[manual.__name__] = manual
        parser = types.ModuleType("matharena.parser")
        exec(compile(sources["src/matharena/parser.py"], "<verified-official-parser>", "exec"), parser.__dict__)
        namespace = dict(parser.__dict__, defaultdict=defaultdict, logger=logger)
        definitions(sources["src/matharena/utils.py"], {"is_conversation_broken"}, namespace)
        definitions(sources["src/matharena/grader.py"], {
            "extract_numbers", "check_number_proximity_any_order", "check_all_numbers",
            "check_output_length", "extract_and_grade"}, namespace)
        answer, correct, warning = namespace["extract_and_grade"](
            [{"role": "assistant", "content": request["response"]}], request["output_tokens"],
            request["answer"], request["config"])
        return {"predicted": str(answer) if answer is not None else None, "correct": bool(correct), "warning": warning}
    if request["task"] == "livecodebench-v6":
        style = definitions(sources["lcb_runner/lm_styles.py"], {"LMStyle"}, {"Enum": Enum})["LMStyle"]
        namespace = definitions(sources["lcb_runner/utils/extraction_utils.py"], {"extract_code"}, {"LMStyle": style})
        code = namespace["extract_code"](request["response"], style.OpenAIChat)
        tests = {}
        exec(compile(sources["lcb_runner/evaluation/testing_util.py"], "<verified-official-testing>", "exec"), tests)
        result, metadata = tests["run_test"]({"input_output": json.dumps(request["tests"])}, code, timeout=6)
        import numpy as np
        metrics = definitions(sources["lcb_runner/evaluation/pass_k_utils.py"],
            {"estimate_pass_at_k", "compute_metrics_from_results"}, {"np": np})
        score = metrics["compute_metrics_from_results"]({0: [result]}, k_list=[1])
        return {"predicted": code, "correct": bool(score["pass@1"]), "pass_at_1": float(score["pass@1"]),
            "test_results": [int(value) for value in result], "test_metadata": metadata}
    raise RuntimeError("unsupported official scoring task")


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    request = json.loads(sys.stdin.buffer.read(128 * 1024 * 1024 + 1))
    output = sys.stdout
    with contextlib.redirect_stdout(io.StringIO()):
        result = run(request)
    output.write(json.dumps(result, default=str))
    output.flush()


if __name__ == "__main__":
    main()
