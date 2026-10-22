import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter, TokenType
from tokenizers import Tokenizer, models

from mfq.evaluation.reference import ReferenceWriter
from mfq.formats.assets import TOKENIZER_GGUF_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.formats.tokenizer_identity import tokenizer_identity


def _identity(names):
    tokenizer = Tokenizer(models.WordLevel(
        {name: index for index, name in enumerate(names)}, unk_token="<unk>"))
    return tokenizer_identity(tokenizer)


def _write_case(root, case, source):
    reference_names = ["<unk>", "<s>", "</s>"] + [f"<0x{value:02X}>" for value in range(256)]
    candidate_names = list(reference_names)
    if case == "reordered":
        candidate_names[100], candidate_names[101] = candidate_names[101], candidate_names[100]
    identity = _identity(reference_names)
    assert len(candidate_names) == len(reference_names)
    assert (_identity(candidate_names) == identity) is (case != "reordered")

    tokenizer_path = root / "tokenizer.gguf"
    writer = GGUFWriter(tokenizer_path, "identity-fixture")
    writer.add_tokenizer_model("llama")
    writer.add_token_list(candidate_names)
    writer.add_token_scores([0.0] * len(candidate_names))
    writer.add_token_types(
        [TokenType.UNKNOWN, TokenType.CONTROL, TokenType.CONTROL] + [TokenType.BYTE] * 256)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    # The contract must be checked before this container's absent model weights.
    model_path = root / "candidate.mfq"
    assets = {TOKENIZER_GGUF_ASSET: tokenizer_path.read_bytes()} if source == "embedded" else {}
    save(model_path, FileHeader(version=2, model_arch="identity-fixture"), assets)

    tokens = np.array([[100, 101, 100, 101]], dtype="<i4")
    reference_path = root / "reference.logits"
    with ReferenceWriter(reference_path, tokens, len(reference_names)) as reference:
        reference.logits(0, 0, np.zeros((1, len(reference_names)), dtype=np.float32))
        reference.commit()
    metadata = {"vocab_size": len(reference_names)}
    if case != "missing":
        metadata["identity"] = identity
    document = {
        "format": "mfq.perplexity-logits-manifest.v1",
        "reference": {"magic": "_logit3_", "file": {
            "name": reference_path.name,
            "bytes": reference_path.stat().st_size,
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        }},
        "dataset": {"id": "identity-fixture", "sha256": hashlib.sha256(b"fixture").hexdigest()},
        "model": {"label": "identity-fixture", "precision": {"classification": "FP32"}},
        "tokenizer": metadata,
        "evaluation": {
            "protocol": "llama.cpp-non-strided-second-half-v1",
            "n_ctx": 4, "n_batch": 4, "n_ubatch": 4, "n_seq": 1, "chunks": 1,
            "target_start": 3, "score_count_per_chunk": 1, "scored_tokens": 1,
            "input_token_ids_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
            "attention": "native_model_default", "kv_cache_dtype": "native_model_default",
        },
    }
    manifest_path = root / "reference.manifest.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    arguments = ["--model", str(model_path), "--kl-base", str(reference_path),
                 "--kl-manifest", str(manifest_path)]
    if source == "external":
        arguments.extend(["--tokenizer", str(tokenizer_path)])
    if case == "matching":
        arguments.extend(["--ctx-size", "8"])
    return arguments


@pytest.mark.parametrize("source", ["embedded", "external"])
@pytest.mark.parametrize("case, message", [
    ("reordered", "KLD reference/model tokenizer mapping mismatch"),
    ("missing", "logits manifest lacks tokenizer identity; regenerate the reference"),
    ("matching", "--ctx-size differs from the logits generation contract"),
])
def test_metal_evaluator_checks_tokenizer_identity_before_model_load(tmp_path, source, case, message):
    executable = os.environ.get("MFQ_METAL_PERPLEXITY_TEST")
    if executable is None:
        pytest.skip("MFQ_METAL_PERPLEXITY_TEST required")
    binary = Path(executable).resolve()
    assert binary.is_file(), f"Metal evaluator does not exist: {binary}"
    process = subprocess.run(
        [str(binary), *_write_case(tmp_path, case, source)],
        capture_output=True, text=True, timeout=30,
    )
    output = process.stdout + process.stderr
    assert process.returncode == 1, output
    assert f"mfq-perplexity: error: {message}" in process.stderr, output
    assert "mfq-perplexity: integrated KLD" not in output, output
    assert "cpp_kl_result" not in output, output
