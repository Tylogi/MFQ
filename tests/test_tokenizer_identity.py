import hashlib
import json
import struct

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from mfq.evaluation.reference import tokenize_windows
from mfq.formats.tokenizer_identity import TOKENIZER_IDENTITY_FORMAT, tokenizer_identity


def test_reference_identity_rejects_same_sized_reordered_mapping(tmp_path):
    identities = []
    for name, vocab in (("reference", {"<unk>": 0, "a": 1, "b": 2}),
                        ("candidate", {"<unk>": 0, "a": 2, "b": 1})):
        root = tmp_path / name
        root.mkdir()
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(root / "tokenizer.json"))
        (root / "tokenizer_config.json").write_text("{}")
        tokens, metadata = tokenize_windows(root, " ".join(["a", "b"] * 32), 32, 1)
        identities.append(metadata["identity"])
        assert metadata["vocab_size"] == 3
        assert tokens[0, 0] == vocab["a"]
    assert identities[0] != identities[1]


def test_identity_matches_native_encoding_and_output_vocabulary_padding():
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "Ġhello": 1, "你好": 2}, unk_token="<unk>"))
    payload = (b"mfq.tokenizer-vocabulary.v1\0\x04\x00\x00\x00"
               b"\x05\x00\x00\x00<unk>\x07\x00\x00\x00\xc4\xa0hello"
               b"\x06\x00\x00\x00\xe4\xbd\xa0\xe5\xa5\xbd\x06\x00\x00\x00[PAD3]")
    assert tokenizer_identity(tokenizer, 4) == {
        "format": TOKENIZER_IDENTITY_FORMAT, "sha256": hashlib.sha256(payload).hexdigest()}
    reordered = Tokenizer(models.WordLevel({"你好": 2, "<unk>": 0, "Ġhello": 1}, unk_token="<unk>"))
    assert tokenizer_identity(reordered, 4) == tokenizer_identity(tokenizer, 4)
    with pytest.raises(ValueError, match="vocabulary size"):
        tokenizer_identity(tokenizer, 2)


def test_identity_matches_hf_to_native_gguf_import(tmp_path):
    from gguf import GGUFReader

    from mfq.server.runtime.hf_tokenizer import _write_tokenizer_gguf

    tokenizer = Tokenizer(models.BPE({"a": 0, "b": 1, "ab": 2}, [("a", "b")]))
    tokenizer.add_special_tokens(["<pad>"])
    path = _write_tokenizer_gguf(output=tmp_path / "tokenizer.gguf", model_name="fixture",
        tokenizer=json.loads(tokenizer.to_str()), tokenizer_config={"pad_token": "<pad>"},
        config={"model_type": "qwen3_5", "vocab_size": 6}, generation_config={})
    tokens = GGUFReader(path).get_field("tokenizer.ggml.tokens").contents()
    payload = TOKENIZER_IDENTITY_FORMAT.encode() + b"\0" + struct.pack("<I", len(tokens))
    for token in tokens:
        encoded = token.encode("utf-8")
        payload += struct.pack("<I", len(encoded)) + encoded
    assert tokenizer_identity(tokenizer, 6)["sha256"] == hashlib.sha256(payload).hexdigest()


def test_identity_rejects_token_text_not_preserved_by_native_c_strings():
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "a\0b": 1}, unk_token="<unk>"))
    with pytest.raises(ValueError, match="NUL"):
        tokenizer_identity(tokenizer)
