"""Stable token-ID mapping identity shared with the native tokenizer."""

from __future__ import annotations

import hashlib
import struct

TOKENIZER_IDENTITY_FORMAT = "mfq.tokenizer-vocabulary.v1"


def tokenizer_identity(tokenizer, vocabulary_size: int | None = None) -> dict[str, str]:
    vocabulary = tokenizer.get_vocab()
    minimum = max(vocabulary.values(), default=-1) + 1
    size = minimum if vocabulary_size is None else vocabulary_size
    if not 0 < size <= 0xFFFFFFFF or minimum > size:
        raise ValueError("invalid tokenizer identity vocabulary size")
    digest = hashlib.sha256(TOKENIZER_IDENTITY_FORMAT.encode("ascii") + b"\0")
    digest.update(struct.pack("<I", size))
    for token_id in range(size):
        token = tokenizer.id_to_token(token_id)
        if token is None:
            token = f"[PAD{token_id}]"
        if not token:
            token = f"[EMPTY_{token_id}]"
        if "\0" in token:
            raise ValueError("native tokenizer identity does not support NUL in token text")
        encoded = token.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    return {"format": TOKENIZER_IDENTITY_FORMAT, "sha256": digest.hexdigest()}
