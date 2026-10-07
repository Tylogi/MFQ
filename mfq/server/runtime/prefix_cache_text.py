from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _tokenizer(path: str, modified: int, size: int):
    from tokenizers import Tokenizer
    from mfq.formats.assets import HF_TOKENIZER_JSON_ASSET
    from mfq.formats.io import open_mmap

    source = Path(path)
    try:
        if source.is_dir():
            return Tokenizer.from_file(str(source / 'tokenizer.json'))
        with open_mmap(source) as store:
            return Tokenizer.from_str(store.read_blob(HF_TOKENIZER_JSON_ASSET).decode('utf-8'))
    except Exception as error:
        raise ValueError('cache tokenizer is unavailable') from error


def decode_prefix_cache_tokens(path: Path, tokens: list[int]) -> str:
    source = path / 'tokenizer.json' if path.is_dir() else path
    status = source.stat()
    return _tokenizer(str(path), status.st_mtime_ns, status.st_size).decode(tokens, skip_special_tokens=False)
