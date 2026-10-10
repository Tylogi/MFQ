from __future__ import annotations

import os
import hashlib
import re
import shutil
import struct
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_HASH = re.compile(r"[0-9a-f]{64}")
_BUCKET = re.compile(r"[0-9a-f]{2}")
_HEADER = struct.Struct("<8sI32s32s32sIIQ32s")
_TEXT_HEADER = struct.Struct("<8s32s32sI32s")


@dataclass(frozen=True)
class _Block:
    path: Path
    stat: os.stat_result
    namespace: str
    digest: str
    parent: str
    tokens: int
    block_size: int
    payload_bytes: int
    text_stat: os.stat_result | None = None

    @property
    def size(self) -> int:
        return self.stat.st_size + (self.text_stat.st_size if self.text_stat else 0)


def _text_record(path: Path, namespace: str, digest: str, count: int, *, read: bool = False):
    try:
        descriptor = os.open(str(path) + '.tokens', os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(descriptor, 'rb') as stream:
            status = os.fstat(stream.fileno())
            raw = stream.read(_TEXT_HEADER.size)
            if len(raw) != _TEXT_HEADER.size or not 0 < count <= 65536:
                return None
            magic, compatibility, identity, tokens, checksum = _TEXT_HEADER.unpack(raw)
            if magic != b'MFQTXT1\0' or compatibility.hex() != namespace or identity.hex() != digest or tokens != count:
                return None
            if status.st_size != _TEXT_HEADER.size + count * 8:
                return None
            if not read:
                return status
            payload = stream.read(count * 8)
            if hashlib.sha256(payload).digest() != checksum:
                return None
            values = [item[0] for item in struct.iter_unpack('<q', payload)]
            return values if all(0 <= item <= 0xffffffff for item in values) else None
    except (OSError, ValueError):
        return None


def _scan_blocks(directory: Path) -> list[_Block]:
    blocks: list[_Block] = []
    if directory.is_dir():
        with os.scandir(directory) as namespaces:
            for namespace in namespaces:
                if not _HASH.fullmatch(namespace.name) or not namespace.is_dir(follow_symlinks=False):
                    continue
                with os.scandir(namespace.path) as buckets:
                    for bucket in buckets:
                        if not _BUCKET.fullmatch(bucket.name) or not bucket.is_dir(follow_symlinks=False):
                            continue
                        with os.scandir(bucket.path) as files:
                            for file in files:
                                if not file.name.endswith(".mfqkv") or not file.is_file(follow_symlinks=False):
                                    continue
                                try:
                                    descriptor = os.open(file.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                                    with os.fdopen(descriptor, "rb") as stream:
                                        status = os.fstat(stream.fileno())
                                        raw = stream.read(_HEADER.size)
                                    if len(raw) != _HEADER.size:
                                        continue
                                    magic, version, compatibility, digest, parent, size, tokens, payload, _ = _HEADER.unpack(raw)
                                    if magic != b"MFQKVB1\0" or version != 1 or compatibility.hex() != namespace.name:
                                        continue
                                    if file.name != digest.hex() + ".mfqkv" or bucket.name != digest.hex()[:2]:
                                        continue
                                    if not 0 < tokens <= size <= 65536 or status.st_size != _HEADER.size + payload:
                                        continue
                                    path = Path(file.path)
                                    blocks.append(_Block(path, status, namespace.name, digest.hex(), parent.hex(), tokens, size, payload,
                                        _text_record(path, namespace.name, digest.hex(), tokens)))
                                except (FileNotFoundError, OSError):
                                    continue
    return blocks


def _remove_block(block: _Block) -> tuple[int, bool] | None:
    try:
        status = block.path.stat(follow_symlinks=False)
        if (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns) != (
            block.stat.st_dev, block.stat.st_ino, block.stat.st_size, block.stat.st_mtime_ns):
            return None
        block.path.unlink()
        released = block.stat.st_size
        complete = True
        if block.text_stat is not None:
            try:
                text = Path(str(block.path) + '.tokens')
                current = text.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino, current.st_size) == (block.text_stat.st_dev, block.text_stat.st_ino, block.text_stat.st_size):
                    text.unlink()
                    released += current.st_size
                else:
                    complete = False
            except FileNotFoundError:
                pass
            except OSError:
                complete = False
        return released, complete
    except OSError:
        return None


def maintain_prefix_disk_budget(directory: Path, limit: int | None, *, evict: bool) -> dict[str, int]:
    blocks = _scan_blocks(directory)
    used = sum(block.size for block in blocks)
    ancestor = directory
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    capacity = (free + used) // 2 if limit is None else min(limit, (free + used) * 9 // 10)
    capacity = min(capacity, max(0, free + used - (1 << 30)))
    removed = 0
    released = 0
    if evict:
        for block in sorted(blocks, key=lambda item: (item.stat.st_mtime_ns, str(item.path))):
            if used <= capacity:
                break
            result = _remove_block(block)
            if result is None:
                continue
            used -= result[0]
            released += result[0]
            removed += 1
    return {"prefix_cache_total_disk_bytes": used, "prefix_cache_total_disk_blocks": len(blocks) - removed,
        "prefix_cache_total_disk_max_bytes": capacity, "released_bytes": released, "evicted_blocks": removed}


def prefix_cache_identity(directory: Path, namespace: str) -> dict[str, str | int | None]:
    result: dict[str, str | int | None] = {"model_name": None, "source_file": None, "architecture": None, "codec": None, "context_size": None, "model_path": None}
    if not _HASH.fullmatch(namespace) or (directory / namespace).is_symlink():
        return result
    try:
        descriptor = os.open(directory / namespace / "identity.txt", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            text = stream.read(32769)
        if len(text) > 32768:
            return result
        lines = text.decode("utf-8").splitlines()
        if len(lines) < 2 or lines[0] != "mfq-cache-identity-v1" or lines[1] != "namespace=" + namespace:
            return result
        for line in lines[2:]:
            key, _, value = line.partition("=")
            if key in {"codec", "architecture"}:
                result[key] = value[:256]
            elif key == "context" and value.isdigit():
                result["context_size"] = int(value)
            elif key == "model_path":
                result["model_path"] = value
            elif key in {"shard", "source"} and value.startswith("0:"):
                name = value[2:].rsplit(":", 2)[0]
                result["source_file"] = name
                result["model_name"] = re.sub(r"-\d{5}-of-\d{5}$", "", Path(name).stem)
    except (OSError, UnicodeError, ValueError):
        pass
    return result


def inspect_prefix_cache(directory: Path, namespace: str | None = None, *, offset: int = 0, limit: int = 100,
    query: str = '', search_in: str = 'all', text_filter: str = 'all', chain_filter: str = 'all', sort: str = 'recent',
    decode_text: Callable[[list[int]], str] | None = None) -> dict:
    blocks = _scan_blocks(directory)
    groups: dict[str, list[_Block]] = {}
    for block in blocks:
        groups.setdefault(block.namespace, []).append(block)
    data = []
    details = []
    matched_blocks = 0
    search_text_unavailable = False
    needle = query.strip().casefold()
    for identity, items in groups.items():
        if namespace is not None and namespace != identity:
            continue
        by_hash = {item.digest: item for item in items}
        lengths: dict[str, tuple[int, bool]] = {}
        def length(digest: str) -> tuple[int, bool]:
            path = []
            visited = set()
            current = digest
            while current not in lengths and current in by_hash and current not in visited:
                visited.add(current)
                path.append(current)
                current = by_hash[current].parent
            count, complete = lengths.get(current, (0, current == '0' * 64))
            for key in reversed(path):
                count += by_hash[key].tokens
                lengths[key] = count, complete
            return lengths.get(digest, (0, False))
        parents = {item.parent for item in items}
        tips = [item for item in items if item.digest not in parents]
        metadata = prefix_cache_identity(directory, identity)
        metadata.pop('model_path', None)
        data.append({"id": identity, **metadata,
            "bytes": sum(item.size for item in items), "blocks": len(items), "prefixes": len(tips),
            "text_blocks": sum(item.text_stat is not None for item in items),
            "block_tokens": max(item.block_size for item in items),
            "max_prefix_tokens": max((length(item.digest)[0] for item in tips if length(item.digest)[1]), default=0),
            "incomplete_prefixes": sum(not length(item.digest)[1] for item in tips),
            "last_used_at": datetime.fromtimestamp(max(item.stat.st_mtime for item in items), timezone.utc).isoformat()})
        if namespace is not None:
            filtered = []
            for item in items:
                if text_filter != 'all' and (item.text_stat is not None) != (text_filter == 'saved'):
                    continue
                if chain_filter != 'all' and length(item.digest)[1] != (chain_filter == 'complete'):
                    continue
                matches = not needle or search_in != 'text' and needle in item.digest
                if not matches and search_in != 'id' and item.text_stat is not None:
                    if decode_text is None:
                        search_text_unavailable = True
                    else:
                        tokens = _text_record(item.path, identity, item.digest, item.tokens, read=True)
                        if tokens is not None:
                            try:
                                matches = needle in decode_text(tokens).casefold()
                            except (OSError, ValueError, ImportError):
                                search_text_unavailable = True
                if matches:
                    filtered.append(item)
            key = (lambda item: (length(item.digest)[0], item.digest)) if sort == 'length' else (
                (lambda item: (item.size, item.digest)) if sort == 'size' else (lambda item: (item.stat.st_mtime_ns, item.digest)))
            ordered = sorted(filtered, key=key, reverse=sort != 'oldest')
            matched_blocks = len(ordered)
            details = [{"id": item.digest, "parent": item.parent, "tokens": item.tokens,
                "prefix_tokens": length(item.digest)[0], "complete_chain": length(item.digest)[1],
                "payload_bytes": item.payload_bytes, "bytes": item.size, "text_available": item.text_stat is not None,
                "last_used_at": datetime.fromtimestamp(item.stat.st_mtime, timezone.utc).isoformat()}
                for item in ordered[offset:offset + limit]]
    data.sort(key=lambda item: item["last_used_at"], reverse=True)
    return {"directory": str(directory), "total_bytes": sum(item.size for item in blocks),
        "total_blocks": len(blocks), "data": data, "blocks": details, "offset": offset, "limit": limit,
        "matched_blocks": matched_blocks, "search_text_unavailable": search_text_unavailable}


def purge_prefix_cache(directory: Path, namespace: str | None = None) -> dict[str, int]:
    blocks = [item for item in _scan_blocks(directory) if namespace is None or item.namespace == namespace]
    results = [_remove_block(item) for item in blocks]
    return {"released_bytes": sum(item[0] for item in results if item is not None),
        "removed_blocks": sum(item is not None for item in results),
        "failed_blocks": sum(item is None or not item[1] for item in results)}


def read_prefix_cache_tokens(directory: Path, namespace: str, digest: str) -> dict:
    if not _HASH.fullmatch(namespace) or not _HASH.fullmatch(digest):
        return {"available": False, "reason": "invalid_cache_id"}
    blocks = {item.digest: item for item in _scan_blocks(directory) if item.namespace == namespace}
    if digest not in blocks:
        return {"available": False, "reason": "cache_not_found"}
    chain = []
    visited = set()
    current = digest
    total = 0
    while current != '0' * 64:
        if current not in blocks or current in visited:
            return {"available": False, "reason": "incomplete_chain"}
        visited.add(current)
        item = blocks[current]
        total += item.tokens
        if total > 1048576:
            return {"available": False, "reason": "prefix_too_large"}
        tokens = _text_record(item.path, namespace, current, item.tokens, read=True)
        if tokens is None:
            return {"available": False, "reason": "text_invalid" if item.text_stat is not None else "text_not_saved"}
        chain.append(tokens)
        current = item.parent
    return {"available": True, "tokens": [token for values in reversed(chain) for token in values],
        **prefix_cache_identity(directory, namespace)}
