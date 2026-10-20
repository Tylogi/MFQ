"""Resolve shared Engram shards without copying them into each precision tier."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import PurePosixPath

_SHARD = re.compile(r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.[mM][fF][qQ]$")
_ENGRAM = re.compile(r"^(.+)-Engram-.+$", re.IGNORECASE)


def is_engram_shard(name: str) -> bool:
    match = _SHARD.fullmatch(PurePosixPath(name).name)
    return match is not None and _ENGRAM.fullmatch(match['stem']) is not None


def is_engram_table(name: str) -> bool:
    return name.endswith(('.associative_memory.embedding.weight', '.engram.embed.weight'))


def with_engram_shards(selected: Sequence[str], available: Sequence[str]) -> list[str]:
    """Add shared shards for missing indices in each selected MFQ shard family.

    Prefer ordinary shards, preserving legacy complete checkpoints. Shared files
    must belong to the same model, split count and adjacent Engram directory.
    Multiple table precisions require an explicit choice, never a guessed one.
    """
    result = dict.fromkeys(selected)
    groups: dict[tuple[PurePosixPath, str, int], set[int]] = {}
    tables = []
    for name in available:
        path = PurePosixPath(name)
        match = _SHARD.fullmatch(path.name)
        if not match:
            continue
        index, count = int(match['index']), int(match['count'])
        if not 1 <= index <= count:
            continue
        groups.setdefault((path.parent, match['stem'], count), set()).add(index)
        table = _ENGRAM.fullmatch(match['stem'])
        if table:
            tables.append((name, path.parent, table[1], index, count))
    visited = set()
    for name in selected:
        path = PurePosixPath(name)
        match = _SHARD.fullmatch(path.name)
        if not match or is_engram_shard(name):
            continue
        count = int(match['count'])
        key = (path.parent, match['stem'], count)
        if key in visited:
            continue
        visited.add(key)
        local = groups.get(key, set())
        candidates: dict[int, list[str]] = {}
        directories = {path.parent, path.parent / 'Engram', path.parent.parent / 'Engram'}
        for table_name, parent, family, index, table_count in tables:
            if (parent in directories and table_count == count and index not in local
                    and match['stem'].startswith(family + '-')):
                candidates.setdefault(index, []).append(table_name)
        for index, names in candidates.items():
            explicit = [value for value in names if value in result]
            if len(explicit) == 1:
                continue
            if len(names) != 1:
                raise ValueError(f'ambiguous shared Engram shard {index}: {", ".join(names)}')
            result[names[0]] = None
    return list(result)
