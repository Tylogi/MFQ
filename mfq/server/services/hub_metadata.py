from __future__ import annotations

import json
import re
import struct
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from mfq.formats.assets import MODEL_CONFIG_ASSET, is_asset_record
from mfq.formats.compat import canonical_dtype
from mfq.server.state.catalog import _is_always_streamed_tensor
from mfq.server.state.model_weights import (
    estimated_resident_weight_bytes as estimated_resident_weight_bytes,
)


class IncompleteHeaderError(ValueError):
    pass


@dataclass(frozen=True)
class MfqMetadata:
    architecture: str
    config: dict[str, Any]
    weight_bytes: int
    ssd_ple_bytes: int
    weight_bytes_by_dtype: dict[str, int] = field(default_factory=dict)
    has_mtp_weights: bool = False
    config_span: tuple[int, int] | None = None
    last_legacy_block: int = -1


def inspect_mfq_header(data: bytes, file_size: int) -> MfqMetadata:
    offset = 0

    def read(fmt: str) -> tuple:
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(data):
            raise IncompleteHeaderError('Incomplete MFQ header')
        result = struct.unpack_from(fmt, data, offset)
        offset += size
        return result

    def text() -> str:
        nonlocal offset
        size, = read('<I')
        if size > 1 << 20:
            raise ValueError('MFQ metadata string is too large')
        if offset + size > len(data):
            raise IncompleteHeaderError('Incomplete MFQ header')
        value = data[offset:offset + size].decode('utf-8')
        offset += size
        return value

    magic, version = read('<4sI')
    if magic != b'MFQ1' or version not in {1, 2}:
        raise ValueError('Unsupported MFQ header')
    architecture = text()
    extra = {}
    if version >= 2:
        count, = read('<I')
        if count > 4096:
            raise ValueError('Too many MFQ metadata entries')
        for _ in range(count):
            key, value = text(), text()
            extra[key] = json.loads(value)
    count, = read('<I')
    if count > 100000:
        raise ValueError('Too many MFQ tensor records')
    payload_bytes = weight_bytes = ple_bytes = 0
    by_dtype: dict[str, int] = {}
    names = set()
    mtp = False
    config_record = None
    last_legacy_block = -1
    for _ in range(count):
        name, dtype = text(), text()
        size, = read('<Q')
        if name in names:
            raise ValueError('Duplicate MFQ tensor record')
        names.add(name)
        payload_bytes += size
        if is_asset_record(name):
            if name == MODEL_CONFIG_ASSET and 0 < size <= 65536:
                config_record = (payload_bytes - size, size)
            continue
        mtp = mtp or name.startswith(('predictor.', 'mtp.', 'model.mtp.'))
        block = re.match(r'^blk\.(\d+)\.', name)
        if block:
            last_legacy_block = max(last_legacy_block, int(block[1]))
        if _is_always_streamed_tensor(name) and dtype in {'NINT', 'F16', 'BF16', 'F32', 'F8_E4M3'}:
            ple_bytes += size
        else:
            weight_bytes += size
            dtype = canonical_dtype(dtype)
            by_dtype[dtype] = by_dtype.get(dtype, 0) + size
    if offset + payload_bytes != file_size:
        raise ValueError('MFQ file size does not match its tensor index')
    config = extra.get('hf_config')
    span = (offset + config_record[0], config_record[1]) if config_record else None
    if not isinstance(config, dict) and span and span[0] + span[1] <= len(data):
        with suppress(ValueError, UnicodeError):
            config = json.loads(data[span[0]:span[0] + span[1]])
        span = None
    return MfqMetadata(architecture, config if isinstance(config, dict) else {}, weight_bytes, ple_bytes, by_dtype, mtp, span, last_legacy_block)


async def _read_range(client: httpx.AsyncClient, url: str, params: dict[str, str] | None,
                      file_size: int, offset: int, length: int) -> bytes:
    async with client.stream('GET', url, params=params, headers={
        'Range': f'bytes={offset}-{offset + length - 1}', 'Accept-Encoding': 'identity',
    }) as response:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get('content-range') != f'bytes {offset}-{offset + length - 1}/{file_size}':
            raise ValueError('The hub did not honor the bounded MFQ metadata range')
        data = bytearray()
        async for chunk in response.aiter_bytes():
            if len(data) + len(chunk) > length:
                raise ValueError('MFQ metadata response exceeds its requested range')
            data.extend(chunk)
    if len(data) != length:
        raise ValueError('Incomplete MFQ metadata range response')
    return bytes(data)


async def read_mfq_metadata(
    client: httpx.AsyncClient, url: str, params: dict[str, str] | None, file_size: int,
) -> MfqMetadata:
    length = min(65536, file_size)
    while length <= min(1 << 20, file_size):
        data = await _read_range(client, url, params, file_size, 0, length)
        try:
            metadata = inspect_mfq_header(data, file_size)
        except IncompleteHeaderError:
            next_length = min(length * 4, 1 << 20, file_size)
            if next_length <= length:
                raise
            length = next_length
            continue
        if not metadata.config and metadata.config_span:
            try:
                start, size = metadata.config_span
                config = json.loads(await _read_range(client, url, params, file_size, start, size))
                if isinstance(config, dict):
                    metadata = replace(metadata, config=config)
            except (httpx.HTTPError, ValueError, UnicodeError):
                pass
        return metadata
    raise ValueError('MFQ tensor index exceeds the metadata read limit')
