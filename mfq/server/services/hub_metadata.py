from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import httpx

from mfq.formats.assets import is_asset_record
from mfq.server.state.catalog import _is_always_streamed_tensor


class IncompleteHeaderError(ValueError):
    pass


@dataclass(frozen=True)
class MfqMetadata:
    architecture: str
    config: dict[str, Any]
    weight_bytes: int
    ssd_ple_bytes: int


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
    names = set()
    for _ in range(count):
        name, dtype = text(), text()
        size, = read('<Q')
        if name in names:
            raise ValueError('Duplicate MFQ tensor record')
        names.add(name)
        payload_bytes += size
        if is_asset_record(name):
            continue
        if _is_always_streamed_tensor(name) and dtype in {'NINT', 'F16', 'BF16', 'F32', 'F8_E4M3'}:
            ple_bytes += size
        else:
            weight_bytes += size
    if offset + payload_bytes != file_size:
        raise ValueError('MFQ file size does not match its tensor index')
    config = extra.get('hf_config')
    return MfqMetadata(architecture, config if isinstance(config, dict) else {}, weight_bytes, ple_bytes)


async def read_mfq_metadata(
    client: httpx.AsyncClient, url: str, params: dict[str, str] | None, file_size: int,
) -> MfqMetadata:
    length = min(65536, file_size)
    while length <= min(1 << 20, file_size):
        async with client.stream('GET', url, params=params, headers={
            'Range': f'bytes=0-{length - 1}', 'Accept-Encoding': 'identity',
        }) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get('content-range') != f'bytes 0-{length - 1}/{file_size}':
                raise ValueError('The hub did not honor the bounded MFQ metadata range')
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > length:
                    raise ValueError('MFQ metadata response exceeds its requested range')
                data.extend(chunk)
        try:
            return inspect_mfq_header(bytes(data), file_size)
        except IncompleteHeaderError:
            next_length = min(length * 4, 1 << 20, file_size)
            if next_length <= length:
                raise
            length = next_length
    raise ValueError('MFQ tensor index exceeds the metadata read limit')
