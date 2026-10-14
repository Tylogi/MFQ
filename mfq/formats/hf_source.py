"""Architecture-neutral canonical view over an HF Safetensors checkpoint."""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mfq.architectures.tensor_schema import (
    canonical_source_tensor_map,
    graph_spec_for_source_names,
)
from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET, model_graph_asset
from mfq.formats.io import BFloat16Array, Float8E4M3Array, MfqTensor
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.mx import MXFP4_DTYPE, MXFP8_DTYPE, MxTensor


_DTYPE_LAYOUTS = {
    "BOOL": np.dtype("?"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "F8_E4M3": np.dtype("u1"),
    "F8_E4M3FN": np.dtype("u1"),
    "F8_E5M2": np.dtype("u1"),
    "F8_E8M0": np.dtype("u1"),
}

_EXPERT_WEIGHT = re.compile(
    r"^((?:model|predictor)\.block\.\d+\.mlp\.experts)\."
    r"(\d+)\.(gate|up|down)\.weight$"
)


@dataclass(frozen=True)
class HfSourceRecord:
    """One validated tensor range inside a Safetensors shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    offset: int
    nbytes: int


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _weight_map(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json_object(index_path)
        raw = index.get("weight_map")
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Safetensors index requires a non-empty weight_map")
        if not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in raw.items()
        ):
            raise ValueError("Safetensors weight_map entries must be strings")
        return dict(raw)
    shards = tuple(sorted(root.glob("*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no Safetensors shards found under {root}")
    result: dict[str, str] = {}
    for shard in shards:
        for name in _shard_records(shard):
            if name in result:
                raise ValueError(f"duplicate Safetensors tensor: {name}")
            result[name] = shard.name
    return result


def _shard_records(path: Path) -> dict[str, HfSourceRecord]:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"truncated Safetensors header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        raw_header = stream.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError(f"truncated Safetensors header: {path}")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError(f"invalid Safetensors header: {path}")
    data_start = 8 + int(header_size)
    result: dict[str, HfSourceRecord] = {}
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"invalid Safetensors entry: {name}")
        dtype = str(raw.get("dtype", ""))
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if (
            dtype not in _DTYPE_LAYOUTS
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"invalid Safetensors entry: {name}")
        dimensions = tuple(int(value) for value in shape)
        begin, end = (int(value) for value in offsets)
        nbytes = end - begin
        expected = math.prod(dimensions) * _DTYPE_LAYOUTS[dtype].itemsize
        if begin < 0 or end < begin or nbytes != expected:
            raise ValueError(f"invalid Safetensors payload range: {name}")
        result[name] = HfSourceRecord(
            name=name,
            dtype=dtype,
            shape=dimensions,
            path=path,
            offset=data_start + begin,
            nbytes=nbytes,
        )
    return result


def _source_records(root: Path, weight_map: Mapping[str, str]) -> dict[str, HfSourceRecord]:
    records: dict[str, HfSourceRecord] = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    for shard_name, names in sorted(by_shard.items()):
        shard_path = root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Safetensors shard is missing: {shard_path}")
        shard = _shard_records(shard_path)
        for name in names:
            try:
                records[name] = shard[name]
            except KeyError as error:
                raise KeyError(
                    f"Safetensors index references missing tensor {name!r} in {shard_name}"
                ) from error
    return records


def _tagged_dense(record: HfSourceRecord) -> np.ndarray:
    with record.path.open("rb", buffering=0) as stream:
        stream.seek(record.offset)
        payload = stream.read(record.nbytes)
    if len(payload) != record.nbytes:
        raise EOFError(f"short Safetensors read for {record.name}")
    value = np.frombuffer(payload, dtype=_DTYPE_LAYOUTS[record.dtype]).copy()
    value = value.reshape(record.shape)
    if record.dtype == "BF16":
        return value.view(BFloat16Array)
    if record.dtype in {"F8_E4M3", "F8_E4M3FN"}:
        return value.view(Float8E4M3Array)
    return value


def _mx_scale_name(weight_name: str) -> str:
    return weight_name.removesuffix(".weight") + ".weight_scale"


def _dequantize_mx(tensor: MxTensor) -> np.ndarray:
    scales = np.asarray(tensor.scales, dtype=np.uint8)
    exponents = scales.astype(np.int16) - 127
    decoded_scales = np.ldexp(np.ones(scales.shape, dtype=np.float32), exponents)
    decoded_scales[scales == 255] = np.nan
    rows, columns = tensor.shape
    if tensor.dtype == MXFP4_DTYPE:
        table = np.asarray(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6),
            dtype=np.float32,
        )
        packed = np.asarray(tensor.values, dtype=np.uint8)
        codes = np.empty((rows, columns), dtype=np.uint8)
        codes[:, 0::2] = packed & 15
        codes[:, 1::2] = packed >> 4
        values = table[codes]
        expanded = np.repeat(decoded_scales, 32, axis=1)
    else:
        raw = np.asarray(tensor.values, dtype=np.uint8)
        sign = np.where(raw & 0x80, -1.0, 1.0).astype(np.float32)
        exponent = ((raw >> 3) & 15).astype(np.int16)
        mantissa = (raw & 7).astype(np.float32)
        values = sign * np.where(
            exponent == 0,
            np.ldexp(mantissa, -9),
            np.ldexp(1.0 + mantissa / 8.0, exponent - 7),
        )
        values[(raw & 0x7F) == 0x7F] = np.nan
        scale_rows, scale_columns = scales.shape
        row_block = 1 if scale_rows == rows else (32 if scale_rows == (rows + 31) // 32 else 128)
        column_block = 32 if scale_columns == columns // 32 else 128
        expanded = np.repeat(np.repeat(decoded_scales, row_block, axis=0), column_block, axis=1)
        expanded = expanded[:rows, :columns]
    return np.ascontiguousarray(values * expanded, dtype=np.float16)


class HfSourceTensorStore(Mapping[str, MfqTensor]):
    """Expose every registered HF architecture through one canonical store.

    Architecture code is not involved in opening Safetensors. It consumes the
    same canonical names used for MFQ files; the existing schema registry is
    consulted once at this storage boundary.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        config_path = self.root / "config.json"
        self._config_bytes = config_path.read_bytes()
        config = json.loads(self._config_bytes)
        if not isinstance(config, dict):
            raise ValueError("HF config.json must be an object")
        weight_map = _weight_map(self.root)
        self._source = _source_records(self.root, weight_map)
        source_names = tuple(self._source)
        self._aliases = canonical_source_tensor_map(config, source_names)
        graph = graph_spec_for_source_names(config, source_names)
        if graph is None:
            raise ValueError("model architecture has no canonical tensor schema")
        self._assets: dict[str, bytes] = {
            MODEL_CONFIG_ASSET: self._config_bytes,
            MODEL_GRAPH_ASSET: model_graph_asset(graph.as_dict()).data,
        }
        grouped: dict[str, dict[int, str]] = {}
        for canonical in self._aliases:
            match = _EXPERT_WEIGHT.match(canonical)
            if match is None:
                continue
            aggregate = f"{match.group(1)}.{match.group(3)}.weight"
            grouped.setdefault(aggregate, {})[int(match.group(2))] = canonical
        self._expert_groups: dict[str, tuple[str, ...]] = {}
        for aggregate, members in grouped.items():
            ids = sorted(members)
            if ids != list(range(len(ids))):
                raise ValueError(f"expert IDs are not contiguous for {aggregate}")
            self._expert_groups[aggregate] = tuple(members[index] for index in ids)
        self._names = tuple(
            dict.fromkeys((*self._assets, *self._aliases, *self._expert_groups))
        )
        self.legacy_semantics = None

    def _mx_tensor(self, canonical: str, record: HfSourceRecord) -> MxTensor | None:
        if not canonical.endswith(".weight") or record.dtype not in {
            "I8",
            "F8_E4M3",
            "F8_E4M3FN",
        }:
            return None
        scale_source = self._aliases.get(_mx_scale_name(canonical))
        if scale_source is None:
            return None
        scale_record = self._source[scale_source]
        if scale_record.dtype != "F8_E8M0":
            return None
        values = np.ascontiguousarray(_tagged_dense(record), dtype=np.uint8)
        scales = np.ascontiguousarray(_tagged_dense(scale_record), dtype=np.uint8)
        if len(record.shape) != 2:
            raise ValueError(f"native MX weight must be rank 2: {canonical}")
        logical_shape = (
            (record.shape[0], record.shape[1] * 2)
            if record.dtype == "I8"
            else record.shape
        )
        dtype = MXFP4_DTYPE if record.dtype == "I8" else MXFP8_DTYPE
        return MxTensor(dtype, logical_shape, values, scales)

    def _tensor(self, canonical: str) -> MfqTensor:
        source_name = self._aliases[canonical]
        record = self._source[source_name]
        mx_tensor = self._mx_tensor(canonical, record)
        return _tagged_dense(record) if mx_tensor is None else mx_tensor

    def _expert_bank(self, name: str) -> MfqTensor:
        members = self._expert_groups[name]
        tensors = tuple(self._tensor(member) for member in members)
        if all(isinstance(value, MxTensor) for value in tensors):
            mx_tensors = tuple(value for value in tensors if isinstance(value, MxTensor))
            out, width = mx_tensors[0].shape
            if any(
                value.dtype != mx_tensors[0].dtype
                or value.shape != (out, width)
                for value in mx_tensors
            ):
                raise ValueError(f"inconsistent native MX expert shapes for {name}")
            try:
                merged = MxTensor(
                    mx_tensors[0].dtype,
                    (len(mx_tensors) * out, width),
                    np.concatenate(tuple(value.values for value in mx_tensors), axis=0),
                    np.concatenate(tuple(value.scales for value in mx_tensors), axis=0),
                )
                return MfeTensor(
                    (len(mx_tensors), out, width),
                    (MfePool(np.arange(len(mx_tensors), dtype=np.int32), merged),),
                )
            except ValueError:
                return np.stack(tuple(_dequantize_mx(value) for value in mx_tensors))
        if not all(isinstance(value, np.ndarray) for value in tensors):
            raise TypeError(f"inconsistent expert storage for {name}")
        arrays = tuple(value for value in tensors if isinstance(value, np.ndarray))
        result = np.ascontiguousarray(np.stack(arrays))
        if all(isinstance(value, BFloat16Array) for value in arrays):
            return result.view(BFloat16Array)
        return result

    def __getitem__(self, name: str) -> MfqTensor:
        if name in self._assets:
            return self._assets[name]
        if name in self._expert_groups:
            return self._expert_bank(name)
        return self._tensor(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def close(self) -> None:
        """Match the MFQ store lifecycle; tensors are read with scoped I/O."""

    def __enter__(self) -> HfSourceTensorStore:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


__all__ = ["HfSourceRecord", "HfSourceTensorStore"]
