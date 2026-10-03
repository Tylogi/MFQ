from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
import pytest

from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.mxfp4_sq import (
    SQ1_PALETTE_NIBBLES,
    SQ2_PALETTE_NIBBLES,
    SQ3_PALETTE_NIBBLES,
    mxfp4_sq_blob_nbytes,
    quantize_mxfp4_sq,
    write_mxfp4_sq_blob,
)

_HEADER = struct.Struct("<4sBBHQQ")


def _source(rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260911 + rows * 100 + columns)
    packed = rng.integers(0, 256, size=(rows, columns // 2), dtype=np.uint8)
    scales = rng.integers(119, 123, size=(rows, columns // 32), dtype=np.uint8)
    return packed, scales


def _unpack(payload: bytes, offset: int, count: int, bits: int) -> np.ndarray:
    result = np.empty(count, dtype=np.uint8)
    mask = (1 << bits) - 1
    for index in range(count):
        bit = index * bits
        value = payload[offset + bit // 8]
        if bit % 8 + bits > 8:
            value |= payload[offset + bit // 8 + 1] << 8
        result[index] = (value >> (bit % 8)) & mask
    return result


def _decode_native(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    magic, version, base, reserved, rows, columns = _HEADER.unpack_from(payload)
    assert (magic, version, reserved) == (b"SQV2", 2, 0)
    blocks = columns // 32
    q_used_nbytes = (rows * 2 + 7) // 8
    q_nbytes = (q_used_nbytes + 3) & ~3
    q = _unpack(payload, _HEADER.size, rows, 2) + 1
    offset = _HEADER.size + q_nbytes
    symbol_rows = []
    for bits in q.tolist():
        count = columns * int(bits)
        symbol_rows.append(_unpack(payload, offset, columns, int(bits)))
        offset += (count + 7) // 8

    sq_rows = int(np.count_nonzero(q < 4))
    sq4_rows = rows - sq_rows
    selectors = _unpack(payload, offset, sq_rows * blocks, 1).reshape(sq_rows, blocks)
    offset += (sq_rows * blocks + 7) // 8
    state_scales = _unpack(payload, offset, sq_rows * 8, 2).reshape(sq_rows, 8)
    offset += sq_rows * 2
    state_palettes = _unpack(payload, offset, sq_rows * 8, 5).reshape(sq_rows, 8)
    offset += sq_rows * 5
    native_scales = np.frombuffer(
        payload,
        dtype=np.uint8,
        count=sq4_rows * blocks,
        offset=offset,
    ).reshape(sq4_rows, blocks)

    output = np.empty((rows, columns // 2), dtype=np.uint8)
    output_scales = np.empty((rows, blocks), dtype=np.uint8)
    tables = {
        1: SQ1_PALETTE_NIBBLES,
        2: SQ2_PALETTE_NIBBLES,
        3: SQ3_PALETTE_NIBBLES,
    }
    sq_row = 0
    sq4_row = 0
    for row, raw_bits in enumerate(q.tolist()):
        bits = int(raw_bits)
        symbols = symbol_rows[row].reshape(blocks, 32)
        if bits == 4:
            nibbles = symbols
            output_scales[row] = native_scales[sq4_row]
            sq4_row += 1
        else:
            if bits == 1:
                low_tag = (
                    np.bitwise_xor.reduce(symbols[:, 0::2], axis=1)
                    | (np.bitwise_xor.reduce(symbols[:, 1::2], axis=1) << 1)
                )
            else:
                low_tag = np.bitwise_xor.reduce(symbols & 3, axis=1)
            tags = low_tag | (selectors[sq_row] << 2)
            output_scales[row] = base + state_scales[sq_row, tags]
            palettes = state_palettes[sq_row, tags]
            nibbles = tables[bits][palettes[:, None], symbols]
            sq_row += 1
        output[row] = (
            nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        ).reshape(columns // 2)
    assert offset + sq4_rows * blocks == len(payload)
    return output, output_scales


def test_unified_quantizer_accepts_one_q_per_neuron_and_is_deterministic() -> None:
    packed, scales = _source(4, 64)
    q = np.asarray((1, 2, 3, 4), dtype=np.uint8)

    first = quantize_mxfp4_sq(packed, scales, row_q_bits=q)
    second = quantize_mxfp4_sq(packed, scales, row_q_bits=q)

    assert first.payload == second.payload
    assert first.shape == (4, 64)
    assert first.row_q_bits == (1, 2, 3, 4)
    assert len(first.payload) == mxfp4_sq_blob_nbytes(q, 4, 64)
    assert first.descriptor.distribution_entropy == pytest.approx(2.0)

    decoded, decoded_scales = _decode_native(first.payload)
    np.testing.assert_array_equal(decoded[3], packed[3])
    np.testing.assert_array_equal(decoded_scales[3], scales[3])
    source = decode_mxfp4(packed, scales, device="cpu").double().numpy()
    reconstruction = decode_mxfp4(decoded, decoded_scales, device="cpu").double().numpy()
    row_sse = np.square(source - reconstruction).sum(axis=1)
    assert np.isfinite(row_sse).all()
    assert row_sse[3] == 0.0


@pytest.mark.parametrize("bits", (1, 2, 3, 4))
def test_uniform_q_preset_uses_canonical_container(bits: int) -> None:
    packed, scales = _source(3, 96)
    tensor = quantize_mxfp4_sq(packed, scales, row_q_bits=bits)

    assert tensor.payload[:4] == b"SQV2"
    assert tensor.row_q_bits == (bits,) * 3
    assert len(tensor.payload) == mxfp4_sq_blob_nbytes(bits, 3, 96)
    decoded, decoded_scales = _decode_native(tensor.payload)
    if bits == 4:
        np.testing.assert_array_equal(decoded, packed)
        np.testing.assert_array_equal(decoded_scales, scales)


def test_writer_accepts_uniform_or_explicit_q_but_not_both(tmp_path: Path) -> None:
    packed, scales = _source(4, 64)
    target = tmp_path / "weight.sq"
    expected = quantize_mxfp4_sq(packed, scales, row_q_bits=(1, 2, 3, 4))

    written = write_mxfp4_sq_blob(
        target,
        packed,
        scales,
        row_q_bits=(1, 2, 3, 4),
    )

    assert written == len(expected.payload)
    assert target.read_bytes() == expected.payload
    with pytest.raises(ValueError, match="either bits or row_q_bits"):
        write_mxfp4_sq_blob(target, packed, scales, bits=2, row_q_bits=(2,) * 4)


@pytest.mark.parametrize(
    "q",
    (
        (1, 2, 3),
        (1, 2, 3, 5),
        (1, 2, 3, 257),
        (1.0, 2.0, 3.0, 4.0),
    ),
)
def test_quantizer_rejects_invalid_q_maps(q) -> None:
    packed, scales = _source(4, 64)
    with pytest.raises(ValueError, match="q"):
        quantize_mxfp4_sq(packed, scales, row_q_bits=q)


def test_quantizer_rejects_invalid_native_source_and_scale_base() -> None:
    packed, scales = _source(4, 64)
    bad_scales = scales.copy()
    bad_scales[0, 0] = 255
    with pytest.raises(ValueError, match="NaN scale"):
        quantize_mxfp4_sq(packed, bad_scales)
    with pytest.raises(ValueError, match="geometry"):
        quantize_mxfp4_sq(packed, scales[:, :1])
    with pytest.raises(ValueError, match="scale base"):
        quantize_mxfp4_sq(packed, scales, matrix_scale_base=252)


def test_lossless_q4_accepts_highest_finite_native_scale() -> None:
    packed, scales = _source(2, 64)
    scales.fill(254)

    tensor = quantize_mxfp4_sq(packed, scales, row_q_bits=4)
    decoded, decoded_scales = _decode_native(tensor.payload)

    assert tensor.matrix_scale_base == 251
    np.testing.assert_array_equal(decoded, packed)
    np.testing.assert_array_equal(decoded_scales, scales)


@pytest.mark.parametrize(
    ("bits", "name", "expected"),
    (
        (1, "kMxfp4Sq1PaletteNibbles", SQ1_PALETTE_NIBBLES),
        (2, "kMxfp4Sq2PaletteNibbles", SQ2_PALETTE_NIBBLES),
        (3, "kMxfp4Sq3PaletteNibbles", SQ3_PALETTE_NIBBLES),
    ),
)
def test_production_palettes_match_unified_metal_tables(
    bits: int,
    name: str,
    expected: np.ndarray,
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc/backends/metal/ops/mlx_mxfp4_sq.h").read_text()
    match = re.search(rf"{name}\s*\{{(.*?)\}};", source, re.S)
    assert match is not None
    actual = np.asarray([int(value) for value in re.findall(r"\d+", match.group(1))])
    np.testing.assert_array_equal(actual.reshape(32, 1 << bits), expected)
