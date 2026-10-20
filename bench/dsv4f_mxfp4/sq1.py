#!/usr/bin/env python3
"""Search the one-bit member of the unified native-output MXFP4-SQ family.

Each block stores 32 binary symbols.  Even- and odd-lane symbol parities
provide two in-band state bits; one explicit bit supplies the third.  The
eight per-neuron states retain the same two-bit E8M0 scale offset and five-bit
frozen palette selector used by SQ2/SQ3.  Decoded values are always legal
MXFP4 E2M1 values.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bench.dsv4f_mxfp4.adaptive_sq import (
    E2M1_NIBBLES,
    E2M1_VALUES,
    NIBBLE_VALUES,
    _raw_gate_up_native,
    _unpack_source,
)
from bench.dsv4f_mxfp4.sq2 import (
    SQ2_TAG_ORDERS,
    _greedy_assignments,
    _refine_from_assignments,
)
from bench.dsv4f_mxfp4.sq3 import _pack_block_selectors, _pack_fixed_width
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint


SQ1_PALETTE_LEVELS = np.asarray(
    list(itertools.combinations(range(len(E2M1_VALUES)), 2)),
    dtype=np.int16,
)
SQ1_PALETTE_VALUES = E2M1_VALUES[SQ1_PALETTE_LEVELS]
SQ1_PALETTE_NIBBLES = E2M1_NIBBLES[SQ1_PALETTE_LEVELS]

# Filled from the real-weight design split by ``screen_sq1_catalog``.  It is a
# format constant rather than per-model metadata.
SQ1_FIXED32_PALETTE_IDS = np.asarray(
    (
        12, 26, 38, 11, 13, 58, 22, 48,
        25, 23, 67, 21, 37, 24, 75, 20,
        8, 59, 68, 9, 76, 7, 19, 36,
        46, 55, 63, 70, 35, 47, 56, 45,
    ),
    dtype=np.int16,
)


@dataclass(frozen=True)
class Sq1RowSolution:
    error_sse: float
    state_scale_offsets: tuple[int, ...]
    state_palette_ids: tuple[int, ...]
    block_tags: np.ndarray
    refinement_steps: int


@dataclass(frozen=True)
class Sq1Encoding:
    matrix_scale_base: int
    symbols: np.ndarray
    block_tags: np.ndarray
    state_scale_offsets: np.ndarray
    state_palette_codes: np.ndarray
    packed_symbols: np.ndarray
    packed_block_selectors: np.ndarray
    packed_state_scales: np.ndarray
    packed_state_palettes: np.ndarray
    packed_mxfp4: np.ndarray
    native_scale_raw: np.ndarray
    searched_sse: float


def _candidate_parity_errors(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    state_scales: np.ndarray,
    state_palette_ids: np.ndarray,
) -> np.ndarray:
    target = NIBBLE_VALUES[source_nibbles_row] * np.exp2(
        source_scale_row[:, None].astype(np.int16) - 127
    )
    levels = SQ1_PALETTE_VALUES[state_palette_ids] * np.exp2(
        state_scales[:, None].astype(np.int16) - 127
    )
    costs = np.square(target[None, :, :, None] - levels[:, None, None, :])
    symbols = costs.argmin(axis=3).astype(np.uint8)
    base_cost = np.take_along_axis(costs, symbols[..., None], axis=3)[..., 0]
    alternative = np.take_along_axis(costs, (1 - symbols)[..., None], axis=3)[..., 0]
    penalty = alternative - base_cost
    even_penalty = penalty[:, :, 0::2].min(axis=2)
    odd_penalty = penalty[:, :, 1::2].min(axis=2)
    parity = (
        (symbols[:, :, 0::2].sum(axis=2) & 1)
        | ((symbols[:, :, 1::2].sum(axis=2) & 1) << 1)
    )
    base_error = base_cost.sum(axis=2)
    result = np.empty((4, len(state_scales), target.shape[0]), dtype=np.float64)
    for tag in range(4):
        delta = parity ^ np.uint8(tag)
        result[tag] = (
            base_error
            + np.where(delta & 1, even_penalty, 0.0)
            + np.where(delta & 2, odd_penalty, 0.0)
        )
    return result


def _quantize_for_tag(
    target: np.ndarray,
    levels: np.ndarray,
    tag: int,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.square(target[:, :, None] - levels[None, None, :])
    symbols = costs.argmin(axis=2).astype(np.uint8)
    selected = np.take_along_axis(costs, symbols[..., None], axis=2)[..., 0]
    alternative = np.take_along_axis(costs, (1 - symbols)[..., None], axis=2)[..., 0]
    penalty = alternative - selected
    parity = (
        (symbols[:, 0::2].sum(axis=1) & 1)
        | ((symbols[:, 1::2].sum(axis=1) & 1) << 1)
    )
    delta = parity ^ np.uint8(tag)
    for block in range(symbols.shape[0]):
        if delta[block] & 1:
            index = int(penalty[block, 0::2].argmin()) * 2
            selected[block, index] = alternative[block, index]
            symbols[block, index] ^= 1
        if delta[block] & 2:
            index = int(penalty[block, 1::2].argmin()) * 2 + 1
            selected[block, index] = alternative[block, index]
            symbols[block, index] ^= 1
    return symbols, selected.sum(axis=1)


def solve_sq1_row(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    *,
    matrix_scale_base: int,
    palette_ids: np.ndarray = SQ1_FIXED32_PALETTE_IDS,
    maximum_steps: int = 8,
) -> Sq1RowSolution:
    catalog = np.asarray(palette_ids, dtype=np.int16)
    scale_values = np.arange(matrix_scale_base, matrix_scale_base + 4, dtype=np.uint8)
    state_scales = np.repeat(scale_values, len(catalog))
    state_palettes = np.tile(catalog, len(scale_values))
    errors = _candidate_parity_errors(
        source_nibbles_row,
        source_scale_row,
        state_scales,
        state_palettes,
    )
    target = NIBBLE_VALUES[source_nibbles_row] * np.exp2(
        source_scale_row[:, None].astype(np.int16) - 127
    )
    starts = [_greedy_assignments(errors, order) for order in SQ2_TAG_ORDERS]
    for values in (
        source_scale_row,
        target.mean(axis=1),
        np.square(target).sum(axis=1),
        np.abs(target).max(axis=1),
    ):
        clusters = np.empty(target.shape[0], dtype=np.uint8)
        order = np.argsort(values, kind="stable")
        clusters[order] = np.minimum(
            7, np.arange(target.shape[0]) * 8 // target.shape[0]
        )
        starts.extend(clusters ^ np.uint8(mask) for mask in range(8))
    error, states, tags, steps = min(
        (
            _refine_from_assignments(errors, start, maximum_steps=maximum_steps)
            for start in starts
        ),
        key=lambda item: item[0],
    )
    return Sq1RowSolution(
        error_sse=error,
        state_scale_offsets=tuple(
            int(state_scales[index]) - matrix_scale_base for index in states
        ),
        state_palette_ids=tuple(int(state_palettes[index]) for index in states),
        block_tags=tags,
        refinement_steps=steps,
    )


def screen_sq1_catalog(
    source_nibbles: np.ndarray,
    source_scales: np.ndarray,
    *,
    row_indices: np.ndarray,
    matrix_scale_base: int,
) -> np.ndarray:
    counts: Counter[int] = Counter()
    all_ids = np.arange(len(SQ1_PALETTE_VALUES), dtype=np.int16)
    for row in np.asarray(row_indices, dtype=np.int64):
        solution = solve_sq1_row(
            source_nibbles[int(row)],
            source_scales[int(row)],
            matrix_scale_base=matrix_scale_base,
            palette_ids=all_ids,
            maximum_steps=6,
        )
        for tag, palette in enumerate(solution.state_palette_ids):
            if np.any(solution.block_tags == tag):
                counts[palette] += int(np.count_nonzero(solution.block_tags == tag))
    selected = [palette for palette, _ in counts.most_common(32)]
    if len(selected) < 32:
        selected.extend(
            index for index in range(len(SQ1_PALETTE_VALUES)) if index not in selected
        )
    return np.asarray(selected[:32], dtype=np.int16)


def quantize_mxfp4_sq1(
    packed: np.ndarray,
    source_scale_raw: np.ndarray,
    *,
    matrix_scale_base: int | None = None,
    palette_ids: np.ndarray = SQ1_FIXED32_PALETTE_IDS,
    row_indices: np.ndarray | None = None,
) -> tuple[torch.Tensor, Sq1Encoding]:
    source_nibbles, _ = _unpack_source(packed, source_scale_raw)
    scales = np.asarray(source_scale_raw, dtype=np.uint8)
    if row_indices is not None:
        selected = np.asarray(row_indices, dtype=np.int64)
        source_nibbles = source_nibbles[selected]
        scales = scales[selected]
    base = int(scales.min()) if matrix_scale_base is None else int(matrix_scale_base)
    if int(scales.max()) > base + 3:
        raise ValueError("source E8M0 range exceeds the four-value SQ window")
    catalog = np.asarray(palette_ids, dtype=np.int16)
    solutions = [
        solve_sq1_row(
            source_nibbles[row],
            scales[row],
            matrix_scale_base=base,
            palette_ids=catalog,
        )
        for row in range(source_nibbles.shape[0])
    ]
    rows, blocks, block_size = source_nibbles.shape
    tags = np.stack([solution.block_tags for solution in solutions])
    state_scales = np.asarray(
        [solution.state_scale_offsets for solution in solutions], dtype=np.uint8
    )
    palette_to_code = {int(value): index for index, value in enumerate(catalog)}
    state_palettes = np.asarray(
        [
            [palette_to_code[value] for value in solution.state_palette_ids]
            for solution in solutions
        ],
        dtype=np.uint8,
    )
    symbols = np.empty((rows, blocks, block_size), dtype=np.uint8)
    output_nibbles = np.empty_like(symbols)
    output_scales = np.empty((rows, blocks), dtype=np.uint8)
    measured_sse = 0.0
    for row in range(rows):
        target = NIBBLE_VALUES[source_nibbles[row]] * np.exp2(
            scales[row, :, None].astype(np.int16) - 127
        )
        for tag in range(8):
            selected = tags[row] == tag
            if not np.any(selected):
                continue
            palette = int(solutions[row].state_palette_ids[tag])
            exponent = base + int(state_scales[row, tag])
            levels = SQ1_PALETTE_VALUES[palette] * math.ldexp(1.0, exponent - 127)
            values, errors = _quantize_for_tag(target[selected], levels, tag & 3)
            symbols[row, selected] = values
            output_nibbles[row, selected] = SQ1_PALETTE_NIBBLES[palette][values]
            output_scales[row, selected] = exponent
            measured_sse += float(errors.sum())
    expected_sse = float(sum(solution.error_sse for solution in solutions))
    if not math.isclose(measured_sse, expected_sse, rel_tol=2e-9, abs_tol=1e-10):
        raise RuntimeError("MXFP4-SQ1 search/materialization SSE mismatch")
    packed_mxfp4 = (
        output_nibbles[:, :, 0::2] | (output_nibbles[:, :, 1::2] << 4)
    ).reshape(rows, blocks * 16)
    reconstruction = decode_mxfp4(packed_mxfp4, output_scales, device="cpu")
    encoding = Sq1Encoding(
        matrix_scale_base=base,
        symbols=symbols.reshape(rows, blocks * block_size),
        block_tags=tags,
        state_scale_offsets=state_scales,
        state_palette_codes=state_palettes,
        packed_symbols=_pack_fixed_width(symbols, 1).reshape(rows, -1),
        packed_block_selectors=_pack_block_selectors(tags >> 2),
        packed_state_scales=_pack_fixed_width(state_scales, 2),
        packed_state_palettes=_pack_fixed_width(state_palettes, 5),
        packed_mxfp4=packed_mxfp4,
        native_scale_raw=output_scales,
        searched_sse=expected_sse,
    )
    return reconstruction.contiguous(), encoding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--design-rows", type=int, default=32)
    parser.add_argument("--holdout-rows", type=int, default=32)
    args = parser.parse_args()
    checkpoint = V4FCheckpoint(args.model)
    packed, scales = _raw_gate_up_native(checkpoint, args.layer, args.expert)
    nibbles, _ = _unpack_source(packed, scales)
    base = int(scales.min())
    design = np.linspace(0, len(nibbles) // 2 - 1, args.design_rows, dtype=np.int64)
    holdout = np.linspace(len(nibbles) // 2, len(nibbles) - 1, args.holdout_rows, dtype=np.int64)
    catalog = screen_sq1_catalog(
        nibbles, scales, row_indices=design, matrix_scale_base=base
    )
    reconstruction, encoding = quantize_mxfp4_sq1(
        packed,
        scales,
        matrix_scale_base=base,
        palette_ids=catalog,
        row_indices=holdout,
    )
    source_packed = np.asarray(packed)[holdout]
    source_scales = np.asarray(scales)[holdout]
    source = decode_mxfp4(source_packed, source_scales, device="cpu")
    error = float((source.double() - reconstruction.double()).square().sum())
    energy = float(source.double().square().sum())
    print(f"palette_ids={catalog.tolist()}")
    print(f"palette_nibbles={SQ1_PALETTE_NIBBLES[catalog].reshape(-1).tolist()}")
    print(f"rows={len(holdout)} sse={error:.9f} snr_db={10.0 * math.log10(energy / error):.6f}")
    print(f"search_sse={encoding.searched_sse:.9f}")


if __name__ == "__main__":
    main()
