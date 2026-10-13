from __future__ import annotations

import numpy as np

from bench.dsv4f_mxfp4.sq1 import quantize_mxfp4_sq1
from mfq.formats.mxfp4_sq import build_mxfp4_sq, pack_mxfp4_sq


def test_sq1_solver_builds_a_mixed_sq1_sq4_tensor() -> None:
    rng = np.random.default_rng(20260911)
    packed = rng.integers(0, 256, (2, 32), dtype=np.uint8)
    scales = np.asarray([[120, 121], [122, 123]], dtype=np.uint8)

    _reconstruction, encoding = quantize_mxfp4_sq1(
        packed,
        scales,
        matrix_scale_base=120,
        row_indices=np.asarray([0]),
    )
    tensor = build_mxfp4_sq(
        row_q_bits=(1, 4),
        input_size=64,
        matrix_scale_base=120,
        row_symbols=(
            encoding.packed_symbols[0].tobytes(),
            packed[1].tobytes(),
        ),
        sq_block_selectors=(encoding.block_tags >> 2).astype(np.uint8),
        sq_state_scales=encoding.state_scale_offsets,
        sq_state_palettes=encoding.state_palette_codes,
        sq4_native_scales=scales[1:2],
    )

    assert tensor.row_q_bits == (1, 4)
    assert tensor.format_version == 2
    assert tensor.bits == 0
    assert tensor.descriptor.distribution_entropy == 1.0
    assert pack_mxfp4_sq(tensor) == tensor.payload
