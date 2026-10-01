"""Exact regression tests for fixed-table reuse, bounded CUDA search and packing."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mfq.formats.nvq import (
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_512,
    NVQ3_D4_1024,
    _pack_bits,
    pack_jsc_group64,
    pack_nvq,
)
from mfq.quantize import nvq_jsc as jsc

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
SPECS = [NVQ2_E8, NVQ2_E8_1024, NVQ2_E8_4096, NVQ3_D4, NVQ3_D4_512, NVQ3_D4_1024]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("importance_kind", ["none", "columns", "matrix"])
def test_fixed_scales_reused_without_changing_weighted_assignment(
    monkeypatch, device, importance_kind
):
    weight = torch.randn((5, 50), generator=torch.Generator().manual_seed(20261001)) * 0.03
    weight[-1].zero_()
    importance = np.linspace(0, 2, 50, dtype=np.float32)
    importance[::7] = 0
    if importance_kind == "matrix":
        importance = np.broadcast_to(importance, weight.shape).copy()
        importance[1, ::3] = 7
        importance[2] = 0
    elif importance_kind == "none":
        importance = None
    tables = jsc.initial_jsc_tables(jsc.NvqJscConfig(banks=2, spec=NVQ3_D4_512))
    search = jsc._search
    calls = []

    def counted_search(*args, **kwargs):
        calls.append(1)
        return search(*args, **kwargs)

    monkeypatch.setattr(jsc, "_search", counted_search)
    options = dict(importance=importance, device=device, search_steps=3, group_chunk=6)
    actual = jsc.quantize_nvq_jsc_fixed(weight, tables, **options)
    assert len(calls) == 2
    assign = jsc._assign_groups

    def uncached(*args, **kwargs):
        kwargs.pop("raw_scales", None)
        return assign(*args, **kwargs)

    monkeypatch.setattr(jsc, "_assign_groups", uncached)
    calls.clear()
    expected = jsc.quantize_nvq_jsc_fixed(weight, tables, **options)
    assert len(calls) == 8
    assert pack_nvq(actual) == pack_nvq(expected)


@CUDA
@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.codebook)
@pytest.mark.parametrize("width", [640, 2560])
@pytest.mark.parametrize("importance_kind", ["none", "columns", "matrix"])
def test_row_aligned_batches_and_cuda_packing_are_exact(spec, width, importance_kind):
    # Real model widths; enough rows to cross multiple batch boundaries. The
    # reference submits individual complete row batches, never >4096 groups.
    ng = math.ceil(width / 24)
    rows_per_chunk = 4096 // ng
    rows = 2 * rows_per_chunk + 1
    generator = torch.Generator().manual_seed(20261001)
    weight = torch.randn((rows, width), generator=generator) * 0.03
    weight[-1].zero_()
    importance = torch.linspace(0, 2, width)
    importance[::11] = 0
    if importance_kind == "matrix":
        importance = importance.expand_as(weight).clone()
        importance[1, ::3] = 7
        importance[2] = 0
    elif importance_kind == "none":
        importance = None
    tables = jsc.initial_jsc_tables(
        jsc.NvqJscConfig(banks=4 if spec.vector_size == 8 else 2, spec=spec)
    )
    options = dict(device="cuda", group_chunk=4096, search_steps=5)
    actual = jsc.quantize_nvq_jsc_fixed(weight, tables, importance=importance, **options)
    parts = []
    for start in range(0, rows, rows_per_chunk):
        stop = start + rows_per_chunk
        part_importance = importance
        if importance_kind == "matrix":
            part_importance = importance[start:stop]
        parts.append(
            jsc.quantize_nvq_jsc_fixed(
                weight[start:stop], tables, importance=part_importance, **options
            )
        )
    for field in ("neuron_scale", "state", "indices", "signs"):
        np.testing.assert_array_equal(
            getattr(actual, field), np.concatenate([getattr(part, field) for part in parts])
        )
    # Different submission boundaries must preserve the same discrete result.
    smaller = jsc.quantize_nvq_jsc_fixed(
        weight, tables, importance=importance, **{**options, "group_chunk": 2048}
    )
    assert pack_nvq(smaller) == pack_nvq(actual)
    assert pack_nvq(actual, device="cuda") == pack_nvq(actual)
    if spec == NVQ2_E8_4096:
        actual.storage_layout = "streams"
        assert pack_nvq(actual, device="cuda") == pack_nvq(actual)


@CUDA
@pytest.mark.parametrize("bits", range(1, 17))
@pytest.mark.parametrize("count", [0, 1, 7, 9, 1025])
def test_cuda_bitpacking_matches_cpu(bits, count):
    rng = np.random.default_rng(20261001)
    values = rng.integers(0, 1 << bits, count, dtype=np.uint16)
    assert _pack_bits(values, bits, device="cuda") == _pack_bits(values, bits)


@CUDA
def test_cuda_search_rejects_a_row_larger_than_the_safe_submission():
    from mfq.quantize.cuda._ext import ext

    value = torch.zeros((1, 24), device="cuda")
    book = torch.ones((256, 4), dtype=torch.int8, device="cuda")
    with pytest.raises(RuntimeError, match="1..4096 groups per row"):
        ext().nvq_search(value, value, book, 4097, 24, 4, 3, 1.0)


@CUDA
def test_cuda_bank_validation_still_rejects_invalid_mapping():
    from mfq.quantize.cuda._ext import ext

    value = torch.ones((4, 24), device="cuda")
    anchor = torch.ones(4, device="cuda")
    alpha = torch.ones(16, device="cuda")
    bank = torch.zeros(16, dtype=torch.uint8, device="cuda")
    book = torch.ones((4, 8, 256), dtype=torch.int8, device="cuda")
    with pytest.raises(RuntimeError, match="every bank must own exactly four states"):
        ext().nvq2j_assign(value, value, anchor, alpha, bank, book, 24, 2, 1)


@CUDA
@pytest.mark.parametrize("width", [1, 23, 24, 25, 50])
def test_group64_partial_vectors_and_noncontiguous_inputs(width):
    rng = np.random.default_rng(20261001)
    ng, nv = math.ceil(width / 24), math.ceil(width / 8)
    state = rng.integers(0, 16, (3, ng * 2), dtype=np.uint8)[:, ::2]
    indices = rng.integers(0, 4096, (3, nv * 2), dtype=np.uint16)[:, ::2]
    signs = rng.integers(0, 128, (3, nv * 2), dtype=np.uint8)[:, ::2]
    args = (state, indices, signs)
    assert pack_jsc_group64(*args, neuron_len=width, device="cuda") == pack_jsc_group64(
        *args, neuron_len=width
    )


@CUDA
@pytest.mark.parametrize("spec", [NVQ2_E8_1024, NVQ2_E8_4096])
def test_fused_training_search_batches_match_single_submissions(spec):
    from mfq.quantize.cuda._ext import ext

    generator = torch.Generator().manual_seed(20261001)
    value = torch.rand((15, 24), generator=generator).cuda()
    weight = torch.rand((15, 24), generator=generator).cuda()
    weight[:, ::5] = 0
    tables = jsc.initial_jsc_tables(jsc.NvqJscConfig(banks=4, spec=spec))
    books = torch.as_tensor(tables.codebooks, device="cuda", dtype=torch.int8)
    qmax = books.amax((1, 2)).float()
    actual = ext().nvq2j_search_banks(value, weight, books, qmax, 3, 2, 3, 6)
    parts = [
        ext().nvq2j_search_banks(value[i : i + 6], weight[i : i + 6], books, qmax, 3, 2, 3, 4096)
        for i in range(0, 15, 6)
    ]
    for field, expected in enumerate(actual):
        assert torch.equal(expected, torch.cat([part[field] for part in parts]))


@CUDA
@pytest.mark.parametrize("profile,spec", [("NVQ2J-XL", NVQ2_E8_4096), ("NVQ3J-512", NVQ3_D4_512)])
def test_cuda_export_keeps_imatrix_and_exact_serialized_bytes(tmp_path, monkeypatch, profile, spec):
    from mfq.tools import quantize_gguf_to_mfq as exporter

    weight = torch.randn((16, 50), generator=torch.Generator().manual_seed(20261001)) * 0.03
    importance = np.geomspace(0.01, 100, 50, dtype=np.float32)
    tables = jsc.initial_jsc_tables(
        jsc.NvqJscConfig(banks=4 if spec.vector_size == 8 else 2, spec=spec)
    )
    options = dict(
        row_chunk=8,
        quant_backend="cuda",
        device="cuda",
        group_chunk=4096,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        importance_rows=lambda _start, _end: importance,
        jsc_tables=tables,
        search_steps=5,
        calibration_mode="none",
    )
    actual = tmp_path / "cuda-packed.blob"
    expected = tmp_path / "cpu-packed.blob"
    exporter._write_nvq_blob(weight, tuple(weight.shape), profile, actual, **options)
    cpu_bits = exporter._pack_nvq_bits
    cpu_group64 = exporter.pack_jsc_group64

    def bits(values, width, **_kwargs):
        return cpu_bits(values, width)

    def group64(*args, **kwargs):
        kwargs.pop("device", None)
        return cpu_group64(*args, **kwargs)

    monkeypatch.setattr(exporter, "_pack_nvq_bits", bits)
    monkeypatch.setattr(exporter, "pack_jsc_group64", group64)
    exporter._write_nvq_blob(weight, tuple(weight.shape), profile, expected, **options)
    assert actual.read_bytes() == expected.read_bytes()
