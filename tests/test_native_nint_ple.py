"""Matched native model-graph gate; requires the locally built Metal test."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.assets import model_config_asset
from mfq.formats.header import FileHeader
from mfq.formats.nint import NintSpec, NintTensor


def _models(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(20261001)
    hidden, streams, experts, intermediate = 128, 2, 2, 32
    config = dict(
        model_type="qwen4_exp_text", vocab_size=32, hidden_size=hidden,
        num_hidden_layers=1, max_position_embeddings=64,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        full_attention_interval=2, layer_types=["linear_attention"],
        hc_count=streams, hc_lowrank=4, partial_rotary_factor=0.5,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, num_experts=experts, num_experts_per_tok=1,
        moe_intermediate_size=intermediate, shared_expert_intermediate_size=intermediate,
        indexer_n_heads=1, indexer_head_dim=64, indexer_compress_ratio=2,
        indexer_budget=8, indexer_kv_heads=1, ple_conv_kernel_size=4,
        ple_embed_dim=hidden, ple_layer_ids=[1], ngram_size=3,
        ngram_vocab_size_base=8, heads_per_ngram=4, split_ngram_parts=2,
        hidden_act="silu", output_gate_type="silu", rms_norm_eps=1e-6,
        mtp_num_hidden_layers=0, eos_token_id=7,
    )
    tensors = {}

    def weight(name, shape, scale=0.03):
        tensors[name] = rng.normal(scale=scale, size=shape).astype(np.float16)

    def zero(name, shape):
        tensors[name] = np.zeros(shape, dtype=np.float16)

    def mixer(root, injection):
        zero(root + ".pre.norm.weight", (streams * hidden,))
        weight(root + ".pre.down.weight", (4, streams * hidden))
        weight(root + ".pre.up.weight", (streams * hidden, 4))
        if injection:
            weight(root + ".post.inject.weight", (streams, streams * hidden))

    weight("model.token_embedding.weight", (32, hidden))
    weight("model.output.weight", (32, hidden))
    mixer("model.mhc", False)
    block = "model.block.0"
    mixer(block + ".attention.mhc", True)
    mixer(block + ".mlp.mhc", True)
    gdn = block + ".linear_attention"
    for suffix, shape in {
        "qkv": (512, hidden), "gate": (256, hidden),
        "alpha": (2, hidden), "beta": (2, hidden), "output": (hidden, 256),
    }.items():
        weight(gdn + "." + suffix + ".weight", shape)
    zero(gdn + ".dt_bias", (2,))
    zero(gdn + ".a", (2,))
    tensors[gdn + ".norm.weight"] = np.ones(128, dtype=np.float16)
    weight(gdn + ".conv.weight", (512, 1, 4))
    mlp = block + ".mlp"
    weight(mlp + ".router.weight", (experts, hidden))
    weight(mlp + ".experts.gate_up.weight", (experts, 2 * intermediate, hidden))
    weight(mlp + ".experts.down.weight", (experts, hidden, intermediate))
    weight(mlp + ".shared_expert.router.weight", (1, hidden))
    for suffix, shape in {"gate": (intermediate, hidden), "up": (intermediate, hidden),
                          "down": (hidden, intermediate)}.items():
        weight(mlp + ".shared_expert." + suffix + ".weight", shape)
    ple = block + ".position_embedding"
    weight(ple + ".key.weight", (streams * hidden, hidden))
    weight(ple + ".value.weight", (hidden, hidden))
    for suffix in ("key_norm", "query_norm", "conv_norm"):
        zero(ple + "." + suffix + ".weight", (streams * hidden,))
    weight(ple + ".conv.weight", (streams * hidden, 1, 4))
    ngram = ple + ".ngram"
    # Keep uint64 hash semantics, signed modulo and EOS routing in the graph.
    tensors[ngram + ".layer_multipliers"] = np.array(
        [6364136223846793005, 1442695040888963407, 3202034522624059733], dtype=np.int64)
    tensors[ngram + ".head_offsets"] = np.arange(8, dtype=np.int64) * 8
    tensors[ngram + ".head_vocab_sizes"] = np.full(8, 8, dtype=np.int64)
    fp8, nint = dict(tensors), dict(tensors)
    fp8[ngram + ".weight_scale"] = np.array([2.0], dtype=np.float16)
    for shard in range(2):
        rows, width, gs = 32, 16, (5 if shard == 0 else 7)
        groups = (width + gs - 1) // gs
        qbits = np.arange(rows, dtype=np.uint8) % 8 + 1
        kbits = np.arange(rows, dtype=np.uint8) % 4 + 5
        codes = np.zeros((rows, groups, gs), dtype=np.uint8)
        for row in range(rows):
            codes[row].reshape(-1)[:] = (np.arange(groups * gs) + row + shard) % min(4, 1 << int(qbits[row]))
        table = NintTensor(
            spec=NintSpec(4, gs, 6), shape=(rows, width), axis=0, q=codes,
            neuron_scale=np.ones(rows, dtype=np.float32),
            neuron_min=np.zeros(rows, dtype=np.float32),
            sub_scale=np.ones((rows, groups), dtype=np.uint8),
            sub_min=np.zeros((rows, groups), dtype=np.uint8), neuron_len=width,
            row_q_bits=qbits, row_sub_bits=kbits,
        )
        name = ngram + f".shard.{shard}.weight"
        nint[name] = table
        # These small integers are exactly representable, including the
        # separate FP8 scale. Thus model logits must match bit-for-bit.
        fp8[name] = np.array([0x00, 0x30, 0x38, 0x3C], dtype=np.uint8)[
            codes.reshape(rows, -1)[:, :width]].view(io.Float8E4M3Array)
    asset = model_config_asset({"model_type": "qwen4_exp", "text_config": config})
    paths = (tmp_path / "fp8.mfq", tmp_path / "nintv2.mfq")
    for path, values in zip(paths, (fp8, nint), strict=True):
        values[asset.name] = asset.data
        io.save(path, FileHeader(model_arch="qwen4_exp", num_tensors=len(values)), values)
    return paths


def test_native_qwen_ple_mixed_qk_matches_scaled_fp8_graph(tmp_path):
    executable = Path(__file__).resolve().parents[1] / "build/csrc/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise the native model graph")
    fp8, nint = _models(tmp_path)
    result = subprocess.run([str(executable), "--qwen-ple", str(fp8), str(nint)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "batch logits passed" in result.stdout


def test_native_qwen_ple_rejects_mixed_fp8_nint_shards(tmp_path):
    executable = Path(__file__).resolve().parents[1] / "build/csrc/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise the native model graph")
    fp8, nint = _models(tmp_path)
    header, tensors = io.load(nint)
    _, original = io.load(fp8)
    name = "model.block.0.position_embedding.ngram.shard.1.weight"
    tensors[name] = original[name]
    mixed = tmp_path / "mixed.mfq"
    io.save(mixed, header, tensors)
    result = subprocess.run([str(executable), "--qwen-ple", str(fp8), str(mixed)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "cannot mix FP8 and NINT shards" in result.stderr


def _mhc_models(tmp_path, *, adaptive, projections):
    _, source = _models(tmp_path)
    header, tensors = io.load(source)
    reference, packed = dict(tensors), dict(tensors)
    names = [name for name in tensors if ".mhc." in name and
             name.endswith((".pre.down.weight", ".pre.up.weight", ".post.inject.weight"))]
    if projections == "down":
        # Match EWQ's first failure while retaining dense companion matrices.
        names = ["model.block.0.attention.mhc.pre.down.weight"]
    for name in names:
        rows, width = tensors[name].shape
        gs = 7
        groups = (width + gs - 1) // gs
        qbits = np.arange(rows, dtype=np.uint8) % 8 + 1 if adaptive else np.full(rows, 4, dtype=np.uint8)
        kbits = np.arange(rows, dtype=np.uint8) % 4 + 5 if adaptive else np.full(rows, 6, dtype=np.uint8)
        codes = np.empty((rows, groups, gs), dtype=np.uint8)
        for row in range(rows):
            codes[row].reshape(-1)[:] = (np.arange(groups * gs) * 3 + row) % min(8, 1 << int(qbits[row]))
        table = NintTensor(
            spec=NintSpec(4, gs, 6), shape=(rows, width), axis=0, q=codes,
            neuron_scale=np.full(rows, 1 / 8192, dtype=np.float32),
            neuron_min=np.full(rows, 1 / 4096, dtype=np.float32),
            sub_scale=np.broadcast_to(np.arange(groups, dtype=np.uint8) % 7 + 1, (rows, groups)).copy(),
            sub_min=np.ones((rows, groups), dtype=np.uint8),
            neuron_len=width,
            row_q_bits=qbits if adaptive else None,
            row_sub_bits=kbits if adaptive else None,
        )
        scales = table.neuron_scale[:, None, None] * table.sub_scale[:, :, None]
        minima = table.neuron_min[:, None, None] * table.sub_min[:, :, None]
        reference[name] = ((scales * codes - minima).reshape(rows, -1)[:, :width]).astype(np.float16)
        packed[name] = table
    paths = (tmp_path / "dense_mhc.mfq", tmp_path / "nint_mhc.mfq")
    for path, values in zip(paths, (reference, packed), strict=True):
        io.save(path, header, values)
    return paths


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("projections", ["down", "all"])
def test_native_qwen_nint_mhc_matches_dense_graph(tmp_path, adaptive, projections):
    executable = Path(__file__).resolve().parents[1] / "build/csrc/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise the native model graph")
    reference, packed = _mhc_models(tmp_path, adaptive=adaptive, projections=projections)
    result = subprocess.run([str(executable), "--qwen-ple", str(reference), str(packed)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "batch logits passed" in result.stdout


@pytest.mark.parametrize("name", [
    "model.block.0.position_embedding.ngram.head_offsets",
    "model.block.0.attention.mhc.pre.norm.weight",
])
def test_native_qwen_nint_mhc_does_not_relax_metadata_loading(tmp_path, name):
    executable = Path(__file__).resolve().parents[1] / "build/csrc/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise the native model graph")
    reference, packed = _mhc_models(tmp_path, adaptive=True, projections="down")
    header, tensors = io.load(packed)
    tensors[name] = tensors["model.block.0.attention.mhc.pre.down.weight"]
    invalid = tmp_path / "invalid_metadata.mfq"
    io.save(invalid, header, tensors)
    result = subprocess.run([str(executable), "--qwen-ple", str(reference), str(invalid)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert f"requires a dense tensor: {name} (got NINT)" in result.stderr


def _cuda_models(tmp_path):
    fp8, nint = _models(tmp_path)
    header, tensors = io.load(fp8)
    prefix = "model.block.0.position_embedding.ngram"
    scale = float(tensors.pop(prefix + ".weight_scale")[0])
    # Use dense F16 as the CUDA oracle: same exact values, no FP8 scale
    # binding dependence and no use of the NINT loader/kernel under test.
    for shard in range(2):
        name = prefix + f".shard.{shard}.weight"
        raw = np.asarray(tensors[name], dtype=np.uint8)
        assert set(np.unique(raw)).issubset({0x00, 0x30, 0x38, 0x3C})
        lut = np.zeros(256, dtype=np.float32)
        lut[[0x00, 0x30, 0x38, 0x3C]] = [0.0, 0.5, 1.0, 1.5]
        tensors[name] = (lut[raw] * scale).astype(np.float16)
    baseline = tmp_path / "dense.mfq"
    io.save(baseline, header, tensors)
    return baseline, nint


def test_cuda_dense_fixture_matches_canonical_nint_rows(tmp_path):
    baseline, nint = _cuda_models(tmp_path)
    _, dense = io.load(baseline)
    _, packed = io.load(nint)
    for shard in range(2):
        name = f"model.block.0.position_embedding.ngram.shard.{shard}.weight"
        table = packed[name]
        scales = table.neuron_scale[:, None, None] * table.sub_scale[:, :, None]
        minima = table.neuron_min[:, None, None] * table.sub_min[:, :, None]
        reconstructed = (scales * table.q - minima).reshape(32, -1)[:, :16]
        np.testing.assert_array_equal(dense[name], reconstructed)


def test_cuda_qwen_nint_ple_matches_dense_native_graph(tmp_path):
    executable = os.environ.get("MFQ_NINT_PLE_CUDA_DECODE")
    if not executable:
        pytest.skip("MFQ_NINT_PLE_CUDA_DECODE must point to the built CUDA mfq-decode")
    baseline, nint = _cuda_models(tmp_path)
    outputs = []
    for model in (baseline, nint):
        process = subprocess.run(
            [executable, "--model", str(model), "--ctx-size", "64", "--check-flash-next"],
            capture_output=True, text=True, timeout=180)
        assert process.returncode == 0, process.stdout + process.stderr
        payload = next(line.removeprefix("flash_next_check ")
                       for line in process.stdout.splitlines()
                       if line.startswith("flash_next_check "))
        outputs.append(json.loads(payload))
    assert outputs[0].keys() == outputs[1].keys()
    for name, reference in outputs[0].items():
        if name == "architecture":
            assert reference == outputs[1][name] == "qwen4_exp"
            continue
        assert reference["shape"] == outputs[1][name]["shape"]
        np.testing.assert_array_equal(reference["data"], outputs[1][name]["data"])
