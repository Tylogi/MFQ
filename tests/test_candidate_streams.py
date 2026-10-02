from dataclasses import fields, replace
import io

import numpy as np
import pytest

from mfq.quantize.candidate_streams import select_parts, write_merged
from mfq.calibration.alphaq import ALPHAQ_PROFILES
from mfq.calibration.evaluator import NINT_EXPERT_PROFILES
from mfq.formats.io import _pack_tensor
from mfq.formats.nint import NintTensor
from mfq.formats.nvq import NvqJscTensor
from mfq.formats.nvq1_l import Nvq1LTensor, NVQ1_L_T8_S3
from mfq.formats.nvq1_s import Nvq1STensor, NVQ1_S, NVQ1_S_SYNTHETIC_BANKS
from mfq.quantize.nvq_jsc import NvqJscConfig, initial_jsc_tables
from mfq.tools.quantize_hf_to_mfq import _NVQ_SPECS


def tensor(profile, width):
    rng = np.random.default_rng(773)
    rows = 24
    anchor = rng.random(rows).astype(np.float16).astype(np.float32)
    base = dict(shape=(rows, width), axis=0, neuron_len=width, neuron_scale=anchor)
    if profile.startswith("NINT"):
        spec = NINT_EXPERT_PROFILES[profile]
        ng = (width+spec.groupsize-1)//spec.groupsize
        return NintTensor(**base, spec=spec, neuron_min=anchor/2,
            q=rng.integers(0, 1 << spec.bits, (rows, ng, spec.groupsize), dtype=np.uint8),
            sub_scale=rng.integers(0, 1 << spec.sub_bits, (rows, ng), dtype=np.uint8),
            sub_min=rng.integers(0, 1 << spec.sub_bits, (rows, ng), dtype=np.uint8))
    ng = (width+23)//24
    if profile in ("NVQ1-S", "NVQ1-L"):
        spec = NVQ1_S if profile == "NVQ1-S" else NVQ1_L_T8_S3
        cls = Nvq1STensor if profile == "NVQ1-S" else Nvq1LTensor
        return cls(**base, spec=spec,
            sub_scale=rng.integers(0, 1 << spec.sub_bits, (rows, ng), dtype=np.uint8),
            indices=rng.integers(0, 1 << spec.index_bits, (rows, width//8), dtype=np.uint16),
            delta_sign=rng.integers(0, 2, (rows, ng), dtype=np.uint8),
            codebook=NVQ1_S_SYNTHETIC_BANKS if profile == "NVQ1-S" else None)
    spec = _NVQ_SPECS[profile]
    tables = initial_jsc_tables(NvqJscConfig(spec=spec, banks=4 if profile.startswith("NVQ2") else 2))
    return NvqJscTensor(**base, base_spec=spec, scale_lut=tables.scale_lut,
        bank_for_state=tables.bank_for_state, codebooks=tables.codebooks,
        state=rng.integers(0, 16, (rows, ng), dtype=np.uint8),
        indices=rng.integers(0, spec.codebook_entries, (rows, width//spec.vector_size), dtype=np.uint16),
        signs=rng.integers(0, 128, (rows, width//8), dtype=np.uint8))


def sliced(value, rows):
    update = {f.name: getattr(value, f.name)[rows] for f in fields(value)
              if f.name in {"q", "neuron_scale", "neuron_min", "sub_scale", "sub_min",
                            "row_q_bits", "row_sub_bits", "indices", "delta_sign", "state", "signs"}}
    return replace(value, **update, shape=(len(rows), value.neuron_len))


@pytest.mark.parametrize("profile", ALPHAQ_PROFILES)
@pytest.mark.parametrize("width", (640, 2560))
def test_select_and_merge_candidate_chunks_preserve_canonical_bytes(profile, width):
    value = tensor(profile, width)
    dtype, blob = _pack_tensor(value)
    selected = np.r_[np.arange(16, 24), np.arange(8)]
    reference = _pack_tensor(sliced(value, selected))[1]
    assert b"".join(select_parts(blob, dtype, [2, 0], 8)) == reference
    first = _pack_tensor(sliced(value, np.arange(16)))[1]
    last = _pack_tensor(sliced(value, np.arange(16, 24)))[1]
    output = io.BytesIO()
    size = write_merged(output, [(last, dtype, [0]), (first, dtype, [0])], 8)
    assert output.getvalue() == reference
    assert size == len(reference)


def test_candidate_selection_rejects_missing_or_duplicate_experts():
    dtype, blob = _pack_tensor(tensor("NINT5", 640))
    for selected in ([0, 0], [3], [-1], []):
        with pytest.raises(ValueError):
            select_parts(blob, dtype, selected, 8)
