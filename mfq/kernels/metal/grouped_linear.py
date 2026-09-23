"""Single-dispatch packed linear projections for Apple silicon.

The decode path concatenates packed NINT8-0 and VQ-family streams once.
NINT pair/triple groups use a separate zero-copy metadata-driven operator;
they never enter the older heterogeneous decoder below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.kernels.metal.moe import (
    _DESCRIPTOR_SIZE,
    _FAMILY,
    _FAMILY_NINT8_ZERO,
    _FAMILY_VQ,
    _K,
    _LOCAL_EXPERT,
    _OUT,
    _Q8_NG,
    _Q8_Q_OFFSET,
    _Q8_SCALE_OFFSET,
    _VQ_ANCHOR_OFFSET,
    _VQ_AUX_MODE,
    _VQ_AUX_OFFSET,
    _VQ_BANK_OFFSET,
    _VQ_CODE_BANK_MODE,
    _VQ_CODE_BANKS,
    _VQ_CODEBOOK_OFFSET,
    _VQ_ENTRIES,
    _VQ_GROUPS_PER_SUPER,
    _VQ_GS,
    _VQ_HAS_TABLE_BANKS,
    _VQ_INDEX_BITS,
    _VQ_INDICES_OFFSET,
    _VQ_NG,
    _VQ_NSUPER,
    _VQ_NVEC,
    _VQ_PARAMETER_OFFSET,
    _VQ_ROTATION_VARIANT,
    _VQ_SCALE_OFFSET,
    _VQ_STATE_BANK_OFFSET,
    _VQ_STATE_BITS,
    _VQ_STATE_OFFSET,
    _VQ_STATES,
    _VQ_VECTOR_SIZE,
    _join,
    _size,
)
from mfq.kernels.metal.nint import MetalNintWeight, _NINT_HEADER_SOURCE
from mfq.kernels.metal.nint8_zero import MetalNint8ZeroWeight
from mfq.kernels.metal.vq import _BITSTREAM_HEADER, MetalVqWeight, signed_hadamard

PackedLinearWeight: TypeAlias = MetalNintWeight | MetalNint8ZeroWeight | MetalVqWeight
HeterogeneousLinearWeight: TypeAlias = MetalNint8ZeroWeight | MetalVqWeight


def _nint_group_input_names(projections: int) -> list[str]:
    names: list[str] = []
    for projection in range(projections):
        suffix = str(projection)
        names.extend(
            (
                f"q_packed_{suffix}",
                f"row_q_layout_{suffix}",
                f"row_q_byte_offsets_{suffix}",
                f"sub_scale_{suffix}",
                f"sub_min_{suffix}",
                f"neuron_scale_{suffix}",
                f"neuron_min_{suffix}",
            )
        )
    names.extend(("x", "params"))
    return names


def _nint_group_source(projections: int) -> str:
    parts = [
        r"""
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint OUTPUTS_PER_SIMD = 2u;
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;
    constexpr uint CHUNKS = (uint(GS) + 3u) / 4u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    if (output_base >= uint(MAX_OUT)) {
        return;
    }
"""
    ]
    for projection in range(projections):
        suffix = str(projection)
        parts.append(
            f"""
    bool active_{suffix} = output_base < uint(P{suffix}_OUT);
    uint metadata_bases_{suffix}[OUTPUTS_PER_SIMD];
    uint q_widths_{suffix}[OUTPUTS_PER_SIMD];
    uint q_byte_offsets_{suffix}[OUTPUTS_PER_SIMD];
    uint q_bit_shifts_{suffix}[OUTPUTS_PER_SIMD];
    float neuron_scales_{suffix}[OUTPUTS_PER_SIMD];
    float neuron_minimums_{suffix}[OUTPUTS_PER_SIMD];
    float accumulators_{suffix}[OUTPUTS_PER_SIMD][M];
    for (uint output_row = 0u;
         output_row < OUTPUTS_PER_SIMD;
         ++output_row) {{
        uint output = min(
            output_base + output_row,
            uint(P{suffix}_OUT) - 1u);
        metadata_bases_{suffix}[output_row] = output * uint(NG);
        uint row_layout = uint(row_q_layout_{suffix}[output]);
        q_widths_{suffix}[output_row] = row_layout & 15u;
        q_byte_offsets_{suffix}[output_row] =
            row_q_byte_offsets_{suffix}[output];
        q_bit_shifts_{suffix}[output_row] = row_layout >> 4u;
        neuron_scales_{suffix}[output_row] = neuron_scale_{suffix}[output];
        neuron_minimums_{suffix}[output_row] = neuron_min_{suffix}[output];
        for (uint row = 0u; row < uint(M); ++row) {{
            accumulators_{suffix}[output_row][row] = 0.0f;
        }}
    }}
"""
        )
    parts.append(
        r"""
    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sums[M];
        for (uint row = 0u; row < uint(M); ++row) {
            activation_sums[row] = 0.0f;
        }
"""
    )
    for projection in range(projections):
        suffix = str(projection)
        parts.append(
            f"""
        float quantized_dots_{suffix}[OUTPUTS_PER_SIMD][M];
        for (uint output_row = 0u;
             output_row < OUTPUTS_PER_SIMD;
             ++output_row) {{
            for (uint row = 0u; row < uint(M); ++row) {{
                quantized_dots_{suffix}[output_row][row] = 0.0f;
            }}
        }}
"""
        )
    parts.append(
        r"""
        for (uint chunk = 0u; chunk < CHUNKS; ++chunk) {
            uint group_element = chunk * 4u;
            uint column = group * uint(GS) + group_element;
"""
    )
    for projection in range(projections):
        suffix = str(projection)
        parts.append(
            f"""
            uint4 codes_{suffix}[OUTPUTS_PER_SIMD];
            if (active_{suffix}) {{
                for (uint output_row = 0u;
                     output_row < OUTPUTS_PER_SIMD;
                     ++output_row) {{
                    codes_{suffix}[output_row] = mfq_nint_read_row_value4(
                        q_packed_{suffix},
                        q_byte_offsets_{suffix}[output_row],
                        q_bit_shifts_{suffix}[output_row],
                        column,
                        q_widths_{suffix}[output_row]);
                }}
            }}
"""
        )
    parts.append(
        r"""
            for (uint row = 0u; row < uint(M); ++row) {
                uint input_base = row * uint(K) + column;
                float4 activation = float4(0.0f);
                if (group_element + 3u < uint(GS) &&
                    column + 3u < uint(K)) {
                    const vec<T, 4> packed_activation =
                        *reinterpret_cast<device const vec<T, 4>*>(
                            x + input_base);
                    activation = float4(packed_activation);
                } else {
                    activation.x = column < uint(K)
                        ? float(x[input_base]) : 0.0f;
                    activation.y = group_element + 1u < uint(GS) &&
                            column + 1u < uint(K)
                        ? float(x[input_base + 1u]) : 0.0f;
                    activation.z = group_element + 2u < uint(GS) &&
                            column + 2u < uint(K)
                        ? float(x[input_base + 2u]) : 0.0f;
                    activation.w = group_element + 3u < uint(GS) &&
                            column + 3u < uint(K)
                        ? float(x[input_base + 3u]) : 0.0f;
                }
                activation_sums[row] +=
                    activation.x + activation.y
                    + activation.z + activation.w;
"""
    )
    for projection in range(projections):
        suffix = str(projection)
        parts.append(
            f"""
                if (active_{suffix}) {{
                    for (uint output_row = 0u;
                         output_row < OUTPUTS_PER_SIMD;
                         ++output_row) {{
                        quantized_dots_{suffix}[output_row][row] += dot(
                            activation,
                            float4(codes_{suffix}[output_row]));
                    }}
                }}
"""
        )
    parts.append(
        r"""
            }
        }
"""
    )
    for projection in range(projections):
        suffix = str(projection)
        parts.append(
            f"""
        if (active_{suffix}) {{
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {{
                uint metadata_index =
                    metadata_bases_{suffix}[output_row] + group;
                float scale = neuron_scales_{suffix}[output_row]
                    * float(sub_scale_{suffix}[metadata_index]);
                float minimum = neuron_minimums_{suffix}[output_row]
                    * float(sub_min_{suffix}[metadata_index]);
                for (uint row = 0u; row < uint(M); ++row) {{
                    accumulators_{suffix}[output_row][row] = fma(
                        scale,
                        quantized_dots_{suffix}[output_row][row],
                        fma(
                            -minimum,
                            activation_sums[row],
                            accumulators_{suffix}[output_row][row]));
                }}
            }}
        }}
"""
        )
    parts.append("    }\n")
    if projections == 2:
        parts.append(
            r"""
    if (uint(SWIGLU) != 0u) {
        if (active_0 && active_1) {
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                uint output = output_base + output_row;
                for (uint row = 0u; row < uint(M); ++row) {
                    float gate = float(T(simd_sum(
                        accumulators_0[output_row][row])));
                    float up = float(T(simd_sum(
                        accumulators_1[output_row][row])));
                    if (lane == 0u && output < uint(P0_OUT)) {
                        if (params[0] > 0.0f) {
                            gate = min(gate, params[0]);
                            up = clamp(up, -params[0], params[0]);
                        }
                        y[row * uint(P0_OUT) + output] =
                            T((gate / (1.0f + exp(-gate))) * up);
                    }
                }
            }
        }
        return;
    }
"""
        )
    for projection in range(projections):
        suffix = str(projection)
        parts.append(
            f"""
    if (active_{suffix}) {{
        for (uint output_row = 0u;
             output_row < OUTPUTS_PER_SIMD;
             ++output_row) {{
            uint output = output_base + output_row;
            for (uint row = 0u; row < uint(M); ++row) {{
                float total = simd_sum(
                    accumulators_{suffix}[output_row][row]);
                if (lane == 0u && output < uint(P{suffix}_OUT)) {{
                    y[row * uint(TOTAL_OUT)
                        + uint(P{suffix}_OFFSET) + output] = T(total);
                }}
            }}
        }}
    }}
"""
        )
    return "".join(parts)


_NINT_GROUP_KERNELS: dict[int, object] = {}


def _nint_group_kernel(projections: int):
    kernel = _NINT_GROUP_KERNELS.get(projections)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mfq_nint_metadata_grouped_p{projections}",
            input_names=_nint_group_input_names(projections),
            output_names=["y"],
            header=_NINT_HEADER_SOURCE,
            source=_nint_group_source(projections),
            compile_options={"math_mode": "fast"},
        )
        _NINT_GROUP_KERNELS[projections] = kernel
    return kernel


_GROUPED_LINEAR_HEADER = (
    _BITSTREAM_HEADER
    + r"""
template <
    typename DescriptorPtr,
    typename Q8Ptr,
    typename Q8ScalePtr,
    typename VqIndicesPtr,
    typename VqStatePtr,
    typename VqAuxPtr,
    typename VqAnchorsPtr,
    typename VqCodebooksPtr,
    typename VqScalesPtr,
    typename VqStateBankPtr,
    typename VqBanksPtr,
    typename VqParametersPtr
>
inline float mfq_grouped_linear_decode_weight(
    DescriptorPtr descriptors,
    Q8Ptr q8_q,
    Q8ScalePtr q8_scales,
    VqIndicesPtr vq_indices,
    VqStatePtr vq_state,
    VqAuxPtr vq_aux,
    VqAnchorsPtr vq_anchors,
    VqCodebooksPtr vq_codebooks,
    VqScalesPtr vq_scales,
    VqStateBankPtr vq_state_to_codebank,
    VqBanksPtr vq_banks,
    VqParametersPtr vq_parameters,
    uint descriptor_base,
    ulong output,
    uint column,
    uint K
) {
    uint family = uint(descriptors[descriptor_base]);
    if (family == 2u) {
        uint groups = uint(descriptors[descriptor_base + 4u]);
        uint group = column >> 5;
        return float(q8_scales[
            uint(descriptors[descriptor_base + 6u])
                + output * groups + group
        ]) * float(q8_q[
            uint(descriptors[descriptor_base + 5u])
                + output * K + column
        ]);
    }

    uint groupsize = uint(descriptors[descriptor_base + 4u]);
    uint groups = uint(descriptors[descriptor_base + 5u]);
    uint vector_size = uint(descriptors[descriptor_base + 6u]);
    return mfq_vq_decode_weight(
        vq_indices + uint(descriptors[descriptor_base + 18u]),
        vq_state + uint(descriptors[descriptor_base + 19u]),
        vq_aux + uint(descriptors[descriptor_base + 20u]),
        vq_anchors + uint(descriptors[descriptor_base + 21u]),
        vq_codebooks + uint(descriptors[descriptor_base + 22u]),
        vq_scales + uint(descriptors[descriptor_base + 23u]),
        vq_state_to_codebank + uint(descriptors[descriptor_base + 24u]),
        vq_banks + uint(descriptors[descriptor_base + 25u]),
        vq_parameters + uint(descriptors[descriptor_base + 26u]),
        uint(output),
        column,
        groupsize,
        groups,
        vector_size,
        uint(descriptors[descriptor_base + 7u]),
        uint(descriptors[descriptor_base + 8u]),
        uint(descriptors[descriptor_base + 9u]),
        uint(descriptors[descriptor_base + 10u]),
        uint(descriptors[descriptor_base + 11u]),
        uint(descriptors[descriptor_base + 12u]),
        uint(descriptors[descriptor_base + 13u]),
        uint(descriptors[descriptor_base + 14u]),
        uint(descriptors[descriptor_base + 15u]),
        uint(descriptors[descriptor_base + 16u]),
        uint(descriptors[descriptor_base + 17u]),
        (K + 7u) / 8u
    );
}
"""
)


_GROUPED_LINEAR_SOURCE = r"""
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint input_row = workgroup / uint(TOTAL_TILES);
    uint global_tile = workgroup - input_row * uint(TOTAL_TILES);
    if (input_row >= uint(ROWS)) {
        return;
    }

    uint projection = 0u;
    while (
        projection + 1u < uint(PROJECTIONS)
        && global_tile >= uint(projection_tile_offsets[projection + 1u])
    ) {
        ++projection;
    }
    uint local_tile =
        global_tile - uint(projection_tile_offsets[projection]);
    uint descriptor_base = projection * uint(DESCRIPTOR_SIZE);
    uint output_width = uint(descriptors[descriptor_base + 2u]);
    uint output_base =
        local_tile * ROWS_PER_TG + simd_group * ROWS_PER_SIMD;
    uint output_offset = uint(projection_output_offsets[projection]);
    uint family = uint(descriptors[descriptor_base]);
    uint rotation_variant = uint(descriptors[descriptor_base + 27u]);
    uint input_base = (
        rotation_variant * uint(ROWS) + input_row
    ) * uint(K);

    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation = float(x[input_base + column]);
        for (uint local_row = 0u; local_row < ROWS_PER_SIMD; ++local_row) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }
            float weight = mfq_grouped_linear_decode_weight(
                descriptors,
                q8_q,
                q8_scales,
                vq_indices,
                vq_state,
                vq_aux,
                vq_anchors,
                vq_codebooks,
                vq_scales,
                vq_state_to_codebank,
                vq_banks,
                vq_parameters,
                descriptor_base,
                output,
                column,
                uint(K)
            );
            accumulators[local_row] = fma(
                activation,
                weight,
                accumulators[local_row]
            );
        }
    }

    for (uint local_row = 0u; local_row < ROWS_PER_SIMD; ++local_row) {
        float total = simd_sum(accumulators[local_row]);
        uint output = output_base + local_row;
        if (lane == 0u && output < output_width) {
            y[
                input_row * uint(TOTAL_OUT)
                + output_offset + output
            ] = T(total);
        }
    }
"""


_GROUPED_LINEAR_KERNEL = mx.fast.metal_kernel(
    name="mfq_heterogeneous_grouped_linear",
    input_names=[
        "descriptors",
        "projection_tile_offsets",
        "projection_output_offsets",
        "q8_q",
        "q8_scales",
        "vq_indices",
        "vq_state",
        "vq_aux",
        "vq_anchors",
        "vq_codebooks",
        "vq_scales",
        "vq_state_to_codebank",
        "vq_banks",
        "vq_parameters",
        "x",
    ],
    output_names=["y"],
    header=_GROUPED_LINEAR_HEADER,
    source=_GROUPED_LINEAR_SOURCE,
    compile_options={"math_mode": "fast"},
)


@dataclass(frozen=True)
class MetalNintLinearGroupWeight:
    """Zero-copy pair/triple of canonical metadata-driven NINT weights."""

    weights: tuple[MetalNintWeight, ...]
    output_widths: tuple[int, ...]
    neuron_len: int
    groupsize: int
    groups: int

    @classmethod
    def from_weights(
        cls,
        weights: tuple[MetalNintWeight, ...],
    ) -> MetalNintLinearGroupWeight:
        if len(weights) not in (2, 3):
            raise ValueError("grouped NINT requires two or three projections")
        first = weights[0]
        if any(
            weight.neuron_len != first.neuron_len
            or weight.groupsize != first.groupsize
            or weight.groups != first.groups
            for weight in weights[1:]
        ):
            raise ValueError("grouped NINT projections must share matrix geometry")
        return cls(
            weights=weights,
            output_widths=tuple(int(weight.out) for weight in weights),
            neuron_len=int(first.neuron_len),
            groupsize=int(first.groupsize),
            groups=int(first.groups),
        )

    @property
    def projections(self) -> int:
        return len(self.weights)

    @property
    def total_out(self) -> int:
        return sum(self.output_widths)

    @property
    def packed_nbytes(self) -> int:
        return sum(weight.packed_nbytes for weight in self.weights)


@dataclass(frozen=True)
class MetalLinearGroupWeight:
    """Concatenated packed buffers for ordinary heterogeneous projections."""

    descriptors: mx.array
    projection_tile_offsets: mx.array
    projection_output_offsets: mx.array
    q8_q: mx.array
    q8_scales: mx.array
    vq_indices: mx.array
    vq_state: mx.array
    vq_aux: mx.array
    vq_anchors: mx.array
    vq_codebooks: mx.array
    vq_scales: mx.array
    vq_state_to_codebank: mx.array
    vq_banks: mx.array
    vq_parameters: mx.array
    descriptor_values: np.ndarray
    rotation_specs: tuple[tuple[mx.array, int, int], ...]
    output_widths: tuple[int, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    neuron_len: int
    total_out: int
    total_tiles: int

    @classmethod
    def from_weights(
        cls,
        weights: tuple[HeterogeneousLinearWeight, ...],
    ) -> MetalLinearGroupWeight:
        if len(weights) < 2:
            raise ValueError("grouped linear requires at least two weights")
        neuron_len = int(weights[0].neuron_len)
        if neuron_len <= 0 or any(int(weight.neuron_len) != neuron_len for weight in weights):
            raise ValueError("grouped linear weights must share one input width")

        descriptors = np.zeros(
            (len(weights), _DESCRIPTOR_SIZE),
            dtype=np.int32,
        )
        streams: dict[str, list[mx.array]] = {
            "q8_q": [],
            "q8_scales": [],
            "vq_indices": [],
            "vq_state": [],
            "vq_aux": [],
            "vq_anchors": [],
            "vq_codebooks": [],
            "vq_scales": [],
            "vq_state_to_codebank": [],
            "vq_banks": [],
            "vq_parameters": [],
        }
        offsets = {name: 0 for name in streams}
        rotation_variants: dict[tuple[int, int], int] = {}
        rotation_specs: list[tuple[mx.array, int, int]] = []
        output_widths: list[int] = []
        output_shapes: list[tuple[int, ...]] = []

        for projection, weight in enumerate(weights):
            descriptor = descriptors[projection]
            descriptor[_LOCAL_EXPERT] = 0
            descriptor[_OUT] = int(weight.out)
            descriptor[_K] = neuron_len
            output_widths.append(int(weight.out))
            output_shapes.append(
                tuple(map(int, weight.output_shape))
                if isinstance(weight, MetalVqWeight)
                else (int(weight.out),)
            )

            if isinstance(weight, MetalNint8ZeroWeight):
                descriptor[_FAMILY] = _FAMILY_NINT8_ZERO
                descriptor[_Q8_NG] = weight.groups
                descriptor[_Q8_Q_OFFSET] = offsets["q8_q"]
                descriptor[_Q8_SCALE_OFFSET] = offsets["q8_scales"]
                streams["q8_q"].append(weight.q)
                streams["q8_scales"].append(weight.scales)
                offsets["q8_q"] += _size(weight.q)
                offsets["q8_scales"] += _size(weight.scales)
                continue

            if isinstance(weight, MetalVqWeight):
                descriptor[_FAMILY] = _FAMILY_VQ
                descriptor[_VQ_GS] = weight.groupsize
                descriptor[_VQ_NG] = weight.groups
                descriptor[_VQ_VECTOR_SIZE] = weight.vector_size
                descriptor[_VQ_NVEC] = weight.vectors
                descriptor[_VQ_INDEX_BITS] = weight.index_bits
                descriptor[_VQ_STATE_BITS] = weight.state_bits
                descriptor[_VQ_STATES] = weight.states
                descriptor[_VQ_ENTRIES] = weight.entries
                descriptor[_VQ_CODE_BANKS] = weight.code_banks
                descriptor[_VQ_AUX_MODE] = weight.aux_mode
                descriptor[_VQ_CODE_BANK_MODE] = weight.code_bank_mode
                descriptor[_VQ_HAS_TABLE_BANKS] = int(weight.table_banks > 1)
                descriptor[_VQ_GROUPS_PER_SUPER] = weight.groups_per_super
                descriptor[_VQ_NSUPER] = weight.supergroups
                descriptor[_VQ_INDICES_OFFSET] = offsets["vq_indices"]
                descriptor[_VQ_STATE_OFFSET] = offsets["vq_state"]
                descriptor[_VQ_AUX_OFFSET] = offsets["vq_aux"]
                descriptor[_VQ_ANCHOR_OFFSET] = offsets["vq_anchors"]
                descriptor[_VQ_CODEBOOK_OFFSET] = offsets["vq_codebooks"]
                descriptor[_VQ_SCALE_OFFSET] = offsets["vq_scales"]
                descriptor[_VQ_STATE_BANK_OFFSET] = offsets["vq_state_to_codebank"]
                descriptor[_VQ_BANK_OFFSET] = offsets["vq_banks"]
                descriptor[_VQ_PARAMETER_OFFSET] = offsets["vq_parameters"]
                if weight.rotation_block:
                    key = (weight.rotation_block, weight.rotation_seed)
                    variant = rotation_variants.get(key, 0)
                    if variant == 0:
                        variant = len(rotation_specs) + 1
                        rotation_variants[key] = variant
                        rotation_specs.append(
                            (
                                weight.rotation_signs,
                                weight.rotation_block,
                                weight.rotation_seed,
                            )
                        )
                    descriptor[_VQ_ROTATION_VARIANT] = variant
                streams["vq_indices"].append(weight.indices_packed)
                streams["vq_state"].append(weight.state_packed)
                streams["vq_aux"].append(weight.aux_packed)
                streams["vq_anchors"].append(weight.anchors)
                streams["vq_codebooks"].append(weight.codebooks)
                streams["vq_scales"].append(weight.scale_lut)
                streams["vq_state_to_codebank"].append(weight.state_to_codebank)
                streams["vq_banks"].append(weight.bank_ids)
                streams["vq_parameters"].append(weight.parameters)
                offsets["vq_indices"] += _size(weight.indices_packed) + 2
                offsets["vq_state"] += _size(weight.state_packed) + 2
                offsets["vq_aux"] += _size(weight.aux_packed) + 2
                offsets["vq_anchors"] += _size(weight.anchors)
                offsets["vq_codebooks"] += _size(weight.codebooks)
                offsets["vq_scales"] += _size(weight.scale_lut)
                offsets["vq_state_to_codebank"] += _size(weight.state_to_codebank)
                offsets["vq_banks"] += _size(weight.bank_ids)
                offsets["vq_parameters"] += _size(weight.parameters)
                continue

            raise TypeError(f"unsupported grouped linear weight {type(weight).__name__}")

        widths = tuple(output_widths)
        output_offsets = np.zeros((len(widths) + 1,), dtype=np.int32)
        output_offsets[1:] = np.cumsum(widths, dtype=np.int64)
        tile_counts = np.asarray(
            [(width + 7) // 8 for width in widths],
            dtype=np.int32,
        )
        tile_offsets = np.zeros((len(widths) + 1,), dtype=np.int32)
        tile_offsets[1:] = np.cumsum(tile_counts, dtype=np.int64)

        return cls(
            descriptors=mx.array(descriptors),
            projection_tile_offsets=mx.array(tile_offsets),
            projection_output_offsets=mx.array(output_offsets),
            q8_q=_join(streams["q8_q"], dtype=mx.int8),
            q8_scales=_join(streams["q8_scales"], dtype=mx.float16),
            vq_indices=_join(
                streams["vq_indices"],
                dtype=mx.uint8,
                padding=2,
            ),
            vq_state=_join(
                streams["vq_state"],
                dtype=mx.uint8,
                padding=2,
            ),
            vq_aux=_join(
                streams["vq_aux"],
                dtype=mx.uint8,
                padding=2,
            ),
            vq_anchors=_join(streams["vq_anchors"], dtype=mx.float32),
            vq_codebooks=_join(streams["vq_codebooks"], dtype=mx.int8),
            vq_scales=_join(streams["vq_scales"], dtype=mx.float32),
            vq_state_to_codebank=_join(
                streams["vq_state_to_codebank"],
                dtype=mx.uint8,
            ),
            vq_banks=_join(streams["vq_banks"], dtype=mx.uint8),
            vq_parameters=_join(
                streams["vq_parameters"],
                dtype=mx.float32,
            ),
            descriptor_values=descriptors,
            rotation_specs=tuple(rotation_specs),
            output_widths=widths,
            output_shapes=tuple(output_shapes),
            neuron_len=neuron_len,
            total_out=int(output_offsets[-1]),
            total_tiles=int(tile_offsets[-1]),
        )

    @property
    def projections(self) -> int:
        return len(self.output_widths)

    @property
    def packed_nbytes(self) -> int:
        arrays = (
            self.descriptors,
            self.projection_tile_offsets,
            self.projection_output_offsets,
            self.q8_q,
            self.q8_scales,
            self.vq_indices,
            self.vq_state,
            self.vq_aux,
            self.vq_anchors,
            self.vq_codebooks,
            self.vq_scales,
            self.vq_state_to_codebank,
            self.vq_banks,
            self.vq_parameters,
        )
        return sum(int(array.nbytes) for array in arrays)


GroupedLinearWeight: TypeAlias = MetalNintLinearGroupWeight | MetalLinearGroupWeight


def _nint_group_dispatch(
    weight: MetalNintLinearGroupWeight,
    x: mx.array | np.ndarray,
    *,
    swiglu: bool,
    limit: float,
) -> tuple[mx.array, tuple[int, ...], int]:
    source = x if isinstance(x, mx.array) else mx.array(x)
    if source.ndim < 1 or int(source.shape[-1]) != weight.neuron_len:
        raise ValueError("grouped NINT input must end in the shared weight width")
    if source.dtype not in (mx.float16, mx.float32):
        source = source.astype(mx.float16)
    prefix = tuple(int(value) for value in source.shape[:-1])
    rows = int(source.size) // weight.neuron_len
    if rows < 1 or rows > 6:
        raise ValueError("grouped NINT metadata operator supports one through six rows")
    if swiglu and (
        weight.projections != 2
        or weight.output_widths[0] != weight.output_widths[1]
    ):
        raise ValueError("grouped NINT SwiGLU requires two equal-width projections")
    source = mx.contiguous(source.reshape((rows, weight.neuron_len)))
    inputs: list[mx.array] = []
    for projection in weight.weights:
        inputs.extend(
            (
                projection.q_packed,
                projection.row_q_layout,
                projection.row_q_byte_offsets,
                projection.sub_scale,
                projection.sub_min,
                projection.neuron_scale,
                projection.neuron_min,
            )
        )
    inputs.extend((source, mx.array([limit], dtype=mx.float32)))
    templates: list[tuple[str, object]] = [
        ("T", source.dtype),
        ("GS", weight.groupsize),
        ("NG", weight.groups),
        ("K", weight.neuron_len),
        ("M", rows),
        ("MAX_OUT", max(weight.output_widths)),
        ("TOTAL_OUT", weight.total_out),
    ]
    if weight.projections == 2:
        templates.append(("SWIGLU", int(swiglu)))
    offset = 0
    for projection, width in enumerate(weight.output_widths):
        templates.extend(
            (
                (f"P{projection}_OUT", width),
                (f"P{projection}_OFFSET", offset),
            )
        )
        offset += width
    output_width = weight.output_widths[0] if swiglu else weight.total_out
    threadgroups = (max(weight.output_widths) + 15) // 16
    result = _nint_group_kernel(weight.projections)(
        inputs=inputs,
        template=templates,
        grid=(threadgroups * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, output_width)],
        output_dtypes=[source.dtype],
    )[0]
    return result, prefix, rows


def grouped_linear_matmul(
    weight: GroupedLinearWeight,
    x: mx.array | np.ndarray,
) -> tuple[mx.array, ...]:
    """Apply all packed projections to a shared input in one Metal dispatch."""

    if isinstance(weight, MetalNintLinearGroupWeight):
        result, prefix, _ = _nint_group_dispatch(
            weight,
            x,
            swiglu=False,
            limit=0.0,
        )
        outputs: list[mx.array] = []
        offset = 0
        for width in weight.output_widths:
            outputs.append(
                result[:, offset : offset + width].reshape((*prefix, width))
            )
            offset += width
        return tuple(outputs)

    source = x if isinstance(x, mx.array) else mx.array(x)
    if source.ndim < 1 or int(source.shape[-1]) != weight.neuron_len:
        raise ValueError("grouped linear input must end in the shared weight width")
    if source.dtype not in (mx.float16, mx.float32):
        source = source.astype(mx.float16)
    source = mx.contiguous(source)
    prefix = tuple(int(value) for value in source.shape[:-1])
    rows = int(source.size) // weight.neuron_len
    if rows == 0:
        return tuple(
            mx.zeros((*prefix, *shape), dtype=source.dtype) for shape in weight.output_shapes
        )
    flattened = source.reshape((rows, weight.neuron_len))
    variants = [flattened]
    variants.extend(
        signed_hadamard(flattened, signs, block) for signs, block, _ in weight.rotation_specs
    )
    execution_input = mx.contiguous(mx.concatenate(variants, axis=0))
    result = _GROUPED_LINEAR_KERNEL(
        inputs=[
            weight.descriptors,
            weight.projection_tile_offsets,
            weight.projection_output_offsets,
            weight.q8_q,
            weight.q8_scales,
            weight.vq_indices,
            weight.vq_state,
            weight.vq_aux,
            weight.vq_anchors,
            weight.vq_codebooks,
            weight.vq_scales,
            weight.vq_state_to_codebank,
            weight.vq_banks,
            weight.vq_parameters,
            execution_input,
        ],
        template=[
            ("T", source.dtype),
            ("ROWS", rows),
            ("PROJECTIONS", weight.projections),
            ("K", weight.neuron_len),
            ("TOTAL_OUT", weight.total_out),
            ("TOTAL_TILES", weight.total_tiles),
            ("DESCRIPTOR_SIZE", _DESCRIPTOR_SIZE),
        ],
        grid=(rows * weight.total_tiles * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, weight.total_out)],
        output_dtypes=[source.dtype],
    )[0]
    outputs: list[mx.array] = []
    offset = 0
    for width, shape in zip(
        weight.output_widths,
        weight.output_shapes,
        strict=True,
    ):
        outputs.append(result[:, offset : offset + width].reshape((*prefix, *shape)))
        offset += width
    return tuple(outputs)


def grouped_linear_swiglu(
    weight: MetalNintLinearGroupWeight,
    x: mx.array | np.ndarray,
    *,
    limit: float = 0.0,
) -> mx.array:
    """Run a NINT gate/up pair and SwiGLU in one metadata-driven dispatch."""

    if not np.isfinite(limit) or float(limit) < 0.0:
        raise ValueError("grouped NINT SwiGLU limit must be finite and non-negative")
    result, prefix, _ = _nint_group_dispatch(
        weight,
        x,
        swiglu=True,
        limit=float(limit),
    )
    return result.reshape((*prefix, weight.output_widths[0]))


__all__ = [
    "GroupedLinearWeight",
    "HeterogeneousLinearWeight",
    "MetalLinearGroupWeight",
    "MetalNintLinearGroupWeight",
    "PackedLinearWeight",
    "grouped_linear_matmul",
    "grouped_linear_swiglu",
]
