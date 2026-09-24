#include "vq.h"
#include "format.h"
#include "nvq_codebooks.generated.h"

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

enum class NvqMatmulPath {
    Gemv,
    Mmq,
    OnlineF16,
    DequantGemm,
};

static NvqMatmulPath select_nvq_matmul_path(const NvqWeight & w, int M) {
    const bool e8_family =
        w.format == 2 || w.format == 5 || w.format == 7 ||
        w.format == 8 || w.format == 9 ||
        w.format == 13 || w.format == 14;

    if (w.format == 8 && M >= 14 && M <= 16 && w.neuron_len >= 2 * w.out) {
        return NvqMatmulPath::DequantGemm;
    }

    if (w.format == 9 && M >= 13) {
        const bool wide_output = w.out * 8 >= w.neuron_len * 21;
        const bool wide_mmq =
            w.out >= 4096 && (w.out >= 2 * w.neuron_len || w.neuron_len >= 2 * w.out);
        if (M <= 15) {
            if (wide_mmq) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M == 16) {
            if (w.out >= 4096) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M <= 31) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            if (w.out >= 4096 && w.out >= w.neuron_len) return NvqMatmulPath::Mmq;
            return NvqMatmulPath::DequantGemm;
        }
        if (M == 32) {
            return w.out >= 4096 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 47) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 48) {
            return wide_output ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 63) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 64) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            return w.out >= 8192 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        return NvqMatmulPath::DequantGemm;
    }

    if (M <= 13) return NvqMatmulPath::Gemv;
    if (M == 14) {
        return w.out >= 2048 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
    }
    if (M == 15) {
        if (e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        return NvqMatmulPath::DequantGemm;
    }
    if (M == 16) {
        if (e8_family && w.out >= 6144) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        if ((w.format == 3 || w.format == 10 ||
             w.format == 12 || w.format == 15) &&
            w.out >= 4096 && w.neuron_len >= 8192) {
            return NvqMatmulPath::Mmq;
        }
        return NvqMatmulPath::DequantGemm;
    }

    const bool wide_expansion = e8_family && w.out >= 3 * w.neuron_len;
    if (wide_expansion && ((M >= 17 && M <= 31) || (M >= 33 && M <= 47))) {
        return NvqMatmulPath::OnlineF16;
    }
    if (M == 32 && e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
    if (M == 48 && e8_family && w.out >= 12288) return NvqMatmulPath::Mmq;
    if (M == 64 && (w.format == 7 || w.format == 9) && w.out >= 8192) {
        return NvqMatmulPath::Mmq;
    }
    return NvqMatmulPath::DequantGemm;
}
static int cpu_nvq_index_bits(int format) {
    switch (format) {
        case 1: return 11;
        case 7: return 7;
        case 8: case 12: return 9;
        case 9: return 6;
        case 13: case 15: return 10;
        case 14: return 12;
        case 2: case 3: case 5: case 10: case 11: return 8;
        default:
            throw std::runtime_error(
                "CPU dense offload does not support NVQ kernel format " +
                std::to_string(format));
    }
}

static bool cpu_nvq_d4(int format) {
    return format == 3 || format == 10 || format == 11 ||
        format == 12 || format == 15;
}

static const int8_t * cpu_nvq_codebook(
        const int8_t * metadata,
        int format,
        uint32_t state) {
    constexpr int64_t header = 64;
    if (format == 1 || format == 2 || format == 3 || format == 8) {
        return metadata;
    }
    const uint32_t bank = format == 11
        ? state & 1u
        : static_cast<uint8_t>(metadata[36 + state]);
    if (format == 5) return metadata + header + bank * 256 * 8;
    if (format == 10 || format == 11) {
        return metadata + header + bank * 256 * 4;
    }
    if (format == 12) return metadata + header + bank * 512 * 4;
    if (format == 13) return metadata + header + bank * 1024 * 8;
    if (format == 14) return metadata + header + bank * 4096 * 8;
    if (format == 15) return metadata + header + bank * 1024 * 4;
    throw std::runtime_error("unsupported CPU NVQ codebook format");
}

static float cpu_nvq_scale(
        const int8_t * metadata,
        int format,
        float anchor,
        uint32_t state) {
    if (format == 1) {
        return anchor * static_cast<float>(state) * 0.125f;
    }
    if (format == 8) {
        return anchor * static_cast<float>(state) * 0.03125f;
    }
    if (format == 2 || format == 3) {
        return anchor * static_cast<float>(state);
    }
    if (format == 11) {
        return anchor * static_cast<float>((state >> 1) + 1);
    }
    const int64_t lut_offset = (format == 7 || format == 9) ? 8 : 4;
    return anchor * cpu_half_from_bytes(
        metadata, lut_offset + static_cast<int64_t>(state) * 2);
}

static int cpu_nvq_parity7(uint32_t value) {
    value &= 0x7fu;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return static_cast<int>(value & 1u);
}

static void cpu_decode_nvq_group(
        const NvqWeight & w,
        int row,
        int group,
        uint32_t state,
        int8_t values[64]) {
    std::fill(values, values + 64, static_cast<int8_t>(0));
    const int format = static_cast<int>(w.kernel_format);
    const bool d4 = cpu_nvq_d4(format);
    const int vector_size = d4 ? 4 : 8;
    const int nvec = static_cast<int>(
        (w.neuron_len + vector_size - 1) / vector_size);
    const int nsign = static_cast<int>((w.neuron_len + 7) / 8);
    const auto * indices = w.indices_packed.data_ptr<uint8_t>();
    const auto * aux = w.aux_packed.data_ptr<uint8_t>();
    const int bits = cpu_nvq_index_bits(format);
    const int8_t * metadata = w.codebook.data_ptr<int8_t>();
    const int8_t * bank = (format == 7 || format == 9)
        ? nullptr
        : cpu_nvq_codebook(metadata, format, state);
    for (int chunk = 0; chunk < 6; ++chunk) {
        const int vector8 = group * 3 + (chunk >> 1);
        const int vector = d4 ? group * 6 + chunk : vector8;
        if (vector >= nvec) continue;
        const int64_t index_linear =
            static_cast<int64_t>(row) * nvec + vector;
        const uint32_t code = bits == 8
            ? indices[index_linear]
            : cpu_load_packed_bits(
                indices, w.indices_packed.numel(), index_linear * bits, bits);
        int decoded[4] = {0, 0, 0, 0};
        if (format == 7) {
            constexpr int64_t header = 64;
            constexpr int64_t first_state_bytes = 8 * 4;
            constexpr int64_t second_offset = header + 8 * first_state_bytes;
            constexpr int64_t second_state_bytes = 16 * 4;
            const int8_t * source = (chunk & 1) == 0
                ? metadata + header + state * first_state_bytes +
                    (code & 7u) * 4
                : metadata + second_offset + state * second_state_bytes +
                    (code >> 3) * 4;
            for (int index = 0; index < 4; ++index) decoded[index] = source[index];
        } else if (format == 9) {
            constexpr int64_t header = 64;
            const int8_t * source = metadata + header +
                (static_cast<int64_t>(state) * 64 + code) * 8 +
                (chunk & 1) * 4;
            for (int index = 0; index < 4; ++index) decoded[index] = source[index];
        } else {
            const int8_t * source = bank +
                static_cast<int64_t>(code) * vector_size +
                (d4 ? 0 : (chunk & 1) * 4);
            for (int index = 0; index < 4; ++index) decoded[index] = source[index];
            if (format == 1 || format == 8) {
                const int64_t delta_linear =
                    static_cast<int64_t>(row) * w.ng + group;
                const bool negative = cpu_load_packed_bits(
                    aux, w.aux_packed.numel(), delta_linear, 1) != 0;
                const int delta = negative ? -1 : 1;
                if (format == 8) {
                    source = metadata +
                        static_cast<int64_t>(negative) * 512 * 8 +
                        static_cast<int64_t>(code) * 8 + (chunk & 1) * 4;
                    for (int index = 0; index < 4; ++index) {
                        decoded[index] = 32 * source[index] + 5 * delta;
                    }
                } else {
                    for (int index = 0; index < 4; ++index) {
                        decoded[index] = 8 * decoded[index] + delta;
                    }
                }
            } else {
                if (vector8 >= nsign) continue;
                const int64_t sign_linear =
                    static_cast<int64_t>(row) * nsign + vector8;
                const uint32_t mask7 = cpu_load_packed_bits(
                    aux, w.aux_packed.numel(), sign_linear * 7, 7);
                const uint32_t last =
                    static_cast<uint32_t>(cpu_nvq_parity7(mask7)) ^
                    ((format == 2 && w.sign_mode != 0)
                        ? ((code >> 7) & 1u) : 0u);
                const uint32_t mask8 = mask7 | (last << 7);
                const int sign_base = (chunk & 1) * 4;
                for (int index = 0; index < 4; ++index) {
                    if (((mask8 >> (sign_base + index)) & 1u) != 0) {
                        decoded[index] = -decoded[index];
                    }
                }
            }
        }
        const int destination = chunk * 4;
        for (int index = 0; index < 4; ++index) {
            values[destination + index] = static_cast<int8_t>(decoded[index]);
        }
    }
}

static mfq_tensor_backend::Tensor nvq_matmul_cpu(
        const NvqWeight & w,
        mfq_tensor_backend::Tensor x) {
    MFQ_RUNTIME_CHECK(!x.is_cuda(), "CPU NVQ GEMV requires CPU activations");
    MFQ_RUNTIME_CHECK(
        !w.indices_packed.is_cuda() && !w.sub_scale_packed.is_cuda() &&
        !w.neuron_scale.is_cuda() && !w.codebook.is_cuda(),
        "CPU NVQ GEMV requires CPU-resident weights");
    MFQ_RUNTIME_CHECK(w.gs == 24,
        "CPU NVQ GEMV currently requires 24-value groups");
    auto activation = cpu_quantize_activation(
        std::move(x), w.neuron_len, w.gs, true);
    const int64_t rows = activation.rows;
    const int64_t outputs = w.out;
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    mfq_half * output = result.data_ptr<mfq_half>();
    const uint8_t * scales = w.sub_scale_packed.data_ptr<uint8_t>();
    const float * anchors = w.neuron_scale.data_ptr<float>();
    const int8_t * metadata = w.codebook.data_ptr<int8_t>();
    const int64_t scale_bytes = w.sub_scale_packed.numel();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            for (int group = 0; group < w.ng; ++group) {
                const int64_t scale_linear = neuron * w.ng + group;
                const uint32_t state = cpu_load_packed_bits(
                    scales, scale_bytes,
                    scale_linear * w.sub_bits,
                    static_cast<int>(w.sub_bits));
                const float scale = cpu_nvq_scale(
                    metadata, static_cast<int>(w.kernel_format),
                    anchors[neuron], state);
                alignas(64) int8_t weights[64];
                cpu_decode_nvq_group(
                    w, static_cast<int>(neuron), group, state, weights);
                for (int64_t row = 0; row < rows; ++row) {
                    const int8_t * quantized = activation.values.data() +
                        (row * activation.groups + group) * 64;
                    const int32_t dot = cpu_dot_s8_s8_64(
                        weights, quantized,
                        activation.sums[static_cast<size_t>(
                            row * activation.groups + group)]);
                    const float activation_scale = activation.scales[
                        static_cast<size_t>(row * activation.groups + group)];
                    accumulators[static_cast<size_t>(row)] = std::fma(
                        scale * activation_scale,
                        static_cast<float>(dot),
                        accumulators[static_cast<size_t>(row)]);
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] =
                    mfq_half(accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}
mfq_tensor_backend::Tensor nvq_matmul(const NvqWeight & w, mfq_tensor_backend::Tensor x) {
    if (!x.is_cuda()) return nvq_matmul_cpu(w, std::move(x));
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (g_kl_mmq_mode != KlMmqMode::Default) {
        MFQ_RUNTIME_CHECK(
            M >= 16,
            "KLD common NVQ MMQ requires at least 16 activation rows");
        x = kl_mmq_prepare_activation(x);
        ++g_kl_mmq_dense_calls;
        return g_profiler.measure("kld_mmq.nvq.fp16", [&]() {
            return nvq_gemm_f16_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode);
        });
    }
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.gemv", [&]() {
            return nvq_gemv_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.mma24", [&]() {
            return nvq_mmq_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::OnlineF16) {
        return g_profiler.measure("nvq.gemm_online_f16", [&]() {
            return nvq_gemm_f16_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode);
        });
    }
    auto weight = g_profiler.measure("nvq.dequant", [&]() { return nvq_dequant(w); });
    return g_profiler.measure("nvq.gemm", [&]() {
        return nint_cublas_gemm_nt_f32acc_cuda(x, weight);
    });
}

mfq_tensor_backend::Tensor nvq_matmul_input_mul(
    const NvqWeight & w,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor gate,
    int mode) {
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ input gate mode must be 1(sigmoid) or 2(silu)");
    if (!x.is_cuda()) {
        x = x.contiguous().to(mfq_tensor_backend::kFloat16);
        gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
        MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
        auto value = mode == 1
            ? x * mfq_tensor_backend::sigmoid(gate)
            : x * mfq_tensor_backend::silu(gate);
        return nvq_matmul_cpu(w, value);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
    MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    int M = (int)x.size(0);
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.gemv_gate", [&]() {
            return nvq_gemv_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = w.workspace(M);
        return g_profiler.measure("nvq.mma24_gate", [&]() {
            return nvq_mmq_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    mfq_tensor_backend::Tensor value = mode == 1 ? x * mfq_tensor_backend::sigmoid(gate) : x * mfq_tensor_backend::silu(gate);
    return nvq_matmul(w, value);
}

bool nvq_pair_compatible(const NvqWeight & first, const NvqWeight & second) {
    return first.format == second.format && first.kernel_format == second.kernel_format &&
           first.gs == second.gs &&
           first.neuron_len == second.neuron_len && first.ng == second.ng;
}

bool nvq_fusion_enabled() {
    const char * disable = std::getenv("MFQ_DISABLE_NVQ_FUSION");
    if (disable == nullptr) disable = std::getenv("MFQ_DISABLE_NIQ_FUSION");
    return disable == nullptr || disable[0] != '1';
}

mfq_tensor_backend::Tensor nvq_matmul_multi2(
    const NvqWeight & first,
    const NvqWeight & second,
    mfq_tensor_backend::Tensor x) {
    if (!nvq_pair_compatible(first, second)) {
        throw std::runtime_error("NVQ multi-projection requires compatible formats and input layouts");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), first.neuron_len);
    const int M = (int)x.size(0);
    if (M > 8) return mfq_tensor_backend::cat({nvq_matmul(first, x), nvq_matmul(second, x)}, -1);
    NvqWorkspace & ws = first.workspace(M);
    return g_profiler.measure("nvq.gemv_multi2", [&]() {
        return nvq_gemv_multi2_ws_cuda(
            first.indices_packed, first.aux_packed, first.sub_scale_packed,
            first.neuron_scale, first.codebook,
            second.indices_packed, second.aux_packed, second.sub_scale_packed,
            second.neuron_scale, second.codebook,
            x, first.neuron_len, first.gs,
            first.sub_bits, first.kernel_format, first.sign_mode,
            second.sub_bits, second.kernel_format, second.sign_mode,
            ws.qx, ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_matmul_swiglu(
    const NvqWeight & gate,
    const NvqWeight & up,
    mfq_tensor_backend::Tensor x) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out) {
        throw std::runtime_error("NVQ SwiGLU requires compatible equal-width gate/up weights");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), gate.neuron_len);
    if (x.size(0) != 1) {
        auto pair = nvq_matmul_multi2(gate, up, x);
        auto parts = pair.split_with_sizes({gate.out, up.out}, -1);
        return mfq_tensor_backend::silu(parts[0]) * parts[1];
    }
    NvqWorkspace & ws = gate.workspace(1);
    return g_profiler.measure("nvq.gemv_swiglu", [&]() {
        return nvq_gemv_swiglu_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode,
            ws.qx, ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_ffn_swiglu_down(
    const NvqWeight & gate,
    const NvqWeight & up,
    const NvqWeight & down,
    mfq_tensor_backend::Tensor x,
    MfqOptional<mfq_tensor_backend::Tensor> residual) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out || gate.out != down.neuron_len) {
        throw std::runtime_error("NVQ fused FFN weight layouts are incompatible");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), gate.neuron_len);
    if (x.size(0) != 1 || (down.gs != 24 && down.gs != 28 && down.gs != 32)) {
        auto output = nvq_matmul(down, nvq_matmul_swiglu(gate, up, x));
        return residual.has_value()
            ? acc_cuda(residual.value(), output)
            : output;
    }
    NvqWorkspace & input_ws = gate.workspace(1);
    NvqWorkspace & output_ws = down.workspace(1);
    if (!input_ws.swiglu_scratch.defined() || input_ws.swiglu_scratch.numel() < gate.out) {
        input_ws.swiglu_scratch = mfq_tensor_backend::empty(
            {gate.out}, gate.indices_packed.options().dtype(
                mfq_tensor_backend::kFloat32));
    }
    g_profiler.measure("nvq.ffn_swiglu_quant", [&]() {
        nvq_ffn_swiglu_quant_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode, down.gs,
            input_ws.qx, input_ws.xscale,
            output_ws.qx, output_ws.xscale, input_ws.swiglu_scratch);
        return 0;
    });
    return g_profiler.measure(
        residual.has_value() ? "nvq.gemv_qx_residual" : "nvq.gemv_qx",
        [&]() {
        if (residual.has_value()) {
            return nvq_gemv_qx_residual_ws_cuda(
                down.indices_packed, down.aux_packed, down.sub_scale_packed,
                down.neuron_scale, down.codebook,
                down.neuron_len, down.gs, down.sub_bits,
                down.kernel_format, down.sign_mode,
                output_ws.qx, output_ws.xscale, residual.value());
        }
        return nvq_gemv_qx_ws_cuda(
            down.indices_packed, down.aux_packed, down.sub_scale_packed,
            down.neuron_scale, down.codebook,
            down.neuron_len, down.gs, down.sub_bits, down.kernel_format, down.sign_mode,
            output_ws.qx, output_ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_embedding(const NvqWeight & w, mfq_tensor_backend::Tensor token_ids) {
    return nvq_embedding_lookup_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, token_ids, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}
static int nvq_vector_size(int format) {
    return (format == 3 || format == 10 || format == 12 || format == 15) ? 4 : 8;
}

static int nvq_index_bits(int format) {
    return format == 1 ? 11 :
        (format == 8 ? 9 :
         (format == 7 ? 7 :
          (format == 9 ? 6 :
           (format == 12 ? 9 :
            ((format == 13 || format == 15) ? 10 :
             (format == 14 ? 12 : 8))))));
}

static bool nvq_delta_format(int format) {
    return format == 1 || format == 8;
}

static bool nvq_no_aux_format(int format) {
    return format == 7 || format == 9;
}

static void copy_packed_bits(
        const std::vector<uint8_t> & source,
        size_t source_bit,
        std::vector<uint8_t> & destination,
        size_t destination_bit,
        size_t bit_count) {
    if (bit_count == 0) return;
    if (source_bit + bit_count >
            source.size() * 8 ||
        destination_bit + bit_count >
            destination.size() * 8) {
        throw std::runtime_error(
            "packed tensor-parallel bit copy is out of bounds");
    }
    if ((source_bit & 7u) == 0 &&
        (destination_bit & 7u) == 0) {
        const size_t full_bytes =
            bit_count / 8;
        if (full_bytes != 0) {
            std::memcpy(
                destination.data() +
                    destination_bit / 8,
                source.data() +
                    source_bit / 8,
                full_bytes);
            source_bit += full_bytes * 8;
            destination_bit += full_bytes * 8;
            bit_count -= full_bytes * 8;
        }
    }
    while (bit_count >= 8) {
        const size_t source_byte =
            source_bit >> 3;
        const int source_shift =
            static_cast<int>(source_bit & 7u);
        uint16_t source_word =
            source[source_byte];
        if (source_shift != 0 &&
            source_byte + 1 < source.size()) {
            source_word |=
                static_cast<uint16_t>(
                    source[source_byte + 1])
                << 8;
        }
        const uint8_t value =
            static_cast<uint8_t>(
                source_word >> source_shift);
        const size_t destination_byte =
            destination_bit >> 3;
        const int destination_shift =
            static_cast<int>(
                destination_bit & 7u);
        destination[destination_byte] |=
            static_cast<uint8_t>(
                value << destination_shift);
        if (destination_shift != 0 &&
            destination_byte + 1 <
                destination.size()) {
            destination[destination_byte + 1] |=
                static_cast<uint8_t>(
                    value >>
                    (8 - destination_shift));
        }
        source_bit += 8;
        destination_bit += 8;
        bit_count -= 8;
    }
    for (size_t bit = 0; bit < bit_count; ++bit) {
        if ((source[source_bit >> 3] >>
             (source_bit & 7u)) & 1u) {
            destination[destination_bit >> 3] |=
                static_cast<uint8_t>(
                    1u << (destination_bit & 7u));
        }
        ++source_bit;
        ++destination_bit;
    }
}

static std::vector<uint8_t> slice_packed_rows(
        const std::vector<uint8_t> & source,
        int64_t source_rows,
        int64_t source_items_per_row,
        int64_t row_begin,
        int64_t row_end,
        int64_t item_begin,
        int64_t item_count,
        int bits) {
    if (bits == 0) return {};
    if (source_rows <= 0 || source_items_per_row <= 0 ||
        row_begin < 0 || row_begin >= row_end ||
        row_end > source_rows || item_begin < 0 ||
        item_count <= 0 ||
        item_begin + item_count > source_items_per_row) {
        throw std::runtime_error("invalid packed tensor-parallel slice");
    }
    const int64_t destination_rows = row_end - row_begin;
    const size_t destination_items =
        static_cast<size_t>(destination_rows) * item_count;
    std::vector<uint8_t> destination(
        (destination_items * static_cast<size_t>(bits) + 7) / 8,
        0);
    for (int64_t row = 0; row < destination_rows; ++row) {
        const size_t source_item =
            static_cast<size_t>(row + row_begin) *
                source_items_per_row +
            static_cast<size_t>(item_begin);
        const size_t destination_item =
            static_cast<size_t>(row) * item_count;
        copy_packed_bits(
            source,
            source_item * static_cast<size_t>(bits),
            destination,
            destination_item * static_cast<size_t>(bits),
            static_cast<size_t>(item_count) *
                static_cast<size_t>(bits));
    }
    return destination;
}

NvqCpu slice_nvq_cpu(
        const NvqCpu & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NVQ");
    if (axis != TensorParallelAxis::Output &&
        axis != TensorParallelAxis::Input) {
        throw std::runtime_error("NVQ shard requires an output or input axis");
    }
    NvqCpu result = source;
    int64_t row_begin = 0;
    int64_t row_end = source.out;
    int64_t group_begin = 0;
    int64_t group_count = source.ng;
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid NVQ output shard");
        }
        row_begin = begin;
        row_end = end;
        result.out = static_cast<int>(end - begin);
        result.shape[0] = result.out;
    } else {
        if (begin < 0 || begin >= end || end > source.ng) {
            throw std::runtime_error("invalid NVQ input shard");
        }
        group_begin = begin;
        group_count = end - begin;
        result.ng = static_cast<int>(group_count);
        const int64_t element_begin = begin * source.gs;
        const int64_t element_end =
            std::min<int64_t>(end * source.gs, source.neuron_len);
        result.neuron_len =
            static_cast<int>(element_end - element_begin);
        result.shape[1] = result.neuron_len;
        const int vector_size = nvq_vector_size(source.format);
        if (element_begin % vector_size != 0 ||
            element_begin % 8 != 0) {
            throw std::runtime_error(
                "NVQ input shard does not preserve vector boundaries");
        }
        result.nvec =
            (result.neuron_len + vector_size - 1) / vector_size;
        result.nsign = (result.neuron_len + 7) / 8;
    }

    result.neuron_scale_h.assign(
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_begin),
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_end));
    result.sub_scale_packed = slice_packed_rows(
        source.sub_scale_packed,
        source.out, source.ng,
        row_begin, row_end,
        group_begin, group_count,
        source.sub_bits);

    const int vector_size = nvq_vector_size(source.format);
    const int64_t element_begin = group_begin * source.gs;
    const int64_t vector_begin = element_begin / vector_size;
    const int64_t vector_count =
        axis == TensorParallelAxis::Output
        ? source.nvec
        : result.nvec;
    result.indices_packed = slice_packed_rows(
        source.indices_packed,
        source.out, source.nvec,
        row_begin, row_end,
        vector_begin, vector_count,
        nvq_index_bits(source.format));

    if (nvq_delta_format(source.format)) {
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.ng,
            row_begin, row_end,
            group_begin, group_count,
            1);
    } else if (nvq_no_aux_format(source.format)) {
        result.aux_packed.clear();
    } else {
        const int64_t sign_begin = element_begin / 8;
        const int64_t sign_count =
            axis == TensorParallelAxis::Output
            ? source.nsign
            : result.nsign;
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.nsign,
            row_begin, row_end,
            sign_begin, sign_count,
            7);
    }
    return result;
}

static std::vector<int8_t> expand_nvq_codebook(
    int format, const uint16_t * packed, size_t count) {
    const int dims = format == 3 ? 4 : 8;
    const int expected = format == 1 ? 2048 : (format == 8 ? 1024 : 256);
    if ((int)count != expected) throw std::runtime_error("NVQ codebook entry count mismatch");
    const int digit_bits = format == 3 ? 3 : 2;
    std::vector<int8_t> codebook(count * (size_t)dims);
    for (size_t row = 0; row < count; ++row) {
        uint16_t word = packed[row];
        for (int i = 0; i < dims; ++i) {
            int digit = (word >> (digit_bits * i)) & ((1 << digit_bits) - 1);
            const bool ternary = format == 1 || format == 8;
            int value = ternary ? digit - 1 : 2 * digit + 1;
            if ((ternary && (value < -1 || value > 1)) ||
                (!ternary && (value < 1 || value > (format == 2 ? 7 : 15)))) {
                throw std::runtime_error("invalid NVQ codebook digit");
            }
            codebook[row * (size_t)dims + i] = (int8_t)value;
        }
    }
    return codebook;
}

static std::vector<int8_t> expand_npq0_s_runtime_lut(
    const std::vector<uint8_t> & packed) {
    constexpr size_t kMetadataBytes = 64;
    constexpr size_t kStates = 4;
    constexpr size_t kEntries = 8;
    constexpr size_t kSubvector = 4;
    constexpr size_t kPackedBytes = kMetadataBytes + 2 * kStates * kEntries * kSubvector;
    constexpr size_t kRuntimeBytes = kMetadataBytes + kStates * 64 * 8;
    if (packed.size() != kPackedBytes) {
        throw std::runtime_error("NPQ0-S packed table size mismatch");
    }
    std::vector<int8_t> runtime(kRuntimeBytes);
    std::memcpy(runtime.data(), packed.data(), kMetadataBytes);
    const int8_t * first = reinterpret_cast<const int8_t *>(packed.data() + kMetadataBytes);
    const int8_t * second = first + kStates * kEntries * kSubvector;
    for (size_t state = 0; state < kStates; ++state) {
        for (size_t second_index = 0; second_index < kEntries; ++second_index) {
            for (size_t first_index = 0; first_index < kEntries; ++first_index) {
                const size_t index = first_index | (second_index << 3);
                int8_t * destination = runtime.data() + kMetadataBytes
                    + (state * 64 + index) * 8;
                std::memcpy(
                    destination,
                    first + (state * kEntries + first_index) * kSubvector,
                    kSubvector);
                std::memcpy(
                    destination + kSubvector,
                    second + (state * kEntries + second_index) * kSubvector,
                    kSubvector);
            }
        }
    }
    return runtime;
}

static std::vector<uint16_t> read_codebook_words(
    const std::vector<uint8_t> & blob, size_t & off, size_t count) {
    size_t nbytes = count * sizeof(uint16_t);
    if (off + nbytes > blob.size()) throw std::runtime_error("truncated NVQ custom codebook");
    std::vector<uint16_t> words(count);
    std::memcpy(words.data(), blob.data() + off, nbytes);
    off += nbytes;
    return words;
}

static std::vector<uint8_t> take_bytes(
    const std::vector<uint8_t> & blob, size_t & off, size_t count, const char * label) {
    if (off + count > blob.size()) throw std::runtime_error(std::string("truncated NVQ ") + label);
    std::vector<uint8_t> result(count);
    std::memcpy(result.data(), blob.data() + off, count);
    off += count;
    return result;
}

static void store_compact_bits_cpu(
        std::vector<uint8_t> & destination,
        size_t bit,
        int bits,
        uint32_t value) {
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    const int bytes = (shift + bits + 7) >> 3;
    if (bits <= 0 || bits > 24 || byte + bytes > destination.size() ||
            value >= (1u << bits)) {
        throw std::runtime_error("invalid compact bitstream write");
    }
    const uint32_t shifted = value << shift;
    for (int index = 0; index < bytes; ++index) {
        destination[byte + index] |= static_cast<uint8_t>(
            shifted >> (index * 8));
    }
}

NvqCpu unpack_nvq(const std::vector<uint8_t> & blob, const std::string & dtype) {
    if (blob.size() < 20) throw std::runtime_error("truncated NVQ header");
    NvqCpu t;
    const bool nvq_family = dtype == "NVQ";
    const bool npq_family = dtype == "NPQ";
    const bool nvq_magic =
        (std::memcmp(blob.data(), "NVQ1", 4) == 0) ||
        (std::memcmp(blob.data(), "NIQ1", 4) == 0);
    const bool nvq1_l_magic = std::memcmp(blob.data(), "NQ1L", 4) == 0;
    const bool nvq1_s_magic = std::memcmp(blob.data(), "NQ1S", 4) == 0;
    const bool npq0_l_magic = std::memcmp(blob.data(), "NPQL", 4) == 0;
    const bool npq0_s_magic = std::memcmp(blob.data(), "NPQS", 4) == 0;
    if ((nvq_family && !(nvq_magic || nvq1_l_magic || nvq1_s_magic)) ||
        (npq_family && !(npq0_l_magic || npq0_s_magic)) ||
        (!nvq_family && !npq_family)) {
        throw std::runtime_error("compact VQ family/payload mismatch: " + dtype);
    }
    size_t off = 4;
    const uint8_t profile = blob[off++];
    if (nvq1_l_magic) {
        t.format = 1;
    } else if (nvq1_s_magic) {
        t.format = 8;
    } else if (npq0_l_magic) {
        t.format = 7;
    } else if (npq0_s_magic) {
        t.format = 9;
    } else {
        constexpr uint8_t kIndexParity = 0x80;
        constexpr uint8_t kCustom = 0x40;
        constexpr uint8_t kJsc = 0x20;
        const bool jsc = (profile & kJsc) != 0;
        const int codebook_id = profile & ~(kIndexParity | kCustom | kJsc);
        if (jsc) {
            t.format = codebook_id == 1 ? 5 :
                (codebook_id == 2 ? 10 :
                 (codebook_id == 3 ? 12 :
                  (codebook_id == 4 ? 13 :
                   (codebook_id == 5 ? 14 :
                    (codebook_id == 6 ? 15 : 0)))));
        } else {
            t.format = codebook_id == 1 ? 2 :
                (codebook_id == 2 ? 3 : 0);
        }
        if (t.format == 0) {
            throw std::runtime_error("unsupported NVQ payload profile");
        }
    }
    t.sub_bits = blob[off++];
    t.gs = (int)read_u16_from(blob, off);
    t.axis = read_i32_from(blob, off);
    t.neuron_len = read_i32_from(blob, off);
    uint32_t ndim = read_u32_from(blob, off);
    if (ndim == 0 || ndim > 8) throw std::runtime_error("invalid NVQ ndim");
    t.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) t.shape[i] = read_i64_from(blob, off);
    t.out = (int)read_u32_from(blob, off);
    if (t.gs != 24 || t.sub_bits < 1 || t.sub_bits > 8) {
        throw std::runtime_error("C++ NVQ runtime requires gs24 and sub_bits in [1,8]");
    }
    if (t.axis != 0 || t.shape.size() != 2 || t.shape[0] != t.out || t.shape[1] != t.neuron_len) {
        throw std::runtime_error("C++ NVQ runtime requires row-major rank-2 axis=0 weights");
    }

    bool custom = false;
    bool group64_storage = false;
    if (t.format == 7) {
        if (profile != 1 || t.sub_bits != 3) {
            throw std::runtime_error("unsupported NPQ0-L profile");
        }
    } else if (t.format == 1) {
        if (profile != 1 && profile != 2) throw std::runtime_error("unsupported NVQ1-L profile");
        custom = profile == 2;
    } else if (t.format == 8) {
        if (profile != 1 || t.sub_bits != 4) {
            throw std::runtime_error("unsupported NVQ1-S profile");
        }
        custom = true;
    } else if (t.format == 9) {
        if (profile != 2 || t.sub_bits != 2) {
            throw std::runtime_error("unsupported NPQ0-S profile");
        }
    } else {
        constexpr uint8_t kIndexParity = 0x80;
        constexpr uint8_t kCustom = 0x40;
        constexpr uint8_t kJsc = 0x20;
        const bool jsc = (profile & kJsc) != 0;
        int codebook_id = profile & ~(kIndexParity | kCustom | kJsc);
        int expected_id =
            t.format == 13 ? 4 :
            (t.format == 14 ? 5 :
             (t.format == 15 ? 6 :
              ((t.format == 2 || t.format == 5)
               ? 1
               : (t.format == 12 ? 3 : 2))));
        if (codebook_id != expected_id) throw std::runtime_error("NVQ payload profile/codebook mismatch");
        t.sign_mode = (profile & kIndexParity) ? 1 : 0;
        if (t.sign_mode && t.format != 2) throw std::runtime_error("NVQ index parity requires NVQ2");
        custom = (profile & kCustom) != 0;
        if (t.format == 5 || t.format == 10 || t.format == 12 ||
            t.format == 13 || t.format == 14 || t.format == 15) {
            if (!jsc || custom || t.sign_mode || t.sub_bits != 4) {
                throw std::runtime_error("invalid NVQ-JSC profile flags");
            }
        } else if (jsc) {
            throw std::runtime_error("NVQ-JSC flag requires a JSC payload profile");
        }
    }

    const size_t codebook_count = t.format == 1 ? 2048 : (t.format == 8 ? 1024 : 256);
    if (t.format == 7) {
        constexpr size_t kMetadataBytes = 64;
        constexpr size_t kTableBytes = 832;
        if (off + kTableBytes > blob.size()) {
            throw std::runtime_error("truncated NPQ0-L product tables");
        }
        const uint8_t * header = blob.data() + off;
        const uint8_t expected[6] = {1, 8, 3, 4, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (header[i] != expected[i]) {
                throw std::runtime_error("unsupported NPQ0-L table profile");
            }
        }
        if (header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid NPQ0-L reserved table bytes");
        }
        for (int state = 0; state < 8; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 8 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NPQ0-L scale LUT must be finite and non-negative");
            }
        }
        for (size_t i = 24; i < kMetadataBytes; ++i) {
            if (header[i] != 0) {
                throw std::runtime_error("invalid NPQ0-L reserved table bytes");
            }
        }
        auto metadata = take_bytes(blob, off, kTableBytes, "NPQ0-L product tables");
        t.codebook.resize(metadata.size());
        std::memcpy(t.codebook.data(), metadata.data(), metadata.size());
    } else if (t.format == 9) {
        constexpr size_t kMetadataBytes = 64;
        constexpr size_t kTableBytes = 320;
        if (off + kTableBytes > blob.size()) {
            throw std::runtime_error("truncated NPQ0-S product tables");
        }
        const uint8_t * header = blob.data() + off;
        const uint8_t expected[6] = {2, 4, 3, 3, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (header[i] != expected[i]) {
                throw std::runtime_error("unsupported NPQ0-S table profile");
            }
        }
        if (header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid NPQ0-S reserved table bytes");
        }
        for (int state = 0; state < 4; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 8 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NPQ0-S scale LUT must be finite and non-negative");
            }
        }
        for (size_t i = 16; i < kMetadataBytes; ++i) {
            if (header[i] != 0) {
                throw std::runtime_error("invalid NPQ0-S reserved table bytes");
            }
        }
        auto metadata = take_bytes(blob, off, kTableBytes, "NPQ0-S product tables");
        t.codebook = expand_npq0_s_runtime_lut(metadata);
    } else if (t.format == 5 || t.format == 10 || t.format == 12 ||
               t.format == 13 || t.format == 14 || t.format == 15) {
        constexpr size_t kHeaderBytes = 64;
        const int vector_size = nvq_vector_size(t.format);
        const size_t entries =
            t.format == 12 ? 512 :
            ((t.format == 13 || t.format == 15) ? 1024 :
             (t.format == 14 ? 4096 : 256));
        const size_t kBankBytes = entries * (size_t)vector_size;
        if (off + kHeaderBytes > blob.size()) {
            throw std::runtime_error("truncated NVQ2J metadata header");
        }
        const uint8_t * header = blob.data() + off;
        const int banks = header[1];
        const int metadata_version = header[0];
        if ((metadata_version != 1 && metadata_version != 2) ||
                (banks != 1 && banks != 2 && banks != 4) || header[2] != 16) {
            throw std::runtime_error("unsupported NVQ2J metadata dimensions");
        }
        if (metadata_version == 1) {
            if (header[52] != 0) {
                throw std::runtime_error("invalid NVQ-JSC v1 storage layout");
            }
        } else {
            if (t.format != 14 || header[52] != 1) {
                throw std::runtime_error(
                    "NVQ-JSC group64 storage requires NVQ2J-XL");
            }
            group64_storage = true;
        }
        if (header[3] != 0 && header[3] != 1) {
            throw std::runtime_error("invalid NVQ-JSC state mode");
        }
        for (int state = 0; state < 16; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 4 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NVQ-JSC scale LUT must be finite and non-negative");
            }
            if (header[36 + state] >= banks) {
                throw std::runtime_error("NVQ-JSC state references a missing bank");
            }
            if (header[3] == 1) {
                const uint8_t expected_bank = (uint8_t)(state % banks);
                const int rank = state / banks;
                const float expected_scale = banks == 1
                    ? (float)state
                    : (vector_size == 4
                        ? (float)(rank + 1)
                        : 15.0f * (float)(rank + 1) / (float)(16 / banks));
                const mfq_half expected_half(expected_scale);
                uint16_t expected_alpha_h;
                std::memcpy(&expected_alpha_h, &expected_half, sizeof(expected_alpha_h));
                if (header[36 + state] != expected_bank || alpha_h != expected_alpha_h) {
                    throw std::runtime_error("invalid analytic NVQ-JSC state tables");
                }
            }
        }
        const size_t reserved_begin = metadata_version == 1 ? 52 : 53;
        for (size_t i = reserved_begin; i < kHeaderBytes; ++i) {
            if (header[i] != 0) throw std::runtime_error("invalid NVQ-JSC reserved metadata bytes");
        }
        const size_t metadata_bytes = kHeaderBytes + (size_t)banks * kBankBytes;
        auto metadata = take_bytes(blob, off, metadata_bytes, "JSC metadata");
        for (int bank = 0; bank < banks; ++bank) {
            const size_t base = kHeaderBytes + (size_t)bank * kBankBytes;
            for (size_t entry = 0; entry < entries; ++entry) {
                bool nonzero = false;
                for (int dim = 0; dim < vector_size; ++dim) {
                    const uint8_t value =
                        metadata[base + (size_t)entry * vector_size + dim];
                    if (value > 127) throw std::runtime_error("NVQ-JSC codebook value exceeds int8");
                    nonzero = nonzero || value != 0;
                }
                if (!nonzero) throw std::runtime_error("NVQ-JSC codeword must not be all zero");
            }
        }
        t.codebook.resize(metadata.size());
        std::memcpy(t.codebook.data(), metadata.data(), metadata.size());
    } else if (custom) {
        auto words = read_codebook_words(blob, off, codebook_count);
        t.codebook = expand_nvq_codebook(t.format, words.data(), words.size());
    } else if (t.format == 1) {
        t.codebook = expand_nvq_codebook(
            1, mfq::nvq_codebooks::kNvq1LCodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq1LCodebookPacked) / sizeof(uint16_t));
    } else if (t.format == 2) {
        t.codebook = expand_nvq_codebook(
            2, mfq::nvq_codebooks::kNvq2CodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq2CodebookPacked) / sizeof(uint16_t));
    } else {
        t.codebook = expand_nvq_codebook(
            3, mfq::nvq_codebooks::kNvq3CodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq3CodebookPacked) / sizeof(uint16_t));
    }

    t.ng = (t.neuron_len + t.gs - 1) / t.gs;
    const int vector_size = nvq_vector_size(t.format);
    t.nvec = (t.neuron_len + vector_size - 1) / vector_size;
    t.nsign = (t.neuron_len + 7) / 8;
    size_t anchor_bytes = (size_t)t.out * 2;
    if (off + anchor_bytes > blob.size()) throw std::runtime_error("truncated NVQ neuron anchors");
    t.neuron_scale_h.resize(t.out);
    std::memcpy(t.neuron_scale_h.data(), blob.data() + off, anchor_bytes);
    for (uint16_t anchor : t.neuron_scale_h) {
        if ((anchor & 0x7c00u) == 0x7c00u || (anchor & 0x8000u) != 0u) {
            throw std::runtime_error(
                "NVQ neuron anchors must be finite and non-negative");
        }
    }
    off += anchor_bytes;
    size_t sub_bytes = ((size_t)t.out * t.ng * t.sub_bits + 7) / 8;
    const size_t index_bits = (size_t)nvq_index_bits(t.format);
    size_t index_bytes = ((size_t)t.out * t.nvec * index_bits + 7) / 8;
    const bool delta_format = t.format == 1 || t.format == 8;
    const bool no_aux_format = t.format == 7 || t.format == 9;
    size_t aux_count = delta_format ? (size_t)t.out * t.ng :
        (no_aux_format ? 0 : (size_t)t.out * t.nsign);
    size_t aux_bits = delta_format ? 1 : (no_aux_format ? 0 : 7);
    size_t aux_bytes = (aux_count * aux_bits + 7) / 8;
    if (group64_storage) {
        const size_t record_count = static_cast<size_t>(t.out) * t.ng;
        const auto records = take_bytes(
            blob, off, record_count * 8, "group64 stream");
        t.sub_scale_packed.assign(sub_bytes, 0);
        t.indices_packed.assign(index_bytes, 0);
        t.aux_packed.assign(aux_bytes, 0);
        for (size_t linear = 0; linear < record_count; ++linear) {
            uint64_t packed = 0;
            std::memcpy(&packed, records.data() + linear * 8, sizeof(packed));
            const uint32_t state = static_cast<uint32_t>(packed >> 60);
            store_compact_bits_cpu(
                t.sub_scale_packed, linear * t.sub_bits, t.sub_bits, state);
            const size_t row = linear / static_cast<size_t>(t.ng);
            const size_t group = linear - row * static_cast<size_t>(t.ng);
            for (int local = 0; local < 3; ++local) {
                const uint32_t segment = static_cast<uint32_t>(
                    (packed >> (local * 20)) & 0xfffffu);
                const uint32_t index = segment & 0xfffu;
                const uint32_t mask8 = segment >> 12;
                uint32_t parity = mask8 & 0x7fu;
                parity ^= parity >> 4;
                parity ^= parity >> 2;
                parity ^= parity >> 1;
                if ((mask8 >> 7) != (parity & 1u)) {
                    throw std::runtime_error(
                        "invalid NVQ-JSC group64 parity bit");
                }
                const size_t local_vector = group * 3 + local;
                if (local_vector >= static_cast<size_t>(t.nvec)) {
                    if (segment != 0) {
                        throw std::runtime_error(
                            "NVQ-JSC group64 padding must be zero");
                    }
                    continue;
                }
                const size_t vector = row * static_cast<size_t>(t.nvec) +
                    local_vector;
                store_compact_bits_cpu(
                    t.indices_packed, vector * index_bits,
                    static_cast<int>(index_bits), index);
                store_compact_bits_cpu(
                    t.aux_packed, vector * 7, 7, mask8 & 0x7fu);
            }
        }
    } else {
        t.sub_scale_packed = take_bytes(blob, off, sub_bytes, "sub-scale stream");
        t.indices_packed = take_bytes(blob, off, index_bytes, "index stream");
        t.aux_packed = take_bytes(blob, off, aux_bytes, "aux stream");
    }
    if (off != blob.size()) throw std::runtime_error("invalid NVQ blob tail");
    return t;
}

constexpr int64_t kNvq2JscXlGroupExecKernelFormat = 16;
constexpr int64_t kNvq3JscLGroupExecKernelFormat = 17;

static std::vector<uint8_t> repack_nvq2_exec_metadata(const NvqCpu & c) {
    if ((c.format != 2 && c.format != 5) || c.nvec != c.nsign) {
        throw std::runtime_error("NVQ2 execution metadata requires one sign record per vector");
    }
    const size_t count = (size_t)c.out * c.nvec;
    if (c.indices_packed.size() != count) {
        throw std::runtime_error("NVQ2 index stream length mismatch during execution repack");
    }
    std::vector<uint8_t> metadata(count * 2);
    for (size_t linear = 0; linear < count; ++linear) {
        const size_t bit = linear * 7;
        const size_t byte = bit >> 3;
        const int shift = (int)(bit & 7);
        uint16_t word = c.aux_packed[byte];
        if (byte + 1 < c.aux_packed.size()) {
            word |= (uint16_t)c.aux_packed[byte + 1] << 8;
        }
        const uint8_t index = c.indices_packed[linear];
        const uint8_t mask7 = (uint8_t)((word >> shift) & 0x7f);
        uint8_t parity = mask7;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        const uint8_t last = (parity & 1u) ^
            (c.sign_mode ? ((index >> 7) & 1u) : 0u);
        metadata[2 * linear] = c.indices_packed[linear];
        metadata[2 * linear + 1] = mask7 | (last << 7);
    }
    return metadata;
}

static uint32_t load_compact_bits_cpu(
        const std::vector<uint8_t> & data, size_t bit, int bits) {
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = byte < data.size() ? data[byte] : 0;
    if (byte + 1 < data.size()) word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < data.size()) word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

static void put_compact_bits96(
        uint32_t (&words)[3], int bit, int bits, uint32_t value) {
    const int word = bit >> 5;
    const int shift = bit & 31;
    words[word] |= value << shift;
    if (shift + bits > 32) words[word + 1] |= value >> (32 - shift);
}

static uint32_t compact_parity7(uint32_t value) {
    value &= 0x7fu;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return value & 1u;
}

// NVQ2J-XL needs exactly 64 bits per gs24 group (three 12-bit indices, three
// 8-bit sign masks, and one 4-bit state). NVQ3J-L stores three indices in
// a 96-bit record per gs24 group (six 10-bit indices, three 8-bit sign masks,
// and one 4-bit state).
// This replaces the two independently bit-packed streams on CUDA; it does not
// keep a second execution copy of the model weights.
static std::vector<uint8_t> repack_extended_jsc_group_metadata(
        const NvqCpu & c,
        const std::array<uint8_t, 16> * state_remap = nullptr,
        const std::array<std::array<uint16_t, 4096>, 4> * index_remap = nullptr) {
    if (c.format != 14 && c.format != 15) {
        throw std::runtime_error("extended JSC group metadata requires NVQ2J-XL or NVQ3J-L");
    }
    const int index_bits = c.format == 14 ? 12 : 10;
    const int vectors_per_group = c.format == 14 ? 3 : 6;
    const size_t record_bytes = c.format == 14 ? 8 : 12;
    const size_t group_count = static_cast<size_t>(c.out) * c.ng;
    std::vector<uint8_t> metadata(group_count * record_bytes, 0);
    mfq_parallel_for(0, static_cast<int64_t>(group_count), 4096, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t row = linear / c.ng;
            const int group = static_cast<int>(linear - row * c.ng);
            const size_t vector_base = static_cast<size_t>(row) * c.nvec +
                static_cast<size_t>(group) * vectors_per_group;
            const size_t sign_base = static_cast<size_t>(row) * c.nsign +
                static_cast<size_t>(group) * 3;
            const uint32_t source_state = load_compact_bits_cpu(
                c.sub_scale_packed, static_cast<size_t>(linear) * c.sub_bits,
                c.sub_bits);
            const uint32_t state = state_remap == nullptr
                ? source_state
                : (*state_remap)[source_state];
            uint8_t * destination = metadata.data() +
                static_cast<size_t>(linear) * record_bytes;
            if (c.format == 14) {
                uint64_t packed = 0;
                for (int local = 0; local < 3; ++local) {
                    const size_t vector = vector_base + local;
                    const uint32_t source_index = vector < static_cast<size_t>((row + 1) * c.nvec)
                        ? load_compact_bits_cpu(c.indices_packed, vector * index_bits, index_bits)
                        : 0;
                    const uint32_t source_bank = static_cast<uint8_t>(
                        c.codebook[36 + source_state]);
                    const uint32_t index = index_remap == nullptr
                        ? source_index
                        : (*index_remap)[source_bank][source_index];
                    const size_t sign = sign_base + local;
                    const uint32_t mask7 = sign < static_cast<size_t>((row + 1) * c.nsign)
                        ? load_compact_bits_cpu(c.aux_packed, sign * 7, 7)
                        : 0;
                    const uint32_t mask8 = mask7 | (compact_parity7(mask7) << 7);
                    const uint32_t segment = index | (mask8 << index_bits);
                    packed |= static_cast<uint64_t>(segment) << (local * 20);
                }
                packed |= static_cast<uint64_t>(state) << 60;
                std::memcpy(destination, &packed, sizeof(packed));
            } else {
                uint32_t words[3] = {0, 0, 0};
                for (int local = 0; local < 6; ++local) {
                    const size_t vector = vector_base + local;
                    const uint32_t index = vector < static_cast<size_t>((row + 1) * c.nvec)
                        ? load_compact_bits_cpu(c.indices_packed, vector * index_bits, index_bits)
                        : 0;
                    put_compact_bits96(words, local * index_bits, index_bits, index);
                }
                for (int local = 0; local < 3; ++local) {
                    const size_t sign = sign_base + local;
                    const uint32_t mask7 = sign < static_cast<size_t>((row + 1) * c.nsign)
                        ? load_compact_bits_cpu(c.aux_packed, sign * 7, 7)
                        : 0;
                    const uint32_t mask8 = mask7 | (compact_parity7(mask7) << 7);
                    put_compact_bits96(words, 60 + local * 8, 8, mask8);
                }
                put_compact_bits96(words, 84, 4, state);
                std::memcpy(destination, words, sizeof(words));
            }
        }
    });
    return metadata;
}

static bool nvq2_exec_layout_enabled() {
    const char * disable = std::getenv("MFQ_DISABLE_NVQ2_EXEC");
    if (disable == nullptr) disable = std::getenv("MFQ_DISABLE_NIQ2_EXEC");
    return disable == nullptr || disable[0] != '1';
}

static bool extended_jsc_group_layout_enabled() {
    const char * enable = std::getenv("MFQ_NVQ_EXTENDED_GROUP_EXEC");
    return enable != nullptr && enable[0] == '1';
}

static std::vector<int8_t> reorder_extended_e8_codebook_for_stage3(
        const NvqCpu & c,
        std::array<uint8_t, 16> & state_remap,
        std::array<std::array<uint16_t, 4096>, 4> & index_remap) {
    constexpr size_t metadata_bytes = 64;
    constexpr size_t bank_bytes = 4096 * 8;
    constexpr int bank_count = 4;
    if (c.format != 14 ||
        c.codebook.size() != metadata_bytes + bank_count * bank_bytes ||
        static_cast<uint8_t>(c.codebook[1]) != bank_count) {
        throw std::runtime_error("extended E8 runtime layout requires NVQ2J-XL");
    }

    uint64_t counts[bank_count] = {0, 0, 0, 0};
    std::vector<uint64_t> index_counts(bank_count * 4096, 0);
    const size_t groups = static_cast<size_t>(c.out) * c.ng;
    for (size_t linear = 0; linear < groups; ++linear) {
        const uint32_t state = load_compact_bits_cpu(
            c.sub_scale_packed, linear * c.sub_bits, c.sub_bits);
        const uint32_t bank = static_cast<uint8_t>(c.codebook[36 + state]);
        if (bank >= bank_count) {
            throw std::runtime_error("NVQ2J-XL state references an invalid bank");
        }
        ++counts[bank];
        const size_t row = linear / c.ng;
        const size_t group = linear - row * c.ng;
        const size_t vector_base = row * c.nvec + group * 3;
        const size_t vector_end = (row + 1) * c.nvec;
        for (int local = 0; local < 3; ++local) {
            const size_t vector = vector_base + local;
            if (vector >= vector_end) break;
            const uint32_t index = load_compact_bits_cpu(
                c.indices_packed, vector * 12, 12);
            ++index_counts[bank * 4096 + index];
        }
    }
    std::array<int, bank_count> order = {0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        return counts[left] > counts[right];
    });
    std::vector<int8_t> reordered = c.codebook;
    int runtime_bank[bank_count] = {0, 0, 0, 0};
    for (int destination = 0; destination < bank_count; ++destination) {
        const int source = order[destination];
        runtime_bank[source] = destination;
        std::array<int, 4096> index_order;
        std::iota(index_order.begin(), index_order.end(), 0);
        std::stable_sort(index_order.begin(), index_order.end(),
            [&](int left, int right) {
                return index_counts[source * 4096 + left] >
                    index_counts[source * 4096 + right];
            });
        for (int destination_index = 0;
             destination_index < 4096;
             ++destination_index) {
            const int source_index = index_order[destination_index];
            index_remap[source][source_index] =
                static_cast<uint16_t>(destination_index);
            std::memcpy(
                reordered.data() + metadata_bytes + destination * bank_bytes +
                    destination_index * 8,
                c.codebook.data() + metadata_bytes + source * bank_bytes +
                    source_index * 8,
                8);
        }
    }
    int bank_state_count[bank_count] = {0, 0, 0, 0};
    for (int source_state = 0; source_state < 16; ++source_state) {
        const uint8_t source_bank = static_cast<uint8_t>(
            c.codebook[36 + source_state]);
        const int rank_in_bank = bank_state_count[source_bank]++;
        if (rank_in_bank >= 4) {
            throw std::runtime_error(
                "NVQ2J-XL runtime bank has more than four scale states");
        }
        const int destination_bank = runtime_bank[source_bank];
        const int destination_state = rank_in_bank * bank_count + destination_bank;
        state_remap[source_state] = static_cast<uint8_t>(destination_state);
        std::memcpy(
            reordered.data() + 4 + destination_state * sizeof(uint16_t),
            c.codebook.data() + 4 + source_state * sizeof(uint16_t),
            sizeof(uint16_t));
        reordered[36 + destination_state] = static_cast<int8_t>(destination_bank);
    }
    for (int bank = 0; bank < bank_count; ++bank) {
        if (bank_state_count[bank] != 4) {
            throw std::runtime_error(
                "NVQ2J-XL runtime bank must have four scale states");
        }
    }
    return reordered;
}

static std::vector<uint8_t> remap_extended_e8_sub_scale_states(
        const NvqCpu & c,
        const std::array<uint8_t, 16> & state_remap) {
    if (c.format != 14 || c.sub_bits != 4) {
        throw std::runtime_error(
            "extended E8 scale-state remap requires 4-bit NVQ2J-XL states");
    }
    const size_t state_count = static_cast<size_t>(c.out) * c.ng;
    const size_t expected_bytes = (state_count + 1) / 2;
    if (c.sub_scale_packed.size() != expected_bytes) {
        throw std::runtime_error("NVQ2J-XL scale-state stream length mismatch");
    }
    std::vector<uint8_t> remapped(expected_bytes, 0);
    mfq_parallel_for(0, static_cast<int64_t>(expected_bytes), 4096,
        [&](int64_t begin, int64_t end) {
            for (int64_t byte = begin; byte < end; ++byte) {
                const uint8_t source = c.sub_scale_packed[byte];
                const size_t low_state = static_cast<size_t>(byte) * 2;
                const uint8_t low = state_remap[source & 0x0fu];
                const uint8_t high = low_state + 1 < state_count
                    ? state_remap[source >> 4]
                    : 0;
                remapped[byte] = low | static_cast<uint8_t>(high << 4);
            }
        });
    return remapped;
}

NvqWeight to_device_nvq(const NvqCpu & c, bool cuda) {
    NvqWeight w;
    std::array<uint8_t, 16> e8_state_remap = {};
    auto e8_index_remap =
        std::make_unique<std::array<std::array<uint16_t, 4096>, 4>>();
    std::vector<int8_t> e8_runtime_codebook;
    std::vector<uint8_t> e8_runtime_sub_scale;
    const bool use_extended_e8 =
        cuda && c.format == 14 && extended_jsc_group_layout_enabled();
    if (use_extended_e8) {
        e8_runtime_codebook = reorder_extended_e8_codebook_for_stage3(
            c, e8_state_remap, *e8_index_remap);
        e8_runtime_sub_scale = remap_extended_e8_sub_scale_states(
            c, e8_state_remap);
    }
    w.format = c.format;
    w.kernel_format = c.format;
    w.sign_mode = c.sign_mode;
    w.sub_bits = c.sub_bits;
    w.gs = c.gs;
    w.out = c.out;
    w.ng = c.ng;
    w.neuron_len = c.neuron_len;
    w.shape = c.shape;
    if (cuda && (c.format == 14 || c.format == 15) &&
        extended_jsc_group_layout_enabled()) {
        auto metadata = repack_extended_jsc_group_metadata(
            c,
            use_extended_e8 ? &e8_state_remap : nullptr,
            use_extended_e8 ? e8_index_remap.get() : nullptr);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = mfq_tensor_backend::empty(
            {0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8));
        w.kernel_format = c.format == 14
            ? kNvq2JscXlGroupExecKernelFormat
            : kNvq3JscLGroupExecKernelFormat;
    } else if (cuda && (c.format == 2 || c.format == 5) &&
               nvq2_exec_layout_enabled()) {
        auto metadata = repack_nvq2_exec_metadata(c);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = mfq_tensor_backend::empty(
            {0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8));
        w.kernel_format = c.format == 5 ? 6 : 4;
    } else if (
        c.format == 10 && c.codebook.size() >= 64 &&
        c.codebook[3] == 1 && c.codebook[1] == 2) {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
        w.kernel_format = 11;
    } else {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
    }
    w.sub_scale_packed = use_extended_e8
        ? cpu_u8_tensor(
            e8_runtime_sub_scale, {(int64_t)e8_runtime_sub_scale.size()})
        : cpu_u8_tensor(
            c.sub_scale_packed, {(int64_t)c.sub_scale_packed.size()});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    if (
        c.format == 5 || c.format == 7 || c.format == 9 ||
        c.format == 10 || c.format == 12 ||
        c.format == 13 || c.format == 14 || c.format == 15) {
        if (use_extended_e8) {
            w.codebook = cpu_i8_tensor(
                e8_runtime_codebook, {(int64_t)e8_runtime_codebook.size()});
        } else {
            w.codebook = cpu_i8_tensor(
                c.codebook, {(int64_t)c.codebook.size()});
        }
    } else {
        const int dims = nvq_vector_size(c.format);
        const int entries = c.format == 1 ? 2048 : (c.format == 8 ? 1024 : 256);
        w.codebook = cpu_i8_tensor(c.codebook, {entries, dims});
    }
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.indices_packed = w.indices_packed.to(target).contiguous();
        w.aux_packed = w.aux_packed.to(target).contiguous();
        w.sub_scale_packed =
            w.sub_scale_packed.to(target).contiguous();
        w.neuron_scale = w.neuron_scale.to(target).contiguous();
        w.codebook = w.codebook.to(target).contiguous();
        (void)w.workspace(1);
    }
    return w;
}

NvqWeight to_gpu_nvq(const NvqCpu & c) {
    return to_device_nvq(c, true);
}

NvqWeight to_cuda_device_nvq(
        const NvqCpu & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nvq(c, true);
}

NvqWeight to_cpu_nvq(const NvqCpu & c) {
    return to_device_nvq(c, false);
}
static size_t packed_nbytes(size_t count, int bits) {
    return bits == 0 ? 0 : (count * (size_t)bits + 7) / 8;
}

static std::vector<uint16_t> unpack_nepq_u16_bits(
        const std::vector<uint8_t> & blob,
        size_t & off,
        size_t count,
        int bits,
        const char * label) {
    const size_t nbytes = packed_nbytes(count, bits);
    if (off + nbytes > blob.size()) {
        throw std::runtime_error(std::string("truncated ") + label);
    }
    std::vector<uint16_t> result(count, 0);
    for (size_t index = 0; index < count; ++index) {
        const size_t first_bit = index * static_cast<size_t>(bits);
        uint16_t value = 0;
        for (int bit = 0; bit < bits; ++bit) {
            const size_t source_bit = first_bit + static_cast<size_t>(bit);
            value |= static_cast<uint16_t>(
                ((blob[off + source_bit / 8] >> (source_bit & 7)) & 1u)
                << bit);
        }
        result[index] = value;
    }
    off += nbytes;
    return result;
}

static void configure_nepq_profile(NepqCpu & value, int profile) {
    if (profile == 0) {
        value.profile = 0;
        value.format = 9;
        value.state_bits = 2;
        value.index_bits = 6;
        value.aux_bits = 0;
        value.table_bytes = 320;
        value.runtime_table_bytes = 320;
    } else if (profile == 1) {
        value.profile = 1;
        value.format = 7;
        value.state_bits = 3;
        value.index_bits = 7;
        value.aux_bits = 0;
        value.table_bytes = 832;
        value.runtime_table_bytes = 832;
    } else if (profile == 2) {
        value.profile = 2;
        value.format = 8;
        value.state_bits = 4;
        value.index_bits = 9;
        value.aux_bits = 1;
        value.table_bytes = 2048;
        value.runtime_table_bytes = 1024 * 8;
    } else if (profile == 3) {
        value.profile = 3;
        value.format = 1;
        value.state_bits = 3;
        value.index_bits = 11;
        value.aux_bits = 1;
        value.table_bytes = 4096;
        value.runtime_table_bytes = 2048 * 8;
    } else if (profile == 4) {
        value.profile = 4;
        value.format = 9;
        value.state_bits = 2;
        value.index_bits = 6;
        value.aux_bits = 0;
        value.table_bytes = 320;
        value.runtime_table_bytes = 320;
        value.residual = true;
        value.residual_position_bits = 5;
        value.residual_record_bits = 15;
        value.residual_block_vectors = 24;
    } else if (profile == 5) {
        value.profile = 5;
        value.format = 8;
        value.state_bits = 4;
        value.index_bits = 9;
        value.aux_bits = 1;
        value.table_bytes = 2048;
        value.runtime_table_bytes = 1024 * 8;
        value.residual = true;
        value.residual_second = true;
        value.residual_position_bits = 4;
        value.residual_record_bits = 14;
        value.residual_block_vectors = 16;
    } else {
        throw std::runtime_error("unsupported NEPQ cohort profile");
    }
}

static std::vector<int8_t> expand_nepq_table(
        const uint8_t * source, const NepqCpu & value) {
    std::vector<uint8_t> packed(
        source, source + static_cast<ptrdiff_t>(value.table_bytes));
    if (value.profile == 0 || value.profile == 4) {
        const uint8_t expected[6] = {2, 4, 3, 3, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (packed[(size_t)i] != expected[i]) {
                throw std::runtime_error("unsupported NEPQ0-S table profile");
            }
        }
        std::vector<int8_t> result(value.table_bytes);
        std::memcpy(result.data(), packed.data(), packed.size());
        return result;
    }
    if (value.profile == 1) {
        const uint8_t expected[6] = {1, 8, 3, 4, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (packed[(size_t)i] != expected[i]) {
                throw std::runtime_error("unsupported NEPQ0-L table profile");
            }
        }
        std::vector<int8_t> result(value.table_bytes);
        std::memcpy(result.data(), packed.data(), packed.size());
        return result;
    }
    std::vector<uint16_t> words((size_t)value.table_bytes / 2);
    std::memcpy(words.data(), packed.data(), packed.size());
    return expand_nvq_codebook(
        (value.profile == 2 || value.profile == 5) ? 8 : 1,
        words.data(), words.size());
}

NepqCpu unpack_nepq(
        const std::vector<uint8_t> & blob,
        const std::string & dtype,
        const std::vector<uint8_t> & runtime_payload) {
    if (blob.size() < 36 || std::memcmp(blob.data(), "NEP1", 4) != 0) {
        throw std::runtime_error("invalid NEPQ cohort header");
    }
    if (dtype != "NEPQ") {
        throw std::runtime_error("unsupported NEPQ cohort dtype: " + dtype);
    }
    NepqCpu value;
    size_t off = 4;
    const uint8_t version = blob[off++];
    const uint8_t profile = blob[off++];
    configure_nepq_profile(value, profile);
    const uint8_t groups_per_supergroup = blob[off++];
    const uint8_t flags = blob[off++];
    value.n_experts = (int)read_u32_from(blob, off);
    value.out_per_expert = (int)read_u32_from(blob, off);
    value.neuron_len = (int)read_u32_from(blob, off);
    value.bank_count = (int)read_u32_from(blob, off);
    value.rotation_block = (int)read_u32_from(blob, off);
    value.rotation_seed = read_u64_from(blob, off);
    if (version != 1 || profile != value.profile ||
        groups_per_supergroup != 4 || (flags & ~1u) != 0 ||
        ((flags & 1u) != 0) != (value.rotation_block != 0)) {
        throw std::runtime_error("unsupported NEPQ cohort profile");
    }
    if (value.n_experts <= 0 || value.out_per_expert <= 0 ||
        value.neuron_len <= 0 || value.neuron_len % 8 != 0 ||
        value.bank_count <= 0 || value.bank_count > 256 ||
        value.n_experts >
            std::numeric_limits<int>::max() / value.out_per_expert) {
        throw std::runtime_error("invalid NEPQ cohort dimensions");
    }
    if (value.rotation_block != 0 &&
        ((value.rotation_block & (value.rotation_block - 1)) != 0 ||
         value.neuron_len % value.rotation_block != 0)) {
        throw std::runtime_error("invalid NEPQ Hadamard block");
    }
    if (value.residual && value.rotation_block == 0) {
        throw std::runtime_error("NEPQ-A requires a Hadamard rotation");
    }
    value.ng = (value.neuron_len + 23) / 24;
    value.nvec = value.neuron_len / 8;
    value.nsuper = (value.ng + 3) / 4;
    const int rows = value.n_experts * value.out_per_expert;

    const size_t all_table_bytes =
        (size_t)value.bank_count * value.table_bytes;
    if (off + all_table_bytes > blob.size()) {
        throw std::runtime_error("truncated NEPQ table pool");
    }
    value.table_pool.reserve(
        (size_t)value.bank_count * value.runtime_table_bytes);
    value.grouped_table_pool.reserve(
        (size_t)value.bank_count *
        ((value.profile == 0 || value.profile == 4)
            ? 2112 : value.runtime_table_bytes));
    for (int bank = 0; bank < value.bank_count; ++bank) {
        const uint8_t * table = blob.data() + off + (size_t)bank * value.table_bytes;
        auto runtime = expand_nepq_table(table, value);
        value.table_pool.insert(
            value.table_pool.end(), runtime.begin(), runtime.end());
        if (value.profile == 0 || value.profile == 4) {
            std::vector<uint8_t> compact(
                table, table + static_cast<ptrdiff_t>(value.table_bytes));
            auto grouped = expand_npq0_s_runtime_lut(compact);
            value.grouped_table_pool.insert(
                value.grouped_table_pool.end(), grouped.begin(), grouped.end());
        } else {
            value.grouped_table_pool.insert(
                value.grouped_table_pool.end(), runtime.begin(), runtime.end());
        }
    }
    off += all_table_bytes;
    const size_t anchor_bytes = (size_t)rows * 2;
    if (off + anchor_bytes > blob.size()) {
        throw std::runtime_error("truncated NEPQ neuron anchors");
    }
    value.neuron_scale_h.resize((size_t)rows);
    std::memcpy(value.neuron_scale_h.data(), blob.data() + off, anchor_bytes);
    for (uint16_t raw : value.neuron_scale_h) {
        if ((raw & 0x7c00u) == 0x7c00u ||
                ((raw & 0x8000u) != 0 && (raw & 0x7fffu) != 0)) {
            throw std::runtime_error(
                "NEPQ neuron anchors must be finite and non-negative");
        }
    }
    off += anchor_bytes;
    value.state_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.ng, value.state_bits),
        "NEPQ state stream");
    value.indices_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.nvec, value.index_bits),
        "NEPQ index stream");
    value.aux_packed = take_bytes(
        blob, off, packed_nbytes((size_t)rows * value.ng, value.aux_bits),
        "NEPQ aux stream");
    value.bank_ids = take_bytes(
        blob, off, (size_t)rows * value.nsuper, "NEPQ bank selectors");
    if (value.residual) {
        const size_t header = off;
        if (header + 64 > blob.size() ||
            std::memcmp(blob.data() + header, "NRA1", 4) != 0) {
            throw std::runtime_error("invalid NEPQ-A residual header");
        }
        off += 4;
        const uint8_t residual_version = blob[off++];
        const uint8_t record_bits = blob[off++];
        const uint8_t position_bits = blob[off++];
        const uint8_t residual_flags = blob[off++];
        const uint32_t dictionary_entries = read_u32_from(blob, off);
        const uint32_t block_count = read_u32_from(blob, off);
        const uint32_t second_count = read_u32_from(blob, off);
        const uint32_t padding_nbytes = read_u32_from(blob, off);
        const uint64_t reserved = read_u64_from(blob, off);
        value.residual_blocks_per_row =
            (value.nvec + value.residual_block_vectors - 1) /
            value.residual_block_vectors;
        const size_t expected_blocks =
            static_cast<size_t>(rows) * value.residual_blocks_per_row;
        const uint8_t expected_flags = value.residual_second ? 1 : 0;
        if (residual_version != 1 ||
            record_bits != value.residual_record_bits ||
            position_bits != value.residual_position_bits ||
            residual_flags != expected_flags ||
            dictionary_entries != 1024 ||
            block_count != expected_blocks ||
            reserved != 0 ||
            (!value.residual_second && second_count != 0)) {
            throw std::runtime_error("unsupported NEPQ-A residual profile");
        }
        off = header + 64;
        const size_t dictionary_values = 1024 * 8;
        const size_t dictionary_nbytes = dictionary_values * sizeof(uint16_t);
        if (off + dictionary_nbytes > blob.size()) {
            throw std::runtime_error("truncated NEPQ-A residual dictionary");
        }
        value.residual_codebook_h.resize(dictionary_values);
        std::memcpy(
            value.residual_codebook_h.data(), blob.data() + off,
            dictionary_nbytes);
        for (size_t row = 0; row < 1024; ++row) {
            for (size_t column = 0; column < 8; ++column) {
                const uint16_t raw = value.residual_codebook_h[row * 8 + column];
                if ((raw & 0x7c00u) == 0x7c00u) {
                    throw std::runtime_error(
                        "NEPQ-A residual dictionary must be finite");
                }
            }
        }
        off += dictionary_nbytes;
        auto first = unpack_nepq_u16_bits(
            blob, off, expected_blocks, value.residual_record_bits,
            "NEPQ-A first residual stream");
        value.residual_first.assign(first.begin(), first.end());
        value.residual_second_dense.assign(expected_blocks, -1);
        if (value.residual_second) {
            auto mask = unpack_nepq_u16_bits(
                blob, off, expected_blocks, 1,
                "NEPQ-A second residual bitmap");
            auto second = unpack_nepq_u16_bits(
                blob, off, second_count, value.residual_record_bits,
                "NEPQ-A second residual stream");
            size_t compact = 0;
            for (size_t block = 0; block < expected_blocks; ++block) {
                if (mask[block] != 0) {
                    if (compact >= second.size()) {
                        throw std::runtime_error(
                            "NEPQ-A second residual count mismatch");
                    }
                    value.residual_second_dense[block] = second[compact++];
                }
            }
            if (compact != second.size()) {
                throw std::runtime_error(
                    "NEPQ-A second residual count mismatch");
            }
        }
        auto validate_record = [&](uint16_t record, size_t block) {
            const uint32_t position_mask =
                (1u << value.residual_position_bits) - 1u;
            const uint32_t position = record & position_mask;
            const uint32_t dictionary_id =
                record >> value.residual_position_bits;
            const size_t block_in_row =
                block % value.residual_blocks_per_row;
            const int available = std::min(
                value.residual_block_vectors,
                value.nvec - static_cast<int>(block_in_row) *
                    value.residual_block_vectors);
            if (dictionary_id >= 1024 || position >=
                    static_cast<uint32_t>(available)) {
                throw std::runtime_error(
                    "NEPQ-A residual record is out of range");
            }
        };
        for (size_t block = 0; block < expected_blocks; ++block) {
            validate_record(value.residual_first[block], block);
            if (value.residual_second_dense[block] >= 0) {
                validate_record(
                    static_cast<uint16_t>(
                        value.residual_second_dense[block]),
                    block);
            }
        }
        if (padding_nbytes > blob.size() - off) {
            throw std::runtime_error("truncated NEPQ-A residual padding");
        }
        const size_t padding_end = off + padding_nbytes;
        for (; off < padding_end; ++off) {
            if (blob[off] != 0) {
                throw std::runtime_error(
                    "NEPQ-A residual padding must be zero");
            }
        }
        const uint32_t position_mask =
            (1u << value.residual_position_bits) - 1u;
        for (size_t block = 0; block < expected_blocks; ++block) {
            const int block_in_row =
                static_cast<int>(block % value.residual_blocks_per_row);
            const int available = std::min(
                value.residual_block_vectors,
                value.nvec - block_in_row * value.residual_block_vectors);
            for (int record : {
                     value.residual_first[block],
                     value.residual_second_dense[block]}) {
                if (record < 0) continue;
                const int position = record & position_mask;
                const int dictionary_id =
                    record >> value.residual_position_bits;
                if (position >= available || dictionary_id >= 1024) {
                    throw std::runtime_error(
                        "invalid NEPQ-A residual record");
                }
            }
        }
    }
    if (off != blob.size()) throw std::runtime_error("invalid NEPQ cohort tail");
    for (uint8_t bank : value.bank_ids) {
        if ((int)bank >= value.bank_count) {
            throw std::runtime_error("NEPQ selector references a missing bank");
        }
    }

    if (value.rotation_block == 0) {
        if (!runtime_payload.empty()) {
            throw std::runtime_error("unexpected NEPQ rotation metadata");
        }
    } else {
        if (runtime_payload.size() != 20 + (size_t)value.neuron_len ||
            std::memcmp(runtime_payload.data(), "HSG1", 4) != 0) {
            throw std::runtime_error("missing NEPQ rotation sign vector");
        }
        size_t runtime_off = 4;
        uint32_t width = read_u32_from(runtime_payload, runtime_off);
        uint32_t block = read_u32_from(runtime_payload, runtime_off);
        uint64_t seed = read_u64_from(runtime_payload, runtime_off);
        if ((int)width != value.neuron_len ||
            (int)block != value.rotation_block || seed != value.rotation_seed) {
            throw std::runtime_error("NEPQ rotation metadata mismatch");
        }
        value.rotation_signs.resize((size_t)value.neuron_len);
        std::memcpy(
            value.rotation_signs.data(), runtime_payload.data() + runtime_off,
            (size_t)value.neuron_len);
        for (int8_t sign : value.rotation_signs) {
            if (sign != -1 && sign != 1) {
                throw std::runtime_error("invalid NEPQ rotation sign");
            }
        }
    }
    return value;
}
NepqWeight to_device_nepq(const NepqCpu & cpu, bool cuda) {
    NepqWeight value;
    value.format = cpu.format;
    value.state_bits = cpu.state_bits;
    value.n_experts = cpu.n_experts;
    value.out_per_expert = cpu.out_per_expert;
    value.neuron_len = cpu.neuron_len;
    value.ng = cpu.ng;
    value.rotation_block = cpu.rotation_block;
    value.rotation_seed = cpu.rotation_seed;
    value.residual = cpu.residual;
    value.residual_position_bits = cpu.residual_position_bits;
    value.residual_block_vectors = cpu.residual_block_vectors;
    value.indices_packed = cpu_u8_tensor(
        cpu.indices_packed, {(int64_t)cpu.indices_packed.size()});
    value.aux_packed = cpu_u8_tensor(
        cpu.aux_packed, {(int64_t)cpu.aux_packed.size()});
    value.state_packed = cpu_u8_tensor(
        cpu.state_packed, {(int64_t)cpu.state_packed.size()});
    value.neuron_scale = cpu_f16_to_f32_tensor(
        cpu.neuron_scale_h, cpu.n_experts * cpu.out_per_expert);
    value.table_pool = cpu_i8_tensor(
        cpu.table_pool, {cpu.bank_count, cpu.runtime_table_bytes});
    const int grouped_stride =
        (cpu.profile == 0 || cpu.profile == 4)
        ? 2112 : cpu.runtime_table_bytes;
    value.grouped_table_pool = cpu_i8_tensor(
        cpu.grouped_table_pool, {cpu.bank_count, grouped_stride});
    value.bank_ids = cpu_u8_tensor(
        cpu.bank_ids, {
            (int64_t)cpu.n_experts * cpu.out_per_expert, cpu.nsuper});
    value.rotation_signs = cpu_i8_tensor(
        cpu.rotation_signs, {(int64_t)cpu.rotation_signs.size()});
    if (cpu.residual) {
        value.residual_codebook = cpu_f16_tensor(
            cpu.residual_codebook_h, {1024, 8});
        value.residual_first = mfq_tensor_backend::from_blob(
            (void *)cpu.residual_first.data(),
            {cpu.n_experts * cpu.out_per_expert,
             cpu.residual_blocks_per_row},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt16)).clone();
        value.residual_second = mfq_tensor_backend::from_blob(
            (void *)cpu.residual_second_dense.data(),
            {cpu.n_experts * cpu.out_per_expert,
             cpu.residual_blocks_per_row},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt16)).clone();
    }
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device());
        value.indices_packed =
            value.indices_packed.to(target).contiguous();
        value.aux_packed =
            value.aux_packed.to(target).contiguous();
        value.state_packed =
            value.state_packed.to(target).contiguous();
        value.neuron_scale =
            value.neuron_scale.to(target).contiguous();
        value.table_pool =
            value.table_pool.to(target).contiguous();
        value.grouped_table_pool =
            value.grouped_table_pool.to(target).contiguous();
        value.bank_ids =
            value.bank_ids.to(target).contiguous();
        value.rotation_signs =
            value.rotation_signs.to(target).contiguous();
        if (cpu.residual) {
            value.residual_codebook =
                value.residual_codebook.to(target).contiguous();
            value.residual_first =
                value.residual_first.to(target).contiguous();
            value.residual_second =
                value.residual_second.to(target).contiguous();
        }
    }
    return value;
}

NepqWeight to_gpu_nepq(const NepqCpu & cpu) {
    return to_device_nepq(cpu, true);
}

NepqWeight to_cpu_nepq(const NepqCpu & cpu) {
    return to_device_nepq(cpu, false);
}
NvqCpu select_nvq_cpu_rows(
        const NvqCpu & source,
        const std::vector<int64_t> & rows) {
    if (source.shape.size() != 2 ||
        source.axis != 0 ||
        source.shape[0] != source.out ||
        rows.empty()) {
        throw std::runtime_error(
            "invalid NVQ MoE output shard source");
    }
    NvqCpu result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.neuron_scale_h.resize(rows.size());
    const int index_bits =
        nvq_index_bits(source.format);
    const bool delta =
        nvq_delta_format(source.format);
    const bool no_aux =
        nvq_no_aux_format(source.format);
    const int aux_items =
        delta ? source.ng :
        (no_aux ? 0 : source.nsign);
    const int aux_bits =
        delta ? 1 : (no_aux ? 0 : 7);
    result.sub_scale_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.sub_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(aux_items),
            aux_bits),
        0);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NVQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.sub_scale_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.sub_bits,
            result.sub_scale_packed,
            destination * source.ng *
                source.sub_bits,
            static_cast<size_t>(source.ng) *
                source.sub_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * index_bits,
            result.indices_packed,
            destination * source.nvec *
                index_bits,
            static_cast<size_t>(source.nvec) *
                index_bits);
        if (aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    aux_items * aux_bits,
                result.aux_packed,
                destination * aux_items *
                    aux_bits,
                static_cast<size_t>(aux_items) *
                    aux_bits);
        }
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
    }
    return result;
}

NepqCpu select_nepq_cpu_rows(
        const NepqCpu & source,
        const std::vector<int64_t> & rows,
        int output_per_expert,
        int selected_experts) {
    const int destination_experts = selected_experts < 0
        ? source.n_experts
        : selected_experts;
    const int source_rows =
        source.n_experts *
        source.out_per_expert;
    if (rows.empty() ||
        output_per_expert <= 0 ||
        destination_experts <= 0 ||
        static_cast<int>(rows.size()) !=
            destination_experts *
                output_per_expert) {
        throw std::runtime_error(
            "invalid NEPQ MoE output shard");
    }
    NepqCpu result = source;
    result.n_experts = destination_experts;
    result.out_per_expert = output_per_expert;
    result.neuron_scale_h.resize(rows.size());
    result.state_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.state_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            source.index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.aux_bits),
        0);
    result.bank_ids.resize(
        rows.size() * source.nsuper);
    if (source.residual) {
        result.residual_first.resize(
            rows.size() * source.residual_blocks_per_row);
        result.residual_second_dense.resize(
            rows.size() * source.residual_blocks_per_row);
    }
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source_rows) {
            throw std::runtime_error(
                "NEPQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.state_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.state_bits,
            result.state_packed,
            destination * source.ng *
                source.state_bits,
            static_cast<size_t>(source.ng) *
                source.state_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * source.index_bits,
            result.indices_packed,
            destination * source.nvec *
                source.index_bits,
            static_cast<size_t>(source.nvec) *
                source.index_bits);
        if (source.aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    source.ng * source.aux_bits,
                result.aux_packed,
                destination * source.ng *
                    source.aux_bits,
                static_cast<size_t>(source.ng) *
                    source.aux_bits);
        }
        std::memcpy(
            result.bank_ids.data() +
                destination * source.nsuper,
            source.bank_ids.data() +
                static_cast<size_t>(source_row) *
                    source.nsuper,
            static_cast<size_t>(source.nsuper));
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
        if (source.residual) {
            std::memcpy(
                result.residual_first.data()
                    + destination * source.residual_blocks_per_row,
                source.residual_first.data()
                    + static_cast<size_t>(source_row)
                        * source.residual_blocks_per_row,
                static_cast<size_t>(source.residual_blocks_per_row)
                    * sizeof(int16_t));
            std::memcpy(
                result.residual_second_dense.data()
                    + destination * source.residual_blocks_per_row,
                source.residual_second_dense.data()
                    + static_cast<size_t>(source_row)
                        * source.residual_blocks_per_row,
                static_cast<size_t>(source.residual_blocks_per_row)
                    * sizeof(int16_t));
        }
    }
    return result;
}
NvqWeight load_nvq_gpu(const mfq::ModelSource & mfq, const std::string & name) {
    const auto & rec = require_tensor(mfq, name);
    return to_gpu_nvq(unpack_nvq(read_tensor(mfq, name), rec.dtype));
}

mfq_tensor_backend::Tensor nvq_dequant(const NvqWeight & w) {
    return nvq_dequant_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}
bool nvq_fused_residual_format(std::int64_t kernel_format) {
    return kernel_format == kNvq2JscXlGroupExecKernelFormat ||
        kernel_format == kNvq3JscLGroupExecKernelFormat;
}
