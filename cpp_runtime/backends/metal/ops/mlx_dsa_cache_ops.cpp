#include "mlx_dsa_cache_ops.h"
#include "mlx_kernel_prepare.h"

#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mfq::metal {
namespace {
using namespace mlx::core;

const fast::CustomKernelFunction& compress_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return fast::metal_kernel("mfq_dsa_compress_window",
            {"kv", "score", "old_kv", "old_score", "norm", "meta", "params"},
            {"latent", "new_kv", "new_score"}, R"MSL(
            const uint D = uint(kv_shape[2]), M = uint(kv_shape[1]);
            const uint R = uint(old_kv_shape[1]);
            const uint P = meta[0], C = (P + M) / R;
            const uint b = threadgroup_position_in_grid.y;
            const uint g = threadgroup_position_in_grid.x;
            const uint tid = thread_index_in_threadgroup;
            if (g == C) {
                for (uint i = tid; i < R * D; i += 256u) {
                    const uint slot = i / D, d = i % D;
                    const int last = int(M) - 1 - int((P + M - 1u + R - slot) % R);
                    const uint out = b * R * D + i;
                    new_kv[out] = last >= 0 ? kv[(b * M + uint(last)) * D + d] : old_kv[out];
                    new_score[out] = last >= 0 ? score[(b * M + uint(last)) * D + d] : old_score[out];
                }
                return;
            }
            threadgroup float pooled[512];
            threadgroup float sums[8];
            threadgroup float inverse;
            float square = 0.0f;
            for (uint d = tid; d < D; d += 256u) {
                float maximum = -INFINITY;
                for (uint j = 0; j < R; ++j) {
                    const int t = int(g * R + j) - int(P);
                    const float s = t >= 0 ? score[(b * M + uint(t)) * D + d]
                        : old_score[(b * R + j) * D + d];
                    maximum = max(maximum, s);
                }
                float total = 0.0f, value = 0.0f;
                for (uint j = 0; j < R; ++j) {
                    const int t = int(g * R + j) - int(P);
                    const float s = t >= 0 ? score[(b * M + uint(t)) * D + d]
                        : old_score[(b * R + j) * D + d];
                    total += metal::precise::exp(s - maximum);
                }
                for (uint j = 0; j < R; ++j) {
                    const int t = int(g * R + j) - int(P);
                    const float s = t >= 0 ? score[(b * M + uint(t)) * D + d]
                        : old_score[(b * R + j) * D + d];
                    const float x = t >= 0 ? kv[(b * M + uint(t)) * D + d]
                        : old_kv[(b * R + j) * D + d];
                    value += x * (metal::precise::exp(s - maximum) / total);
                }
                value = float(T(value));
                pooled[d] = value;
                square += value * value;
            }
            square = simd_sum(square);
            if (thread_index_in_simdgroup == 0u) sums[simdgroup_index_in_threadgroup] = square;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 32u) {
                const float total = simd_sum(tid < 8u ? sums[tid] : 0.0f);
                if (tid == 0u) inverse = metal::precise::rsqrt(total / float(D) + params[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = tid; d < D; d += 256u)
                latent[(b * C + g) * D + d] = T((pooled[d] * inverse) * norm[d]);
            )MSL", "", true, false, options);
    }();
    return kernel;
}

constexpr const char* kPackStoreSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline uchar dsa_fp8_bits(float f) {
    uint bits = as_type<uint>(f), signbit = bits & 0x80000000u;
    bits ^= signbit;
    uchar out;
    if (bits >= (543u << 21)) out = 0x7e;
    else if (bits < (121u << 23)) {
        const uint denorm = 141u << 23;
        out = uchar(as_type<uint>(as_type<float>(bits) + as_type<float>(denorm)) - denorm);
    } else {
        const uint odd = (bits >> 20) & 1u;
        bits += ((uint)(7 - 127) << 23) + 0x7ffff + odd;
        out = uchar(bits >> 20);
    }
    return out | uchar(signbit >> 24);
}
inline float dsa_fp8_round(float x) {
    const float a = min(abs(x), 448.0f);
    const int exp = max(int(floor(log2(max(a, 0x1p-9f)))) - 3, -9);
    const float step = as_type<float>(uint(exp + 127) << 23);
    return sign(x) * min(rint(a / step) * step, 448.0f);
}
struct Params { uint B, M, D, capacity, begin, rotary, gs, e4, normalize, emit; float eps; };
template <typename T>
kernel void dsa_pack_store(device uchar* cache [[buffer(0)]], device const T* x [[buffer(1)]],
    device const float* norm [[buffer(2)]], device const float* cosine [[buffer(3)]],
    device const float* sine [[buffer(4)]], device T* normalized [[buffer(5)]],
    constant Params& p [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float values[512];
    threadgroup float squares[8];
    threadgroup float inverse;
    float square = 0.0f;
    for (uint d = tid; d < p.D; d += 256u) {
        const float v = float(x[row * p.D + d]);
        square += v * v;
    }
    square = simd_sum(square);
    if (lane == 0u) squares[simd] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        const float total = simd_sum(tid < 8u ? squares[tid] : 0.0f);
        if (tid == 0u) inverse = p.normalize ? precise::rsqrt(total / float(p.D) + p.eps) : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < p.D; d += 256u) {
        const float v = float(x[row * p.D + d]);
        const T n = p.normalize ? T((v * inverse) * norm[d]) : T(v);
        values[d] = float(n);
        if (p.emit) normalized[row * p.D + d] = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row_bytes = p.D / 2u + p.D / p.gs;
    const uint base = ((row / p.M) * p.capacity + p.begin + row % p.M) * row_bytes;
    for (uint d = tid; d < p.D; d += 256u) {
        float v = values[d];
        if (d >= p.D - p.rotary) {
            const uint r = d - (p.D - p.rotary), pair = r / 2u;
            const uint a = p.D - p.rotary + pair * 2u;
            const float c = cosine[(row % p.M) * (p.rotary / 2u) + pair];
            const float s = sine[(row % p.M) * (p.rotary / 2u) + pair];
            v = float(T((r & 1u) ? fma(values[a], s, values[a + 1u] * c)
                : fma(values[a], c, -values[a + 1u] * s)));
        }
        float amax = abs(v);
        for (uint mask = 1u; mask < p.gs; mask <<= 1u)
            amax = max(amax, simd_shuffle_xor(amax, mask));
        amax = max(amax, 6.0f * (p.e4 ? 0x1p-9f : 0x1p-126f));
        const int exponent = max(int(ceil(log2(amax / 6.0f))), -126);
        const float scale = p.e4 ? dsa_fp8_round(amax / 6.0f)
            : as_type<float>(uint(exponent + 127) << 23);
        const float scaled = clamp(v / scale, -6.0f, 6.0f), a = abs(scaled);
        uint code = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f)
            + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
        code |= scaled < 0.0f ? 8u : 0u;
        const uint partner = simd_shuffle_xor(code, 1u);
        if ((d & 1u) == 0u) cache[base + d / 2u] = uchar(code | (partner << 4u));
        if (d % p.gs == 0u) cache[base + p.D / 2u + d / p.gs] =
            p.e4 ? dsa_fp8_bits(scale) : uchar(exponent + 127);
    }
}
template [[host_name("dsa_pack_store_f16")]] kernel decltype(dsa_pack_store<half>) dsa_pack_store<half>;
template [[host_name("dsa_pack_store_bf16")]] kernel decltype(dsa_pack_store<bfloat>) dsa_pack_store<bfloat>;
template [[host_name("dsa_pack_store_f32")]] kernel decltype(dsa_pack_store<float>) dsa_pack_store<float>;
)MSL";

struct PackParams {
    std::uint32_t B, M, D, capacity, begin, rotary, gs, e4, normalize, emit;
    float eps;
};

class DsaPackStore final : public Primitive, public MlxPreparableKernel {
public:
    DsaPackStore(Stream stream, int begin, int rotary, int gs, bool e4,
        float eps, bool normalize, bool emit, Dtype dtype)
        : Primitive(stream), begin_(begin), rotary_(rotary), gs_(gs), e4_(e4),
          eps_(eps), normalize_(normalize), emit_(emit), dtype_(dtype) {}
    std::string preparation_key() const override { return name_string(); }
    std::string name_string() const {
        return dtype_ == float16 ? "dsa_pack_store_f16" :
            dtype_ == bfloat16 ? "dsa_pack_store_bf16" : "dsa_pack_store_f32";
    }
    MTL::Library* library() {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return kernel_device(stream().device).get_library("mfq_dsa_pack_store_v1", options,
            [] { return std::string(kPackStoreSource); });
    }
    void prepare_gpu() override {
        (void)kernel_device(stream().device).get_kernel(name_string(), library());
    }
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("DSA packed cache update requires Metal");
    }
    void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
        out[0].copy_shared_buffer(in[0]);
        out[1].set_data(allocator::malloc(std::max<std::size_t>(out[1].nbytes(), 4)));
        PackParams p{std::uint32_t(in[1].shape(0)), std::uint32_t(in[1].shape(1)),
            std::uint32_t(in[1].shape(2)), std::uint32_t(in[0].shape(1)),
            std::uint32_t(begin_), std::uint32_t(rotary_), std::uint32_t(gs_),
            std::uint32_t(e4_), std::uint32_t(normalize_), std::uint32_t(emit_), eps_};
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel_device(stream().device).get_kernel(name_string(), library()));
        encoder.set_output_array(out[0], 0);
        for (int i = 1; i < 5; ++i) encoder.set_input_array(in[i], i);
        encoder.set_output_array(out[1], 5);
        encoder.set_bytes(p, 6);
        encoder.dispatch_threadgroups(MTL::Size(p.B * p.M, 1, 1), MTL::Size(256, 1, 1));
    }
    const char* name() const override { return "DsaPackStore"; }
    bool is_equivalent(const Primitive& other) const override {
        const auto& p = static_cast<const DsaPackStore&>(other);
        return begin_ == p.begin_ && rotary_ == p.rotary_ && gs_ == p.gs_ && e4_ == p.e4_ &&
            eps_ == p.eps_ && normalize_ == p.normalize_ && emit_ == p.emit_ && dtype_ == p.dtype_;
    }
private:
    int begin_, rotary_, gs_;
    bool e4_;
    float eps_;
    bool normalize_, emit_;
    Dtype dtype_;
};
}

MlxDsaCompressedWindow mlx_dsa_compress_window(const array& kv, const array& score,
    const array& partial_kv, const array& partial_score, const array& norm,
    int partial_length, Dtype dtype, float eps) {
    if (kv.ndim() != 3 || kv.shape() != score.shape() || partial_kv.ndim() != 3 ||
        partial_kv.shape() != partial_score.shape() || partial_kv.shape(0) != kv.shape(0) ||
        partial_kv.shape(2) != kv.shape(2) || kv.shape(2) > 512 || kv.shape(2) <= 0 ||
        kv.shape(1) <= 0 || partial_kv.shape(1) != 2 || partial_length < 0 || partial_length >= 2 ||
        norm.size() != kv.shape(2) || !(eps > 0) ||
        (dtype != float16 && dtype != bfloat16 && dtype != float32))
        throw std::invalid_argument("invalid DSA compression window");
    const int count = (partial_length + kv.shape(1)) / 2;
    auto result = compress_kernel()({contiguous(astype(kv, float32)), contiguous(astype(score, float32)),
        contiguous(astype(partial_kv, float32)), contiguous(astype(partial_score, float32)),
        contiguous(astype(norm, float32)), array({partial_length}, int32), array({eps})},
        {{kv.shape(0), count, kv.shape(2)}, partial_kv.shape(), partial_score.shape()},
        {dtype, float32, float32}, {(count + 1) * 256, kv.shape(0), 1}, {256, 1, 1},
        {{"T", dtype}}, std::nullopt, false, {});
    return {std::move(result[0]), std::move(result[1]), std::move(result[2])};
}

MlxDsaPackedCacheUpdate mlx_dsa_norm_rope_pack_store(const array& cache, const array& values,
    const array& norm, const array& cosine, const array& sine, int begin, int rotary,
    int group_size, bool e4m3_scale, float eps, bool normalize, bool return_normalized) {
    if (values.ndim() != 3 || cache.ndim() != 3 || cache.dtype() != uint8 ||
        values.shape(0) <= 0 || values.shape(1) <= 0 || !(eps > 0) ||
        values.shape(2) <= 0 || values.shape(2) > 512 || values.shape(0) != cache.shape(0) ||
        (group_size != 16 && group_size != 32) || values.shape(2) % group_size ||
        cache.shape(2) != values.shape(2) / 2 + values.shape(2) / group_size ||
        begin < 0 || begin + values.shape(1) > cache.shape(1) || rotary < 0 ||
        rotary > values.shape(2) || rotary % 2 || norm.size() != values.shape(2) ||
        cosine.shape() != sine.shape() || cosine.size() != values.shape(1) * rotary / 2 ||
        (values.dtype() != float16 && values.dtype() != bfloat16 && values.dtype() != float32))
        throw std::invalid_argument("invalid DSA packed cache transform");
    auto result = array::make_arrays({cache.shape(), return_normalized ? values.shape() : Shape{0}},
        {uint8, values.dtype()}, std::make_shared<DsaPackStore>(default_stream(default_device()),
            begin, rotary, group_size, e4m3_scale, eps, normalize, return_normalized, values.dtype()),
        {cache, contiguous(values), contiguous(astype(norm, float32)),
            contiguous(astype(cosine, float32)), contiguous(astype(sine, float32))});
    return {std::move(result[0]), std::move(result[1])};
}
}
