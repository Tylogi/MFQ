constexpr char kShortWindow[] = R"CSA2(
inline int csa2_window_index(uint query, uint slot, constant int* meta) {
    const int position = meta[6], length = meta[7], window_size = meta[8];
    const int first = position == 0 ? max(0, int(query) - window_size + 1)
        : position + int(query) - window_size + 1;
    const int absolute = first + int(slot);
    const int origin = position - (meta[3] - length);
    if (absolute < max(0, origin) || absolute > position + int(query)) return -1;
    return length == 1 ? absolute % window_size : absolute - origin;
}
)CSA2";

constexpr char kShortMerge[] = R"CSA2(
    const uint lane = thread_index_in_simdgroup;
    const uint part = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint head = part / (D / 32), d = (part % (D / 32)) * 32 + lane;
    const uint query = threadgroup_position_in_grid.y;
    const int H = meta[0], NS = meta[5];
    if (head >= H) return;
    const size_t base = (size_t(query) * H + head) * NS * (D + 2);
    float maximum = float(sink[head]);
    for (int s = 0; s < NS; ++s) maximum = max(maximum, partial[base + s * (D + 2) + D]);
    float denominator = exp(float(sink[head]) - maximum), acc = 0.0f;
    for (int s = 0; s < NS; ++s) {
        const size_t pos = base + s * (D + 2);
        const float correction = exp(partial[pos + D] - maximum);
        denominator += partial[pos + D + 1] * correction;
        acc += partial[pos + d] * correction;
    }
    out[(query * H + head) * D + d] = half(acc / denominator);
)CSA2";

bool csa2_nax_supported(Stream stream) {
    if (mlx_metal_nax_disabled()) return false;
    if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
        const auto& device = kernel_device(stream.device);
        const auto& architecture = device.get_architecture();
        return !architecture.empty() && device.get_architecture_gen() >=
            (architecture.back() == 'p' ? 18 : 17);
    }
    return false;
}

class Csa2Short final : public Primitive, public MlxPreparableKernel {
public:
    Csa2Short(Stream stream, int dim, int position, int window, bool unpacked = false)
        : Primitive(stream), dim_(dim), position_(position), window_(window), unpacked_(unpacked) {
        nax_ = dim == 512 && csa2_nax_supported(stream);
    }

    std::string preparation_key() const override {
        return "mfq_csa2_short_f16_d" + std::to_string(dim_) + (nax_ ? "_nax_v1" : "_v1")
            + (unpacked_ ? "_unpacked" : "");
    }

    void prepare_gpu() override {
        auto& device = kernel_device(stream().device);
        auto* library = prepared_library();
        for (const auto* suffix : {"_shared", "_scalar", "_merge"})
            (void)device.get_kernel(preparation_key() + suffix, library);
        if (nax_) (void)device.get_kernel(preparation_key() + "_nax", library);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("CSA2 requires Metal");
    }

    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        const auto& query = inputs[0];
        const int length = query.shape(1), heads = query.shape(2);
        const int slots = position_ == 0 ? std::min(window_, length) : window_;
        const int compressed = inputs[3].shape(-1);
        const bool prefill = length > 8 && nax_;
        const int splits = prefill ? 1 : std::max(1, (slots + compressed + 31) / 32);
        const std::array<int, 9> params{heads, slots, compressed, inputs[1].shape(1),
            inputs[2].shape(1), splits, position_, length, window_};
        const float scale = 1.0f / std::sqrt(float(dim_));
        auto partial = array(prefill ? Shape{1} : Shape{1, length, heads, splits, dim_ + 2}, float32, nullptr, {});
        partial.set_data(allocator::malloc(partial.nbytes()));
        auto& output = outputs.front();
        output.set_data(allocator::malloc(output.nbytes()));
        auto& device = kernel_device(stream().device);
        auto* library = prepared_library();
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.add_temporary(partial);
        const bool shared = heads >= 8 && slots + compressed >= 512;
        const bool nax = nax_ && heads % 16 == 0 && (prefill || slots + compressed >= 512);
        encoder.set_compute_pipeline_state(device.get_kernel(
            preparation_key() + (nax ? "_nax" : shared ? "_shared" : "_scalar"), library));
        for (int index = 0; index < 4; ++index) encoder.set_input_array(inputs[index], index);
        encoder.set_bytes(params, 4);
        encoder.set_bytes(scale, 5);
        encoder.set_output_array(partial, 6);
        if (nax) {
            encoder.set_input_array(inputs[4], 7);
            encoder.set_output_array(output, 8);
        }
        const int group = shared ? 8 : 4;
        encoder.dispatch_threadgroups(nax ? MTL::Size(length * splits, heads / 16, 1)
            : MTL::Size((heads + group - 1) / group, length, splits), MTL::Size(128, 1, 1));
        if (prefill) return;
        encoder.set_compute_pipeline_state(device.get_kernel(preparation_key() + "_merge", library));
        encoder.set_input_array(partial, 0);
        encoder.set_input_array(inputs[4], 1);
        encoder.set_bytes(params, 2);
        encoder.set_output_array(output, 3);
        encoder.dispatch_threadgroups(MTL::Size((heads * (dim_ / 32) + 3) / 4, length, 1), MTL::Size(128, 1, 1));
    }

    const char* name() const override { return "Csa2Short"; }
    bool is_equivalent(const Primitive& other) const override {
        const auto& selected = static_cast<const Csa2Short&>(other);
        return dim_ == selected.dim_ && position_ == selected.position_ && window_ == selected.window_
            && nax_ == selected.nax_ && unpacked_ == selected.unpacked_;
    }

private:
    MTL::Library* prepared_library() {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return kernel_device(stream().device).get_library(preparation_key(), options, [this] {
            std::string source = nax_ ? "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                "#define UNROLL _Pragma(\"clang loop unroll(full)\")\n" : "";
            if (unpacked_) source += "#define CSA2_UNPACKED 1\n";
            source += std::string(kCsa2Header) + kShortWindow;
            if (nax_) source += kCsa2NaxRead;
            source += "\nconstant int D = " + std::to_string(dim_) + ";\nconstant int CHUNK = 32;\n";
            const std::string indices =
                "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]], "
                "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], "
                "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], "
                "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n";
            for (const auto& [suffix, body] : {
                     std::pair{"_shared", kCsa2AttentionMma}, std::pair{"_scalar", kCsa2Attention}}) {
                source += "kernel void " + preparation_key() + suffix + "("
                    "const device half* q [[buffer(0)]], const device uchar* window [[buffer(1)]], "
                    "const device uchar* pooled [[buffer(2)]], const device int* ci [[buffer(3)]], "
                    "constant int* meta [[buffer(4)]], constant float* scalep [[buffer(5)]], "
                    "device float* partial [[buffer(6)]], " + indices;
                std::string selected(body);
                const std::string from = "wi[query * W + j]";
                const std::string to = "csa2_window_index(query, uint(j), meta)";
                for (std::size_t at = 0; (at = selected.find(from, at)) != std::string::npos; at += to.size())
                    selected.replace(at, from.size(), to);
                source += selected + "\n}\n";
            }
            if (nax_) {
                source += "kernel void " + preparation_key() + "_nax("
                    "const device half* q [[buffer(0)]], const device uchar* window [[buffer(1)]], "
                    "const device uchar* pooled [[buffer(2)]], const device int* ci [[buffer(3)]], "
                    "constant int* meta [[buffer(4)]], constant float* scalep [[buffer(5)]], "
                    "device float* partial [[buffer(6)]], const device float* sink [[buffer(7)]], "
                    "device half* final_output [[buffer(8)]], " + indices;
                source += std::string(kCsa2Nax) + "\n}\n";
            }
            source += "kernel void " + preparation_key() + "_merge("
                "const device float* partial [[buffer(0)]], const device float* sink [[buffer(1)]], "
                "constant int* meta [[buffer(2)]], device half* out [[buffer(3)]], " + indices;
            source += std::string(kShortMerge) + "\n}\n";
            return source;
        });
    }

    int dim_, position_, window_;
    bool nax_ = false;
    bool unpacked_ = false;
};
