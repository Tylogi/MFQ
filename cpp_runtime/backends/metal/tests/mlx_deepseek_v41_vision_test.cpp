#include "mlx_axial_patch_vision.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void append(std::string& bytes, T value) {
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void append_string(std::string& bytes, const std::string& value) {
    append<std::uint32_t>(bytes, static_cast<std::uint32_t>(value.size()));
    bytes += value;
}

std::vector<float> values(int count, int salt) {
    std::vector<float> result(count);
    for (int index = 0; index < count; ++index)
        result[index] = static_cast<float>((index * 7 + salt) % 37 - 18) * 0.01f;
    return result;
}

struct Fixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("mfq-v41-vision-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".mfq");
    std::vector<std::pair<std::string, std::string>> records;

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    void add(const std::string& name, const std::vector<std::int64_t>& shape,
        const std::vector<float>& data) {
        std::string blob;
        append<std::uint32_t>(blob, static_cast<std::uint32_t>(shape.size()));
        for (auto dimension : shape) append<std::int64_t>(blob, dimension);
        for (auto value : data) append<float>(blob, value);
        records.emplace_back(name, std::move(blob));
    }

    void affine(const std::string& name, int output, int input, int salt, bool bias = true) {
        add(name + ".weight", {output, input}, values(output * input, salt));
        if (bias) add(name + ".bias", {output}, values(output, salt + 1));
    }

    void norm(const std::string& name) {
        std::vector<float> data(8);
        for (int index = 0; index < 8; ++index) data[index] = 1.0f + (index % 3) * 0.02f;
        add(name + ".weight", {8}, data);
    }

    void write() const {
        std::string bytes("MFQ1", 4);
        append<std::uint32_t>(bytes, 2);
        append_string(bytes, "unit-test");
        append<std::uint32_t>(bytes, 0);
        append<std::uint32_t>(bytes, static_cast<std::uint32_t>(records.size()));
        for (const auto& [name, blob] : records) {
            append_string(bytes, name);
            append_string(bytes, "F32");
            append<std::uint64_t>(bytes, blob.size());
        }
        for (const auto& record : records) bytes += record.second;
        std::ofstream stream(path, std::ios::binary);
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw std::runtime_error("cannot write vision fixture");
    }
};

void test_official_vision_reference() {
    Fixture fixture;
    fixture.affine("vision.patch_embedding", 8, 12, 1);
    fixture.norm("vision.block.0.norm1");
    fixture.affine("vision.block.0.attention.qkv", 24, 8, 3);
    fixture.affine("vision.block.0.attention.output", 8, 8, 5);
    fixture.norm("vision.block.0.norm2");
    fixture.affine("vision.block.0.mlp.gate_up", 24, 8, 7, false);
    fixture.affine("vision.block.0.mlp.down", 8, 12, 8, false);
    fixture.norm("vision.output_norm");
    fixture.affine("vision.aligner.input", 16, 72, 9);
    fixture.affine("vision.aligner.output", 16, 16, 11);
    fixture.write();
    const mfq::metal::MfqContainer model(fixture.path);
    const mfq::metal::MlxAxialPatchVisionConfig config{8, 1, 1, 12, 2, 3, 16, 10000.0, 1e-6};
    auto tower = mfq::metal::MlxAxialPatchVisionTower::load(model, config);
    const auto patches = values(240, 13);
    auto output = tower.encode(mlx::core::array(patches.begin(), mlx::core::Shape{20, 12}), 4, 5);
    output.eval();
    constexpr std::array<float, 64> expected{
        0.0371742807f, 0.121525779f, 0.0961211026f, 0.180472583f,
        -0.105175875f, -0.274209738f, -0.189858243f, -0.166728422f,
        -0.0823769271f, -0.306721926f, -0.222370416f, -0.138018921f,
        -0.0597964302f, 0.0245550871f, 0.0799301118f, -0.205718368f,
        -0.199794695f, -0.107896425f, -0.199546829f, -0.107648537f,
        -0.385750234f, -0.236529350f, -0.144631088f, 0.00735121593f,
        0.0992494822f, -0.149375558f, -0.0574772581f, 0.0344210193f,
        0.0475550741f, 0.139453351f, 0.0764370263f, -0.201664686f,
        0.142890066f, 0.222523272f, 0.320547998f, 0.400181234f,
        0.109814420f, -0.0761131793f, 0.00352000631f, 0.0944621563f,
        0.174095348f, -0.248481899f, -0.168848693f, -0.0892155021f,
        -0.212876320f, -0.133243099f, -0.0940255076f, -0.384392291f,
        0.0387553312f, 0.117877364f, 0.0744187534f, 0.153540790f,
        -0.137337133f, -0.166595563f, -0.0874735191f, 0.0477273427f,
        0.126849383f, -0.221955329f, -0.142833292f, -0.0637112334f,
        -0.0370800272f, 0.0420420319f, 0.0902444273f, -0.200633496f,
    };
    if (output.shape() != mlx::core::Shape{4, 16})
        throw std::runtime_error("DeepSeek-V4.1 padded image output shape changed");
    const auto* actual = output.data<float>();
    for (std::size_t index = 0; index < expected.size(); ++index)
        if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 5e-4f + std::abs(expected[index]) * 1e-3f)
            throw std::runtime_error("DeepSeek-V4.1 vision differs from the official reference at " + std::to_string(index)
                + ": actual=" + std::to_string(actual[index]) + " expected=" + std::to_string(expected[index]));
}

} // namespace

int main() try {
    mlx::core::set_default_device(mlx::core::Device::gpu);
    test_official_vision_reference();
    std::cout << "MFQ DeepSeek-V4.1 official vision reference passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
