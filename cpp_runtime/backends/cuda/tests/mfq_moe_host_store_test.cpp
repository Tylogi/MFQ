#include "../storage/moe_host_store.h"

#include <cstring>
#include <iostream>

int main() {
#ifdef _WIN32
    return 77;
#else
    using namespace mfq::cuda;
    try {
        auto pattern = (std::filesystem::temp_directory_path() /
            "mfq-moe-host-test-XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
        } cleanup{pattern};
        auto bytes = tensor<std::uint8_t>({99, 1, 2, 3, 4, 5}).narrow(0, 1, 5);
        auto offsets = tensor<std::int64_t>({0, 1023, 4097, 99999}).reshape({2, 2});
        auto scales = tensor<float>({0.25f, -2.0f, 0.0f});
        auto empty_field = empty({0}, TensorOptions().dtype(kUInt8));
        std::vector<Tensor*> fields{&bytes, &offsets, &scales, &empty_field};
        std::vector<Tensor> expected;
        for (const auto* field : fields) expected.push_back(field->clone());
        const auto size = mmap_moe_host_fields(fields, pattern);
        MFQ_RUNTIME_CHECK(size >= 49 && std::filesystem::is_empty(pattern),
            "SSD staging files must be unlinked immediately");
        for (std::size_t i = 0; i < fields.size(); ++i) {
            const auto& field = *fields[i];
            MFQ_RUNTIME_CHECK(field.sizes() == expected[i].sizes() &&
                field.scalar_type() == expected[i].scalar_type() &&
                field.is_cpu() && !field.options().pinned(), "mapped field layout changed");
            MFQ_RUNTIME_CHECK(field.numel() == 0 ||
                std::memcmp(field.data_ptr(), expected[i].data_ptr(), field.nbytes()) == 0,
                "mapped field data changed");
        }
        auto retained = offsets.narrow(0, 1, 1);
        for (auto* field : fields) *field = Tensor();
        MFQ_RUNTIME_CHECK(retained.data_ptr<std::int64_t>()[0] == 4097 &&
                retained.data_ptr<std::int64_t>()[1] == 99999,
            "mapped storage did not survive its original tensor views");

        auto noncontiguous = expected[1].transpose(0, 1);
        std::vector<Tensor*> invalid{&noncontiguous};
        bool rejected = false;
        try { mmap_moe_host_fields(invalid, pattern); }
        catch (const std::exception&) { rejected = true; }
        MFQ_RUNTIME_CHECK(rejected, "noncontiguous storage was accepted");
#ifdef __linux__
        struct statfs filesystem{};
        if (statfs("/dev/shm", &filesystem) == 0 && filesystem.f_type == TMPFS_MAGIC) {
            std::vector<Tensor*> ramdisk{&expected[0]};
            rejected = false;
            try { mmap_moe_host_fields(ramdisk, "/dev/shm"); }
            catch (const std::exception&) { rejected = true; }
            MFQ_RUNTIME_CHECK(rejected, "RAM disk offload was accepted");
        }
#endif
        std::cout << "SSD MoE fields: exact bytes, offsets, view lifetime and validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
