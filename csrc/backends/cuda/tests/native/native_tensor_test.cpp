#include "native/tensor/tensor.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <future>
#include <vector>

int main() {
    using namespace mfq::cuda;

    const auto before=tensor_host_bytes.load();
    Tensor retained_view;
    {
        auto storage=empty({16},TensorOptions{}.dtype(kFloat32));
        retained_view=storage.narrow(0,3,4);
        assert(tensor_host_bytes.load()==before+16*sizeof(float));
    }
    assert(tensor_host_bytes.load()==before+16*sizeof(float));
    retained_view={};
    assert(tensor_host_bytes.load()==before);

    auto values = zeros({2, 3, 4}, TensorOptions{}.dtype(kFloat32));
    assert(values.defined());
    assert(values.is_cpu());
    assert(values.is_contiguous());
    assert(values.numel() == 24);
    values.fill_(3.0);
    assert(values.data_ptr<float>()[17] == 3.0f);

    auto empty_blob = from_blob(
        static_cast<void*>(nullptr), {0}, TensorOptions{}.dtype(kFloat32));
    assert(empty_blob.defined());
    assert(empty_blob.numel() == 0);
    assert(empty_blob.clone().numel() == 0);
    bool rejected_null_nonempty = false;
    try {
        (void)from_blob(
            static_cast<void*>(nullptr), {1}, TensorOptions{}.dtype(kFloat32));
    } catch (const std::invalid_argument&) {
        rejected_null_nonempty = true;
    }
    assert(rejected_null_nonempty);

    const auto reshaped = values.reshape({4, 6});
    assert(reshaped.size(0) == 4);
    assert(reshaped.size(1) == 6);
    assert(reshaped.data_ptr() == values.data_ptr());

    const auto transposed = reshaped.transpose(0, 1);
    assert(transposed.size(0) == 6);
    assert(transposed.size(1) == 4);
    assert(!transposed.is_contiguous());

    const auto selected = values.narrow(1, 1, 2).select(2, 3);
    assert(selected.size(0) == 2);
    assert(selected.size(1) == 2);
    assert(selected.byte_offset() == (4 + 3) * sizeof(float));

    auto copy = values.clone();
    assert(copy.data_ptr() != values.data_ptr());
    assert(copy.data_ptr<float>()[17] == 3.0f);
    copy.zero_();
    assert(copy.data_ptr<float>()[17] == 0.0f);
    assert(values.data_ptr<float>()[17] == 3.0f);

    auto scalar = full({1}, 9.0, TensorOptions{}.dtype(kInt64));
    assert(scalar.item<std::int64_t>() == 9);

    auto sequence = tensor<float>({0, 1, 2, 3, 4, 5}).reshape({2, 3});
    auto materialized = sequence.transpose(0, 1).reshape({6});
    const std::array<float, 6> expected{0, 3, 1, 4, 2, 5};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        assert(materialized.data_ptr<float>()[index] == expected[index]);
    }

    auto half = tensor<float>({1.5f, -2.25f}).to(kFloat16);
    assert(std::abs(half.select(0, 0).item<float>() - 1.5f) == 0.0f);
    assert(std::abs(half.select(0, 1).item<float>() + 2.25f) == 0.0f);
    auto restored = half.to(kFloat32);
    assert(restored.data_ptr<float>()[0] == 1.5f);
    assert(restored.data_ptr<float>()[1] == -2.25f);

    auto dtype_only = sequence.to(TensorOptions{}.dtype(kFloat64));
    assert(dtype_only.is_cpu());
    assert(dtype_only.scalar_type() == kFloat64);
    auto like_options = TensorOptions{}.dtype(kInt32);
    auto like_dtype_only = empty_like(sequence, &like_options);
    assert(like_dtype_only.device() == sequence.device());
    assert(like_dtype_only.scalar_type() == kInt32);

    const auto serialized = pickle_save(sequence);
    const auto roundtrip = pickle_load(serialized).toTensor();
    assert(roundtrip.sizes() == sequence.sizes());
    assert(roundtrip.scalar_type() == sequence.scalar_type());
    assert(roundtrip.data_ptr<float>()[5] == 5.0f);

    // A live narrow view must keep its block out of the idle pool, even
    // when the original tensor is released and another worker allocates.
    const auto fixed_host_bytes=tensor_host_bytes.load();
    const auto options=TensorOptions{}.dtype(kUInt8);
    constexpr std::int64_t large_bytes=(2u<<20)+17;
    auto large=empty({large_bytes},options);
    auto* original=large.data_ptr<std::uint8_t>();
    original[0]=71;original[large_bytes-1]=93;
    auto retained=large.narrow(0,0,large_bytes);
    large={};
    auto separate=empty({large_bytes},options);
    assert(separate.data_ptr()!=original && retained.data_ptr<std::uint8_t>()[0]==71);
    retained={};
    auto reused=empty({large_bytes},options);
    assert(reused.data_ptr()==original && reused.nbytes()==large_bytes);
    assert(reused.data_ptr<std::uint8_t>()[large_bytes-1]==93);
    reused={};separate={};

    std::vector<std::future<void>> workers;
    for(int worker=0;worker<8;++worker)workers.push_back(std::async(std::launch::async,[&,worker] {
        for(int repeat=0;repeat<24;++repeat) {
            auto value=empty({large_bytes},options);
            auto* bytes=value.data_ptr<std::uint8_t>();
            bytes[0]=std::uint8_t(worker);bytes[large_bytes-1]=std::uint8_t(repeat);
            auto view=value.narrow(0,0,large_bytes);value={};
            assert(view.data_ptr<std::uint8_t>()[0]==worker &&
                   view.data_ptr<std::uint8_t>()[large_bytes-1]==repeat);
        }
    }));
    for(auto& worker:workers)worker.get();
    {
        std::vector<Tensor> allocations;
        for(int i=0;i<20;++i)allocations.push_back(empty({4u<<20},options));
    }
    assert(tensor_host_bytes.load()==fixed_host_bytes+(64u<<20));
}
