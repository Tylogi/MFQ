// Tensor adapter for the embedding kernels: validation, allocation, pointer
// conversion and stream acquisition.  Compiled by the native ops target and by
// the Python extension, so these checks exist once while the kernel unit stays
// backend-free.

#include "kernels/kernels.h"
#include "kernels/quant_ops.h"

#include <cstdint>
#include <limits>

mfq_tensor_backend::Tensor embedding_lookup_cuda(
        mfq_tensor_backend::Tensor weight,
        mfq_tensor_backend::Tensor token_ids) {
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() && weight.dim() == 2,
        "embedding weight must be contiguous CUDA rank-2");
    MFQ_RUNTIME_CHECK(
        weight.scalar_type() == mfq_tensor_backend::kFloat32 ||
        weight.scalar_type() == mfq_tensor_backend::kFloat16,
        "embedding weight must be fp16 or fp32");
    MFQ_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == mfq_tensor_backend::kInt64 &&
        token_ids.device() == weight.device(),
        "embedding token IDs must be contiguous CUDA int64 on the weight device");
    const int vocabulary = static_cast<int>(weight.size(0));
    const int width = static_cast<int>(weight.size(1));
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(shape, weight.options());
    if (token_count == 0) {
        return output;
    }
    const auto weight_type =
        weight.scalar_type() == mfq_tensor_backend::kFloat32
            ? mfq::cuda::ScalarType::float32
            : mfq::cuda::ScalarType::float16;
    mfq::cuda::kernels::dense_embedding_lookup(
        weight.data_ptr(),
        weight_type,
        token_ids.data_ptr<int64_t>(),
        output.data_ptr(),
        token_count,
        width,
        vocabulary,
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint_selected_rows_cuda(
        mfq_tensor_backend::Tensor packed,
        mfq_tensor_backend::Tensor descriptors, int64_t width) {
    MFQ_RUNTIME_CHECK(packed.is_cuda() && packed.is_contiguous() &&
        packed.scalar_type() == mfq_tensor_backend::kUInt8 && packed.dim() == 1,
        "NINT selected payload must be CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(descriptors.is_cuda() && descriptors.is_contiguous() &&
        descriptors.scalar_type() == mfq_tensor_backend::kInt32 &&
        descriptors.dim() == 2 && descriptors.size(1) == 6 &&
        descriptors.device() == packed.device(),
        "NINT selected descriptors must be CUDA int32 [rows,6] on the payload device");
    MFQ_RUNTIME_CHECK(width > 0 && width <= std::numeric_limits<int>::max() &&
        descriptors.size(0) <= std::numeric_limits<int>::max() / 6 &&
        static_cast<uint64_t>(descriptors.size(0)) * width <= std::numeric_limits<uint32_t>::max(),
        "NINT selected row dimensions exceed bounds");
    const MfqCudaGuard guard(packed.device());
    auto output = mfq_tensor_backend::empty({descriptors.size(0), width},
        packed.options().dtype(mfq_tensor_backend::kFloat16));
    if (descriptors.size(0) == 0) return output;
    MFQ_RUNTIME_CHECK(packed.numel() > 0, "NINT selected payload is empty");
    const size_t total = static_cast<size_t>(descriptors.size(0)) * width;
    mfq::cuda::kernels::nint_selected_rows(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(descriptors.data_ptr<int32_t>()),
        nullptr,
        reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
        total, static_cast<int>(width), mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void nint_selected_rows_into_cuda(
        mfq_tensor_backend::Tensor packed, mfq_tensor_backend::Tensor descriptors,
        mfq_tensor_backend::Tensor inverse, mfq_tensor_backend::Tensor output) {
    MFQ_RUNTIME_CHECK(packed.is_cuda() && packed.is_contiguous() &&
        packed.scalar_type() == mfq_tensor_backend::kUInt8 && packed.dim() == 1 && packed.numel() > 0,
        "NINT selected payload must be nonempty CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(descriptors.is_cuda() && descriptors.is_contiguous() &&
        descriptors.scalar_type() == mfq_tensor_backend::kInt32 && descriptors.dim() == 2 &&
        descriptors.size(1) == 6 && descriptors.device() == packed.device(),
        "NINT selected descriptors must be CUDA int32 [rows,6]");
    MFQ_RUNTIME_CHECK(inverse.is_cuda() && inverse.is_contiguous() &&
        inverse.scalar_type() == mfq_tensor_backend::kInt64 && inverse.dim() == 1 &&
        inverse.device() == packed.device() && output.device() == packed.device() &&
        output.is_contiguous() && output.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output.dim() == 2 && output.size(0) == inverse.numel() && output.size(1) > 0 &&
        output.size(1) <= std::numeric_limits<int>::max() &&
        output.numel() <= std::numeric_limits<uint32_t>::max(),
        "NINT selected inverse/output geometry differs");
    if (!output.numel()) return;
    const MfqCudaGuard guard(packed.device());
    mfq::cuda::kernels::nint_selected_rows(
        packed.data_ptr<uint8_t>(), reinterpret_cast<const uint32_t*>(descriptors.data_ptr<int32_t>()),
        inverse.data_ptr<int64_t>(), reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
        output.numel(), static_cast<int>(output.size(1)), mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

mfq_tensor_backend::Tensor nint_embedding_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor token_ids,
        int64_t width,
        int64_t group_size) {
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1,
        "NINT embedding bitstream must be contiguous CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT embedding q metadata is invalid");
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT embedding subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT embedding neuron metadata is invalid");
    MFQ_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == mfq_tensor_backend::kInt64,
        "NINT embedding token IDs must be contiguous CUDA int64");
    const int vocabulary = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    MFQ_RUNTIME_CHECK(
        vocabulary > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT embedding dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == vocabulary &&
        row_q_bit_offsets.numel() == vocabulary &&
        neuron_scale.numel() == vocabulary &&
        neuron_minimum.numel() == vocabulary,
        "NINT embedding row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device() &&
        bitstream.device() == token_ids.device(),
        "NINT embedding tensors must share one CUDA device");
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(
        shape, bitstream.options().dtype(mfq_tensor_backend::kFloat16));
    if (token_count == 0) {
        return output;
    }
    mfq::cuda::kernels::nint_embedding_lookup(
        bitstream.data_ptr<uint8_t>(),
        row_q_bits.data_ptr<uint8_t>(),
        row_q_bit_offsets.data_ptr<int64_t>(),
        subgroup_scale.data_ptr<uint8_t>(),
        subgroup_minimum.data_ptr<uint8_t>(),
        neuron_scale.data_ptr<float>(),
        neuron_minimum.data_ptr<float>(),
        token_ids.data_ptr<int64_t>(),
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
        token_count,
        vocabulary,
        groups,
        group_size,
        width,
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_embedding_lookup_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor token_ids,
        int64_t width) {
    MFQ_RUNTIME_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == mfq_tensor_backend::kUInt8 &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 embedding q must be contiguous CUDA uint8 [V,G,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == mfq_tensor_backend::kFloat16 &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 embedding scale must be contiguous CUDA fp16 [V,G]");
    MFQ_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == mfq_tensor_backend::kInt64 &&
        token_ids.device() == quantized.device() &&
        scale.device() == quantized.device(),
        "NINT8-0 embedding tensors must share one CUDA device");
    MFQ_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 embedding width is invalid");
    const int vocabulary = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(
        shape, quantized.options().dtype(mfq_tensor_backend::kFloat16));
    if (token_count == 0) {
        return output;
    }
    mfq::cuda::kernels::nint8_zero_embedding_lookup(
        reinterpret_cast<const int8_t *>(
            quantized.data_ptr<uint8_t>()),
        reinterpret_cast<const __half *>(scale.data_ptr<mfq_half>()),
        token_ids.data_ptr<int64_t>(),
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
        token_count,
        vocabulary,
        groups,
        width,
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
