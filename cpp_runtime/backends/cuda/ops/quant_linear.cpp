#include "quant_linear.h"

#include "format.h"
#include "fp8_sq.h"
#include "moe.h"
#include "mx.h"
#include "mxfp4_sq.h"
#include "nint.h"
#include "vq.h"
#include "mfq_format_compat.h"
#include "mfe_expert_store.h"
#include "moe_cache_policy.h"
#include "moe_cache_transfer.h"
#include "nvq_codebooks.generated.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define MFQ_CPU_X86_GNU 1
#endif

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

int64_t read_i64_from(const std::vector<uint8_t> & b, size_t & off) {
    int64_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

uint32_t read_u32_from(const std::vector<uint8_t> & b, size_t & off) {
    uint32_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

bool g_mfq_drop_file_cache = false;

const mfq::TensorMetadata& require_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto* tensor = source.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    return *tensor;
}

bool has_tensor(
        const mfq::ModelSource& source,
        std::string_view name) noexcept {
    return source.find_tensor(name) != nullptr;
}

bool has_tensor_prefix(
        const mfq::ModelSource& source,
        std::string_view prefix) {
    for (const auto& tensor : source.tensors()) {
        if (tensor.name == prefix ||
            (tensor.name.size() > prefix.size() &&
             tensor.name.compare(0, prefix.size(), prefix) == 0 &&
             tensor.name[prefix.size()] == '.')) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> read_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto& tensor = require_tensor(source, name);
    if (tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model tensor is too large to read");
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(tensor.nbytes));
    source.read_range_into(
        name, 0, reinterpret_cast<std::byte*>(result.data()), result.size());
    if (g_mfq_drop_file_cache) source.drop_file_cache();
    return result;
}

std::vector<std::uint8_t> read_asset(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    std::vector<std::uint8_t> result(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

std::string read_asset_text(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    return {
        reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
    std::vector<mfq_tensor_backend::Tensor> outputs);

std::vector<mfq::TensorParallelSlice>
plan_moe_expert_parallel_slices(
    int64_t extent,
    const std::string & name);

MfeWeight load_mfe_gpu(
    const mfq::ModelSource & mfq, const std::string & name,
    bool cacheable,
    int layer_id,
    const std::string & projection_role);

static mfq_tensor_backend::Tensor dequant_nint_dense_f32(
    const NintWeight & weight);

mfq_tensor_backend::Tensor load_dense_gpu(const mfq::ModelSource & mfq, const std::string & name) {
    MfqCudaGuard guard(
        active_weight_load_device());
    const auto & rec = require_tensor(mfq, name);
    auto blob = read_tensor(mfq, name);
    if (rec.dtype == "NINT8-0") {
        const auto source = unpack_nint8_zero(blob);
        if (g_loading_cpu_layer) {
            return dequant_nint8_zero_cpu(source);
        }
        const auto packed = to_gpu_nint8_zero(source);
        auto dense = nint8_zero_dequant_cuda(
            packed.q_packed,
            packed.q8_zero_scale,
            packed.neuron_len)
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
        if (packed.shape.size() != 2 ||
            dense.size(0) != packed.shape[0] ||
            dense.size(1) != packed.shape[1]) {
            throw std::runtime_error(
                "NINT8-0 dense tensor shape mismatch: " + name);
        }
        return dense;
    }
    if (rec.dtype == "NINT") {
        auto dense = dequant_nint_dense_f32(load_nint_gpu(mfq, name));
        return g_loading_cpu_layer
            ? dense.cpu().contiguous()
            : dense;
    }
    if (rec.dtype != "F32" && rec.dtype != "BF16" &&
            rec.dtype != "F16" && rec.dtype != "I64" &&
            rec.dtype != "I32") {
        throw std::runtime_error(
            "unsupported dense dtype for C++ runtime: " + rec.dtype +
            " tensor " + name);
    }
    size_t off = 0;
    uint32_t ndim = read_u32_from(blob, off);
    std::vector<int64_t> shape(ndim);
    int64_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        shape[i] = read_i64_from(blob, off);
        numel *= shape[i];
    }
    mfq_tensor_backend::Tensor t;
    if (rec.dtype == "F32") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone();
    } else if (rec.dtype == "BF16") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)).clone().to(mfq_tensor_backend::kFloat32);
    } else if (rec.dtype == "F16") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone().to(mfq_tensor_backend::kFloat32);
    } else if (rec.dtype == "I64") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone();
    } else if (rec.dtype == "I32") {
        t = mfq_tensor_backend::from_blob(blob.data() + off, shape, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32)).clone();
    } else {
        throw std::runtime_error("unsupported dense dtype for C++ runtime: " + rec.dtype + " tensor " + name);
    }
    (void)numel;
    return g_loading_cpu_layer
        ? t.contiguous()
        : t.to(mfq_tensor_backend::kCUDA).contiguous();
}

static mfq_tensor_backend::Tensor load_dense_linear_cpu(
        const mfq::ModelSource & mfq,
        const std::string & name) {
    const auto & rec = require_tensor(mfq, name);
    auto blob = read_tensor(mfq, name);
    size_t off = 0;
    const uint32_t ndim = read_u32_from(blob, off);
    std::vector<int64_t> shape(ndim);
    for (uint32_t index = 0; index < ndim; ++index) {
        shape[index] = read_i64_from(blob, off);
    }
    if (shape.size() != 2) {
        throw std::runtime_error(
            "dense linear tensor must be rank 2: " + name);
    }
    mfq_tensor_backend::Tensor value;
    if (rec.dtype == "BF16") {
        value = mfq_tensor_backend::from_blob(
            blob.data() + off, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)).clone();
    } else if (rec.dtype == "F16") {
        value = mfq_tensor_backend::from_blob(
            blob.data() + off, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone();
    } else if (rec.dtype == "F32") {
        value = mfq_tensor_backend::from_blob(
            blob.data() + off, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone();
    } else {
        throw std::runtime_error(
            "unsupported dense linear dtype: " + rec.dtype +
            " tensor " + name);
    }
    return value.contiguous();
}

static mfq_tensor_backend::Tensor load_dense_linear_gpu(
        const mfq::ModelSource & mfq,
        const std::string & name) {
    MfqCudaGuard guard(active_weight_load_device());
    return load_dense_linear_cpu(mfq, name)
        .to(mfq_tensor_backend::kCUDA).contiguous();
}

static NintWeight cat_weights(const std::vector<NintWeight> & ws) {
    if (ws.empty()) throw std::runtime_error("empty NINT group");
    const auto & a = ws[0];
    std::vector<mfq_tensor_backend::Tensor> qp, rqb, rqoff, q8s, ss, sm, ns, nm;
    int64_t out = 0;
    int64_t q_bit_base = 0;
    for (const auto & w : ws) {
        if (w.ng != a.ng || w.gs != a.gs ||
            w.neuron_len != a.neuron_len || w.q8_zero != a.q8_zero) {
            throw std::runtime_error("cannot group NINT tensors with different input layout");
        }
        qp.push_back(w.q_packed);
        if (!w.q8_zero) {
            rqb.push_back(w.row_q_bits);
            rqoff.push_back(w.row_q_bit_offsets + q_bit_base);
            q_bit_base += w.q_packed.numel() * 8;
        }
        if (w.q8_zero) {
            q8s.push_back(w.q8_zero_scale);
            out += w.out;
            continue;
        }
        ss.push_back(w.sub_scale);
        sm.push_back(w.sub_min);
        ns.push_back(w.neuron_scale);
        nm.push_back(w.neuron_min);
        out += w.out;
    }
    NintWeight g;
    g.out = out;
    g.ng = a.ng;
    g.gs = a.gs;
    g.bits = a.bits;
    g.neuron_len = a.neuron_len;
    g.q8_zero = a.q8_zero;
    g.shape = a.shape;
    g.shape[0] = out;
    g.q_packed = mfq_tensor_backend::cat(qp, 0).contiguous();
    if (!a.q8_zero) {
        g.row_q_bits = mfq_tensor_backend::cat(rqb, 0).contiguous();
        g.row_q_bit_offsets = mfq_tensor_backend::cat(rqoff, 0).contiguous();
    }
    if (a.q8_zero) {
        g.q8_zero_scale = mfq_tensor_backend::cat(q8s, 0).contiguous();
        return g;
    }
    g.sub_scale = mfq_tensor_backend::cat(ss, 0).contiguous();
    g.sub_min = mfq_tensor_backend::cat(sm, 0).contiguous();
    g.neuron_scale = mfq_tensor_backend::cat(ns, 0).contiguous();
    g.neuron_min = mfq_tensor_backend::cat(nm, 0).contiguous();
    return g;
}

static NintLinearGroup make_linear_group(const std::vector<NintWeight> & ws) {
    NintLinearGroup g;
    for (const auto & w : ws) g.outs.push_back(w.out);
    bool same = !ws.empty();
    for (size_t i = 1; i < ws.size(); ++i) {
        if (ws[i].ng != ws[0].ng || ws[i].gs != ws[0].gs ||
            ws[i].neuron_len != ws[0].neuron_len ||
            ws[i].q8_zero != ws[0].q8_zero) {
            same = false;
            break;
        }
    }
    if (same) {
        g.w = cat_weights(ws);
        if (g.w.q8_zero) {
            g.projection_w.reserve(ws.size());
            int64_t offset = 0;
            for (const auto & source : ws) {
                NintWeight projection = g.w;
                projection.out = source.out;
                projection.shape = source.shape;
                projection.q_packed =
                    g.w.q_packed.narrow(0, offset, source.out);
                projection.q8_zero_scale =
                    g.w.q8_zero_scale.narrow(0, offset, source.out);
                projection.workspaces.clear();
                g.projection_w.push_back(std::move(projection));
                offset += source.out;
            }
            MFQ_RUNTIME_CHECK(offset == g.w.out,
                        "Q8 projection views do not cover grouped output");
        }
    } else {
        std::vector<NintWeight> cur;
        std::vector<int64_t> cur_outs;
        auto flush = [&]() {
            if (cur.empty()) return;
            g.split_w.push_back(cur.size() == 1 ? cur[0] : cat_weights(cur));
            g.split_outs.push_back(cur_outs);
            cur.clear();
            cur_outs.clear();
        };
        for (const auto & w : ws) {
            bool append = !cur.empty() &&
                w.ng == cur[0].ng && w.gs == cur[0].gs &&
                w.neuron_len == cur[0].neuron_len &&
                w.q8_zero == cur[0].q8_zero;
            if (!append) flush();
            cur.push_back(w);
            cur_outs.push_back(w.out);
        }
        flush();
    }
    return g;
}

mfq_tensor_backend::Tensor run_quant_linear_shard(
        const QuantLinearShard & shard,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        int gate_mode) {
    MfqCudaGuard guard(shard.device);
    if (shard.kind == QuantLinearKind::Nint) {
        return gate.has_value()
            ? nint_matmul_input_mul(
                shard.nint, x, gate.value(), gate_mode)
            : nint_matmul(shard.nint, x);
    }
    if (shard.kind == QuantLinearKind::Nvq) {
        return gate.has_value()
            ? nvq_matmul_input_mul(
                shard.nvq, x, gate.value(), gate_mode)
            : nvq_matmul(shard.nvq, x);
    }
    if (shard.kind == QuantLinearKind::Mxfp4) {
        MFQ_RUNTIME_CHECK(
            !gate.has_value(),
            "MXFP4 tensor-parallel linear does not support input gating");
        return mxfp4_matmul(shard.mxfp4, x);
    }
    if (shard.kind == QuantLinearKind::Dense) {
        auto local = x.to(shard.dense.scalar_type());
        if (gate.has_value()) {
            MFQ_RUNTIME_CHECK(
                gate_mode == 1 || gate_mode == 2,
                "dense input gate mode must be sigmoid or SiLU");
            auto local_gate = gate.value().to(local.scalar_type());
            local = gate_mode == 1
                ? local * mfq_tensor_backend::sigmoid(local_gate)
                : local * mfq_tensor_backend::silu(local_gate);
        }
        return mfq_tensor_backend::matmul(local, shard.dense.transpose(0, 1));
    }
    MFQ_RUNTIME_CHECK(
        !gate.has_value(),
        "MXFP8 tensor-parallel linear does not support input gating");
    return mxfp8_matmul(shard.mxfp8, x);
}

mfq_tensor_backend::Tensor tensor_to_cuda_device(
        mfq_tensor_backend::Tensor value,
        int device,
        mfq_tensor_backend::Tensor reusable) {
    if (value.is_cuda() && value.get_device() == device) {
        return value.contiguous();
    }
    const auto reusable_matches = [&]() {
        return reusable.defined() && reusable.is_cuda() &&
            reusable.get_device() == device &&
            reusable.sizes() == value.sizes() &&
            reusable.scalar_type() == value.scalar_type() &&
            reusable.is_contiguous();
    };
    const auto destination_tensor = [&]() {
        if (reusable_matches()) return reusable;
        MfqCudaGuard destination_guard(device);
        return mfq_tensor_backend::empty(
            value.sizes(),
            value.options().device(
                mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, device)));
    };
    if (value.numel() == 0) {
        return destination_tensor();
    }
#if defined(MFQ_NATIVE_CUDA_RUNTIME) && defined(MFQ_HAVE_NCCL)
    if (value.is_cuda() &&
            g_model_parallel_collectives.collectives_enabled) {
        const int source_device = value.get_device();
        const auto source_rank_it = std::find(
            g_model_parallel_collectives.devices.begin(),
            g_model_parallel_collectives.devices.end(),
            source_device);
        const auto destination_rank_it = std::find(
            g_model_parallel_collectives.devices.begin(),
            g_model_parallel_collectives.devices.end(),
            device);
        if (source_rank_it !=
                g_model_parallel_collectives.devices.end() &&
                destination_rank_it !=
                g_model_parallel_collectives.devices.end()) {
            const auto source_stream =
                mfq_get_current_cuda_stream(source_device);
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            {
                MfqCudaGuard source_guard(source_device);
                MFQ_CUDA_CHECK(cudaStreamIsCapturing(
                    source_stream.stream(), &capture_status));
            }
            if (capture_status != cudaStreamCaptureStatusNone) {
                auto source = value.contiguous();
                auto destination = destination_tensor();
                auto& runtime = g_model_parallel_collectives;
                const auto source_rank = static_cast<size_t>(
                    source_rank_it - runtime.devices.begin());
                const auto destination_rank = static_cast<size_t>(
                    destination_rank_it - runtime.devices.begin());
                {
                    MfqCudaGuard source_guard(source_device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.ready[source_rank],
                        source_stream.stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams[source_rank].stream(),
                        runtime.ready[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    // Cross-device event waits make the receiving NCCL stream
                    // participate in the same capture before ncclRecv runs.
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams[destination_rank].stream(),
                        runtime.ready[source_rank], 0));
                }
                MFQ_NCCL_CHECK(ncclGroupStart());
                MFQ_NCCL_CHECK(ncclSend(
                    source.data_ptr(), source.nbytes(), ncclUint8,
                    static_cast<int>(destination_rank),
                    runtime.communicators[source_rank],
                    runtime.streams[source_rank].stream()));
                MFQ_NCCL_CHECK(ncclRecv(
                    destination.data_ptr(), destination.nbytes(), ncclUint8,
                    static_cast<int>(source_rank),
                    runtime.communicators[destination_rank],
                    runtime.streams[destination_rank].stream()));
                MFQ_NCCL_CHECK(ncclGroupEnd());
                {
                    MfqCudaGuard source_guard(source_device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.completed[source_rank],
                        runtime.streams[source_rank].stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        source_stream.stream(),
                        runtime.completed[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    const auto destination_stream =
                        mfq_get_current_cuda_stream(device);
                    MFQ_CUDA_CHECK(cudaEventRecord(
                        runtime.completed[destination_rank],
                        runtime.streams[destination_rank].stream()));
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                        destination_stream.stream(),
                        runtime.completed[destination_rank], 0));
                }
                return destination;
            }
        }
    }
#endif
    MfqCudaGuard guard(device);
    if (reusable_matches()) {
        reusable.copy_(value, true);
        return reusable;
    }
    return value.to(
        value.options().device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device)),
        true, false).contiguous();
}

mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
        std::vector<mfq_tensor_backend::Tensor> outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "cannot reduce an empty model-parallel output");
    }
#ifdef MFQ_HAVE_NCCL
    if (g_model_parallel_collectives.collectives_enabled &&
            outputs.size() ==
                g_model_parallel_collectives.devices.size()) {
        auto & runtime = g_model_parallel_collectives;
        const auto shape = outputs.front().sizes().vec();
        const auto output_dtype = outputs.front().scalar_type();
        const int64_t elements = outputs.front().numel();
        const bool reduce_to_primary =
            model_parallel_reduce_to_primary_enabled();
        // A two-input FP16 sum has the same final FP16 rounding as the former
        // FP32 reduction, while avoiding both conversion passes.
        const bool fp16_reduce =
            model_parallel_fp16_reduce_enabled() &&
            reduce_to_primary &&
            outputs.size() == 2 &&
            output_dtype == mfq_tensor_backend::kFloat16;
        const int primary = model_parallel_primary_device();
        const auto primary_rank_it = std::find(
            runtime.devices.begin(), runtime.devices.end(), primary);
        if (primary_rank_it == runtime.devices.end()) {
            throw std::runtime_error(
                "model-parallel primary device is absent from NCCL ranks");
        }
        const auto primary_rank = static_cast<size_t>(
            primary_rank_it - runtime.devices.begin());
        for (size_t index = 0; index < outputs.size(); ++index) {
            const int device = runtime.devices[index];
            if (!outputs[index].defined() || !outputs[index].is_cuda() ||
                    outputs[index].get_device() != device ||
                    outputs[index].sizes().vec() != shape ||
                    outputs[index].scalar_type() != output_dtype) {
                throw std::runtime_error(
                    "NCCL model-parallel reduction received mismatched shards");
            }
            MfqCudaGuard guard(device);
            if (fp16_reduce) {
                outputs[index] = outputs[index].contiguous();
            }
            const auto producer =
                mfq_get_current_cuda_stream(device);
            MFQ_CUDA_CHECK(cudaEventRecord(
                runtime.ready[index], producer.stream()));
            const auto communication = runtime.streams[index];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                communication.stream(), runtime.ready[index], 0));
            if (fp16_reduce) {
                mfq_cuda_record_stream(outputs[index], communication);
            } else {
                MfqCudaStreamGuard stream_guard(communication);
                auto & buffer = runtime.reduction_buffers[index];
                if (!buffer.defined() || buffer.sizes().vec() != shape ||
                        buffer.get_device() != device ||
                        buffer.scalar_type() != mfq_tensor_backend::kFloat32) {
                    buffer = mfq_tensor_backend::empty(
                        shape,
                        outputs[index].options().dtype(mfq_tensor_backend::kFloat32));
                }
                buffer.copy_(outputs[index], true);
                mfq_cuda_record_stream(outputs[index], communication);
            }
        }

        MFQ_NCCL_CHECK(ncclGroupStart());
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto & buffer = runtime.reduction_buffers[index];
            if (reduce_to_primary) {
                void * reduction_data = fp16_reduce
                    ? outputs[index].data_ptr()
                    : buffer.data_ptr<float>();
                MFQ_NCCL_CHECK(ncclReduce(
                    reduction_data,
                    reduction_data,
                    static_cast<size_t>(elements),
                    fp16_reduce ? ncclFloat16 : ncclFloat32,
                    ncclSum,
                    static_cast<int>(primary_rank),
                    runtime.communicators[index],
                    runtime.streams[index].stream()));
            } else {
                MFQ_NCCL_CHECK(ncclAllReduce(
                    buffer.data_ptr<float>(),
                    buffer.data_ptr<float>(),
                    static_cast<size_t>(elements),
                    ncclFloat32,
                    ncclSum,
                    runtime.communicators[index],
                    runtime.streams[index].stream()));
            }
        }
        MFQ_NCCL_CHECK(ncclGroupEnd());

        mfq_tensor_backend::Tensor result;
        for (size_t index = 0; index < outputs.size(); ++index) {
            MfqCudaGuard guard(runtime.devices[index]);
            const auto communication = runtime.streams[index];
            if (runtime.devices[index] == primary) {
                if (fp16_reduce) {
                    result = outputs[index];
                } else {
                    MfqCudaStreamGuard stream_guard(communication);
                    result = runtime.reduction_buffers[index]
                        .to(output_dtype).contiguous();
                }
            }
            MFQ_CUDA_CHECK(cudaEventRecord(
                runtime.completed[index], communication.stream()));
        }
        MfqCudaGuard primary_guard(primary);
        const auto parent = mfq_get_current_cuda_stream(primary);
        MFQ_CUDA_CHECK(cudaStreamWaitEvent(
            parent.stream(), runtime.completed[primary_rank], 0));
        mfq_cuda_record_stream(result, parent);
        return result;
    }
#endif
    const int primary =
        model_parallel_primary_device();
    MfqCudaGuard primary_guard(primary);
    const auto output_dtype =
        outputs.front().scalar_type();
    mfq_tensor_backend::Tensor reduced;
    for (auto & output : outputs) {
        auto partial =
            tensor_to_cuda_device(output, primary)
                .to(mfq_tensor_backend::kFloat32);
        if (!reduced.defined()) {
            reduced = std::move(partial);
        } else {
            reduced.add_(partial);
        }
    }
    return reduced.to(output_dtype).contiguous();
}

mfq_tensor_backend::Tensor quant_embedding_lookup(
        const QuantLinear & embedding,
        mfq_tensor_backend::Tensor token_ids) {
    MFQ_RUNTIME_CHECK(
        !embedding.tensor_parallel(),
        "quantized embedding does not support tensor-parallel shards");
    if (embedding.is_dense()) {
        auto output_shape = token_ids.sizes().vec();
        output_shape.push_back(embedding.dense.size(1));
        return embedding.dense.index_select(
            0, token_ids.reshape({-1})).reshape(output_shape);
    }
    if (embedding.is_nvq()) {
        return nvq_embedding(embedding.nvq.w, token_ids);
    }
    if (embedding.is_nint()) {
        if (embedding.nint.w.q8_zero) {
            return nint8_zero_embedding_lookup_cuda(
                embedding.nint.w.q_packed,
                embedding.nint.w.q8_zero_scale,
                token_ids, embedding.nint.w.neuron_len);
        }
        return nint_embedding_cuda(
            embedding.nint.w.q_packed,
            embedding.nint.w.row_q_bits,
            embedding.nint.w.row_q_bit_offsets,
            embedding.nint.w.sub_scale,
            embedding.nint.w.sub_min,
            embedding.nint.w.neuron_scale,
            embedding.nint.w.neuron_min,
            token_ids, embedding.nint.w.neuron_len,
            embedding.nint.w.gs);
    }
    if (embedding.is_mxfp4()) {
        return mxfp4_embedding_lookup_cuda(
            embedding.mxfp4.weight.values,
            embedding.mxfp4.weight.scales, token_ids);
    }
    if (embedding.is_mxfp8()) {
        return mxfp8_embedding_lookup_cuda(
            embedding.mxfp8.weight.values,
            embedding.mxfp8.weight.scales, token_ids);
    }
    MFQ_RUNTIME_CHECK(
        !embedding.is_mxfp4_sq() && !embedding.is_fp8_sq(),
        "SQ tensors do not support embedding lookup");
    throw std::runtime_error("unsupported quantized embedding kind");
}

bool tensor_parallel_output_projections_compatible(
        const QuantLinearProjectionRefs & projections) {
    if (projections.size() < 2 ||
            !std::all_of(
                projections.begin(), projections.end(),
                [](const QuantLinear * layer) {
                    return layer != nullptr &&
                        layer->tensor_parallel() &&
                        layer->tensor_parallel_axis ==
                            TensorParallelAxis::Output;
                })) {
        return false;
    }
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (shard_count < 2) return false;
    for (const auto * layer : projections) {
        if (layer->tensor_parallel_shards.size() != shard_count) {
            return false;
        }
        for (size_t shard = 0; shard < shard_count; ++shard) {
            const auto & candidate =
                layer->tensor_parallel_shards[shard];
            const auto & reference =
                projections.front()->tensor_parallel_shards[shard];
            if (candidate.device != reference.device ||
                    candidate.input_begin != reference.input_begin ||
                    candidate.input_end != reference.input_end) {
                return false;
            }
        }
    }
    return true;
}

std::vector<mfq_tensor_backend::Tensor>
forward_tensor_parallel_output_projections(
        mfq_tensor_backend::Tensor x,
        const QuantLinearProjectionRefs & projections) {
    MFQ_RUNTIME_CHECK(
        tensor_parallel_output_projections_compatible(projections),
        "tensor-parallel output projections are incompatible");
    auto output_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)});
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (g_decode_graph_tp_projection_major) {
        std::vector<mfq_tensor_backend::Tensor> local_inputs(shard_count);
        for (size_t launch_position = 0;
             launch_position < shard_count; ++launch_position) {
            const size_t shard = model_parallel_launch_index(
                launch_position, shard_count);
            const int device =
                projections.front()->tensor_parallel_shards[shard].device;
            MfqCudaGuard guard(device);
            local_inputs[shard] = tensor_to_cuda_device(flat, device);
        }

        const int primary = model_parallel_primary_device();
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(projections.size());
        for (const auto * projection : projections) {
            std::vector<mfq_tensor_backend::Tensor> local_outputs(shard_count);
            for (size_t launch_position = 0;
                 launch_position < shard_count; ++launch_position) {
                const size_t shard = model_parallel_launch_index(
                    launch_position, shard_count);
                const auto & weight =
                    projection->tensor_parallel_shards[shard];
                MfqCudaGuard guard(weight.device);
                local_outputs[shard] = run_quant_linear_shard(
                    weight, local_inputs[shard]);
            }
            MfqCudaGuard primary_guard(primary);
            std::vector<mfq_tensor_backend::Tensor> gathered;
            gathered.reserve(shard_count);
            for (auto & output : local_outputs) {
                gathered.push_back(tensor_to_cuda_device(output, primary));
            }
            auto combined = mfq_tensor_backend::cat(
                gathered, -1).contiguous();
            auto shape = output_shape;
            shape.back() = combined.size(-1);
            result.push_back(combined.reshape(shape));
        }
        return result;
    }
    std::vector<std::vector<mfq_tensor_backend::Tensor>>
        local_outputs(projections.size());
    for (auto & outputs : local_outputs) {
        outputs.resize(shard_count);
    }
    for (size_t launch_position = 0;
         launch_position < shard_count; ++launch_position) {
        const size_t shard = model_parallel_launch_index(
            launch_position, shard_count);
        const int device =
            projections.front()->tensor_parallel_shards[shard].device;
        MfqCudaGuard guard(device);
        // All projections consume the same immutable activation. Transfer it
        // once per rank, then keep the independent local projection work on
        // that rank before gathering each output.
        auto local_x = tensor_to_cuda_device(flat, device);
        for (size_t projection = 0;
                projection < projections.size(); ++projection) {
            local_outputs[projection][shard] =
                run_quant_linear_shard(
                    projections[projection]
                        ->tensor_parallel_shards[shard],
                    local_x);
        }
    }

    const int primary = model_parallel_primary_device();
    MfqCudaGuard primary_guard(primary);
    std::vector<mfq_tensor_backend::Tensor> result;
    result.reserve(projections.size());
    for (auto & projection_outputs : local_outputs) {
        std::vector<mfq_tensor_backend::Tensor> gathered;
        gathered.reserve(shard_count);
        for (auto & output : projection_outputs) {
            gathered.push_back(
                tensor_to_cuda_device(output, primary));
        }
        auto combined = mfq_tensor_backend::cat(
            gathered, -1).contiguous();
        auto shape = output_shape;
        shape.back() = combined.size(-1);
        result.push_back(combined.reshape(shape));
    }
    return result;
}

static bool is_nint_linear_dtype(const std::string & dtype) {
    return dtype == "NINT";
}

static bool is_nvq_linear_dtype(const std::string & dtype) {
    return dtype == "NVQ" || dtype == "NPQ";
}

static TensorParallelAxis infer_tensor_parallel_axis(
        const std::string & name) {
    const auto ends_with = [&name](std::string_view suffix) {
        return name.size() >= suffix.size() &&
            std::string_view(name).substr(name.size() - suffix.size()) == suffix;
    };
    if (name == "model.token_embedding.weight" ||
        ends_with(".token_embedding.weight") ||
        ends_with(".text_embedding.weight") ||
        (name.find(".code_embedding.") != std::string::npos &&
         ends_with(".weight"))) {
        return TensorParallelAxis::Mirrored;
    }
    if (name == "model.output.weight" ||
        name.find(".code_output.") != std::string::npos) {
        return TensorParallelAxis::Output;
    }
    if (ends_with(".mlp.down.weight") ||
        ends_with(".mlp.shared_expert.down.weight") ||
        ends_with(".attention.output.weight") ||
        ends_with(".linear_attention.output.weight") ||
        ends_with(".attention.output_a.weight") ||
        ends_with(".attention.output_b.weight") ||
        ends_with(".projector.output.weight")) {
        return TensorParallelAxis::Input;
    }
    return TensorParallelAxis::Output;
}

static std::vector<mfq::TensorParallelSlice>
plan_parallel_slices(
        int64_t extent,
        int64_t preferred_granularity,
        const ParallelConfig & config,
        const std::string & name) {
    (void)name;
    if (!config.enabled()) {
        throw std::runtime_error(
            "parallel slice planning requires at least two ranks");
    }
    int64_t granularity = std::max<int64_t>(
        1, std::min<int64_t>(
            preferred_granularity,
            extent / static_cast<int64_t>(config.devices.size())));
    while (granularity > 1 &&
           extent < static_cast<int64_t>(config.devices.size()) *
               granularity) {
        granularity /= 2;
    }
    auto slices = mfq::plan_tensor_parallel_slices(
        extent, granularity, config.devices, config.split);
    mfq::validate_tensor_parallel_slices(
        slices, extent, granularity);
    return slices;
}

static std::vector<mfq::TensorParallelSlice>
plan_quant_tensor_parallel_slices(
        int64_t extent,
        int64_t preferred_granularity,
        const std::string & name) {
    return plan_parallel_slices(
        extent, preferred_granularity,
        g_tensor_parallel, name);
}

std::vector<mfq::TensorParallelSlice>
plan_moe_expert_parallel_slices(
        int64_t extent,
        const std::string & name) {
    return plan_parallel_slices(
        extent, 1, moe_parallel_config(), name);
}

QuantLinear load_quant_linear(
        const mfq::ModelSource & mfq,
        const std::string & name,
        std::optional<TensorParallelAxis> axis_override,
        const std::vector<mfq::TensorParallelSlice> *
            slices_override) {
    const auto & dtype = require_tensor(mfq, name).dtype;
    QuantLinear result;
    const TensorParallelAxis axis =
        axis_override.value_or(
            infer_tensor_parallel_axis(name));
    if (slices_override != nullptr &&
        axis != TensorParallelAxis::Output) {
        throw std::runtime_error(
            "explicit tensor-parallel slices require an output-axis weight");
    }
    auto select_slices = [&](
            int64_t extent,
            int64_t preferred) {
        if (slices_override == nullptr) {
            return plan_quant_tensor_parallel_slices(
                extent, preferred, name);
        }
        auto slices = *slices_override;
        mfq::validate_tensor_parallel_slices(
            slices, extent, 1);
        if (slices.size() !=
            g_tensor_parallel.devices.size()) {
            throw std::runtime_error(
                "explicit tensor-parallel slice count mismatch");
        }
        for (size_t index = 0;
             index < slices.size(); ++index) {
            if (slices[index].device !=
                g_tensor_parallel.devices[index]) {
                throw std::runtime_error(
                    "explicit tensor-parallel device order mismatch");
            }
        }
        return slices;
    };
    result.tensor_parallel_axis = axis;
    if (is_nint_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nint;
        const auto blob = read_tensor(mfq, name);
        if (dtype == "NINT8-0") {
            const auto cpu = unpack_nint8_zero(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : 4;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint8_zero_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint8_zero_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * 32 : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * 32,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint =
                        to_cuda_device_nint8_zero(
                            shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                if (g_loading_cpu_layer) {
                    result.nint.w = to_device_nint8_zero(cpu, false);
                } else {
                    MfqCudaGuard guard(
                        active_weight_load_device());
                    result.nint.w =
                        to_device_nint8_zero(cpu, true);
                }
            }
        } else {
            const auto cpu = unpack_nint(blob);
            result.logical_out = cpu.out;
            result.logical_neuron_len = cpu.neuron_len;
            if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    QuantLinearShard shard;
                    shard.device = slice.device;
                    shard.kind = QuantLinearKind::Nint;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * cpu.gs : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * cpu.gs,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.nint = to_cuda_device_nint(
                        shard_cpu, slice.device);
                    result.tensor_parallel_shards.push_back(
                        std::move(shard));
                }
            } else {
                if (g_loading_cpu_layer) {
                    result.nint.w = to_device_nint(cpu, false);
                } else {
                    MfqCudaGuard guard(
                        active_weight_load_device());
                    result.nint.w =
                        to_device_nint(cpu, true);
                }
            }
        }
    } else if (is_nvq_linear_dtype(dtype)) {
        result.kind = QuantLinearKind::Nvq;
        const auto cpu =
            unpack_nvq(read_tensor(mfq, name), dtype);
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
            axis != TensorParallelAxis::Mirrored) {
            const int64_t extent =
                axis == TensorParallelAxis::Output
                ? cpu.out : cpu.ng;
            const int64_t preferred =
                axis == TensorParallelAxis::Output
                ? 128
                : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
            for (const auto & slice :
                 select_slices(
                     extent, preferred)) {
                auto shard_cpu = slice_nvq_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Nvq;
                shard.output_begin =
                    axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end =
                    axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin =
                    axis == TensorParallelAxis::Input
                    ? slice.begin * cpu.gs : 0;
                shard.input_end =
                    axis == TensorParallelAxis::Input
                    ? std::min<int64_t>(
                        slice.end * cpu.gs,
                        cpu.neuron_len)
                    : cpu.neuron_len;
                shard.nvq = to_cuda_device_nvq(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else {
            if (g_loading_cpu_layer) {
                result.nvq.w = to_device_nvq(cpu, false);
            } else {
                MfqCudaGuard guard(
                    active_weight_load_device());
                result.nvq.w = to_device_nvq(cpu, true);
            }
        }
    } else if (dtype == "MXFP4-SQ") {
        result.kind = QuantLinearKind::Mxfp4Sq;
        if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            throw std::runtime_error(
                "MXFP4-SQ tensor parallelism is not implemented: " + name);
        }
        if (g_loading_cpu_layer) {
            throw std::runtime_error(
                "MXFP4-SQ does not support dense CPU-layer offload: " + name);
        }
        result.mxfp4_sq.weight = to_device_mxfp4_sq(
            read_tensor(mfq, name),
            true,
            active_weight_load_device());
        result.logical_out = result.mxfp4_sq.weight.out;
        result.logical_neuron_len =
            result.mxfp4_sq.weight.neuron_len;
    } else if (mfq::fp8sq::is_dtype(dtype)) {
        result.kind = QuantLinearKind::Fp8Sq;
        if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            throw std::runtime_error(
                "FP8-SQ tensor parallelism is not implemented: " + name);
        }
        if (g_loading_cpu_layer) {
            throw std::runtime_error(
                "FP8-SQ does not support dense CPU-layer offload: " + name);
        }
        result.fp8_sq.weight = to_device_fp8_sq(
            dtype, read_tensor(mfq, name), true,
            active_weight_load_device());
        result.logical_out = result.fp8_sq.weight.out;
        result.logical_neuron_len =
            result.fp8_sq.weight.neuron_len;
    } else if (dtype == "MXFP4") {
        result.kind = QuantLinearKind::Mxfp4;
        const auto cpu = unpack_mxfp4(read_tensor(mfq, name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            const int64_t preferred = axis == TensorParallelAxis::Output
                ? 128 : 32;
            for (const auto & slice : select_slices(extent, preferred)) {
                auto shard_cpu = slice_mxfp4_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp4;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp4 = to_device_mxfp4(
                    shard_cpu, true, slice.device);
                result.tensor_parallel_shards.push_back(std::move(shard));
            }
        } else {
            result.mxfp4.weight = to_device_mxfp4(
                cpu, !g_loading_cpu_layer,
                g_loading_cpu_layer ? -1 : active_weight_load_device());
        }
    } else if (dtype == "MXFP8") {
        result.kind = QuantLinearKind::Mxfp8;
        const auto cpu = unpack_mxfp8(read_tensor(mfq, name));
        result.logical_out = cpu.out;
        result.logical_neuron_len = cpu.neuron_len;
        if (g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            for (const auto & slice : select_slices(extent, 128)) {
                auto shard_cpu = slice_mxfp8_cpu(
                    cpu, axis, slice.begin, slice.end);
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Mxfp8;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.mxfp8 = to_cuda_device_mxfp8(
                    shard_cpu, slice.device);
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else if (g_loading_cpu_layer) {
            result.mxfp8.weight = to_device_mxfp8(cpu, false);
        } else {
            result.mxfp8.weight = to_cuda_device_mxfp8(
                cpu, active_weight_load_device());
        }
    } else if (dtype == "BF16" || dtype == "F16" || dtype == "F32") {
        result.kind = QuantLinearKind::Dense;
        result.dense_small_m_rowwise = name.rfind("predictor.", 0) == 0;
        auto cpu = load_dense_linear_cpu(mfq, name);
        result.logical_out = cpu.size(0);
        result.logical_neuron_len = cpu.size(1);
        // Keep native floating-point linears whole on the primary TP rank.
        // Splitting these matrices changes the cuBLAS GEMM geometry and causes
        // materially larger drift than the weight formats under test. They
        // are a small fraction of the model; routed experts and MXFP8/NINT/NVQ
        // weights remain sharded.
        const char * shard_native_float_env =
            std::getenv("MFQ_TP_SHARD_NATIVE_FLOAT");
        const bool shard_native_float =
            shard_native_float_env != nullptr &&
            std::atoi(shard_native_float_env) != 0;
        if (shard_native_float && g_tensor_parallel.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.size(0) : cpu.size(1);
            for (const auto & slice : select_slices(extent, 128)) {
                QuantLinearShard shard;
                shard.device = slice.device;
                shard.kind = QuantLinearKind::Dense;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.size(0);
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.size(1);
                const int64_t dimension =
                    axis == TensorParallelAxis::Output ? 0 : 1;
                MfqCudaGuard guard(slice.device);
                shard.dense = cpu.narrow(
                        dimension, slice.begin,
                        slice.end - slice.begin)
                    .to(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, slice.device))
                    .contiguous();
                result.tensor_parallel_shards.push_back(
                    std::move(shard));
            }
        } else if (g_loading_cpu_layer) {
            result.dense = cpu.contiguous();
        } else {
            MfqCudaGuard guard(active_weight_load_device());
            result.dense = cpu.to(mfq_tensor_backend::kCUDA).contiguous();
        }
    } else {
        throw std::runtime_error(
            "linear tensor must be NINT/NVQ/MXFP4-SQ/MXFP8-SQ/FP8-128SQ/MXFP4/MXFP8/BF16/F16/F32: " +
            name + " dtype=" + dtype);
    }
    return result;
}

bool is_quant_dtype(const std::string & dtype) {
    return is_nint_linear_dtype(dtype) ||
        is_nvq_linear_dtype(dtype) || dtype == "MXFP4-SQ" ||
        mfq::fp8sq::is_dtype(dtype) ||
        dtype == "MXFP4" || dtype == "MXFP8";
}

QuantLinearGroup make_quant_group(
        std::vector<QuantLinear> layers,
        bool preserve_projection_boundaries) {
    if (layers.empty()) throw std::runtime_error("empty quantized linear group");
    QuantLinearGroup result;
    result.decode_branch_parallel = !preserve_projection_boundaries;
    result.outs.reserve(layers.size());
    bool all_nint = true;
    std::vector<NintWeight> nint_weights;
    nint_weights.reserve(layers.size());
    for (const auto & layer : layers) {
        result.outs.push_back(layer.out());
        all_nint = all_nint && layer.is_nint();
        if (layer.is_nint() && !layer.tensor_parallel()) {
            nint_weights.push_back(layer.nint.w);
        }
    }
    const char * disable_nint_group =
        std::getenv("MFQ_DIAGNOSTIC_DISABLE_NINT_GROUP");
    const bool diagnostic_keep_nint_separate =
        disable_nint_group != nullptr && disable_nint_group[0] == '1';
    const bool any_tensor_parallel = std::any_of(
        layers.begin(), layers.end(),
        [](const QuantLinear & layer) {
            return layer.tensor_parallel();
        });
    if (all_nint && !preserve_projection_boundaries &&
        !diagnostic_keep_nint_separate && !g_loading_cpu_layer &&
        !any_tensor_parallel) {
        result.nint_grouped = true;
        result.nint = make_linear_group(nint_weights);
    } else {
        result.layers = std::move(layers);
        result.nvq_prefix2 = !g_loading_cpu_layer && result.layers.size() >= 2 &&
            !any_tensor_parallel &&
            result.layers[0].is_nvq() && result.layers[1].is_nvq() &&
            nvq_pair_compatible(result.layers[0].nvq.w, result.layers[1].nvq.w);
    }
    return result;
}

static bool quant_linear_pair_compatible(const QuantLinear & a, const QuantLinear & b) {
    if (a.tensor_parallel() || b.tensor_parallel()) {
        return a.tensor_parallel() && b.tensor_parallel() &&
            a.kind == b.kind &&
            a.tensor_parallel_axis == b.tensor_parallel_axis &&
            a.tensor_parallel_shards.size() ==
                b.tensor_parallel_shards.size();
    }
    if (a.kind != b.kind) return false;
    if (a.is_nvq()) return nvq_pair_compatible(a.nvq.w, b.nvq.w);
    if (a.is_mxfp8()) {
        return a.mxfp8.weight.neuron_len == b.mxfp8.weight.neuron_len;
    }
    if (a.is_mxfp4()) {
        return a.mxfp4.weight.neuron_len == b.mxfp4.weight.neuron_len;
    }
    if (a.is_mxfp4_sq()) {
        return a.mxfp4_sq.weight.neuron_len ==
            b.mxfp4_sq.weight.neuron_len;
    }
    if (a.is_fp8_sq()) {
        return a.fp8_sq.weight.dtype == b.fp8_sq.weight.dtype &&
            a.fp8_sq.weight.neuron_len == b.fp8_sq.weight.neuron_len &&
            a.fp8_sq.weight.block_rows == b.fp8_sq.weight.block_rows &&
            a.fp8_sq.weight.block_columns == b.fp8_sq.weight.block_columns &&
            a.fp8_sq.weight.scale_kind == b.fp8_sq.weight.scale_kind;
    }
    if (a.is_dense()) {
        return a.dense.size(1) == b.dense.size(1);
    }
    const auto & x = a.nint.w;
    const auto & y = b.nint.w;
    return x.ng == y.ng && x.gs == y.gs &&
        x.neuron_len == y.neuron_len && x.q8_zero == y.q8_zero;
}

QuantLinearGroup load_quant_group(
    const mfq::ModelSource & mfq, const std::vector<std::string> & names,
    size_t required_compatible_prefix,
    const std::vector<mfq::TensorParallelSlice> *
        slices_override,
    bool preserve_projection_boundaries) {
    std::vector<QuantLinear> layers;
    layers.reserve(names.size());
    for (const auto & name : names) {
        layers.push_back(
            load_quant_linear(
                mfq, name,
                slices_override != nullptr
                    ? std::optional<TensorParallelAxis>(
                        TensorParallelAxis::Output)
                    : std::nullopt,
                slices_override));
    }
    if (required_compatible_prefix > layers.size()) {
        throw std::runtime_error("invalid required quantized-group prefix length");
    }
    // A compatible prefix is fused opportunistically by make_quant_group().
    // Mixed-precision Q/K and gate/up pairs remain separate QuantLinear
    // branches and preserve the recipe-selected layouts.
    return make_quant_group(
        std::move(layers), preserve_projection_boundaries);
}

static std::vector<mfq::TensorParallelSlice>
tensor_parallel_output_slices_for_input(
        const QuantLinear & input_parallel) {
    if (!input_parallel.tensor_parallel() ||
        input_parallel.tensor_parallel_axis !=
            TensorParallelAxis::Input) {
        return {};
    }
    std::vector<mfq::TensorParallelSlice> result;
    result.reserve(
        input_parallel.tensor_parallel_shards.size());
    for (const auto & shard :
         input_parallel.tensor_parallel_shards) {
        result.push_back({
            shard.device,
            shard.input_begin,
            shard.input_end,
        });
    }
    mfq::validate_tensor_parallel_slices(
        result,
        input_parallel.neuron_len(),
        1);
    return result;
}

QuantLinearGroup load_paired_gate_up(
        const mfq::ModelSource & mfq,
        const std::vector<std::string> & names,
        const QuantLinear & down,
        size_t required_compatible_prefix,
        bool preserve_projection_boundaries) {
    auto slices =
        tensor_parallel_output_slices_for_input(
            down);
    return slices.empty()
        ? load_quant_group(
            mfq, names,
            required_compatible_prefix,
            nullptr,
            preserve_projection_boundaries)
        : load_quant_group(
            mfq, names,
            required_compatible_prefix,
            &slices,
            preserve_projection_boundaries);
}

DenseLinearGroup make_dense_group(const std::vector<mfq_tensor_backend::Tensor> & ws) {
    if (ws.empty()) throw std::runtime_error("empty dense group");
    DenseLinearGroup g;
    std::vector<mfq_tensor_backend::Tensor> parts;
    parts.reserve(ws.size());
    for (const auto & w : ws) {
        if (w.dim() != 2) throw std::runtime_error("dense linear group expects 2D weights");
        if (!parts.empty() && w.size(1) != parts[0].size(1)) {
            throw std::runtime_error("cannot group dense tensors with different input width");
        }
        g.outs.push_back(w.size(0));
        parts.push_back(w.to(mfq_tensor_backend::kFloat32));
    }
    g.w = mfq_tensor_backend::cat(parts, 0).contiguous();
    return g;
}

static mfq_tensor_backend::Tensor dequant_nint_dense_f32(const NintWeight & w) {
    mfq_tensor_backend::Tensor dense;
    if (w.q8_zero) {
        dense = nint8_zero_dequant_cuda(
            w.q_packed, w.q8_zero_scale, w.neuron_len);
    } else {
        dense = nint_decode_cuda(
            w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
            w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            w.neuron_len, w.gs);
    }
    return dense.to(mfq_tensor_backend::kFloat32).contiguous();
}

static mfq_tensor_backend::Tensor dequant_quant_linear_f32(const QuantLinear & linear) {
    if (!linear.tensor_parallel()) {
        if (linear.is_nint()) {
            return dequant_nint_dense_f32(linear.nint.w);
        }
        if (linear.is_nvq()) {
            return nvq_dequant(linear.nvq.w)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp8()) {
            return mxfp8_dequant_cuda(
                linear.mxfp8.weight.values,
                linear.mxfp8.weight.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp4()) {
            return mxfp4_dequant_cuda(
                linear.mxfp4.weight.values,
                linear.mxfp4.weight.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp4_sq()) {
            const auto & weight = linear.mxfp4_sq.weight;
            return mxfp4_sq_dequant_cuda(
                weight.blob,
                weight.row_q,
                weight.row_symbol_byte_offsets,
                weight.row_auxiliary,
                weight.bits,
                weight.out,
                weight.neuron_len,
                weight.matrix_scale_base,
                weight.q_sum,
                weight.sq4_rows,
                true).contiguous();
        }
        if (linear.is_fp8_sq()) {
            return dequant_fp8_sq(linear.fp8_sq.weight, true).contiguous();
        }
        if (linear.is_dense()) {
            return linear.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        }
        throw std::runtime_error(
            "unsupported linear kind for FP32 reconstruction");
    }
    if (linear.tensor_parallel_axis != TensorParallelAxis::Output &&
        linear.tensor_parallel_axis != TensorParallelAxis::Input) {
        throw std::runtime_error(
            "cannot reconstruct a mirrored tensor-parallel linear");
    }
    const int primary = model_parallel_primary_device();
    std::vector<mfq_tensor_backend::Tensor> parts;
    parts.reserve(linear.tensor_parallel_shards.size());
    for (const auto & shard : linear.tensor_parallel_shards) {
        MfqCudaGuard shard_guard(shard.device);
        mfq_tensor_backend::Tensor part;
        if (shard.kind == QuantLinearKind::Nint) {
            part = dequant_nint_dense_f32(shard.nint);
        } else if (shard.kind == QuantLinearKind::Nvq) {
            part = nvq_dequant(shard.nvq)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp8) {
            part = mxfp8_dequant_cuda(
                shard.mxfp8.values,
                shard.mxfp8.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp4) {
            part = mxfp4_dequant_cuda(
                shard.mxfp4.values, shard.mxfp4.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Dense) {
            part = shard.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        } else {
            throw std::runtime_error(
                "unsupported tensor-parallel shard kind for reconstruction");
        }
        parts.push_back(
            tensor_to_cuda_device(part, primary)
                .to(mfq_tensor_backend::kFloat32).contiguous());
    }
    MfqCudaGuard primary_guard(primary);
    auto dense = mfq_tensor_backend::cat(
        parts,
        linear.tensor_parallel_axis == TensorParallelAxis::Output
            ? 0 : 1).contiguous();
    if (dense.size(0) != linear.out() ||
        dense.size(1) != linear.neuron_len()) {
        throw std::runtime_error(
            "reconstructed tensor-parallel linear shape mismatch");
    }
    return dense;
}

DenseLinearGroup make_fp32_quant_group(QuantLinearGroup group) {
    DenseLinearGroup dense;
    dense.outs = group.outs;
    if (group.nint_grouped) {
        if (group.nint.split_w.empty()) {
            dense.w = dequant_nint_dense_f32(group.nint.w);
        } else {
            std::vector<mfq_tensor_backend::Tensor> parts;
            parts.reserve(group.nint.split_w.size());
            for (const auto & weight : group.nint.split_w) {
                parts.push_back(dequant_nint_dense_f32(weight));
            }
            dense.w = mfq_tensor_backend::cat(parts, 0).contiguous();
        }
    } else {
        std::vector<mfq_tensor_backend::Tensor> parts;
        parts.reserve(group.layers.size());
        for (const auto & linear : group.layers) {
            parts.push_back(dequant_quant_linear_f32(linear));
        }
        dense.w = mfq_tensor_backend::cat(parts, 0).contiguous();
    }
    MFQ_RUNTIME_CHECK(
        dense.w.dim() == 2 &&
            dense.w.size(0) ==
                std::accumulate(
                    dense.outs.begin(), dense.outs.end(), int64_t{0}),
        "FP32 compressor projection shape mismatch");
    return dense;
}

mfq_tensor_backend::Tensor quant_linear_reference_weight(
        const QuantLinear& linear) {
    if (linear.is_nint()) {
        const auto& weight = linear.nint.w;
        return weight.q8_zero
            ? nint8_zero_dequant_cuda(
                  weight.q_packed, weight.q8_zero_scale,
                  weight.neuron_len)
            : nint_decode_cuda(
                  weight.q_packed, weight.row_q_bits,
                  weight.row_q_bit_offsets, weight.sub_scale,
                  weight.sub_min, weight.neuron_scale,
                  weight.neuron_min, weight.neuron_len, weight.gs);
    }
    if (linear.is_nvq()) return nvq_dequant(linear.nvq.w);
    if (linear.is_mxfp4()) {
        return mxfp4_dequant_cuda(
            linear.mxfp4.weight.values, linear.mxfp4.weight.scales);
    }
    if (linear.is_mxfp4_sq()) {
        const auto& weight = linear.mxfp4_sq.weight;
        return mxfp4_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.row_auxiliary, weight.bits, weight.out,
            weight.neuron_len, weight.matrix_scale_base,
            weight.q_sum, weight.sq4_rows, false);
    }
    if (linear.is_fp8_sq()) {
        return dequant_fp8_sq(linear.fp8_sq.weight, false);
    }
    if (linear.is_mxfp8()) {
        return mxfp8_cpu_reference(linear.mxfp8.weight);
    }
    if (linear.is_dense()) return linear.dense;
    throw std::runtime_error("unsupported linear reference format");
}
