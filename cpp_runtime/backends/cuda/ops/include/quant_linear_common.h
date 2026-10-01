#pragma once

#include "format.h"
#include "fp8_sq.h"
#include "mx.h"
#include "mxfp4_sq.h"
#include "moe_types.h"
#include "nint.h"
#include "vq.h"

#include "cuda_execution.h"
#include "mfq_cuda_ops.h"
#include "mfq/model_source.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct QuantLinear;
struct QuantLinearGroup;
struct DenseLinearGroup;

struct MfqDropFileCacheGuard {
    CudaExecutionContext& execution;
    bool previous;

    MfqDropFileCacheGuard(
            bool enabled, CudaExecutionContext& context)
        : execution(context), previous(context.drop_file_cache) {
        execution.drop_file_cache = enabled;
    }

    ~MfqDropFileCacheGuard() {
        execution.drop_file_cache = previous;
    }
};

bool decode_branch_parallel_enabled(
    bool serial_branches,
    std::int64_t rows);

mfq_tensor_backend::Tensor run_nint_linear(
    CudaExecutionContext& execution, const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> gate = mfq_nullopt, int mode = 0);
mfq_tensor_backend::Tensor run_nvq_linear(
    CudaExecutionContext& execution, const NvqWeight& weight,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> gate = mfq_nullopt, int mode = 0);

mfq_tensor_backend::Tensor tensor_to_cuda_device(
    CudaExecutionContext& execution,
    mfq_tensor_backend::Tensor value,
    int device,
    mfq_tensor_backend::Tensor reusable = {});
mfq_tensor_backend::Tensor run_quant_linear_shard(
    CudaExecutionContext& execution,
    const struct QuantLinearShard& shard,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> gate = mfq_nullopt,
    int gate_mode = 0);

struct DecodeGraphBranchScope {
    CudaExecutionContext& execution;
    bool previous;

    explicit DecodeGraphBranchScope(CudaExecutionContext& context)
        : execution(context),
          previous(context.decode_graph_serial_branches) {
        execution.decode_graph_serial_branches = true;
    }
    ~DecodeGraphBranchScope() {
        execution.decode_graph_serial_branches = previous;
    }
};

struct DecodeGraphTpProjectionScope {
    CudaExecutionContext& execution;
    bool previous;

    explicit DecodeGraphTpProjectionScope(CudaExecutionContext& context)
        : execution(context),
          previous(context.decode_graph_tp_projection_major) {
        execution.decode_graph_tp_projection_major = true;
    }
    ~DecodeGraphTpProjectionScope() {
        execution.decode_graph_tp_projection_major = previous;
    }
};

struct CudaIndependentBranchExecutor {
    using Stream =
        decltype(mfq_get_stream_from_pool(false));

    int device = -1;
    cudaEvent_t ready = nullptr;
    std::vector<Stream> streams;
    std::vector<cudaEvent_t> completed;

    ~CudaIndependentBranchExecutor() {
        for (cudaEvent_t event : completed) {
            if (event != nullptr) {
                (void)cudaEventDestroy(event);
            }
        }
        if (ready != nullptr) {
            (void)cudaEventDestroy(ready);
        }
    }

    bool ensure(size_t branches, const Stream & parent) {
        const int parent_device = parent.device_index();
        if (device >= 0 && device != parent_device) {
            return false;
        }
        if (streams.size() >= branches) {
            return true;
        }
        cudaStreamCaptureStatus capture_status =
            cudaStreamCaptureStatusNone;
        MFQ_CUDA_CHECK(cudaStreamIsCapturing(
            parent.stream(), &capture_status));
        if (capture_status != cudaStreamCaptureStatusNone) {
            return false;
        }
        device = parent_device;
        if (ready == nullptr) {
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &ready, cudaEventDisableTiming));
        }
        while (streams.size() < branches) {
            streams.push_back(
                mfq_get_stream_from_pool(false, device));
            cudaEvent_t event = nullptr;
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(
                &event, cudaEventDisableTiming));
            completed.push_back(event);
        }
        return true;
    }

    template <typename Fn>
    bool run(
            size_t branches,
            Fn && fn,
            std::vector<mfq_tensor_backend::Tensor> & outputs) {
        if (branches < 2) {
            return false;
        }
        const Stream parent =
            mfq_get_current_cuda_stream();
        if (!ensure(branches, parent)) {
            return false;
        }

        outputs.resize(branches);
        MFQ_CUDA_CHECK(cudaEventRecord(
            ready, parent.stream()));
        for (size_t index = 0; index < branches; ++index) {
            const Stream branch_stream = streams[index];
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                branch_stream.stream(), ready, 0));
            {
                MfqCudaStreamGuard guard(
                    branch_stream);
                outputs[index] = fn(index);
            }
            MFQ_CUDA_CHECK(cudaEventRecord(
                completed[index], branch_stream.stream()));
        }
        for (size_t index = 0; index < branches; ++index) {
            MFQ_CUDA_CHECK(cudaStreamWaitEvent(
                parent.stream(), completed[index], 0));
            if (outputs[index].defined()) {
                mfq_cuda_record_stream(outputs[index], parent);
            }
        }
        return true;
    }
};
