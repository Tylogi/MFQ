#pragma once

#include "nint.h"

struct NvqWorkspace {
    int M = 0;
    int K_pad = 0;
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
    mfq_tensor_backend::Tensor swiglu_scratch;
};

struct NvqWeight {
    mfq_tensor_backend::Tensor indices_packed;
    mfq_tensor_backend::Tensor aux_packed;
    mfq_tensor_backend::Tensor sub_scale_packed;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor codebook;
    int64_t format = 0;
    int64_t kernel_format = 0;
    int64_t sign_mode = 0;
    int64_t sub_bits = 0;
    int64_t gs = 0;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t neuron_len = 0;
    std::vector<int64_t> shape;
    mutable std::unordered_map<int, NvqWorkspace> workspaces;

    NvqWorkspace & workspace(int M) const {
        int K_pad = (int)(ng * gs);
        auto it = workspaces.find(M);
        if (it != workspaces.end() && it->second.K_pad == K_pad) return it->second;
        NvqWorkspace ws;
        ws.M = M;
        ws.K_pad = K_pad;
        const auto workspace_options = indices_packed.options();
        ws.qx = mfq_tensor_backend::empty(
            {M, K_pad}, workspace_options.dtype(mfq_tensor_backend::kInt8));
        ws.xscale = mfq_tensor_backend::empty(
            {M, ng}, workspace_options.dtype(mfq_tensor_backend::kFloat32));
        return workspaces.emplace(M, std::move(ws)).first->second;
    }
};

bool nvq_fusion_enabled();
mfq_tensor_backend::Tensor nvq_matmul_multi2(
    const NvqWeight& first,
    const NvqWeight& second,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul_swiglu(
    const NvqWeight& gate,
    const NvqWeight& up,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul(
    const NvqWeight& weight, mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul_input_mul(
    const NvqWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);

struct NvqLinear {
    NvqWeight w;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul(w, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    mfq_tensor_backend::Tensor forward_input_mul(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate, int mode) const {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul_input_mul(
            w, x.reshape({-1, x.size(-1)}), gate.reshape({-1, gate.size(-1)}), mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

struct NvqCpu {
    int format = 0;
    int sign_mode = 0;
    int sub_bits = 0;
    int gs = 0;
    int axis = 0;
    int neuron_len = 0;
    int out = 0;
    int ng = 0;
    int nvec = 0;
    int nsign = 0;
    std::vector<int64_t> shape;
    std::vector<uint8_t> indices_packed;
    std::vector<uint8_t> aux_packed;
    std::vector<uint8_t> sub_scale_packed;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<int8_t> codebook;
};

struct NepqCpu {
    int profile = -1;
    int format = 0;
    int state_bits = 0;
    int index_bits = 0;
    int aux_bits = 0;
    int table_bytes = 0;
    int runtime_table_bytes = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int bank_count = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
    int ng = 0;
    int nvec = 0;
    int nsuper = 0;
    std::vector<uint8_t> indices_packed;
    std::vector<uint8_t> aux_packed;
    std::vector<uint8_t> state_packed;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<int8_t> table_pool;
    std::vector<int8_t> grouped_table_pool;
    std::vector<uint8_t> bank_ids;
    std::vector<int8_t> rotation_signs;
    bool residual = false;
    bool residual_second = false;
    int residual_position_bits = 0;
    int residual_record_bits = 0;
    int residual_block_vectors = 0;
    int residual_blocks_per_row = 0;
    std::vector<uint16_t> residual_codebook_h;
    std::vector<int16_t> residual_first;
    std::vector<int16_t> residual_second_dense;
};

struct NepqWeight {
    mfq_tensor_backend::Tensor indices_packed;
    mfq_tensor_backend::Tensor aux_packed;
    mfq_tensor_backend::Tensor state_packed;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor table_pool;
    mfq_tensor_backend::Tensor grouped_table_pool;
    mfq_tensor_backend::Tensor bank_ids;
    mfq_tensor_backend::Tensor rotation_signs;
    int format = 0;
    int state_bits = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int ng = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
    bool residual = false;
    int residual_position_bits = 0;
    int residual_block_vectors = 0;
    mfq_tensor_backend::Tensor residual_codebook;
    mfq_tensor_backend::Tensor residual_first;
    mfq_tensor_backend::Tensor residual_second;
};

NvqCpu unpack_nvq(const std::vector<uint8_t>& blob, const std::string& dtype);
NvqCpu slice_nvq_cpu(
    const NvqCpu& source, TensorParallelAxis axis, int64_t begin, int64_t end);
NvqCpu select_nvq_cpu_rows(
    const NvqCpu& source, const std::vector<int64_t>& rows);
NvqWeight to_device_nvq(const NvqCpu& source, bool cuda);
NvqWeight to_gpu_nvq(const NvqCpu& source);
NvqWeight to_cuda_device_nvq(const NvqCpu& source, int device);
NvqWeight to_cpu_nvq(const NvqCpu& source);
NepqCpu unpack_nepq(
    const std::vector<uint8_t>& blob,
    const std::string& dtype,
    const std::vector<uint8_t>& runtime_payload);
NepqCpu select_nepq_cpu_rows(
    const NepqCpu& source,
    const std::vector<int64_t>& rows,
    int output_per_expert,
    int selected_experts = -1);
NepqWeight to_device_nepq(const NepqCpu& source, bool cuda);
NepqWeight to_gpu_nepq(const NepqCpu& source);
NepqWeight to_cpu_nepq(const NepqCpu& source);
NvqWeight load_nvq_gpu(
    const mfq::ModelSource& source, const std::string& name);

bool nvq_pair_compatible(const NvqWeight& first, const NvqWeight& second);
mfq_tensor_backend::Tensor nvq_dequant(const NvqWeight& weight);
mfq_tensor_backend::Tensor nvq_embedding(
    const NvqWeight& weight, mfq_tensor_backend::Tensor token_ids);
