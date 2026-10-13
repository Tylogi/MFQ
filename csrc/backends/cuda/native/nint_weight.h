#pragma once
#include "native/tensor/backend.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// Quantized NINT weight plus its per-M activation workspace.  Shared by the
// kernels that read it and the ops that own it.

struct Workspace {
    int M = 0;
    int K_pad = 0;
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
};

struct NintWeight {
    mfq_tensor_backend::Tensor q_packed;
    mfq_tensor_backend::Tensor row_q_bits;
    mfq_tensor_backend::Tensor row_q_bit_offsets;
    mfq_tensor_backend::Tensor q8_zero_scale;
    mfq_tensor_backend::Tensor sub_scale;
    mfq_tensor_backend::Tensor sub_min;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor neuron_min;
    // Experimental routed row records; populated only by the comparison bench.
    mfq_tensor_backend::Tensor route_metadata;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t gs = 0;
    int64_t bits = 0;
    int64_t neuron_len = 0;
    int64_t q_expert_stride = 0;
    int format_version = 2;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
    bool q8_zero = false;
    // Proven once from CPU row metadata; false for unclassified/grouped weights.
    bool aligned_q8 = false;
    std::vector<int64_t> shape;
    mutable std::unordered_map<int, Workspace> workspaces;

    Workspace & workspace(int M) const {
        int K_pad = (int)(ng * gs);
        auto it = workspaces.find(M);
        if (it != workspaces.end() && it->second.K_pad == K_pad) return it->second;
        Workspace ws;
        ws.M = M;
        ws.K_pad = K_pad;
        const auto workspace_options = q_packed.options();
        ws.qx = mfq_tensor_backend::empty(
            {M, K_pad}, workspace_options.dtype(mfq_tensor_backend::kInt8));
        ws.xscale = mfq_tensor_backend::empty(
            {M, ng}, workspace_options.dtype(mfq_tensor_backend::kFloat32));
        auto res = workspaces.emplace(M, std::move(ws));
        return res.first->second;
    }

};
