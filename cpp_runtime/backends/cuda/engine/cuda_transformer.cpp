#include "cuda_transformer.h"

std::string layer_name(const std::string & templ, int i) {
    std::string s = templ;
    auto p = s.find("{i}");
    if (p != std::string::npos) s.replace(p, 3, std::to_string(i));
    return s;
}

mfq_tensor_backend::Tensor qwen_rms_norm(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor weight,
        double eps,
        double weight_offset) {
    if (!x.is_cuda()) {
        auto xf = x.contiguous().to(mfq_tensor_backend::kFloat32);
        auto wf = weight.contiguous().to(mfq_tensor_backend::kFloat32);
        if (weight_offset != 0.0) wf = wf + weight_offset;
        auto inverse = mfq_tensor_backend::rsqrt(
            xf.square().mean(-1, true) + eps);
        return (xf * inverse * wf).contiguous();
    }
    return rms_norm_offset_cuda(x, weight, eps, weight_offset);
}

mfq_tensor_backend::Tensor qwen_rms_norm_bf16(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor weight,
        double eps,
        double weight_offset) {
    auto input = x.contiguous().to(mfq_tensor_backend::kBFloat16);
    const char * fused_env = std::getenv("MFQ_MINICPM_FUSED_BF16_RMSNORM");
    if (input.is_cuda() &&
            (fused_env == nullptr || fused_env[0] != '0')) {
        return qwen_rms_norm_bf16_cuda(
            input, weight.contiguous(), eps, weight_offset);
    }
    auto xf = input.to(mfq_tensor_backend::kFloat32);
    auto inverse = mfq_tensor_backend::rsqrt(
        xf.square().mean(-1, true) + eps);
    auto normalized = (xf * inverse).to(mfq_tensor_backend::kBFloat16);
    auto scale = weight.contiguous().to(mfq_tensor_backend::kBFloat16);
    if (weight_offset != 0.0) scale = scale + weight_offset;
    return (scale * normalized).contiguous();
}

mfq_tensor_backend::Tensor gemma_rms_norm_f16(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor weight,
        double eps,
        double weight_offset) {
    MFQ_RUNTIME_CHECK(
        x.scalar_type() == mfq_tensor_backend::kFloat16,
        "gemma_rms_norm_f16: activation must remain f16");
    return rms_norm_f16_cuda(
        x.contiguous(), weight, eps, weight_offset);
}

void prepare_ffn_workspaces(FFN & f) {
    if (g_loading_cpu_layer) return;
    if (f.down.tensor_parallel()) return;
    if (f.gate_up.nvq_prefix2 && f.gate_up.layers.size() == 2 && f.down.is_nvq() &&
        f.gate_up.outs.size() == 2 && f.gate_up.outs[0] == f.gate_up.outs[1] &&
        f.gate_up.outs[0] == f.down.nvq.w.neuron_len) {
        NvqWorkspace & ws = f.gate_up.layers[0].nvq.w.workspace(1);
        ws.swiglu_scratch = mfq_tensor_backend::empty(
            {f.gate_up.outs[0]}, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        (void)f.down.nvq.w.workspace(1);
    }
    if (f.important_neurons) {
        prepare_ffn_workspaces(*f.important_neurons);
    }
}

void load_important_neuron_branch(
        const mfq::ModelSource & mfq,
        int64_t hidden_size,
        int64_t intermediate_size,
        FFN & f,
        const std::string & down_name,
        const std::string & gate_name,
        const std::string & up_name) {
    const std::string down_high = down_name + ".in_high";
    const std::string gate_high = gate_name + ".in_high";
    const std::string up_high = up_name + ".in_high";
    const bool has_down = has_tensor(mfq, down_high);
    const bool has_gate = has_tensor(mfq, gate_high);
    const bool has_up = has_tensor(mfq, up_high);
    if (!has_down && !has_gate && !has_up) {
        return;
    }
    if (!has_down || !has_gate || !has_up) {
        throw std::runtime_error(
            "important-neuron FFN requires matching gate/up/down .in_high records");
    }
    if (f.is_moe) {
        throw std::runtime_error(
            "important-neuron records are unsupported on routed MoE FFNs");
    }

    auto high = std::make_unique<FFN>();
    high->down = load_quant_linear(
        mfq, down_high, TensorParallelAxis::Input);
    high->gate_up = load_paired_gate_up(
        mfq, {gate_high, up_high}, high->down);
    high->geglu = f.geglu;
    high->swiglu_limit = f.swiglu_limit;

    if (f.gate_up.outs.size() != 2 ||
        high->gate_up.outs.size() != 2 ||
        f.gate_up.outs[0] != f.gate_up.outs[1] ||
        high->gate_up.outs[0] != high->gate_up.outs[1] ||
        f.down.out() != hidden_size ||
        high->down.out() != hidden_size ||
        f.down.neuron_len() != f.gate_up.outs[0] ||
        high->down.neuron_len() != high->gate_up.outs[0] ||
        f.down.neuron_len() + high->down.neuron_len() !=
            intermediate_size) {
        throw std::runtime_error(
            "important-neuron FFN tensor shapes disagree with model config");
    }
    f.important_neurons = std::move(high);
}
