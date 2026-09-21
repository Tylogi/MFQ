#pragma once

#include "../../runtime/cuda_transformer.h"
#include "models/qwen35.h"

namespace mfq::cuda::qwen35 {

struct LinearRecurrentInputs {
    mfq_tensor_backend::Tensor qk;
    mfq_tensor_backend::Tensor value;
    mfq_tensor_backend::Tensor qkv;
    mfq_tensor_backend::Tensor gate;
    mfq_tensor_backend::Tensor beta;
    int64_t batch = 0;
    int64_t tokens = 0;
    bool split = false;
};

struct LinearAttentionBlock final : ::Block {
    mfq::models::qwen35::Config qwen_config;
    mfq_tensor_backend::Tensor attn_norm, ffn_norm, conv_weight, conv_bias, dt_bias, a_log, linear_norm;
    bool split_in_proj = false;
    bool split_dense_zab = false;
    bool dense_ab_tail = false;
    bool ab_is_nint = false;
    bool dense_out_proj = false;
    bool tiled_v_heads = false;
    QuantLinearGroup in_proj;
    QuantLinearGroup qkv_proj;
    QuantLinearGroup qkvz_proj;
    QuantLinear z_proj;
    QuantLinearGroup ab_nint_proj;
    DenseLinearGroup ab_proj;
    DenseLinearGroup zab_proj;
    QuantLinear out_proj;
    mfq_tensor_backend::Tensor out_proj_dense;
    FFN ffn;
    mfq_tensor_backend::Tensor conv_state, gdn_state;
    mfq_tensor_backend::Tensor speculative_conv, speculative_gdn;
    bool speculative_pending = false;
    int64_t speculative_ffn_batches = 0;
    int64_t speculative_projection_batches = 0;
    LinearRecurrentInputs speculative_recurrent;
    int64_t speculative_start = -1;
    int64_t speculative_confirmed = 0;
    int64_t speculative_tokens = 0;

    bool supports_speculation() const noexcept override { return true; }
    mfq_tensor_backend::Tensor forward_context(
        mfq_tensor_backend::Tensor input,
        const Block::Context& context,
        const RopeCache& rope) override;
    void commit_speculative() noexcept override;
    void rollback_speculative(int64_t keep_position) override;

    void reset(int64_t B) override {
        clear_speculative();
        if (conv_state.defined() && gdn_state.defined() &&
            conv_state.size(0) == B && gdn_state.size(0) == B) {
            conv_state.zero_();
            gdn_state.zero_();
            return;
        }
        conv_state = mfq_tensor_backend::Tensor();
        gdn_state = mfq_tensor_backend::Tensor();
    }

    void clear_speculative() noexcept;

    mfq_tensor_backend::Tensor forward_cpu(
            mfq_tensor_backend::Tensor x) {
        const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
        const int64_t nk = qwen_config.linear_num_key_heads;
        const int64_t nv = qwen_config.linear_num_value_heads;
        const int64_t dk = qwen_config.linear_key_head_dim;
        const int64_t dv = qwen_config.linear_value_head_dim;
        const int64_t ksz = qwen_config.linear_k_size();
        const int64_t vsz = qwen_config.linear_v_size();
        const int64_t conv_dim = 2 * ksz + vsz;
        if (!conv_state.defined()) {
            auto f32 = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32);
            conv_state = mfq_tensor_backend::zeros(
                {B, qwen_config.linear_conv_kernel_dim - 1, conv_dim}, f32);
            gdn_state = mfq_tensor_backend::zeros({B, nv, dv, dv}, f32);
        }

        auto residual = x;
        auto xn = qwen_rms_norm(
            x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
            attn_norm, qwen_config.rms_norm_eps, 1.0)
            .reshape({B, T, H});
        mfq_tensor_backend::Tensor qkv, qk_part, v_part, z, alpha_raw, beta_raw;
        if (dense_ab_tail) {
            auto parts = in_proj.forward(xn);
            qkv = parts[0].to(mfq_tensor_backend::kFloat32);
            z = parts[1];
            auto ab = ab_proj.forward(xn);
            alpha_raw = ab[0];
            beta_raw = ab[1];
        } else if (split_in_proj) {
            if (split_dense_zab) {
                auto qkv_parts = qkv_proj.forward(xn);
                qk_part = qkv_parts[0];
                v_part = qkv_parts[1];
                auto zab = zab_proj.forward(xn);
                z = zab[0];
                alpha_raw = zab[1];
                beta_raw = zab[2];
            } else {
                auto qkv_parts = qkv_proj.forward(xn);
                qk_part = qkv_parts[0];
                v_part = qkv_parts[1];
                z = z_proj.forward(xn);
                auto ab = ab_is_nint
                    ? ab_nint_proj.forward(xn)
                    : ab_proj.forward(xn);
                alpha_raw = ab[0];
                beta_raw = ab[1];
            }
            qkv = mfq_tensor_backend::cat(
                {qk_part.to(mfq_tensor_backend::kFloat32), v_part.to(mfq_tensor_backend::kFloat32)}, -1);
        } else {
            auto parts = in_proj.forward(xn);
            qkv = parts[0].to(mfq_tensor_backend::kFloat32);
            z = parts[1];
            alpha_raw = parts[2];
            beta_raw = parts[3];
        }

        auto beta = mfq_tensor_backend::sigmoid(
            beta_raw.to(mfq_tensor_backend::kFloat32).reshape({B, T, nv}));
        auto alpha = alpha_raw.to(mfq_tensor_backend::kFloat32).reshape({B, T, nv});
        auto gate = mfq_tensor_backend::softplus(
            alpha + dt_bias.to(mfq_tensor_backend::kFloat32).reshape({1, 1, nv}));
        auto coefficient = -mfq_tensor_backend::exp(a_log.to(mfq_tensor_backend::kFloat32));
        gate = gate * coefficient.reshape({1, 1, nv});

        auto conv_input = mfq_tensor_backend::cat({conv_state, qkv}, 1);
        conv_state.copy_(conv_input.narrow(
            1, conv_input.size(1) - (qwen_config.linear_conv_kernel_dim - 1),
            qwen_config.linear_conv_kernel_dim - 1));
        auto weight = conv_weight.to(mfq_tensor_backend::kFloat32);
        if (weight.dim() == 3) weight = weight.squeeze(1);
        if (weight.dim() != 2) {
            throw std::runtime_error("CPU SSM convolution weight must be rank 2 or 3");
        }
        if (weight.size(0) != conv_dim && weight.size(1) == conv_dim) {
            weight = weight.transpose(0, 1).contiguous();
        }
        MFQ_RUNTIME_CHECK(
            weight.size(0) == conv_dim &&
            weight.size(1) == qwen_config.linear_conv_kernel_dim,
            "CPU SSM convolution geometry mismatch");
        auto windows = conv_input.unfold(
            1, qwen_config.linear_conv_kernel_dim, 1);
        auto conv = (windows * weight.reshape(
            {1, 1, conv_dim, qwen_config.linear_conv_kernel_dim})).sum(-1);
        if (conv_bias.defined()) {
            conv = conv + conv_bias.to(mfq_tensor_backend::kFloat32).reshape({1, 1, conv_dim});
        }
        conv = mfq_tensor_backend::silu(conv);

        auto q = conv.narrow(-1, 0, ksz)
            .reshape({B, T, nk, dk}).transpose(1, 2).contiguous();
        auto k = conv.narrow(-1, ksz, ksz)
            .reshape({B, T, nk, dk}).transpose(1, 2).contiguous();
        auto v = conv.narrow(-1, 2 * ksz, vsz)
            .reshape({B, T, nv, dv}).transpose(1, 2).contiguous();
        q = q / mfq_tensor_backend::sqrt(q.square().sum(-1, true))
            .clamp_min(qwen_config.rms_norm_eps);
        k = k / mfq_tensor_backend::sqrt(k.square().sum(-1, true))
            .clamp_min(qwen_config.rms_norm_eps);
        if (nk != nv) {
            MFQ_RUNTIME_CHECK(nv % nk == 0, "CPU GDN head ratio is invalid");
            const int64_t repeat = nv / nk;
            if (tiled_v_heads) {
                q = q.repeat({1, repeat, 1, 1});
                k = k.repeat({1, repeat, 1, 1});
            } else {
                q = q.repeat_interleave(repeat, 1);
                k = k.repeat_interleave(repeat, 1);
            }
        }

        auto gate_t = gate.transpose(1, 2).contiguous();
        auto beta_t = beta.transpose(1, 2).contiguous();
        auto state = gdn_state;
        std::vector<mfq_tensor_backend::Tensor> outputs;
        outputs.reserve(static_cast<size_t>(T));
        const double retrieve_scale = 1.0 / std::sqrt(static_cast<double>(dk));
        for (int64_t token = 0; token < T; ++token) {
            auto qt = q.select(2, token);
            auto kt = k.select(2, token);
            auto vt = v.select(2, token);
            state = state * mfq_tensor_backend::exp(gate_t.select(2, token))
                .reshape({B, nv, 1, 1});
            auto state_k = mfq_tensor_backend::matmul(
                state.transpose(-1, -2), kt.unsqueeze(-1)).squeeze(-1);
            auto delta = (vt - state_k) *
                beta_t.select(2, token).unsqueeze(-1);
            state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2);
            outputs.push_back(
                mfq_tensor_backend::matmul(
                    state.transpose(-1, -2), qt.unsqueeze(-1))
                    .squeeze(-1) * retrieve_scale);
        }
        gdn_state = state.contiguous();
        auto y = mfq_tensor_backend::stack(outputs, 2);
        z = z.reshape({B, T, nv, dv}).transpose(1, 2).contiguous();
        auto y_flat = y.reshape({-1, dv}).to(mfq_tensor_backend::kFloat32);
        auto inverse = mfq_tensor_backend::rsqrt(
            y_flat.square().mean(-1, true) +
            qwen_config.rms_norm_eps);
        auto y_norm = (y_flat * inverse *
            linear_norm.to(mfq_tensor_backend::kFloat32)).reshape_as(y);
        auto yf = y_norm.transpose(1, 2).contiguous().reshape({B, T, vsz});
        auto zf = z.transpose(1, 2).contiguous().reshape({B, T, vsz});
        mfq_tensor_backend::Tensor attention_output;
        if (dense_out_proj) {
            auto dtype = out_proj_dense.scalar_type();
            auto gated = yf.to(dtype) * mfq_tensor_backend::silu(zf.to(dtype));
            attention_output = mfq_tensor_backend::matmul(
                gated.reshape({B * T, vsz}),
                out_proj_dense.transpose(0, 1)).reshape({B, T, H});
        } else {
            attention_output = out_proj.forward_input_mul(yf, zf, 2);
        }
        x = (residual.to(mfq_tensor_backend::kFloat32) +
            attention_output.to(mfq_tensor_backend::kFloat32))
            .to(residual.scalar_type()).contiguous();
        residual = x;
        xn = qwen_rms_norm(
            x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
            ffn_norm, qwen_config.rms_norm_eps, 1.0)
            .reshape({B, T, H});
        auto ff = ffn.forward(xn.reshape({B * T, H})).reshape({B, T, H});
        return (residual.to(mfq_tensor_backend::kFloat32) + ff.to(mfq_tensor_backend::kFloat32))
            .to(residual.scalar_type()).contiguous();
    }

    std::array<mfq_tensor_backend::Tensor, 2> forward_attention_cuda(
            mfq_tensor_backend::Tensor x,
            LinearRecurrentInputs* captured = nullptr) {
        int64_t B = x.size(0), T = x.size(1), H = x.size(2);
        int64_t nk = qwen_config.linear_num_key_heads, nv = qwen_config.linear_num_value_heads;
        int64_t dk = qwen_config.linear_key_head_dim, dv = qwen_config.linear_value_head_dim;
        int64_t ksz = qwen_config.linear_k_size(), vsz = qwen_config.linear_v_size();
        int64_t conv_dim = 2 * ksz + vsz;
        if (!conv_state.defined()) {
            conv_state = mfq_tensor_backend::zeros({B, qwen_config.linear_conv_kernel_dim - 1, conv_dim}, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
            gdn_state = mfq_tensor_backend::zeros({B, nv, dv, dv}, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        }
        auto residual = x;
        auto xn = g_profiler.measure("linear.attn_norm", [&]() {
            return qwen_rms_norm(
                x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                attn_norm, qwen_config.rms_norm_eps, 1.0)
                .reshape({B, T, H});
        });
        mfq_tensor_backend::Tensor qkv, qk_part, v_part, z, alpha_raw, beta_raw;
        if (dense_ab_tail) {
            auto parts = g_profiler.measure("linear.in_proj", [&]() { return in_proj.forward(xn); });
            qkv = g_profiler.measure("linear.qkv_cast", [&]() { return parts[0].to(mfq_tensor_backend::kFloat32); });
            z = parts[1];
            auto ab = g_profiler.measure("linear.ab_proj", [&]() { return ab_proj.forward(xn); });
            alpha_raw = ab[0];
            beta_raw = ab[1];
        } else if (split_in_proj) {
            bool shared_projection_input = false;
            if (!shared_projection_input &&
                    !split_dense_zab && ab_is_nint &&
                    tensor_parallel_grouped_projections_enabled() &&
                    tensor_parallel_shared_linear_attention_input_enabled() &&
                    qkv_proj.layers.size() == 2) {
                QuantLinearProjectionRefs projections = {
                    &qkv_proj.layers[0], &qkv_proj.layers[1],
                    &z_proj,
                };
                const bool shared_tp_ab =
                    ab_nint_proj.layers.size() == 2 &&
                    ab_nint_proj.layers[0].tensor_parallel() &&
                    ab_nint_proj.layers[1].tensor_parallel();
                if (shared_tp_ab) {
                    projections.push_back(&ab_nint_proj.layers[0]);
                    projections.push_back(&ab_nint_proj.layers[1]);
                }
                if (tensor_parallel_output_projections_compatible(
                        projections)) {
                    auto parts = g_profiler.measure(
                        "linear.split_projections", [&]() {
                            return forward_tensor_parallel_output_projections(
                                xn, projections);
                        });
                    qk_part = parts[0];
                    v_part = parts[1];
                    z = parts[2];
                    if (shared_tp_ab) {
                        alpha_raw = parts[3];
                        beta_raw = parts[4];
                    } else {
                        auto ab = g_profiler.measure(
                            "linear.ab_proj", [&]() {
                                return ab_nint_proj.forward(xn);
                            });
                        alpha_raw = ab[0];
                        beta_raw = ab[1];
                    }
                    shared_projection_input = true;
                }
            }
            if (!shared_projection_input) {
                if (split_dense_zab) {
                    auto qkv_parts = g_profiler.measure("linear.qkv_proj", [&]() { return qkv_proj.forward(xn); });
                    qk_part = qkv_parts[0];
                    v_part = qkv_parts[1];
                    auto zab = g_profiler.measure("linear.zab_proj", [&]() { return zab_proj.forward(xn); });
                    z = zab[0];
                    alpha_raw = zab[1];
                    beta_raw = zab[2];
                } else {
                    auto qkv_parts = g_profiler.measure("linear.qkv_proj", [&]() { return qkv_proj.forward(xn); });
                    qk_part = qkv_parts[0];
                    v_part = qkv_parts[1];
                    z = g_profiler.measure("linear.z_proj", [&]() { return z_proj.forward(xn); });
                    auto ab = g_profiler.measure("linear.ab_proj", [&]() {
                        return ab_is_nint ? ab_nint_proj.forward(xn) : ab_proj.forward(xn);
                    });
                    alpha_raw = ab[0];
                    beta_raw = ab[1];
                }
            }
        } else {
            auto parts = g_profiler.measure("linear.in_proj", [&]() { return in_proj.forward(xn); });
            qkv = g_profiler.measure("linear.qkv_cast", [&]() { return parts[0].to(mfq_tensor_backend::kFloat32); });
            z = parts[1];
            alpha_raw = parts[2];
            beta_raw = parts[3];
        }
        auto gates = g_profiler.measure("linear.gates_fused", [&]() {
            return linear_gate_beta_cuda(alpha_raw.reshape({B, T, nv}), beta_raw.reshape({B, T, nv}), dt_bias, a_log);
        });
        auto gate_t = gates[0];
        auto beta_t = gates[1];
        if (captured != nullptr) {
            captured->qk = qk_part;
            captured->value = v_part;
            captured->qkv = qkv;
            captured->gate = gate_t;
            captured->beta = beta_t;
            captured->batch = B;
            captured->tokens = T;
            captured->split = split_in_proj;
        }
        auto bias = conv_bias.defined() ? conv_bias : mfq_tensor_backend::empty({0}, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32));
        auto recurrent_step = [&](mfq_tensor_backend::Tensor rq, mfq_tensor_backend::Tensor rk,
                mfq_tensor_backend::Tensor rv, mfq_tensor_backend::Tensor rg,
                mfq_tensor_backend::Tensor rb) {
            const char* transposed_env = std::getenv("MFQ_GDN_TRANSPOSED_STATE");
            const bool transposed = transposed_env == nullptr || transposed_env[0] != '0';
            if (transposed) {
                if (tiled_v_heads) return gdn_inplace_transposed_tiled_cuda(
                    rq.contiguous(), rk.contiguous(), rv.contiguous(), rg, rb, gdn_state);
                return gdn_inplace_transposed_cuda(
                    rq.contiguous(), rk.contiguous(), rv.contiguous(), rg, rb, gdn_state);
            }
            if (tiled_v_heads) return gdn_inplace_tiled_cuda(
                rq.contiguous(), rk.contiguous(), rv.contiguous(), rg, rb, gdn_state);
            return gdn_inplace_cuda(rq.contiguous(), rk.contiguous(), rv.contiguous(), rg, rb, gdn_state);
        };
        mfq_tensor_backend::Tensor q, k, v;
        const char * fused_prefill_env = std::getenv("MFQ_LINEAR_CONV_PREFILL_FUSED");
        bool fused_prefill = T >= 256 && (fused_prefill_env == nullptr || fused_prefill_env[0] != '0');
        if (T > 1 && split_in_proj && fused_prefill) {
            auto qkv_fast = g_profiler.measure("linear.conv_qkv_prefill", [&]() {
                return linear_conv_qkv_prefill_cuda(
                    conv_state,
                    qk_part.contiguous(),
                    v_part.contiguous(),
                    conv_weight,
                    bias,
                    nk,
                    nv,
                    dk,
                    dv,
                    qwen_config.rms_norm_eps);
            });
            q = qkv_fast[0];
            k = qkv_fast[1];
            v = qkv_fast[2];
            // Runtime CUDA graphs retain this state storage address across requests.
            conv_state.copy_(qkv_fast[3]);
        } else if (T == 1 && split_in_proj) {
            auto qkv_fast = g_profiler.measure("linear.conv_qkv_decode", [&]() {
                return linear_conv_qkv_decode_cuda(
                    conv_state,
                    qk_part.contiguous(),
                    v_part.contiguous(),
                    conv_weight,
                    bias,
                    nk,
                    nv,
                    dk,
                    dv,
                    qwen_config.rms_norm_eps);
            });
            q = qkv_fast[0];
            k = qkv_fast[1];
            v = qkv_fast[2];
        } else {
            if (split_in_proj) {
                qkv = g_profiler.measure("linear.qkv_cat", [&]() {
                    return mfq_tensor_backend::cat({qk_part.to(mfq_tensor_backend::kFloat32), v_part.to(mfq_tensor_backend::kFloat32)}, -1);
                });
            }
            mfq_tensor_backend::Tensor conv;
            if (T == 1) {
            conv = g_profiler.measure("linear.conv_silu_decode", [&]() {
                return ssm_conv_silu_decode_cuda(conv_state, qkv, conv_weight, bias);
            });
            } else {
            auto conv_in = g_profiler.measure("linear.conv_input", [&]() { return mfq_tensor_backend::cat({conv_state, qkv}, 1); });
            auto next_conv_state = g_profiler.measure("linear.conv_state_update", [&]() { return conv_in.index({Slice(), Slice(-(qwen_config.linear_conv_kernel_dim - 1), mfq_tensor_backend::indexing::None), Slice()}).contiguous(); });
            conv_state.copy_(next_conv_state);
            conv = g_profiler.measure("linear.conv_silu", [&]() { return ssm_conv_silu_cuda(conv_in, conv_weight, bias, T); });
            }
            q = g_profiler.measure("linear.q_view", [&]() { return conv.index({Slice(), Slice(), Slice(0, ksz)}).reshape({B, T, nk, dk}).transpose(1, 2); });
            k = g_profiler.measure("linear.k_view", [&]() { return conv.index({Slice(), Slice(), Slice(ksz, 2 * ksz)}).reshape({B, T, nk, dk}).transpose(1, 2); });
            v = g_profiler.measure("linear.v_view", [&]() { return conv.index({Slice(), Slice(), Slice(2 * ksz, 2 * ksz + vsz)}).reshape({B, T, nv, dv}).transpose(1, 2); });
            q = g_profiler.measure("linear.q_l2", [&]() {
                return l2_norm_cuda(
                    q.contiguous().reshape({-1, dk}),
                    qwen_config.rms_norm_eps).reshape_as(q);
            });
            k = g_profiler.measure("linear.k_l2", [&]() {
                return l2_norm_cuda(
                    k.contiguous().reshape({-1, dk}),
                    qwen_config.rms_norm_eps).reshape_as(k);
            });
        }
        auto gd = g_profiler.measure("linear.gdn", [&]() {
            return recurrent_step(q, k, v, gate_t, beta_t);
        });
        auto y = gd[0];
        gdn_state = gd[1];
        mfq_tensor_backend::Tensor oo;
        z = g_profiler.measure("linear.z_view", [&]() { return z.reshape({B, T, nv, dv}).transpose(1, 2).contiguous(); });
        auto y_norm = g_profiler.measure("linear.out_norm", [&]() {
            return rms_norm_cuda(
                y.reshape({-1, dv}).to(mfq_tensor_backend::kFloat32),
                linear_norm, qwen_config.rms_norm_eps).reshape_as(y);
        });
        auto yf = g_profiler.measure("linear.y_view", [&]() { return y_norm.transpose(1, 2).contiguous().reshape({B, T, vsz}); });
        auto zf = g_profiler.measure("linear.zf_view", [&]() { return z.transpose(1, 2).contiguous().reshape({B, T, vsz}); });
        if (dense_out_proj) {
            oo = g_profiler.measure("linear.out_proj_gate_dense", [&]() {
                auto dtype = out_proj_dense.scalar_type();
                auto gated = yf.to(dtype) * mfq_tensor_backend::silu(zf.to(dtype));
                return mfq_tensor_backend::matmul(
                    gated.reshape({B * T, vsz}),
                    out_proj_dense.transpose(0, 1)).reshape({B, T, H});
            });
        } else {
            oo = g_profiler.measure("linear.out_proj_gate", [&]() {
                return out_proj.forward_input_mul(yf, zf, 2);
            });
        }
        auto attn_pair = g_profiler.measure("linear.attn_residual_ffn_norm", [&]() {
            auto rr = residual.reshape({-1, H});
            auto oo2 = oo.reshape({-1, H});
            if (oo2.scalar_type() != rr.scalar_type()) {
                oo2 = oo2.to(rr.scalar_type()).contiguous();
            }
            const char * fp32_residual_env =
                std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
            if (fp32_residual_env != nullptr &&
                    fp32_residual_env[0] == '1') {
                return acc_rms_norm_cuda(
                    rr.to(mfq_tensor_backend::kFloat32),
                    oo2.to(mfq_tensor_backend::kFloat32),
                    ffn_norm, qwen_config.rms_norm_eps, 1.0);
            }
            if (rr.scalar_type() == mfq_tensor_backend::kFloat16 && oo2.scalar_type() == mfq_tensor_backend::kFloat16) {
                return acc_rms_norm_f16_cuda(
                    rr, oo2, ffn_norm,
                    qwen_config.rms_norm_eps, 1.0);
            }
            return acc_rms_norm_cuda(
                rr, oo2, ffn_norm,
                qwen_config.rms_norm_eps, 1.0);
        });
        return {attn_pair[0].reshape({B, T, H}),
                attn_pair[1].reshape({B, T, H})};
    }

    void replay_recurrent_cuda(
            const LinearRecurrentInputs& inputs,
            int64_t retained_tokens) {
        MFQ_RUNTIME_CHECK(
            retained_tokens > 0 && retained_tokens <= inputs.tokens &&
                inputs.batch > 0 && inputs.split == split_in_proj,
            "invalid linear-attention recurrent replay");
        const int64_t B = inputs.batch;
        const int64_t T = retained_tokens;
        const int64_t nk = qwen_config.linear_num_key_heads;
        const int64_t nv = qwen_config.linear_num_value_heads;
        const int64_t dk = qwen_config.linear_key_head_dim;
        const int64_t dv = qwen_config.linear_value_head_dim;
        const int64_t ksz = qwen_config.linear_k_size();
        const int64_t vsz = qwen_config.linear_v_size();
        auto bias = conv_bias.defined()
            ? conv_bias
            : mfq_tensor_backend::empty(
                  {0}, mfq_tensor_backend::TensorOptions()
                      .device(mfq_tensor_backend::kCUDA)
                      .dtype(mfq_tensor_backend::kFloat32));
        auto qkv = inputs.split
            ? mfq_tensor_backend::cat(
                  {inputs.qk.narrow(1, 0, T)
                       .to(mfq_tensor_backend::kFloat32),
                   inputs.value.narrow(1, 0, T)
                       .to(mfq_tensor_backend::kFloat32)},
                  -1)
            : inputs.qkv.narrow(1, 0, T);
        auto conv_input = mfq_tensor_backend::cat({conv_state, qkv}, 1);
        conv_state.copy_(conv_input.narrow(
            1,
            conv_input.size(1) - (qwen_config.linear_conv_kernel_dim - 1),
            qwen_config.linear_conv_kernel_dim - 1));
        auto conv = ssm_conv_silu_cuda(
            conv_input, conv_weight, bias, T);
        auto q = conv.narrow(2, 0, ksz)
            .reshape({B, T, nk, dk}).transpose(1, 2);
        auto k = conv.narrow(2, ksz, ksz)
            .reshape({B, T, nk, dk}).transpose(1, 2);
        auto v = conv.narrow(2, 2 * ksz, vsz)
            .reshape({B, T, nv, dv}).transpose(1, 2);
        q = l2_norm_cuda(
            q.contiguous().reshape({-1, dk}), qwen_config.rms_norm_eps)
                .reshape_as(q);
        k = l2_norm_cuda(
            k.contiguous().reshape({-1, dk}), qwen_config.rms_norm_eps)
                .reshape_as(k);
        const char* transposed_env = std::getenv("MFQ_GDN_TRANSPOSED_STATE");
        const bool transposed =
            transposed_env == nullptr || transposed_env[0] != '0';
        const auto gate = inputs.gate.narrow(2, 0, T).contiguous();
        const auto beta = inputs.beta.narrow(2, 0, T).contiguous();
        std::vector<mfq_tensor_backend::Tensor> output;
        if (transposed) {
            output = tiled_v_heads
                ? gdn_inplace_transposed_tiled_cuda(
                      q.contiguous(), k.contiguous(), v.contiguous(),
                      gate, beta, gdn_state)
                : gdn_inplace_transposed_cuda(
                      q.contiguous(), k.contiguous(), v.contiguous(),
                      gate, beta, gdn_state);
        } else {
            output = tiled_v_heads
                ? gdn_inplace_tiled_cuda(
                      q.contiguous(), k.contiguous(), v.contiguous(),
                      gate, beta, gdn_state)
                : gdn_inplace_cuda(
                      q.contiguous(), k.contiguous(), v.contiguous(),
                      gate, beta, gdn_state);
        }
        gdn_state = output[1];
    }

    mfq_tensor_backend::Tensor forward_ffn_cuda(
            mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor xn) {
        const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
        auto ffn_input = xn.reshape({B * T, H});
        auto residual_flat = residual.reshape({B * T, H});
        if (ffn.can_forward_fused_residual(ffn_input, residual_flat)) {
            return g_profiler.measure("linear.ffn_down_residual", [&]() {
                return ffn.forward_fused_residual(
                    ffn_input, residual_flat).reshape({B, T, H});
            });
        }
        auto ff = ffn.forward(ffn_input).reshape({B, T, H});
        return g_profiler.measure("linear.ffn_residual", [&]() {
            auto rr = residual.reshape({-1, H});
            auto ff2 = ff.reshape({-1, H});
            const char * fp32_residual_env =
                std::getenv("MFQ_DIAGNOSTIC_FP32_RESIDUAL");
            if (fp32_residual_env != nullptr &&
                    fp32_residual_env[0] == '1') {
                rr = rr.to(mfq_tensor_backend::kFloat32);
                ff2 = ff2.to(mfq_tensor_backend::kFloat32);
            } else if (ff2.scalar_type() != rr.scalar_type()) {
                ff2 = ff2.to(rr.scalar_type()).contiguous();
            }
            return acc_cuda(rr, ff2).reshape({B, T, H});
        });
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor pos,
            int64_t cache_pos,
            const MfqOptional<mfq_tensor_backend::Tensor> & seq_len,
            const RopeCache & rope,
            const MfqOptional<mfq_tensor_backend::Tensor> & cache_positions = mfq_nullopt,
            const MfqOptional<mfq_tensor_backend::Tensor> & attention_mask = mfq_nullopt) override {
        (void)cache_positions;
        (void)attention_mask;
        if (!x.is_cuda()) return forward_cpu(std::move(x));
        (void)seq_len;
        (void)pos; (void)cache_pos; (void)rope;
        auto attention = forward_attention_cuda(std::move(x));
        return forward_ffn_cuda(std::move(attention[0]), std::move(attention[1]));
    }
};

} // namespace mfq::cuda::qwen35
