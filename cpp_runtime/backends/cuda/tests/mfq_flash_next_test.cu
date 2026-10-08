// Native ABI numerical-test bridge: tests/test_cuda_flash_next.py supplies the
// same NumPy oracle cases to this executable and the production Torch module.
#include "mfq_cuda_attention_ops.h"
#include "mfq/kernels/cuda/glm5_next.h"
#include "mfq/kernels/cuda/qwen4_exp.h"
#include "glm5_next/model.h"
#include "qwen4_exp/model.h"
#include "cuda_execution.h"
#include "mfq_cuda_context.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

mfq_tensor_backend::Tensor attention_glm_mla_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor meta,
    double scale);
mfq_tensor_backend::Tensor attention_dsv4_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor mask,
    mfq_tensor_backend::Tensor sinks, mfq_tensor_backend::Tensor meta,
    double scale);

namespace {
using namespace mfq::cuda;
using Json = nlohmann::json;

void require_constant(
    const Tensor& value, float expected, float tolerance, const char * name) {
    const auto host = value.to(kFloat32).contiguous().cpu();
    float maximum_error = 0.0f;
    for (int64_t index = 0; index < host.numel(); ++index) {
        maximum_error = std::max(
            maximum_error,
            std::abs(host.data_ptr<float>()[index] - expected));
    }
    if (maximum_error > tolerance) {
        throw std::runtime_error(
            std::string(name) + " mismatch, max_abs=" +
            std::to_string(maximum_error));
    }
}

void require_qsa_gqa_means(
    const Tensor& value, float first, float second, float tolerance) {
    const auto host = value.to(kFloat32).contiguous().cpu();
    float maximum_error = 0.0f;
    for (int head = 0; head < 24; ++head) {
        const float expected = head < 12 ? first : second;
        for (int width = 0; width < 256; ++width) {
            maximum_error = std::max(
                maximum_error,
                std::abs(host.data_ptr<float>()[head * 256 + width] - expected));
        }
    }
    if (maximum_error > tolerance) {
        throw std::runtime_error(
            "QSA sparse GQA mismatch, max_abs=" +
            std::to_string(maximum_error));
    }
}

void check_shared_sparse_attention() {
    const auto float_cuda = TensorOptions{}.dtype(kFloat32).device(kCUDA);
    const auto half_cuda = TensorOptions{}.dtype(kFloat16).device(kCUDA);
    const auto int_cuda = TensorOptions{}.dtype(kInt32).device(kCUDA);
    constexpr int selected_count = 2048;
    std::vector<int32_t> selected(selected_count);
    float glm_expected = 0.0f;
    for (int index = 0; index < selected_count; ++index) {
        selected[index] = (index * 37 + 11) % 256;
        glm_expected += static_cast<float>(selected[index]);
    }
    glm_expected /= selected.size();
    const auto indices = tensor(selected)
        .reshape({1, 1, selected_count}).to(int_cuda);
    auto meta = empty({4 << 20}, float_cuda);

    const auto qsa_q = zeros({1, 24, 1, 256}, float_cuda);
    const auto qsa_k = zeros({1, 2, 256, 256}, half_cuda);
    const auto qsa_v = arange(512, float_cuda)
        .reshape({1, 2, 256, 1}).expand({1, 2, 256, 256})
        .contiguous().to(kFloat16);
    require_qsa_gqa_means(
        mfq_qwen4_exp::sparse_gqa_attention(
            qsa_q, qsa_k, qsa_v, indices),
        glm_expected, glm_expected + 256.0f, 6.0e-2f);

    const auto glm_q = zeros({1, 64, 1, 576}, float_cuda);
    const auto glm_kv = arange(256, float_cuda)
        .reshape({1, 256, 1}).expand({1, 256, 576}).contiguous().to(kFloat16);
    require_constant(
        attention_glm_mla_sparse_cuda(glm_q, glm_kv, indices, meta, 1.0),
        glm_expected, 2.0e-2f, "GLM sparse MLA");

    const auto dsv_q = zeros({1, 64, 1, 512}, float_cuda);
    const auto dsv_kv = arange(256, float_cuda)
        .reshape({1, 256, 1}).expand({1, 256, 512}).contiguous().to(kFloat16);
    std::vector<float> mask_values(selected_count, 0.0f);
    float dsv_sum = 0.0f;
    int dsv_count = 0;
    for (int index = 0; index < selected_count; ++index) {
        if (index % 5 == 0) {
            mask_values[index] = -std::numeric_limits<float>::infinity();
        } else {
            dsv_sum += static_cast<float>(selected[index]);
            ++dsv_count;
        }
    }
    const auto mask = tensor(mask_values)
        .reshape({1, 1, selected_count}).to(half_cuda);
    const auto sinks = zeros({64}, float_cuda);
    require_constant(
        attention_dsv4_sparse_cuda(
            dsv_q, dsv_kv, indices, mask, sinks, meta, 1.0),
        dsv_sum / static_cast<float>(dsv_count + 1),
        6.0e-2f, "DSV4 sparse attention");

    const auto all_invalid_mask = full(
        {1, 1, selected_count},
        -std::numeric_limits<float>::infinity(), half_cuda);
    const auto all_invalid_sinks = full(
        {64}, -std::numeric_limits<float>::infinity(), float_cuda);
    require_constant(
        attention_dsv4_sparse_cuda(
            dsv_q, dsv_kv, indices, all_invalid_mask,
            all_invalid_sinks, meta, 1.0),
        0.0f, 0.0f, "fully masked DSV4 sparse attention");
}

Tensor input(const Json& j) {
    if (j.is_null()) return {};
    const auto shape = j.at("shape").get<std::vector<int64_t>>();
    const auto dtype = j.value("dtype", "float32");
    auto host = empty(shape, TensorOptions{}.dtype(dtype == "int64" ? kInt64 : kFloat32));
    if (dtype == "int64") {
        const auto data = j.at("data").get<std::vector<int64_t>>();
        if (data.size() != static_cast<size_t>(host.numel())) throw std::runtime_error("fixture size mismatch");
        std::copy(data.begin(), data.end(), host.data_ptr<int64_t>());
    } else {
        const auto data = j.at("data").get<std::vector<float>>();
        if (data.size() != static_cast<size_t>(host.numel())) throw std::runtime_error("fixture size mismatch");
        std::copy(data.begin(), data.end(), host.data_ptr<float>());
    }
    auto value = host.to(Device{DeviceType::cuda, 0});
    if (dtype == "float16") value = value.to(kFloat16);
    else if (dtype == "bfloat16") value = value.to(kBFloat16);
    else if (dtype != "float32" && dtype != "int64") throw std::runtime_error("unknown fixture dtype");
    if (j.value("noncontiguous", false) && value.dim() >= 2)
        value = value.transpose(-1, -2).contiguous().transpose(-1, -2);
    return value;
}

std::vector<Tensor> run(const std::string& op, const std::vector<Tensor>& a, const Json& p) {
    CudaExecutionContext execution;
    const auto optional = [&](size_t i) -> std::optional<Tensor> {
        return a.at(i).defined() ? std::optional<Tensor>(a.at(i)) : std::nullopt;
    };
    std::optional<double> scale;
    if (p.contains("scale") && !p.at("scale").is_null()) scale = p.at("scale").get<double>();
    if (op == "runtime_gdn") {
        const auto linear = [&](int index) -> mfq::cuda::qwen4_exp::Linear {
            auto w=a.at(index);return [w](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
        };
        mfq::cuda::qwen4_exp::GdnWeights w{linear(1),linear(2),linear(3),linear(4),linear(5),a.at(6),a.at(7),a.at(8),a.at(9)};
        mfq::cuda::qwen4_exp::Gdn block(std::move(w),p.at("key_heads"),p.at("value_heads"),p.at("width"),
            p.at("kernel"),p.value("eps",1e-6),p.value("silu_gate",false));
        std::vector<Tensor> out;
        for (const auto& step:p.at("steps")) {
            if (step.value("reset",false)) block.reset();
            else if (step.value("commit",false)) block.commit();
            else if (step.value("rollback",false)) block.rollback();
            else {
                out.push_back(block.forward(execution,a.at(0).narrow(1,step.at("begin"),step.at("count")),step.value("cache",true),step.value("confirmed",0)));
                out.push_back(block.conv_state().clone());out.push_back(block.recurrent_state().clone());
            }
        }
        return out;
    }
    if (op == "runtime_ngram" || op == "runtime_ple") {
        const auto weights=a.at(op=="runtime_ngram"?1:2);
        std::vector<mfq::cuda::qwen4_exp::Embedding> shards;
        for (int64_t i=0;i<weights.size(0);++i) {
            auto w=weights.select(0,i);
            shards.push_back([w](const Tensor& ids) {
                auto shape=ids.sizes().vec();shape.push_back(w.size(1));
                return w.index_select(0,ids.reshape({-1}).to(kInt64)).reshape(shape);
            });
        }
        mfq::cuda::qwen4_exp::NgramEmbedding embedding(std::move(shards),weights.size(1),weights.size(2),p.at("ngram"),
            p.at("heads_per_ngram"),p.at("eos"),p.at("multipliers").get<std::vector<int64_t>>(),
            p.at("offsets").get<std::vector<int64_t>>(),p.at("vocab").get<std::vector<int64_t>>());
        std::vector<Tensor> out;
        if (op=="runtime_ngram") {
            for (const auto& step:p.at("steps")) {
                if (step.value("reset",false)) embedding.reset();
                else {
                    Tensor ids;
                    out.push_back(embedding.forward(a.at(0).narrow(1,step.at("begin"),step.at("count")),step.value("cache",true),&ids));
                    out.push_back(ids);
                }
            }
        } else {
            const auto linear=[&](int i)->mfq::cuda::qwen4_exp::Linear {
                auto w=a.at(i);return [w](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
            };
            mfq::cuda::qwen4_exp::PleWeights w{linear(3),linear(4),a.at(5),a.at(6),a.at(7),a.at(8)};
            mfq::cuda::qwen4_exp::Ple block(std::move(embedding),std::move(w),p.at("hidden"),p.at("streams"),p.at("ngram"),p.value("eps",1e-6));
            for (const auto& step:p.at("steps")) {
                if (step.value("reset",false)) block.reset();
                else if (step.value("commit",false)) block.commit();
                else if (step.value("rollback",false)) block.rollback();
                else {
                    const int64_t begin=step.at("begin"),count=step.at("count");
                    out.push_back(block.forward(execution,a.at(0).narrow(1,begin,count),a.at(1).narrow(1,begin,count),step.value("cache",true),step.value("confirmed",0)));
                    out.push_back(block.conv_state().clone());
                }
            }
        }
        return out;
    }
    if (op == "runtime_rotary") {
        mfq::cuda::RotaryEmbedding rotary(p.at("rotary"),p.at("maximum"),p.value("base",1e7),
            p.value("sections",std::vector<int64_t>{}),p.value("interleaved",false));
        return {rotary.forward(a.at(0),a.at(1))};
    }
    if (op == "runtime_qsa") {
        const auto linear = [&](int index) -> mfq::cuda::qwen4_exp::Linear {
            auto weight=a.at(index);
            return [weight](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(weight.scalar_type()),weight.transpose(-1,-2));};
        };
        auto rotary=std::make_shared<mfq::cuda::RotaryEmbedding>(p.at("rotary"),p.at("maximum"),p.value("base",1e7),
            p.value("sections",std::vector<int64_t>{}),p.value("interleaved",false));
        mfq::cuda::qwen4_exp::QsaWeights weights{linear(1),linear(2),linear(3),linear(4),linear(5),a.at(6),a.at(7),a.at(8),a.at(9)};
        mfq::cuda::qwen4_exp::QsaConfig config{p.at("heads"),p.at("kv_heads"),p.at("width"),p.at("index_heads"),
            p.at("index_width"),p.at("pool"),p.at("budget"),p.at("maximum"),p.value("eps",1e-6)};
        mfq::cuda::qwen4_exp::Qsa block(std::move(weights),config,rotary);
        Tensor history;
        std::vector<Tensor> out;
        for (const auto& step:p.at("steps")) {
            if (step.value("reset",false)) {block.reset();history={};}
            else if (step.contains("truncate")) {
                block.truncate(step.at("truncate"));
                history=history.narrow(-1,0,step.at("truncate"));
            } else {
                auto pos=a.at(10).narrow(-1,step.at("begin"),step.at("count"));
                const bool cache=step.value("cache",true);
                auto full=cache && history.defined()?cat({history,pos},-1):pos;
                std::vector<Tensor> trace;
                out.push_back(block.forward(execution,a.at(0).narrow(1,step.at("begin"),step.at("count")),pos,full,cache,
                    p.value("trace",false)?&trace:nullptr));
                out.insert(out.end(),trace.begin(),trace.end());
                if (cache) history=full;
            }
        }
        return out;
    }
    if (op == "runtime_select_pooled_blocks") return {mfq::cuda::attention_ops::select_pooled_blocks(
        a.at(0), p.at("query_offset"), p.at("logical_length"), p.at("pool"), p.at("budget"), p.at("tail"))};
    if (op == "runtime_sequence_cache") {
        mfq::cuda::attention_ops::SequenceCache cache(p.at("maximum"), a.at(0).size(-1));
        std::vector<Tensor> out;
        for (const auto& step : p.at("steps")) {
            if (step.contains("truncate")) cache.truncate(step.at("truncate"));
            else if (step.value("reset", false)) cache.reset();
            else out.push_back(cache.append(a.at(0).narrow(1, step.at("begin"), step.at("count"))).clone());
        }
        return out;
    }
    if (op == "runtime_kda") {
        const auto linear = [&](int index) -> mfq::cuda::glm5_next::Linear {
            auto weight = a.at(index);
            return [weight](CudaExecutionContext&, const Tensor& x) { return matmul(x.to(weight.scalar_type()), weight.transpose(-1,-2)); };
        };
        mfq::cuda::glm5_next::KdaWeights weights{linear(1),linear(2),linear(3),linear(4),linear(5),linear(6),linear(7),
            a.at(8),a.at(9),a.at(10),a.at(11),a.at(12),a.at(13)};
        mfq::cuda::glm5_next::Kda block(std::move(weights), p.at("heads"), p.at("width"), p.at("kernel"),
            p.value("lower_bound", -5.0), p.value("eps", 1e-5));
        std::vector<Tensor> out;
        for (const auto& step : p.at("steps")) {
            if (step.value("rollback", false)) block.rollback();
            else if (step.value("commit", false)) block.commit();
            else if (step.value("reset", false)) block.reset();
            else {
                out.push_back(block.forward(execution,a.at(0).narrow(1, step.at("begin"), step.at("count")),
                    step.value("cache", true), step.value("confirmed", 0)));
                out.push_back(block.conv_state().clone());
                out.push_back(block.recurrent_state().clone());
            }
        }
        return out;
    }
    if (op == "qwen4_grouped_rms_norm") return {mfq_qwen4_exp::grouped_rms_norm(a.at(0), a.at(1), p.at("group_size"), p.value("eps", 1e-6))};
    if (op == "qwen4_gated_residual_pre") return mfq_qwen4_exp::gated_residual_pre(a.at(0), a.at(1), a.at(2), a.at(3), optional(4), p.at("hidden_size"), p.value("hc_count", 4), p.value("eps", 1e-6));
    if (op == "qwen4_gated_residual_post") return {mfq_qwen4_exp::gated_residual_post(a.at(0), a.at(1), a.at(2), p.value("hc_count", 4))};
    if (op == "glm5_mhc_pre") return mfq_glm5_next::mhc_pre(a.at(0), a.at(1), a.at(2), a.at(3), p.value("sinkhorn_iterations", 20), p.value("hc_eps", 1e-6), p.value("rms_eps", 1e-5));
    if (op == "glm5_mhc_post") return {mfq_glm5_next::mhc_post(a.at(0), a.at(1), a.at(2), a.at(3))};
    if (op == "glm5_kda_forget_gate") return {mfq_glm5_next::kda_forget_gate(a.at(0), a.at(1), a.at(2), a.at(3), a.at(4), p.at("num_heads"), p.at("head_dim"), p.value("lower_bound", -5.0))};
    if (op == "qsa_block_scores") return {mfq_qwen4_exp::block_scores(a.at(0), a.at(1))};
    if (op == "glm5_kpool_scores") return {mfq_glm5_next::kpool_scores(a.at(0), a.at(1), a.at(2))};
    if (op == "glm5_kpool_states") return {mfq_glm5_next::kpool_states(a.at(0), a.at(1), a.at(2), p.value("pool_size", 4))};
    if (op == "qwen4_ple_dilated_conv_silu") return mfq_qwen4_exp::ple_dilated_conv_silu(a.at(0), a.at(1), optional(2), p.at("dilation"));
    if (op == "qwen4_dense_gqa_attention") return {mfq_qwen4_exp::dense_gqa_attention(a.at(0), a.at(1), a.at(2), p.at("query_offset"))};
    if (op == "qwen4_sparse_gqa_attention") return {mfq_qwen4_exp::sparse_gqa_attention(a.at(0), a.at(1), a.at(2), a.at(3))};
    if (op == "glm5_dense_mla_attention") return {mfq_glm5_next::dense_mla_attention(a.at(0), a.at(1), p.at("query_offset"), scale)};
    if (op == "glm5_sparse_mla_attention") return {mfq_glm5_next::sparse_mla_attention(a.at(0), a.at(1), a.at(2), scale)};
    throw std::runtime_error("unknown Flash-Next operation");
}

Json output(const std::vector<Tensor>& tensors) {
    Json result = Json::array();
    for (const auto& value : tensors) {
        if (!value.defined()) { result.push_back(nullptr); continue; }
        auto host = value.to(kFloat32).contiguous().cpu();
        std::vector<float> data(host.numel());
        if (host.numel()) std::copy_n(host.data_ptr<float>(), host.numel(), data.data());
        result.push_back({{"shape", value.sizes().vec()}, {"data", data},
            {"dtype", value.scalar_type() == kFloat16 ? "float16" :
                value.scalar_type() == kBFloat16 ? "bfloat16" : "float32"}});
    }
    return result;
}
} // namespace

void check_rotary_fusion() {
    using namespace mfq::cuda;
    int ordinary_cases=0,graph_cases=0,switch_cases=0;
    const auto exact=[](const Tensor& actual,const Tensor& expected) {
        const auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type() ||
            std::memcmp(a.data_ptr(),b.data_ptr(),size_t(a.numel()*a.element_size())))
            throw std::runtime_error("fused rotary original output bytes differ, case "+std::to_string(a.numel()));
    };
    const std::array<std::array<int64_t,5>,3> shapes{{{1,24,1,256,96},{2,3,3,128,64},{2,2,35,17,12}}};
    for(auto dtype:{kFloat16,kFloat32,kBFloat16})for(const auto& shape:shapes)
        for(int layout=1;layout<=4;++layout)for(int mode=0;mode<3;++mode) {
            const auto b=shape[0],h=shape[1],t=shape[2],d=shape[3],rotary=shape[4],pairs=rotary/2;
            std::vector<float> values(size_t(b*h*t*d));
            for(size_t i=0;i<values.size();++i) {
                values[i]=float(std::sin(double(i)*.031)*.71);
                if(i%17==0)values[i]=0.f;
                if(i%17==1)values[i]=-0.f;
                if(i%17==2)values[i]=std::numeric_limits<float>::denorm_min();
                if(i%17==3)values[i]=-std::numeric_limits<float>::denorm_min();
            }
            auto x=tensor(values).reshape({b,h,t,d}).to(kCUDA,dtype);
            if(layout==2)x=x.transpose(-1,-2).contiguous().transpose(-1,-2);
            const int64_t axes=layout==1?1:(layout==2?2:(layout==3?3:5));
            const int64_t batches=layout==3?b:1;
            const auto position_shape=layout==1?std::vector<int64_t>{t}:
                (layout==3 || layout==4?std::vector<int64_t>{axes,batches,t}:std::vector<int64_t>{axes,t});
            std::vector<int64_t> positions(size_t(axes*batches*t));
            const std::array<int64_t,9> choices{-5,0,1,96,511,512,2147483647LL,2147483648LL,-2147483649LL};
            for(size_t i=0;i<positions.size();++i)positions[i]=choices[i%choices.size()];
            auto p=tensor(positions).reshape(position_shape).to(kCUDA);
            std::vector<int64_t> sections=mode==0?std::vector<int64_t>{}:std::vector<int64_t>{pairs/3,pairs/3,pairs-2*(pairs/3)};
            RotaryEmbedding original(rotary,512,1e7,sections,mode==2,false);
            RotaryEmbedding fused(rotary,512,1e7,sections,mode==2,true);
            exact(fused.forward(x,p),original.forward(x,p));++ordinary_cases;
            if(mode==2 && layout==3) {
                std::array<StreamHandle,2> streams{stream_from_pool(false),stream_from_pool(false)};
                std::array<Graph,2> graphs;std::array<Tensor,2> outputs;
                for(int variant=0;variant<2;++variant) {
                    StreamGuard guard(streams[variant]);
                    auto& operation=variant?fused:original;
                    graphs[variant].prepare_memory();outputs[variant]=operation.forward(x,p);
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream(0).stream()));
                    outputs[variant]=Tensor{};
                    graphs[variant].capture_begin();outputs[variant]=operation.forward(x,p);graphs[variant].capture_end();
                }
                for(int step=0;step<3;++step) {
                    for(size_t i=0;i<positions.size();++i)positions[i]=choices[(i+step+4)%choices.size()];
                    p.copy_(tensor(positions).reshape(position_shape).to(kCUDA));
                    x.copy_(tensor(values).reshape({b,h,t,d}).to(kCUDA,dtype)*(float(step)-1.f));
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream(0).stream()));
                    for(int variant=0;variant<2;++variant) {
                        StreamGuard guard(streams[variant]);graphs[variant].replay();
                        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(streams[variant].stream()));
                    }
                    exact(outputs[1],outputs[0]);++graph_cases;
                }
            }
            fused.set_fused(false);
            exact(fused.forward(x,p),original.forward(x,p));++switch_cases;
            fused.set_fused(true);
            exact(fused.forward(x,p),original.forward(x,p));++switch_cases;
        }
    std::cout<<"rotary_fused_ordinary_exact_cases="<<ordinary_cases<<" changing_graph_exact_cases="<<graph_cases
             <<" mode_switch_exact_cases="<<switch_cases<<" PASS\n";
}

void check_rotary_normalized_fusion() {
    using namespace mfq::cuda;
    int ordinary_cases=0,cache_cases=0,graph_cases=0;
    const auto exact=[](const Tensor& actual,const Tensor& expected,const char* label) {
        const auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type() ||
            std::memcmp(a.data_ptr(),b.data_ptr(),size_t(a.numel()*a.element_size())))
            throw std::runtime_error(std::string("normalized rotary bytes differ: ")+label);
    };
    const std::array<std::array<int64_t,5>,7> shapes{{
        {1,20,1,128,96},{1,24,1,256,96},{2,3,3,128,64},{2,2,35,17,12},
        {1,3,6,32,16},{1,2,3,64,48},{1,2,3,513,128}}};
    for(auto dtype:{kFloat16,kFloat32})for(const auto& shape:shapes)
        for(int layout=1;layout<=4;++layout)for(int mode=0;mode<3;++mode) {
            const auto b=shape[0],h=shape[1],t=shape[2],d=shape[3],rotary=shape[4],pairs=rotary/2;
            std::vector<float> values(size_t(b*t*h*d*2)),weights(static_cast<size_t>(d));
            for(size_t i=0;i<values.size();++i) {
                values[i]=float(std::sin(double(i)*.031)*.71);
                if(i%17==0)values[i]=0.f;
                if(i%17==1)values[i]=-0.f;
                if(i%17==2)values[i]=std::numeric_limits<float>::denorm_min();
                if(i%17==3)values[i]=-std::numeric_limits<float>::denorm_min();
            }
            for(size_t i=0;i<weights.size();++i)weights[i]=i%11==0?-1.f:float(std::cos(double(i)*.07)*.13);
            auto joined=tensor(values).reshape({b,t,h,2*d}).to(kCUDA,dtype);
            auto x=joined.narrow(-1,0,d),v=joined.narrow(-1,d,d);
            if(layout==2)x=x.transpose(-1,-2).contiguous().transpose(-1,-2);
            if(layout==4)v=v.transpose(-1,-2).contiguous().transpose(-1,-2);
            const auto source=x.clone(),source_value=v.clone();
            auto weight=tensor(weights).to(kCUDA);
            const int64_t axes=layout==1?1:(layout==2?2:(layout==3?3:5));
            const int64_t batches=layout==3?b:1;
            const auto position_shape=layout==1?std::vector<int64_t>{t}:
                (layout==3 || layout==4?std::vector<int64_t>{axes,batches,t}:std::vector<int64_t>{axes,t});
            std::vector<int64_t> positions(size_t(axes*batches*t)),slots(static_cast<size_t>(t));
            const std::array<int64_t,9> choices{-5,0,1,96,511,512,2147483647LL,2147483648LL,-2147483649LL};
            for(size_t i=0;i<positions.size();++i)positions[i]=choices[i%choices.size()];
            for(size_t i=0;i<slots.size();++i)slots[i]=5+3*i;
            const auto index_dtype=layout%2?kInt64:kInt32;
            auto p=tensor(positions).reshape(position_shape).to(kCUDA,index_dtype);
            auto cache_positions=tensor(slots).to(kCUDA,index_dtype);
            std::vector<int64_t> sections=mode==0?std::vector<int64_t>{}:std::vector<int64_t>{pairs/3,pairs/3,pairs-2*(pairs/3)};
            RotaryEmbedding original(rotary,512,1e7,sections,mode==2,false);
            RotaryEmbedding fused(rotary,512,1e7,sections,mode==2,true);
            const auto options=x.options().dtype(kFloat16);
            std::array<Tensor,2> keys{full({b,h,3*t+8,d},-.17,options),full({b,h,3*t+8,d},-.17,options)};
            std::array<Tensor,2> cached_values{full({b,h,3*t+8,d},.23,options),full({b,h,3*t+8,d},.23,options)};
            std::array<Tensor,2> outputs;
            const auto forward=[&](int variant) {
                if(variant) {
                    outputs[1]=fused.forward_normalized(x,weight,p,1e-6);
                    auto unused=fused.forward_normalized(x,weight,p,1e-6,keys[1],cache_positions,v,cached_values[1]);
                    if(unused.defined())throw std::runtime_error("direct cached rotary allocated an output");
                } else {
                    auto norm=grouped_rms_norm_cuda(x.contiguous(),weight,d,1e-6,1.0);
                    outputs[0]=original.forward(norm.permute({0,2,1,3}),p);
                    keys[0].index_copy_(2,cache_positions,outputs[0].to(kFloat16));
                    cached_values[0].index_copy_(2,cache_positions,v.permute({0,2,1,3}).to(kFloat16));
                }
            };
            forward(0);forward(1);
            exact(outputs[1],outputs[0],"output");++ordinary_cases;
            exact(keys[1],keys[0],"keys");exact(cached_values[1],cached_values[0],"values");++cache_cases;
            if(mode==2 && layout==3) {
                std::array<StreamHandle,2> streams{stream_from_pool(false),stream_from_pool(false)};
                std::array<Graph,2> graphs;
                for(int variant=0;variant<2;++variant) {
                    StreamGuard guard(streams[variant]);graphs[variant].prepare_memory();forward(variant);
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(streams[variant].stream()));outputs[variant]=Tensor{};
                    graphs[variant].capture_begin();forward(variant);graphs[variant].capture_end();
                }
                for(int step=0;step<3;++step) {
                    for(size_t i=0;i<positions.size();++i)positions[i]=choices[(i+step+4)%choices.size()];
                    for(size_t i=0;i<slots.size();++i)slots[i]=5+3*i+step;
                    p.copy_(tensor(positions).reshape(position_shape).to(kCUDA,index_dtype));
                    cache_positions.copy_(tensor(slots).to(kCUDA,index_dtype));
                    x.copy_(source*(float(step)-1.f));v.copy_(source_value*(float(step)*.5f-1.f));
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream(0).stream()));
                    for(int variant=0;variant<2;++variant) {
                        StreamGuard guard(streams[variant]);graphs[variant].replay();
                        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(streams[variant].stream()));
                    }
                    try { exact(outputs[1],outputs[0],"graph output"); }
                    catch(const std::exception&) {
                        std::cerr<<"normalized graph mismatch dtype="<<int(dtype)<<" B="<<b<<" H="<<h<<" T="<<t<<" D="<<d<<" step="<<step<<'\n';
                        throw;
                    }
                    exact(keys[1],keys[0],"graph keys");
                    exact(cached_values[1],cached_values[0],"graph values");++graph_cases;
                }
            }
        }
    std::cout<<"normalized_rotary_ordinary_exact_cases="<<ordinary_cases<<" cache_exact_cases="<<cache_cases
             <<" changing_graph_exact_cases="<<graph_cases<<" PASS\n";
}

void check_attention_fusion() {
    using namespace mfq::cuda;
    int ordinary_cases=0,gate_cases=0,graph_cases=0;
    const auto exact=[](const Tensor& actual,const Tensor& expected,const char* label) {
        const auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type() ||
            std::memcmp(a.data_ptr(),b.data_ptr(),size_t(a.numel()*a.element_size())))
            throw std::runtime_error(std::string("attention fusion bytes differ: ")+label);
    };
    const std::array<std::array<int64_t,7>,6> shapes{{
        {1,20,2,1,128,128,163},{1,24,2,1,256,33,41},
        {2,4,2,3,17,17,23},{1,4,2,2,513,17,23},
        {1,2,1,2,1025,9,13},{1,4,2,3,64,0,13}}};
    for(auto dtype:{kFloat16,kFloat32})for(auto gate_dtype:{kFloat16,kFloat32})
        for(bool half_output:{false,true})for(const auto& shape:shapes)for(int layout=0;layout<2;++layout) {
            const auto b=shape[0],h=shape[1],kh=shape[2],t=shape[3],d=shape[4],capacity=shape[5],columns=shape[6];
            const auto make=[&](int64_t count,float scale,float shift) {
                std::vector<float> numbers(static_cast<size_t>(count));
                for(size_t i=0;i<numbers.size();++i) {
                    numbers[i]=float(std::sin(double(i)*.043+shift)*scale);
                    if(i%31==0)numbers[i]=0.f;
                    if(i%31==1)numbers[i]=-0.f;
                    if(i%31==2)numbers[i]=std::numeric_limits<float>::denorm_min();
                    if(i%31==3)numbers[i]=-std::numeric_limits<float>::denorm_min();
                }
                return tensor(numbers).to(kCUDA);
            };
            auto q=make(b*h*t*d,.71f,.13f).reshape({b,h,t,d}).to(dtype);
            auto k=make(b*kh*capacity*d,.37f,.27f).reshape({b,kh,capacity,d}).to(kFloat16);
            auto v=make(b*kh*capacity*d,1.31f,.31f).reshape({b,kh,capacity,d}).to(kFloat16);
            auto joined=make(b*t*h*d*2,17.f,.43f).reshape({b,t,h,2*d}).to(gate_dtype);
            auto gate=joined.narrow(-1,d,d);
            if(!layout)gate=gate.contiguous();
            else {q=q.transpose(-1,-2).contiguous().transpose(-1,-2);v=v.transpose(-1,-2).contiguous().transpose(-1,-2);}
            auto source_q=q.clone(),source_gate=gate.clone();
            const auto index_dtype=layout?kInt64:kInt32;
            std::vector<int64_t> choices{-5,0,capacity-1,capacity,2147483648LL,-2147483649LL};
            std::vector<int64_t> ends(static_cast<size_t>(t));
            for(size_t i=0;i<ends.size();++i)ends[i]=choices[(i+2)%choices.size()];
            auto positions=tensor(ends).to(kCUDA,index_dtype);
            const auto reference=[&] {
                auto selected=arange(columns,positions.options()).reshape({1,1,columns}).expand({b,t,columns});
                selected=where(selected<=positions.reshape({1,t,1}),selected,full_like(selected,-1));
                auto attended=mfq_qwen4_exp::sparse_gqa_attention(q,k,v,selected);
                auto out=(attended.to(kFloat32)*sigmoid(gate.to(kFloat32))).to(half_output?kFloat16:kFloat32);
                exact(mfq_qwen4_exp::attention_gate(attended,gate,half_output),out,"gate");++gate_cases;
                return out;
            };
            const auto candidate=[&] {return mfq_qwen4_exp::causal_gqa_attention_gate(q,k,v,positions,gate,columns,half_output);};
            try {exact(candidate(),reference(),"causal output");}
            catch(const std::exception&) {std::cerr<<"attention fusion mismatch D="<<d<<" dtype="<<int(dtype)
                <<" gate_dtype="<<int(gate_dtype)<<" half_output="<<half_output<<" layout="<<layout<<'\n';throw;}
            ++ordinary_cases;
            if(layout && gate_dtype==kFloat16) {
                auto stream=stream_from_pool(false);Graph graph;Tensor output;
                {StreamGuard guard(stream);graph.prepare_memory();output=candidate();
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));output={};
                    graph.capture_begin();output=candidate();graph.capture_end();}
                for(int step=0;step<3;++step) {
                    for(size_t i=0;i<ends.size();++i)ends[i]=choices[(i+step+3)%choices.size()];
                    positions.copy_(tensor(ends).to(kCUDA,index_dtype));
                    q.copy_(source_q*(float(step)-1.f));gate.copy_(source_gate*(float(step)*.5f-1.f));
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream(0).stream()));
                    {StreamGuard guard(stream);graph.replay();MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));}
                    exact(output,reference(),"changing graph");++graph_cases;
                }
            }
        }
    std::cout<<"attention_fusion_ordinary_exact_cases="<<ordinary_cases<<" gate_exact_cases="<<gate_cases
             <<" changing_graph_exact_cases="<<graph_cases<<" PASS\n";
}

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        const auto context = mfq::cuda::default_context(0);
        auto stream = mfq::cuda::stream_from_pool(false, 0);
        mfq::cuda::StreamGuard stream_guard(stream);
        if(argc==2 && std::string(argv[1])=="--rotary-fused-check") {check_rotary_fusion();return 0;}
        if(argc==2 && std::string(argv[1])=="--rotary-normalized-check") {check_rotary_normalized_fusion();return 0;}
        if(argc==2 && std::string(argv[1])=="--attention-fused-check") {check_attention_fusion();return 0;}
        if (argc == 1) {
            auto q = mfq::cuda::ones({1,1,2,4}, mfq::cuda::TensorOptions{}.device(mfq::cuda::kCUDA));
            auto k = mfq::cuda::ones({1,3,4}, q.options());
            auto got = mfq_qwen4_exp::block_scores(q,k).cpu();
            for (int i = 0; i < 3; ++i)
                if (got.data_ptr<float>()[i] != 4.f) throw std::runtime_error("QSA smoke mismatch");
            check_shared_sparse_attention();
            std::cout << "Flash-Next and shared sparse-attention native smoke passed\n";
            return 0;
        }
        if (std::string(argv[1]) != "--json") throw std::runtime_error("expected --json");
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto request = Json::parse(line);
                std::vector<Tensor> tensors;
                for (const auto& item : request.at("inputs")) tensors.push_back(input(item));
                const auto op = request.at("op").get<std::string>();
                const auto params = request.value("params", Json::object());
                const auto execute = [&] { return run(op, tensors, params); };
                std::vector<Tensor> result;
                mfq::cuda::Graph graph;
                if (request.value("graph", false)) {
                    result = execute();
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq::cuda::current_stream(0).stream()));
                    graph.prepare_memory();
                    result = execute();
                    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq::cuda::current_stream(0).stream()));
                    result.clear();
                    graph.capture_begin();
                    result = execute();
                    graph.capture_end();
                    graph.replay();
                    graph.replay();
                    if (request.contains("replay_inputs")) {
                        const auto& updates = request.at("replay_inputs");
                        if (updates.size() != tensors.size()) throw std::runtime_error("replay fixture count mismatch");
                        for (size_t i = 0; i < tensors.size(); ++i) {
                            if (tensors[i].defined()) tensors[i].copy_(input(updates.at(i)));
                        }
                        graph.replay();
                    }
                } else result = execute();
                std::cout << Json({{"outputs", output(result)}}).dump() << std::endl;
            } catch (const std::exception& e) {
                std::cout << Json({{"error", e.what()}}).dump() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Flash-Next native test failed: " << e.what() << '\n';
        return 1;
    }
}
