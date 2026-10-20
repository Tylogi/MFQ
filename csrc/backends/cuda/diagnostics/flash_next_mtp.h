#pragma once

#include "nlohmann/json.hpp"
#include "generation_result.h"

#include <cstdint>
#include <utility>
#include <vector>

// Architecture-specific CLI diagnostics. Production generation uses the
// common MtpModule path and does not depend on these concrete types.

template <typename Model, typename Predictor>
static int run_flash_next_mtp_check(Model& model, Predictor& mtp) {
    namespace tb=mfq_tensor_backend;
    const MtpTarget target(model);
    MFQ_RUNTIME_CHECK(
        model.metadata.flash_next && model.vocab_size() >= 8 &&
            model.max_position_embeddings() >= 24,
        "Flash-Next MTP diagnostic requires vocab>=8 and context>=24");
    const auto width=mtp.hidden_norm.numel();
    auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    auto ids=tb::tensor(std::vector<int64_t>{1,2,3,4,5,6,7},options).reshape({1,7});
    auto previous=(tb::arange(7*width,options).remainder(31).to(tb::kFloat32)*.01-.15).reshape({1,7,width});
    const auto json_tensor=[](const tb::Tensor& value) {
        auto host=value.to(tb::kFloat32).contiguous().cpu();
        return nlohmann::json{{"shape",host.sizes().vec()},
            {"data",std::vector<float>(host.template data_ptr<float>(),host.template data_ptr<float>()+host.numel())}};
    };
    nlohmann::json result;
    result["architecture"] = model.graph.backbone;
    result["previous"]=json_tensor(previous);
    for (int64_t i=0;i<int64_t(mtp.lengths.size());++i) {
        nlohmann::json row;
        mtp.reset();auto full=mtp.evaluate(target,previous,ids,i);
        row["full"]=json_tensor(full.first);row["multi"]=json_tensor(full.second);
        row["logits"]=json_tensor(model.logits_from_hidden(full.first));
        const auto saved=mtp.lengths;
        row["uncached"]=json_tensor(mtp.evaluate(target,previous,ids,i,false).first);
        MFQ_RUNTIME_CHECK(saved==mtp.lengths,"uncached MTP changed layer positions");
        mtp.reset();std::vector<tb::Tensor> pieces;
        for (auto [begin,count]:std::vector<std::pair<int64_t,int64_t>>{{0,2},{2,1},{3,4}})
            pieces.push_back(mtp.evaluate(target,previous.narrow(1,begin,count),ids.narrow(1,begin,count),i).first);
        row["chunked"]=json_tensor(tb::cat(pieces,1));
        mtp.reset();row["reset"]=json_tensor(mtp.evaluate(target,previous,ids,i).first);
        // Changing batch resets every layer, not just the selected depth.
        row["batch"]=json_tensor(mtp.evaluate(target,previous.repeat({2,1,1}),ids.repeat({2,1}),i).first);
        row["batch_reset"]=json_tensor(mtp.evaluate(target,previous,ids,i).first);
        auto pos=ids.reshape({7})+2;
        if (model.metadata.multi_axis_positions) {
            pos = tb::stack({pos + 10, pos, pos + 2, pos + 4}, 0);
        }
        row["axis"]=json_tensor(mtp.evaluate(target,previous,ids,i,false,pos).first);
        mtp.reset();
        for (int64_t other=0;other<int64_t(mtp.lengths.size());++other)
            if (other!=i) (void)mtp.evaluate(target,previous,ids,other);
        row["independent"]=json_tensor(mtp.evaluate(target,previous,ids,i).first);
        result["layers"].push_back(std::move(row));
    }
    model.reset(1);tb::Tensor raw;
    auto normalized=model.hidden_forward(ids,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&raw);
    result["target_raw"]=json_tensor(raw);result["target_normalized"]=json_tensor(normalized);
    if constexpr(requires { mtp.graph_replays; }) {
        // Restore A after an unrelated B has overwritten every recurrent,
        // PLE and predictor cache. Snapshots must own their saved tensors.
        auto state=model.capture_text_session_state({1,2,3,4,5,6,7});
        mtp.reset();
        (void)mtp.evaluate(target,raw.narrow(1,0,6),ids.narrow(1,1,6));
        auto predictor=mtp.capture_session_state(7,raw.narrow(1,6,1));
        state.mtp=predictor;state.bytes+=predictor.bytes;
        std::vector<CudaPagedPayload> disk{encode_cuda_qwen4_paged_block(state,4,0),encode_cuda_qwen4_paged_block(state,4,1)};
        MFQ_RUNTIME_CHECK(cuda_paged_decodable_blocks({disk.front()})==0 && cuda_paged_decodable_blocks(disk)==2 &&
            cuda_paged_payload_tokens(disk.back())==3,"Qwen4 SSD prefix lost its exact boundary or tail");
        auto cold=decode_cuda_qwen4_paged_session(disk,state.tokens,4);
        auto next=tb::tensor(std::vector<int64_t>{2,3},options).reshape({1,2});
        tb::Tensor extension_raw;
        auto extension=model.hidden_forward(next,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&extension_raw);
        auto pairs=tb::cat({raw.narrow(1,6,1),extension_raw.narrow(1,0,1)},1);
        auto expected=mtp.evaluate(target,pairs,next);
        for(const auto* saved:{&state,&cold}) {
            model.reset(1);(void)model.hidden_forward(ids.narrow(1,0,3));
            model.restore_text_session_state(*saved);mtp.restore_session_state(*saved->mtp);
            tb::Tensor restored_raw;
            auto restored=model.hidden_forward(next,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&restored_raw);
            auto restored_pairs=tb::cat({saved->mtp->last_target_hidden,restored_raw.narrow(1,0,1)},1);
            auto actual=mtp.evaluate(target,restored_pairs,next);
            for(const auto& pair:std::vector<std::pair<tb::Tensor,tb::Tensor>>{
                {restored,extension},{restored_raw,extension_raw},{actual.first,expected.first},{actual.second,expected.second}}) {
                auto a=pair.first.to(tb::kFloat32).contiguous().cpu(),b=pair.second.to(tb::kFloat32).contiguous().cpu();
                MFQ_RUNTIME_CHECK(a.sizes()==b.sizes(),"Qwen4 restored session changed output shape");
                for(int64_t i=0;i<a.numel();++i)MFQ_RUNTIME_CHECK(
                    std::abs(a.template data_ptr<float>()[i]-b.template data_ptr<float>()[i])<=.003+.003*std::abs(b.template data_ptr<float>()[i]),
                    "Qwen4 restored session changed target or predictor output");
            }
        }
        result["session_restore_checked"]=true;
        result["ssd_session_restore_checked"]=true;
    }
    if constexpr(requires { mtp.graph_replays; }) {
        double maximum_error=0;
        for(int tokens=1;tokens<=6;++tokens) {
            const auto prime=[&] {
                mtp.reset();(void)mtp.evaluate(target,previous.narrow(1,0,2),ids.narrow(1,0,2));
            };
            prime();auto expected=mtp.evaluate(target,previous.narrow(1,1,tokens),ids.narrow(1,1,tokens));
            prime();auto actual=mtp.step(target,previous.narrow(1,1,tokens),ids.narrow(1,1,tokens));
            std::vector<std::pair<tb::Tensor,tb::Tensor>> pairs{
                {actual.sample_hidden,expected.first},{actual.chain_hidden,expected.second}};
            if(actual.sample_logits.defined()) {
                pairs.emplace_back(actual.sample_logits,
                    target.logits(expected.first.narrow(1,tokens-1,1).contiguous()));
                MFQ_RUNTIME_CHECK(actual.greedy_token==actual.sample_logits.argmax(-1).template item<int64_t>(),
                    "MTP predictor graph returned stale greedy token");
            }
            for(const auto& [value,reference]:pairs) {
                auto a=value.to(tb::kFloat32).contiguous().cpu(),b=reference.to(tb::kFloat32).contiguous().cpu();
                MFQ_RUNTIME_CHECK(a.sizes()==b.sizes(),"MTP predictor graph changed its output shape");
                for(int64_t i=0;i<a.numel();++i) {
                    const double error=std::abs(double(a.template data_ptr<float>()[i])-b.template data_ptr<float>()[i]);
                    maximum_error=std::max(maximum_error,error);
                    MFQ_RUNTIME_CHECK(std::isfinite(error) && error<=.003+.003*std::abs(b.template data_ptr<float>()[i]),
                        "MTP predictor graph differs from eager evaluation");
                }
            }
        }
        result["predictor_graph_max_abs_error"]=maximum_error;
    }
    // Exercise the production runtime generation path, target rollback and
    // step output limits. The two architectures retain distinct head equations.
    MfqSamplingParams sampling;sampling.max_tokens=12;sampling.temperature=0;sampling.top_k=1;
    std::vector<int64_t> prompt{1,2,3,4,5,6,7},expected,generated;
    model.reset(1);auto current=ids;
    for (int i=0;i<sampling.max_tokens;++i) {
        auto next=model.next_token(current);expected.push_back(next.template item<int64_t>());current=next.reshape({1,1});
    }
    generated=mfq::cuda::diagnostics::check_mtp_steps(model,mtp,prompt,sampling);
    const auto count=generated.size();
    MFQ_RUNTIME_CHECK(count==sampling.max_tokens && generated==expected && mtp.last_cycles>0,
        "Flash-Next MTP greedy tokens disagree with incremental target");
    result["greedy"]=generated;result["cycles"]=mtp.last_cycles;
    result["accepted"]=mtp.last_accepted;result["rejected"]=mtp.last_rejected;
    result["drafted"]=mtp.last_stats.drafted_tokens;
    result["selected_depth"]=mtp.last_stats.selected_depth;
    result["depth_cycles"]=mtp.last_stats.depth_cycles;
    if constexpr(requires { mtp.graph_replays; })result["predictor_graph_replays"]=mtp.graph_replays;
    generated.clear();
    generated=mfq::cuda::diagnostics::check_mtp_steps(model,mtp,prompt,sampling,3);
    const auto stopped=generated.size();
    MFQ_RUNTIME_CHECK(stopped==3 && generated==std::vector<int64_t>(expected.begin(),expected.begin()+3),
        "Flash-Next MTP step emitted extra or incorrect tokens");
    sampling.max_tokens=8;sampling.temperature=.8;sampling.top_k=16;sampling.top_p=.95;
    sampling.presence_penalty=.2;sampling.frequency_penalty=.1;sampling.repetition_penalty=1.05;sampling.seed=20260907;
    MFQ_RUNTIME_CHECK(mfq::cuda::diagnostics::check_mtp_steps(model,mtp,prompt,sampling).size()==8,
        "Flash-Next stochastic MTP failed to generate requested tokens");
    std::cout<<"flash_next_mtp_check "<<result.dump()<<'\n';return 0;
}
