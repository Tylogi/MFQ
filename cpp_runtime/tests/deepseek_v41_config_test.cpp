#include "models/deepseek_v41/config.h"
#include "models/deepseek_v41/causal_lm.h"
#include "models/deepseek_v41/engram.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mfq::models::deepseek_v41::Config;
using mfq::models::deepseek_v41::TensorNames;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

const char* valid_config = R"json({
  "model_type": "deepseek_v41",
  "eos_token_id": [1, 2],
  "image_token_id": 129264,
  "quantization_config": {
    "quant_method": "fp8",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4"
  },
  "text_config": {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 4,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "rms_norm_eps": 1e-20,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "rope_scaling": {
      "rope_type": "yarn",
      "factor": 16,
      "original_max_position_embeddings": 65536
    },
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "norm_topk_prob": true,
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "sliding_window": 128,
    "compress_ratios": [0, 2, 1, 1, 0, 0, 0],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [1, 2],
    "index_source_layer_ids": [1, 2],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 2,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": [1],
    "engram_num_embeddings": [384006168],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [1, 2, 3],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3
  },
  "vision_config": {
    "model_type": "deepseek_v41_vision",
    "num_hidden_layers": 32,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936
  }
})json";

template <class Mutator>
void require_rejected(Mutator mutate, const char* message) {
    auto document = nlohmann::json::parse(valid_config);
    mutate(document);
    try {
        static_cast<void>(Config::from_json(document.dump()));
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

void check_retained_engram_history() {
    using mfq::models::deepseek_v41::EngramHashState;
    const auto make=[] {return EngramHashState(8,3,2,0,{1},{36},{5,7,11,13},
        {0,5,12,23},{3,5,7},{0,1,2,3,4,5,6,7});};
    const auto tokens=[](int batch,int count,int offset) {
        std::vector<int64_t> result(batch*count);
        for(int b=0;b<batch;++b)for(int i=0;i<count;++i)result[b*count+i]=(b*3+offset+i)%8;
        return result;
    };
    for(int batch:{1,2})for(int kept=0;kept<=5;++kept) {
        auto reference=make(),candidate=make();
        auto prompt=tokens(batch,3,0);
        const auto prompt_preview=candidate.preview(prompt.data(),batch,3,0);
        reference.forward(prompt.data(),batch,3,0);
        require(candidate.forward(prompt.data(),batch,3,0).values==prompt_preview.values,
                "Engram preview disagreed with fresh prompt hashes");
        candidate.begin_speculative();
        auto first=tokens(batch,2,3),second=tokens(batch,3,5);
        const auto first_preview=candidate.preview(first.data(),batch,2,3);
        require(candidate.forward(first.data(),batch,2,3).values==first_preview.values,
                "Engram preview changed a speculative token's hashes");
        const auto second_preview=candidate.preview(second.data(),batch,3,5);
        require(candidate.forward(second.data(),batch,3,5).values==second_preview.values,
                "Engram preview changed multi-forward speculative history");
        if(kept) {auto accepted=tokens(batch,kept,3);reference.forward(accepted.data(),batch,kept,3);}
        candidate.rollback_speculative(3+kept);
        auto next=tokens(batch,2,7);
        auto alternative=tokens(batch,3,2);
        candidate.preview(alternative.data(),batch,3,3+kept);
        const auto preview=candidate.preview(next.data(),batch,2,3+kept);
        require(candidate.forward(next.data(),batch,2,3+kept).values==preview.values && preview.values==
                    reference.forward(next.data(),batch,2,3+kept).values,
                "partial Engram rollback changed batched/multi-forward ngram history");
    }
}

void check_image_engram_history() {
    using mfq::models::deepseek_v41::EngramHashState;
    const auto make=[] {return EngramHashState(8,3,2,0,{1},{36},{5,7,11,13},
        {0,5,12,23},{3,5,7},{0,1,2,3,4,5,6,7});};
    const std::vector<int64_t> ids{1,2,3,4,5,6,7};
    const std::vector<uint8_t> mask{0,0,1,1,1,0,0};
    auto full=make(),chunked=make(),fresh=make();
    auto hashes=full.forward(ids.data(),1,int(ids.size()),0,mask.data());
    auto first=chunked.forward(ids.data(),1,4,0,mask.data());
    auto last=chunked.forward(ids.data()+4,1,3,4,mask.data()+4);
    auto combined=first.values;combined.insert(combined.end(),last.values.begin(),last.values.end());
    require(combined==hashes.values,"image Engram history changed across a prefill boundary");
    const auto suffix=fresh.forward(ids.data()+5,1,2,0);
    require(std::equal(suffix.values.begin(),suffix.values.end(),hashes.values.begin()+5*hashes.columns),
            "image span did not terminate the text ngram history");
    const int64_t next=2;
    require(full.forward(&next,1,1,7).values==chunked.forward(&next,1,1,7).values,
            "decode after an image changed Engram history");
}

void check_dspark_requested_head() {
    for(int width:{1,2,5})for(int stages:{1,3}) {
        auto run=[&](int requested) {
            return mfq::models::deepseek_v41::dspark_draft(width,requested,stages,
                [](int64_t count) {
                    std::vector<int> rows(count);
                    for(int i=0;i<count;++i)rows[i]=i+1;
                    return rows;
                },
                [](const auto& rows){return rows;},
                [](const auto&){return 1;},
                [&](std::size_t stage,auto rows,int& pre,int64_t output_rows) {
                    require(rows.size()==std::size_t(width),"DSpark stage lost physical block rows");
                    int total=0;for(auto value:rows)total+=value;
                    for(auto& value:rows)value+=total+int(stage)+pre;
                    ++pre;rows.resize(output_rows);return rows;
                },
                [](auto rows,int pre){for(auto& value:rows)value*=pre;return rows;},
                [&](auto rows) {
                    require(rows.size()==std::size_t(requested),"DSpark projected unused vocabulary rows");
                    for(auto& value:rows)value=value*17+3;
                    return rows;
                },
                [](auto rows,int64_t count){rows.resize(count);return rows;});
        };
        const auto full=run(width);
        for(int requested=1;requested<=width;++requested) {
            const auto prefix=run(requested);
            require(std::equal(prefix.begin(),prefix.end(),full.begin()),
                    "DSpark head pruning changed requested logits");
        }
    }
}
} // namespace

int main() {
    try {
        check_retained_engram_history();
        check_image_engram_history();
        const auto config = Config::from_json(valid_config);
        require(config.causal_encoder_layers == 2, "CED split changed");
        require(config.compress_ratios.size() == 7, "compression schedule changed");
        require(config.is_kv_source(1), "KV source schedule was lost");
        require(config.is_index_source(2), "index source schedule was lost");
        require(config.has_engram(1), "Engram schedule was lost");
        require(config.has_dspark(), "DSpark schedule was lost");
        require(config.has_vision(), "vision contract was lost");
        require(config.eos_token_ids == std::vector<std::int64_t>({1, 2}),
                "EOS array was not preserved");
        require(TensorNames::layer(2, "attention.query.weight") ==
                    "model.block.2.attention.query.weight",
                "layer tensor name changed");
        require(TensorNames::predictor(1, "main_projection.weight") ==
                    "predictor.stage.1.main_projection.weight",
                "predictor tensor name changed");

        require_rejected(
            [](auto& root) { root["model_type"] = "deepseek_v4"; },
            "legacy DeepSeek-V4 alias was accepted");
        require_rejected(
            [](auto& root) {
                root["text_config"]["compress_ratios"] = {0, 2, 1, 0, 0, 0, 0};
            },
            "non-decoder compression after the CED split was accepted");
        require_rejected(
            [](auto& root) {
                root["text_config"]["kv_source_layer_ids"] = {0, 2};
            },
            "zero-ratio KV source was accepted");
        require_rejected(
            [](auto& root) {
                root["quantization_config"]["weight_block_size"] = {128, 128};
            },
            "unsupported source encoding was accepted");

        check_dspark_requested_head();
        std::cout << "DeepSeek-V4.1 common config and requested DSpark head tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DeepSeek-V4.1 common config test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
