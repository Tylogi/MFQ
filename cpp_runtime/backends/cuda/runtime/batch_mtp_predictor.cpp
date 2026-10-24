#include "models/qwen35/mtp.h"
#include "runtime/batch_decode_inputs.h"

namespace {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;

struct FullAttentionPredictorState final : MtpBatchState {
    std::vector<KVCache> caches;
    void trim(int64_t value) override {
        MFQ_RUNTIME_CHECK(value >= 0 && value <= position, "invalid batched predictor trim");
        position = value;
    }
};
}

std::unique_ptr<MtpBatchState> Qwen35Mtp::make_batch_state() const {
    auto state = std::make_unique<FullAttentionPredictorState>();
    state->caches.resize(blocks.size());
    return state;
}

MtpStep Qwen35Mtp::batch_step(const MtpTarget& target, const std::vector<MtpBatchState*>& states,
        Tensor hidden, Tensor ids) {
    const int64_t batch = states.size(), tokens = ids.size(1), width = config.hidden_size;
    MFQ_RUNTIME_CHECK(batch > 0 && ids.size(0) == batch && hidden.size(0) == batch &&
        hidden.size(1) == tokens, "invalid batched Qwen predictor geometry");
    std::vector<int64_t> positions, lengths;
    int64_t maximum = 0;
    for (const auto* state : states) {
        MFQ_RUNTIME_CHECK(state->position + tokens <= config.max_position_embeddings,
                          "batched predictor exceeds context");
        maximum = std::max(maximum, state->position);
        lengths.push_back(state->position + tokens);
        for (int64_t token = 0; token < tokens; ++token) positions.push_back(state->position + token);
    }
    auto pos = tb::tensor(positions, tb::TensorOptions().dtype(tb::kInt64))
        .reshape({batch, tokens}).to(ids.device());
    auto ends = tb::tensor(lengths, tb::TensorOptions().dtype(tb::kInt64)).to(ids.device());
    const auto normalize = [&](Tensor x, const Tensor& weight) {
        return qwen_rms_norm(x.reshape({-1, width}).to(tb::kFloat32), weight,
            config.rms_norm_eps, config.legacy_tensor_layout.norm_weight_offset).reshape(x.sizes());
    };
    auto embedding = normalize(target.embed(ids).to(hidden.scalar_type()), embedding_norm);
    auto current = fusion.forward(*execution, tb::cat({embedding, normalize(hidden, hidden_norm)}, -1));
    mfq::cuda::BatchDecodeInputs inputs(batch, 2, ids.get_device());
    std::vector<mfq::cuda::BatchDecodeRow> rows;
    for (int64_t row = 0; row < batch; ++row) rows.push_back({0, states[row]->position, row});
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        auto* full = dynamic_cast<FullBlock*>(blocks[index].get());
        MFQ_RUNTIME_CHECK(full && !full->sliding, "batched Qwen predictor requires full attention");
        inputs.update_rows(rows);
        for (int64_t row = 0; row < batch; ++row) {
            auto& cache = static_cast<FullAttentionPredictorState&>(*states[row]).caches[index];
            if (!cache.defined()) cache = KVCache(1, full->kv_heads, config.max_position_embeddings,
                full->attention_head_dim, false, ids.device(),
                full->official_bf16 ? tb::kBFloat16 : tb::kFloat16, full->kv_quantization);
            inputs.state(row, 0, cache.k, 2);
            inputs.state(row, 1, cache.v, 2);
        }
        inputs.upload();
        auto pointers = inputs.states().select(2, 0).contiguous();
        auto pages = tb::arange(batch, ids.options().dtype(tb::kInt32)).reshape({batch, 1});
        auto saved = std::move(full->cache);
        full->cache = KVCache::paged_view(pointers.select(0, 0), pointers.select(0, 1), pages,
            batch, full->kv_heads, full->attention_head_dim, config.max_position_embeddings, 1,
            full->official_bf16 ? tb::kBFloat16 : tb::kFloat16);
        full->cache.quantization = full->kv_quantization;
        try {
            current = full->forward(*execution, std::move(current), pos, maximum, ends, *target.rope, pos);
            full->cache = std::move(saved);
        } catch (...) { full->cache = std::move(saved); throw; }
    }
    auto output = normalize(std::move(current), output_norm);
    for (auto* state : states) state->position += tokens;
    return {output, output};
}
