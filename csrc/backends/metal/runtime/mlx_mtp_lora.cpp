#include "mlx_mtp_lora.h"
#include "mlx_resident_budget.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;
thread_local MlxMtpLora* current_lora = nullptr;
const MlxMtpLoraProjection& projection(const std::vector<MlxMtpLoraProjection>& layout, const std::string& name) {
    const auto found = std::find_if(layout.begin(), layout.end(), [&](const auto& item) { return item.name == name; });
    if (found == layout.end()) throw std::invalid_argument("unknown MTP LoRA projection: " + name);
    return *found;
}
array cpu_copy(const array& input, Stream cpu) {
    auto value = contiguous(astype(input, float32, cpu), false, cpu);
    value.eval();
    return array(value.data<float>(), value.shape(), float32);
}
array log_softmax(const array& input, int axis, Stream cpu) {
    return subtract(input, logsumexp(input, axis, true, cpu), cpu);
}
}

MlxMtpLoraParameters::MlxMtpLoraParameters(const std::vector<MlxMtpLoraProjection>& layout,
    array a, array b, std::shared_ptr<MlxAneMatmul> engine)
    : layout_(layout), a_(std::move(a)), b_(std::move(b)), engine_(std::move(engine)) {}
Stream MlxMtpLoraParameters::cpu() const { return engine_->cpu(); }
array MlxMtpLoraParameters::linear(const array& x, const array& weight) const {
    auto shape = x.shape();
    shape.back() = weight.shape(0);
    return reshape((*engine_)(reshape(x, {-1, weight.shape(1)}, cpu()),
        transpose(weight, cpu())), shape, cpu());
}
array MlxMtpLoraParameters::delta(const std::string& name, const array& x) const {
    const auto& item = projection(layout_, name);
    const auto a = slice(a_, {item.input_offset, 0}, {item.input_offset + item.input, a_.shape(1)}, cpu());
    const auto b = slice(b_, {0, item.output_offset}, {b_.shape(0), item.output_offset + item.output}, cpu());
    auto shape = x.shape();
    shape.back() = item.output;
    return reshape((*engine_)((*engine_)(reshape(x, {-1, item.input}, cpu()), a), b), shape, cpu());
}

struct MlxMtpLora::Example final : MtpLoraExample {
    std::vector<array> constants;
    MlxMtpLoraGraph body, final;
    std::size_t bytes() const noexcept override {
        std::size_t total = 0;
        for (const auto& item : constants) total += item.nbytes();
        return total;
    }
};

MlxMtpLora::MlxMtpLora(std::vector<MlxMtpLoraProjection> layout, array frozen_head,
    std::string fingerprint, std::filesystem::path directory, std::size_t hot_bytes, std::size_t disk_bytes, int rank, int routes)
    : vocabulary_(frozen_head.shape(0)), rank_(rank), layout_(std::move(layout)),
      frozen_head_(std::move(frozen_head)), engine_(std::make_shared<MlxAneMatmul>()) {
    if (layout_.empty() || rank_ != 16 || routes <= 0 || routes > 512 || frozen_head_.ndim() != 2)
        throw std::invalid_argument("invalid internal MTP LoRA layout");
    for (auto& item : layout_) {
        if (item.name.empty() || item.input <= 0 || item.output <= 0 ||
            std::count_if(layout_.begin(), layout_.end(), [&](const auto& other) { return item.name == other.name; }) != 1)
            throw std::invalid_argument("duplicate or invalid internal MTP LoRA projection");
        item.input_offset = inputs_;
        item.output_offset = outputs_;
        if (item.name.starts_with("expert.0."))
            training_workspace_ += std::size_t(5) * routes * item.input * item.output * sizeof(float);
        inputs_ += item.input;
        outputs_ += item.output;
        for (auto [m, k, n] : {std::tuple{1, item.input, rank_}, std::tuple{1, rank_, item.output},
            std::tuple{1, item.output, rank_}, std::tuple{rank_, 1, item.output},
            std::tuple{item.input, 1, rank_}})
            engine_->prepare(m, k, n);
    }
    engine_->prepare(1, frozen_head_.shape(1), vocabulary_);
    engine_->prepare(1, vocabulary_, frozen_head_.shape(1));
    frozen_head_ = cpu_copy(frozen_head_, engine_->cpu());
    fingerprint += "\ninternal-ffn-qsa-rank16-v1";
    for (const auto& item : layout_) fingerprint += "\n" + item.name + ":" +
        std::to_string(item.input) + ":" + std::to_string(item.output);
    sessions_ = std::make_unique<MtpLoraSessions>(inputs_, outputs_, rank_, std::move(fingerprint),
        std::move(directory), hot_bytes, disk_bytes,
        [this](const MtpLoraState& state, const MtpLoraBatch& batch) { return train(state, batch); },
        engine_->bytes() + frozen_head_.nbytes(), [layout = layout_, inputs = inputs_](MtpLoraState& state) {
            for (const auto& item : layout) {
                const auto factor = std::sqrt(float(inputs) / item.input);
                for (int row = item.input_offset; row < item.input_offset + item.input; ++row)
                    for (int r = 0; r < state.rank; ++r) state.a[row * state.rank + r] *= factor;
            }
        });
}

MlxMtpLora* MlxMtpLora::current() noexcept { return current_lora; }
MlxMtpLoraScope::MlxMtpLoraScope(MlxMtpLora* value) : previous_(current_lora) { current_lora = value; }
MlxMtpLoraScope::~MlxMtpLoraScope() { current_lora = previous_; }

void MlxMtpLora::begin(const std::string& session) {
    end();
    if (!session.empty()) {
        try { MlxResidentBudgetScope::reserve(std::size_t(inputs_ + outputs_) * rank_ * sizeof(float) * 4 + 112); }
        catch (const std::exception&) { ++budget_skips_; return; }
    }
    state_ = sessions_->begin(session);
    if (state_) { last_version_ = state_->version; last_optimizer_step_ = state_->optimizer_step; }
}
void MlxMtpLora::end() {
    examples_.clear();
    state_.reset();
    a_.reset(); b_.reset();
    routed_banks_.clear();
    device_version_ = UINT64_MAX;
    collect_ = false;
    sessions_->end();
}
void MlxMtpLora::begin_round() {
    examples_.clear();
    depth_ = 0;
    state_ = sessions_->boundary();
    if (state_) { last_version_ = state_->version; last_optimizer_step_ = state_->optimizer_step; }
    collect_ = state_ && sessions_->wants_batch();
    if (collect_) {
        try { MlxResidentBudgetScope::reserve(state_->bytes() * 4 + training_workspace_ + (64ULL << 20)); }
        catch (const std::exception&) { collect_ = false; ++budget_skips_; }
    }
    if (state_ && state_->version > 0 && state_->version != device_version_) {
        a_ = astype(array(state_->a.begin(), Shape{inputs_, rank_}, float32), float16);
        b_ = astype(array(state_->b.begin(), Shape{rank_, outputs_}, float32), float16);
        routed_banks_.clear();
        device_version_ = state_->version;
    }
}
void MlxMtpLora::prepare_linear(const array& weight) {
    if (weight.ndim() != 2) throw std::invalid_argument("MTP training linear is not a matrix");
    eval(weight);
    engine_->prepare(1, weight.shape(1), weight.shape(0));
    engine_->prepare(1, weight.shape(0), weight.shape(1));
    sessions_->set_trainer_bytes(engine_->bytes() + frozen_head_.nbytes());
}
void MlxMtpLora::prepare_attention(int dimension, int keys) {
    engine_->prepare(1, dimension, keys);
    engine_->prepare(1, keys, dimension);
    sessions_->set_trainer_bytes(engine_->bytes() + frozen_head_.nbytes());
}
array MlxMtpLora::routed_apply(const std::string& role, const array& x, const array& base, const array& ids) {
    if (!active()) return base;
    if (!routed_banks_.contains(role)) {
        const auto& first = projection(layout_, "expert.0." + role);
        const int count = std::count_if(layout_.begin(), layout_.end(), [&](const auto& item) {
            return item.name.starts_with("expert.") && item.name.ends_with("." + role);
        });
        auto a = reshape(slice(*a_, {first.input_offset, 0}, {first.input_offset + count * first.input, rank_}),
            {count, first.input, rank_});
        auto b = transpose(reshape(slice(*b_, {0, first.output_offset}, {rank_, first.output_offset + count * first.output}),
            {rank_, count, first.output}), {1, 0, 2});
        routed_banks_.emplace(role, std::pair{std::move(a), std::move(b)});
    }
    const auto& [a, b] = routed_banks_.at(role);
    const int routes = ids.size(), rows = x.size() / a.shape(1);
    auto expert_ids = reshape(astype(ids, int32), {routes});
    auto lhs = rows == routes ? arange(routes, int32) : reshape(broadcast_to(
        expand_dims(arange(rows, int32), 1), ids.shape()), {routes});
    auto features = gather_mm(reshape(astype(x, float16), {rows, 1, a.shape(1)}), a, lhs, expert_ids);
    auto delta = gather_mm(features, b, arange(routes, int32), expert_ids);
    return base + reshape(astype(delta, base.dtype()), base.shape());
}
array MlxMtpLora::apply(const std::string& name, const array& x, const array& base) {
    if (!state_ || state_->version == 0) return base;
    const auto& item = projection(layout_, name);
    const auto a = slice(*a_, {item.input_offset, 0}, {item.input_offset + item.input, rank_});
    const auto b = slice(*b_, {0, item.output_offset}, {rank_, item.output_offset + item.output});
    const auto delta = matmul(matmul(astype(reshape(x, {-1, item.input}), float16), a), b);
    return base + reshape(astype(delta, base.dtype()), base.shape());
}
void MlxMtpLora::capture(std::vector<array> constants, MlxMtpLoraGraph body) {
    ++depth_;
    if (!collect_) return;
    auto example = std::make_shared<Example>();
    example->constants = std::move(constants);
    example->body = std::move(body);
    examples_.push_back(std::move(example));
}
void MlxMtpLora::set_final_graph(MlxMtpLoraGraph final) {
    if (collect_ && !examples_.empty()) examples_.back()->final = std::move(final);
}
void MlxMtpLora::verified(const array& teacher_logits, int accepted, double cycle_ms) {
    sessions_->observe(cycle_ms, accepted + 1, depth_);
    if (!collect_ || examples_.empty()) return;
    const int rows = std::min<int>(accepted + 1, examples_.size());
    MtpLoraBatch batch;
    batch.rows = rows;
    batch.workspace_bytes = state_->bytes() * 2 + training_workspace_ + (64ULL << 20);
    auto teacher = cpu_copy(slice(teacher_logits, {0, 0}, {rows, vocabulary_}), engine_->cpu());
    batch.teacher_logits.assign(teacher.data<float>(), teacher.data<float>() + teacher.size());
    for (int row = 0; row < rows; ++row) {
        auto& example = examples_[row];
        if (!example->body || !example->final) throw std::logic_error("incomplete internal MTP training graph");
        eval(example->constants);
        for (auto& item : example->constants) item = cpu_copy(item, engine_->cpu());
        batch.examples.push_back(example);
        batch.position_weights.push_back(std::pow(0.8f, row));
    }
    sessions_->submit(std::move(batch), state_->version);
    examples_.clear();
    collect_ = false;
}

MtpLoraUpdate MlxMtpLora::train(const MtpLoraState& state, const MtpLoraBatch& batch) {
    if (batch.rows <= 0 || batch.rows > 5 || batch.examples.size() != std::size_t(batch.rows) ||
        batch.teacher_logits.size() != std::size_t(batch.rows) * vocabulary_ ||
        batch.position_weights.size() != std::size_t(batch.rows))
        throw std::invalid_argument("internal MTP training batch mismatch");
    state.validate();
    const auto cpu = engine_->cpu();
    const auto teacher = array(batch.teacher_logits.begin(), Shape{batch.rows, vocabulary_}, float32);
    const auto log_p = log_softmax(teacher, -1, cpu);
    const auto p = exp(log_p, cpu);
    const auto weight_sum = std::accumulate(batch.position_weights.begin(), batch.position_weights.end(), 0.0f);
    if (!(weight_sum > 0)) throw std::invalid_argument("invalid MTP training weights");
    const auto weights = reshape(array(batch.position_weights.begin(), Shape{batch.rows}, float32), {batch.rows, 1}, cpu);
    const auto forward = [&](const std::vector<array>& parameters) {
        MlxMtpLoraParameters params(layout_, parameters[0], parameters[1], engine_);
        std::vector<array> rows;
        for (const auto& opaque : batch.examples) {
            const auto example = std::dynamic_pointer_cast<const Example>(opaque);
            if (!example) throw std::invalid_argument("invalid MTP training example");
            auto hidden = example->body(params, example->constants);
            hidden = example->final(params, {hidden});
            rows.push_back(params.linear(reshape(hidden, {1, frozen_head_.shape(1)}, cpu), frozen_head_));
        }
        return concatenate(rows, 0, cpu);
    };
    const auto loss = [&](const array& logits) {
        return divide(sum(multiply(multiply(p, subtract(log_p, log_softmax(logits, -1, cpu), cpu), cpu),
            weights, cpu), cpu), array(weight_sum), cpu);
    };
    const std::vector<array> parameters{
        array(state.a.begin(), Shape{inputs_, rank_}, float32),
        array(state.b.begin(), Shape{rank_, outputs_}, float32)};
    auto differentiated = value_and_grad([&](const std::vector<array>& args) {
        auto logits = forward(args);
        return std::vector<array>{multiply(loss(logits), array(1024.0f), cpu), logits};
    }, std::vector<int>{0, 1})(parameters);
    eval(differentiated.first);
    eval(differentiated.second);
    MtpLoraUpdate result;
    result.loss_before = differentiated.first[0].item<float>() / 1024;
    std::vector<float> gradients;
    for (auto gradient : differentiated.second) {
        gradient = contiguous(divide(gradient, array(1024.0f), cpu), false, cpu);
        gradient.eval();
        gradients.insert(gradients.end(), gradient.data<float>(), gradient.data<float>() + gradient.size());
    }
    auto candidate = mtp_lora_adam(state, gradients);
    if (!candidate) return result;
    const auto old_log_q = log_softmax(differentiated.first[1], -1, cpu);
    for (int attempt = 0; attempt < 3; ++attempt) {
        const auto updated = forward({array(candidate->a.begin(), Shape{inputs_, rank_}, float32),
            array(candidate->b.begin(), Shape{rank_, outputs_}, float32)});
        auto after = loss(updated);
        auto trust = divide(sum(multiply(multiply(exp(old_log_q, cpu),
            subtract(old_log_q, log_softmax(updated, -1, cpu), cpu), cpu), weights, cpu), cpu), array(weight_sum), cpu);
        eval(after, trust);
        result.loss_after = after.item<float>();
        result.trust_kl = trust.item<float>();
        if (std::isfinite(result.loss_after) && result.loss_after <= result.loss_before + 1e-7 &&
            std::isfinite(result.trust_kl) && result.trust_kl <= 0.02) {
            result.state = std::move(candidate);
            break;
        }
        for (std::size_t i = 0; i < state.a.size(); ++i)
            candidate->a[i] = state.a[i] + 0.5f * (candidate->a[i] - state.a[i]);
        for (std::size_t i = 0; i < state.b.size(); ++i)
            candidate->b[i] = state.b[i] + 0.5f * (candidate->b[i] - state.b[i]);
    }
    return result;
}
bool MlxMtpLora::close(const std::string& id) { return sessions_->close(id); }
bool MlxMtpLora::fork(const std::string& source, const std::string& target) {
    try { MlxResidentBudgetScope::reserve(std::size_t(inputs_ + outputs_) * rank_ * sizeof(float) * 4 + 112); }
    catch (const std::exception&) { ++budget_skips_; return false; }
    return sessions_->fork(source, target);
}
void MlxMtpLora::clear() { end(); sessions_->clear(); }
std::size_t MlxMtpLora::trim(std::size_t bytes) { return sessions_->trim(bytes); }
std::size_t MlxMtpLora::bytes() const { return sessions_->bytes() - frozen_head_.nbytes(); }
std::vector<std::pair<std::string, double>> MlxMtpLora::metrics() const {
    auto result = sessions_->metrics();
    result.emplace_back("mtp_ttt_budget_skips", budget_skips_);
    result.emplace_back("mtp_ttt_active_version", state_ ? state_->version : 0);
    result.emplace_back("mtp_ttt_last_version", last_version_);
    result.emplace_back("mtp_ttt_last_optimizer_step", last_optimizer_step_);
    result.emplace_back("mtp_ttt_rank", rank_);
    result.emplace_back("mtp_ttt_projections", layout_.size());
    return result;
}

} // namespace mfq::metal
