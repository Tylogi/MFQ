#include "mlx_engine.h"
#include "tokenizer.h"

namespace mfq::metal {
using namespace mfq::engine;

MlxEngine::MlxEngine(std::unique_ptr<MlxEngineModel> model,
    std::unique_ptr<TextProcessor> text, EngineInfo info)
    : model_(std::move(model)), text_(std::move(text)), info_(std::move(info)) {
    if (!model_ || !text_) throw std::invalid_argument("Metal engine requires model and tokenizer");
    info_.max_requests = 1;
    info_.chat = text_->chat_template_capabilities();
}
MlxEngine::~MlxEngine() { try { shutdown(); } catch (...) {} }
EngineStatus MlxEngine::status() const {
    return model_ ? requests_.status(duplex_active_) : EngineStatus{0, false};
}
Admission MlxEngine::admit(EngineRequest request) {
    if (!model_) throw std::runtime_error("Metal engine is unloaded");
    return requests_.admit(std::move(request), text_.get(), info_, *this);
}
void MlxEngine::cancel(const RequestId& id) { requests_.cancel(id); }
EngineStepResult MlxEngine::step(const std::vector<RequestId>& eligible) {
    auto result = requests_.step(eligible, *this);
    if (result.has_work && result.events.empty())
        result.wake_at = Clock::now() + std::chrono::milliseconds(1);
    return result;
}
Generation MlxEngine::generate(const RequestId&, ExecutionRequest& request) {
    return consume_mlx_generation([this, &request](MlxGenerationJob& job) {
        model_->generate(request.input, job);
    }, request.output);
}
SessionResult MlxEngine::session(const SessionCommand& command) {
    if (!model_) throw std::runtime_error("Metal engine is unloaded");
    if (command.kind != SessionCommand::Kind::metrics &&
        command.kind != SessionCommand::Kind::trim &&
        command.kind != SessionCommand::Kind::budget && !requests_.empty())
        throw std::runtime_error("session operation requires a quiescent Metal engine");
    return model_->session(command);
}
ControlResult MlxEngine::control(ControlRequest request) {
    if (!model_) throw std::runtime_error("Metal engine is unloaded");
    return std::visit([&](auto value) -> ControlResult {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, DecodeTokens>)
            return text_->decode_tokens(value.tokens, value.excluded);
        else if constexpr (std::is_same_v<T, ScoreText>) {
            if (!info_.probability) throw std::invalid_argument("probability scoring is unavailable for this model");
            if (!requests_.empty() || duplex_active_)
                throw std::runtime_error("probability scoring requires a quiescent Metal engine");
            if (value.prompt.empty() || value.prompt.size() > 262144 || value.continuations.empty() || value.continuations.size() > 64)
                throw std::invalid_argument("score requires a bounded prompt and 1 to 64 continuations");
            const auto& tokenizer = text_->tokenizer();
            auto prompt = tokenizer.tokenize(value.prompt, false, false);
            if (prompt.empty()) throw std::invalid_argument("score prompt tokenized to an empty sequence");
            ScoreTokens prepared;
            prepared.prompt_tokens = prompt.size();
            prepared.next_token = value.next_token;
            std::size_t total = 0;
            for (const auto& continuation : value.continuations) {
                if (continuation.empty() || continuation.size() > 65536)
                    throw std::invalid_argument("score continuations must be nonempty and bounded");
                auto sequence = value.next_token ? prompt : tokenizer.tokenize(value.prompt + continuation, false, false);
                std::vector<std::int64_t> targets;
                if (value.next_token) {
                    targets = tokenizer.tokenize(continuation, false, false);
                    if (targets.size() != 1) throw std::invalid_argument("next_token candidates must encode to exactly one token");
                } else {
                    if (sequence.size() <= prompt.size() || !std::equal(prompt.begin(), prompt.end(), sequence.begin()))
                        throw std::invalid_argument("continuation changes the prompt token boundary; move the trailing whitespace into the continuation");
                    targets.assign(sequence.begin() + prompt.size(), sequence.end());
                }
                if (sequence.size() > 8192 || sequence.size() > static_cast<std::size_t>(info_.max_context))
                    throw std::invalid_argument("score exceeds the loaded context or 8192-token scoring limit");
                total += sequence.size();
                if (total > 131072) throw std::invalid_argument("score exceeds the aggregate token limit");
                prepared.sequences.push_back(std::move(sequence));
                prepared.targets.push_back(std::move(targets));
            }
            return model_->control(std::move(prepared));
        } else if constexpr (std::is_same_v<T, PrepareDuplex>) {
            text_->prepare_duplex_session(value.prompt, value.parameters);
            return std::move(value.parameters);
        } else if constexpr (std::is_same_v<T, PrepareDuplexStep>) {
            text_->prepare_duplex_step(value.text, value.input);
            return std::move(value.input);
        } else {
            if constexpr (!std::is_same_v<T, RuntimeMetrics>)
                if (!requests_.empty()) throw std::runtime_error("duplex conflicts with active generation");
            auto result = model_->control(std::move(value));
            if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) duplex_active_ = true;
            if constexpr (std::is_same_v<T, StopDuplex>) duplex_active_ = false;
            return result;
        }
    }, std::move(request));
}
std::int64_t MlxEngine::reload(std::int64_t context) {
    if (!model_) throw std::runtime_error("Metal engine is unloaded");
    if (context < 1) throw std::invalid_argument("reload context must be positive");
    if (!requests_.empty() || duplex_active_)
        throw std::runtime_error("reload requires a quiescent Metal engine");
    info_.max_context = model_->reload(context);
    return info_.max_context;
}
void MlxEngine::shutdown() {
    auto failure = requests_.clear();
    if (model_ && duplex_active_) {
        try { (void)model_->control(StopDuplex{}); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    duplex_active_ = false;
    // Constraint cursors and native jobs are gone before their resource owners.
    text_.reset();
    model_.reset();
    if (failure) std::rethrow_exception(failure);
}

} // namespace mfq::metal
