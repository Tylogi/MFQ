#include "mlx_engine.h"

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
Admission MlxEngine::admit(EngineRequest&& request) {
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
    if (command.kind != SessionCommand::Kind::metrics && !requests_.empty())
        throw std::runtime_error("session operation requires a quiescent Metal engine");
    return model_->session(command);
}
ControlResult MlxEngine::control(ControlRequest request) {
    if (!model_) throw std::runtime_error("Metal engine is unloaded");
    return std::visit([&](auto value) -> ControlResult {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, DecodeTokens>)
            return text_->decode_tokens(value.tokens, value.excluded);
        else if constexpr (std::is_same_v<T, PrepareDuplex>) {
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
