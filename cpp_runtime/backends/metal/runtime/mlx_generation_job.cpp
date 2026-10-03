#include "mlx_generation_job.h"

namespace mfq::metal {
namespace { struct CancelledWork {}; }

MlxGenerationJob::MlxGenerationJob(Work work)
    : worker_([this, work = std::move(work)] {
        std::exception_ptr failure;
        try { work(*this); }
        catch (const CancelledWork&) {}
        catch (...) { failure = std::current_exception(); }
        std::lock_guard lock(mutex_);
        failure_ = failure;
        done_ = true;
    }) {}

MlxGenerationJob::~MlxGenerationJob() {
    cancel();
    join();
}

bool MlxGenerationJob::push(Event event) {
    std::unique_lock lock(mutex_);
    space_.wait(lock, [this] { return cancelled_ || events_.size() < capacity_; });
    if (cancelled_) return false;
    events_.push_back(std::move(event));
    return true;
}

bool MlxGenerationJob::token(std::int64_t value) { return push(value); }
void MlxGenerationJob::prefill(MfqPrefillTiming timing) {
    if (!push(timing)) throw CancelledWork{};
}
bool MlxGenerationJob::cancelled() const {
    std::lock_guard lock(mutex_);
    return cancelled_;
}
void MlxGenerationJob::cancel() {
    std::lock_guard lock(mutex_);
    cancelled_ = true;
    events_.clear();
    space_.notify_all();
}
std::optional<MlxGenerationJob::Event> MlxGenerationJob::take() {
    std::lock_guard lock(mutex_);
    if (events_.empty()) {
        if (done_ && failure_) std::rethrow_exception(failure_);
        return std::nullopt;
    }
    auto event = std::move(events_.front());
    events_.pop_front();
    space_.notify_one();
    return event;
}
bool MlxGenerationJob::done() const {
    std::lock_guard lock(mutex_);
    return done_ && events_.empty();
}
void MlxGenerationJob::join() { if (worker_.joinable()) worker_.join(); }

mfq::engine::Generation consume_mlx_generation(
    MlxGenerationJob::Work work, mfq::engine::InferenceOutput& output) {
    if (output.stopped()) co_return;
    MlxGenerationJob job(std::move(work));
    mfq::engine::ExecutionCleanup cleanup{output.cleanup_failure, [&] {
        job.cancel();
        job.join();
        (void)job.take();
    }};
    while (!output.stopped()) {
        if (auto event = job.take()) {
            if (auto* token = std::get_if<std::int64_t>(&*event))
                co_yield output.append({*token});
            else co_yield mfq::engine::PrefillProgress{std::get<MfqPrefillTiming>(*event)};
        } else if (job.done()) break;
        else co_yield mfq::StepState::waiting;
    }
    // Cancellation releases a producer paused by backpressure before joining.
    // Normal completion also joins before reading statistics or model state.
    cleanup.finish(); // Also preserves failures during coroutine destruction.
    output.metrics.mtp = job.mtp;
}

} // namespace mfq::metal
