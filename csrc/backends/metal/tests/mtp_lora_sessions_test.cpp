#include "mtp_lora_sessions.h"
#include "mfq_paged_prefix_cache.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
double metric(const mfq::metal::MtpLoraSessions& sessions, const char* name) {
    for (const auto& [key, value] : sessions.metrics()) if (key == name) return value;
    throw std::runtime_error("missing LoRA metric");
}
void wait(const mfq::metal::MtpLoraSessions& sessions) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (metric(sessions, "mtp_ttt_completed") < metric(sessions, "mtp_ttt_submitted")) {
        if (std::chrono::steady_clock::now() >= limit) throw std::runtime_error("LoRA worker timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
template <typename Predicate> void until(Predicate predicate) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= limit) throw std::runtime_error("LoRA persistence timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}

int main() {
    using namespace mfq::metal;
    char temporary[] = "/tmp/mfq-mtp-lora-test-XXXXXX";
    if (!mkdtemp(temporary)) return 1;
    const std::filesystem::path root(temporary);
    try {
        auto original = MtpLoraState::create(64, 65);
        original.version = original.optimizer_step = 7;
        original.b[0] = 0.1f;
        original.first_moment[5] = 0.03f;
        auto encoded = encode_mtp_lora(original, "model/session");
        auto restored = decode_mtp_lora(encoded, "model/session", 64, 65, 16);
        require(restored.a == original.a && restored.b == original.b &&
            restored.first_moment == original.first_moment && restored.version == 7, "snapshot changed optimizer or weights");
        for (int mode = 0; mode < 4; ++mode) {
            auto broken = encoded;
            if (mode == 0) broken[100] ^= 1;
            if (mode == 1) broken.pop_back();
            bool rejected = false;
            try { (void)decode_mtp_lora(broken, mode == 2 ? "other-model" : "model/session", 64, mode == 3 ? 64 : 65, 8); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "corrupt or incompatible snapshot was accepted");
        }
        for (int mode = 0; mode < 3; ++mode) {
            auto invalid = original;
            if (mode == 0) invalid.a[0] = std::numeric_limits<float>::quiet_NaN();
            if (mode == 1) invalid.second_moment[0] = -1;
            if (mode == 2) ++invalid.optimizer_step;
            bool rejected = false;
            try { (void)encode_mtp_lora(invalid, "model/session"); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "invalid optimizer state was persisted");
        }
        std::mutex mutex;
        std::condition_variable condition;
        bool blocked = true, started = false;
        auto trainer = [&](const MtpLoraState& state, const MtpLoraBatch&) {
            std::unique_lock lock(mutex);
            started = true;
            condition.notify_all();
            condition.wait(lock, [&] { return !blocked; });
            auto next = std::make_shared<MtpLoraState>(state);
            ++next->version;
            ++next->optimizer_step;
            next->b[0] += 0.001f;
            return MtpLoraUpdate{next, 1, 0.9, 0.01};
        };
        {
            MtpLoraSessions sessions(64, 65, 16, "model-v1", root, 1 << 20, 1 << 20, trainer);
            auto first = sessions.begin("a");
            require(first->version == 0, "new session not zero initialized");
            require(sessions.submit({}, first->version), "first batch rejected");
            {
                std::unique_lock lock(mutex);
                require(condition.wait_for(lock, std::chrono::seconds(5), [&] { return started; }), "worker did not start");
            }
            require(sessions.boundary()->version == 0, "boundary waited or switched an in-flight adapter");
            require(!sessions.submit({}, 0), "unbounded training queue");
            require(sessions.close("a"), "close did not report an adapter-only asset");
            auto replacement = sessions.begin("a");
            require(replacement->version == 0, "closed session was resurrected");
            {
                std::lock_guard lock(mutex);
                blocked = false;
            }
            condition.notify_all();
            wait(sessions);
            require(sessions.boundary()->version == 0, "stale result attached to recreated session");
            require(sessions.submit({}, 0), "replacement training rejected");
            wait(sessions);
            require(replacement->version == 0 && sessions.boundary()->version == 1,
                "published update mutated active snapshot or did not switch at boundary");
            require(sessions.fork("a", "b"), "fork did not report an adapter-only asset");
            auto forked = sessions.begin("b");
            require(forked->version == 1 && forked->b[0] == 0.001f, "fork lost adapter");
            require(sessions.submit({}, 1), "fork training rejected");
            wait(sessions);
            require(sessions.boundary()->version == 2, "fork update missing");
            require(sessions.begin("a")->version == 1, "fork changed source adapter");
            sessions.end();
        }
        {
            MtpLoraSessions sessions(64, 65, 16, "model-v1", root, 1 << 20, 1 << 20, trainer);
            require(sessions.begin("a")->version == 1, "process restart lost session A");
            require(sessions.begin("b")->version == 2, "process restart lost session B");
            sessions.close("a");
            require(sessions.begin("a")->version == 0, "close did not delete SSD asset");
            sessions.clear();
        }
        {
            MtpLoraSessions sessions(64, 65, 16, "budget-model", root / "budget",
                original.bytes() * 2, 1 << 20, trainer);
            require(!sessions.fork("absent", "new") && metric(sessions, "mtp_ttt_sessions") == 0,
                "fork allocated an unknown source session");
            for (int round = 1; round <= 16; ++round) {
                sessions.begin("short");
                require(sessions.wants_batch() == (round == 16), "short requests reset the training interval");
                sessions.observe(1, 1, 2);
                sessions.end();
            }
            sessions.begin("second");
            sessions.end();
            sessions.begin("third");
            sessions.end();
            require(sessions.bytes() == original.bytes() * 2, "hot budget did not evict the least-used session");
            require(!sessions.close("short") && sessions.close("second"), "hot LRU evicted the wrong session");
            require(sessions.trim(0) == original.bytes() && sessions.bytes() == 0, "manual pressure trim retained a cold asset");
            sessions.clear();
        }
        {
            blocked = true;
            started = false;
            MtpLoraSessions sessions(64, 65, 16, "throttle-model", root / "throttle", 1 << 20, 1 << 20, trainer);
            sessions.begin("busy");
            for (int i = 0; i < 8; ++i) {
                sessions.wants_batch();
                sessions.observe(1, 1, 2);
            }
            require(sessions.submit({}, 0), "throttle fixture could not start training");
            {
                std::unique_lock lock(mutex);
                require(condition.wait_for(lock, std::chrono::seconds(5), [&] { return started; }), "throttle worker did not start");
            }
            for (int i = 0; i < 8; ++i) {
                sessions.wants_batch();
                sessions.observe(2, 1, 2);
            }
            const bool throttled = metric(sessions, "mtp_ttt_throttles") == 1 &&
                metric(sessions, "mtp_ttt_stride") == 32 && metric(sessions, "mtp_ttt_cooldown_rounds") == 128;
            {
                std::lock_guard lock(mutex);
                blocked = false;
            }
            condition.notify_all();
            wait(sessions);
            require(throttled, "inference contention did not throttle background training");
            sessions.clear();
        }
        {
            const auto directory = root / "disk";
            MtpLoraSessions sessions(64, 65, 16, "disk-model", directory,
                1 << 20, (original.bytes() + 112) * 2, trainer);
            const auto file = [&](const std::string& id) {
                return directory / (mfq::cache::block_hash_hex(mfq::cache::sha256("disk-model\n" + id)) + ".lora");
            };
            const auto train_session = [&](const std::string& id) {
                sessions.begin(id);
                require(sessions.submit({}, 0), "disk LRU training rejected");
                wait(sessions);
                require(sessions.boundary()->version == 1, "disk LRU update missing");
                sessions.end();
                until([&] { return std::filesystem::exists(file(id)); });
            };
            train_session("a");
            train_session("b");
            sessions.begin("a");
            sessions.end();
            train_session("c");
            until([&] { return !std::filesystem::exists(file("b")); });
            require(std::filesystem::exists(file("a")) && std::filesystem::exists(file("c")),
                "SSD budget evicted a recently used session");
            until([&] { sessions.trim(0); return sessions.bytes() == 0; });
            require(sessions.begin("b")->version == 1, "RAM eviction lost an adapter whose SSD copy had been evicted");
            sessions.clear();
        }
        std::filesystem::remove_all(root);
        std::cout << "MTP LoRA isolation, stale update, fork, restart and corruption tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
