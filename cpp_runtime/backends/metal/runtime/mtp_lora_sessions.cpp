#include "mtp_lora_sessions.h"
#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>

namespace mfq::metal {
namespace {

void integer(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
}

std::uint64_t integer(std::span<const std::uint8_t> bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= std::uint64_t(bytes[offset + i]) << (i * 8);
    return value;
}

} // namespace

std::vector<std::uint8_t> encode_mtp_lora(const MtpLoraState& state, const std::string& fingerprint) {
    state.validate();
    std::vector<std::uint8_t> bytes{'M','F','Q','T','T','T','1',0};
    bytes.reserve(state.bytes() + 112);
    integer(bytes, state.inputs);
    integer(bytes, state.outputs);
    integer(bytes, state.rank);
    integer(bytes, state.version);
    integer(bytes, state.optimizer_step);
    const auto key = mfq::cache::sha256(fingerprint);
    bytes.insert(bytes.end(), key.begin(), key.end());
    for (const auto* values : {&state.a, &state.b, &state.first_moment, &state.second_moment}) {
        if constexpr (std::endian::native == std::endian::little) {
            const auto* data = reinterpret_cast<const std::uint8_t*>(values->data());
            bytes.insert(bytes.end(), data, data + values->size() * sizeof(float));
        } else for (auto value : *values) {
            std::uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<std::uint8_t>(bits >> (i * 8)));
        }
    }
    const auto hash = mfq::cache::sha256(bytes.data(), bytes.size());
    bytes.insert(bytes.end(), hash.begin(), hash.end());
    return bytes;
}

MtpLoraState decode_mtp_lora(std::span<const std::uint8_t> bytes, const std::string& fingerprint,
    int inputs, int outputs, int rank) {
    const auto expected = std::uint64_t(inputs + outputs) * rank * sizeof(float) * 3 + 112;
    if (bytes.size() != expected || std::memcmp(bytes.data(), "MFQTTT1\0", 8) != 0 ||
        integer(bytes, 8) != std::uint64_t(inputs) || integer(bytes, 16) != std::uint64_t(outputs) ||
        integer(bytes, 24) != std::uint64_t(rank)) throw std::runtime_error("incompatible MTP LoRA snapshot");
    const auto key = mfq::cache::sha256(fingerprint);
    const auto hash = mfq::cache::sha256(bytes.data(), bytes.size() - 32);
    if (!std::equal(key.begin(), key.end(), bytes.begin() + 48) ||
        !std::equal(hash.begin(), hash.end(), bytes.end() - 32))
        throw std::runtime_error("MTP LoRA snapshot fingerprint or checksum mismatch");
    auto state = MtpLoraState::create(inputs, outputs, rank, false);
    state.version = integer(bytes, 32);
    state.optimizer_step = integer(bytes, 40);
    std::size_t cursor = 80;
    for (auto* values : {&state.a, &state.b, &state.first_moment, &state.second_moment}) {
        if constexpr (std::endian::native == std::endian::little) {
            std::memcpy(values->data(), bytes.data() + cursor, values->size() * sizeof(float));
            cursor += values->size() * sizeof(float);
        } else for (auto& value : *values) {
            std::uint32_t bits = 0;
            for (int i = 0; i < 4; ++i) bits |= std::uint32_t(bytes[cursor++]) << (i * 8);
            std::memcpy(&value, &bits, sizeof(bits));
        }
    }
    state.validate();
    return state;
}

struct MtpLoraSessions::Impl {
    struct Asset {
        std::shared_ptr<const MtpLoraState> active, ready;
        std::uint64_t epoch = 0, touched = 0;
        bool dirty = false;
        std::uint64_t rounds = 0;
    };
    struct Job {
        std::string session;
        std::uint64_t epoch;
        std::shared_ptr<const MtpLoraState> state;
        MtpLoraBatch batch;
    };
    int inputs, outputs, rank;
    std::string fingerprint, current;
    std::filesystem::path directory;
    std::size_t hot_limit, disk_limit, trainer_bytes, work_bytes = 0;
    Train train;
    std::function<void(MtpLoraState&)> initialize;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::unordered_map<std::string, Asset> assets;
    std::optional<Job> pending;
    bool stopping = false, busy = false, failed = false;
    bool round_contended = false;
    std::array<double, 6> idle_ms{}, contended_ms{};
    std::array<int, 6> idle_samples{}, contended_samples{};
    std::uint64_t stride = 16, cooldown = 0, throttles = 0;
    std::uint64_t clock = 0, submitted = 0, completed = 0, dropped = 0, updates = 0, applied = 0, rejected = 0, corrupt = 0;
    double loss_before = 0, loss_after = 0, trust_kl = 0;
    std::thread worker;

    std::filesystem::path path(const std::string& session) const {
        return directory / (mfq::cache::block_hash_hex(mfq::cache::sha256(fingerprint + "\n" + session)) + ".lora");
    }
    std::string key(const std::string& session) const { return fingerprint + "\n" + session; }
    std::size_t resident() const {
        std::size_t size = trainer_bytes + work_bytes;
        for (const auto& [id, asset] : assets) {
            if (asset.active) size += asset.active->bytes();
            if (asset.ready) size += asset.ready->bytes();
        }
        return size;
    }
    Asset& get(const std::string& session) {
        if (auto found = assets.find(session); found != assets.end()) return found->second;
        const auto asset_bytes = std::size_t(inputs + outputs) * rank * sizeof(float) * 3;
        trim(hot_limit > asset_bytes ? hot_limit - asset_bytes : 0);
        if (resident() > hot_limit - asset_bytes) throw std::length_error("MTP LoRA hot budget is busy");
        auto state = std::make_shared<MtpLoraState>(MtpLoraState::create(inputs, outputs, rank));
        if (initialize) initialize(*state);
        auto file = path(session);
        std::error_code error;
        const auto expected = state->bytes() + 112;
        if (std::filesystem::is_regular_file(std::filesystem::symlink_status(file, error)) &&
            std::filesystem::file_size(file, error) == expected && !error) {
            try {
                std::vector<std::uint8_t> bytes(expected);
                std::ifstream input(file, std::ios::binary);
                if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
                    throw std::runtime_error("truncated MTP LoRA file");
                *state = decode_mtp_lora(bytes, key(session), inputs, outputs, rank);
                std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now(), error);
            } catch (...) { ++corrupt; }
        } else if (!error && std::filesystem::exists(file)) ++corrupt;
        return assets.emplace(session, Asset{state, {}, ++clock, clock, false}).first->second;
    }
    void save(const std::string& session, const Asset& asset, std::unique_lock<std::mutex>* lock = nullptr) {
        if (!asset.dirty || !asset.active) return;
        if (lock) lock->unlock();
        auto bytes = encode_mtp_lora(*asset.active, key(session));
        auto destination = path(session);
        auto temporary = destination;
        temporary += ".tmp.XXXXXX";
        auto temporary_name = temporary.string();
        int fd = mkstemp(temporary_name.data());
        temporary = temporary_name;
        if (fd < 0) throw std::runtime_error("cannot create MTP LoRA snapshot");
        fcntl(fd, F_SETFD, FD_CLOEXEC);
        std::size_t written = 0;
        while (written < bytes.size()) {
            const auto count = write(fd, bytes.data() + written, bytes.size() - written);
            if (count <= 0) { ::close(fd); std::filesystem::remove(temporary); throw std::runtime_error("cannot write MTP LoRA snapshot"); }
            written += count;
        }
        const auto synced = fsync(fd);
        const auto closed = ::close(fd);
        if (synced != 0 || closed != 0) { std::filesystem::remove(temporary); throw std::runtime_error("cannot flush MTP LoRA snapshot"); }
        if (lock) {
            lock->lock();
            const auto found = assets.find(session);
            if (found == assets.end() || found->second.epoch != asset.epoch ||
                found->second.active->version != asset.active->version) {
                std::filesystem::remove(temporary);
                return;
            }
            std::filesystem::rename(temporary, destination);
            found->second.dirty = false;
            lock->unlock();
        } else std::filesystem::rename(temporary, destination);
        disk_lru(destination);
        if (lock) lock->lock();
    }
    void disk_lru(const std::filesystem::path& protected_file) {
        std::vector<std::filesystem::directory_entry> files;
        std::uint64_t total = 0;
        for (const auto& file : std::filesystem::directory_iterator(directory)) {
            if (!std::filesystem::is_regular_file(file.symlink_status()) || file.path().extension() != ".lora") continue;
            total += file.file_size();
            files.push_back(file);
        }
        std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.last_write_time() < b.last_write_time(); });
        for (const auto& file : files) {
            if (total <= disk_limit) break;
            if (file.path() == protected_file) continue;
            const auto bytes = file.file_size();
            if (std::filesystem::remove(file.path())) total -= bytes;
        }
    }
    std::size_t trim(std::size_t limit) {
        const auto before = resident();
        while (resident() > limit) {
            auto victim = assets.end();
            for (auto it = assets.begin(); it != assets.end(); ++it) {
                if (it->first == current || it->second.dirty ||
                    (pending && it->first == pending->session) || it->second.active.use_count() > 1) continue;
                if (victim == assets.end() || it->second.touched < victim->second.touched) victim = it;
            }
            if (victim == assets.end()) break;
            if (victim->second.active->version > 0 && !std::filesystem::exists(path(victim->first))) {
                victim->second.dirty = true;
                condition.notify_one();
                continue;
            }
            assets.erase(victim);
        }
        return before - resident();
    }
    void run() {
        std::unique_lock lock(mutex);
        for (;;) {
            const auto dirty = [&] { return std::any_of(assets.begin(), assets.end(),
                [](const auto& item) { return item.second.dirty; }); };
            condition.wait(lock, [&] { return stopping || pending.has_value() || dirty(); });
            if (stopping) break;
            if (!pending) {
                auto found = std::find_if(assets.begin(), assets.end(), [](const auto& item) { return item.second.dirty; });
                const auto session = found->first;
                auto asset = found->second;
                work_bytes = asset.active->bytes() + 112;
                try { save(session, asset, &lock); }
                catch (const std::exception& error) {
                    if (!lock.owns_lock()) lock.lock();
                    if (auto item = assets.find(session); item != assets.end() && item->second.epoch == asset.epoch)
                        item->second.dirty = false;
                    std::cerr << "mtp_ttt action=save_failed error=" << error.what() << '\n';
                }
                asset.active.reset();
                asset.ready.reset();
                work_bytes = 0;
                trim(hot_limit);
                continue;
            }
            auto job = std::move(*pending);
            pending.reset();
            busy = true;
            lock.unlock();
            std::optional<MtpLoraUpdate> result;
            try { result = train(*job.state, job.batch); }
            catch (const std::exception& error) { std::cerr << "mtp_ttt action=training_failed error=" << error.what() << '\n'; }
            lock.lock();
            busy = false;
            ++completed;
            work_bytes = 0;
            auto found = assets.find(job.session);
            if (result) {
                loss_before = result->loss_before;
                loss_after = result->loss_after;
                trust_kl = result->trust_kl;
            }
            if (!result) { failed = true; ++rejected; }
            else if (found != assets.end() && found->second.epoch == job.epoch &&
                found->second.active->version == job.state->version) {
                if (result->state && result->state->version == job.state->version + 1 &&
                    result->state->inputs == inputs && result->state->outputs == outputs && result->state->rank == rank) {
                    found->second.ready = result->state; ++updates;
                }
                else ++rejected;
            } else ++dropped;
            condition.notify_all();
        }
    }
};

MtpLoraSessions::MtpLoraSessions(int inputs, int outputs, int rank, std::string fingerprint,
    std::filesystem::path directory, std::size_t hot_bytes, std::size_t disk_bytes,
    Train train, std::size_t trainer_bytes, std::function<void(MtpLoraState&)> initialize) : impl_(std::make_unique<Impl>()) {
    auto initial = MtpLoraState::create(inputs, outputs, rank);
    if (!train || fingerprint.empty() || directory.empty() || hot_bytes < initial.bytes() + trainer_bytes ||
        disk_bytes < initial.bytes() + 112) throw std::invalid_argument("invalid MTP LoRA asset budgets");
    impl_->inputs = inputs;
    impl_->outputs = outputs;
    impl_->rank = rank;
    impl_->fingerprint = std::move(fingerprint);
    impl_->directory = std::move(directory);
    impl_->hot_limit = hot_bytes;
    impl_->disk_limit = disk_bytes;
    impl_->trainer_bytes = trainer_bytes;
    impl_->initialize = std::move(initialize);
    impl_->train = std::move(train);
    std::filesystem::create_directories(impl_->directory);
    std::filesystem::permissions(impl_->directory, std::filesystem::perms::owner_all);
    impl_->worker = std::thread([this] { impl_->run(); });
}

void MtpLoraSessions::set_trainer_bytes(std::size_t bytes) {
    std::lock_guard lock(impl_->mutex);
    if (bytes > impl_->hot_limit) throw std::length_error("MTP trainer exceeds the configured RAM budget");
    impl_->trainer_bytes = bytes;
    impl_->trim(impl_->hot_limit);
}

MtpLoraSessions::~MtpLoraSessions() {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->stopping = true;
        impl_->pending.reset();
    }
    impl_->condition.notify_all();
    impl_->worker.join();
    for (const auto& [session, asset] : impl_->assets) try { impl_->save(session, asset); } catch (...) {}
}

std::shared_ptr<const MtpLoraState> MtpLoraSessions::begin(const std::string& session) {
    std::lock_guard lock(impl_->mutex);
    impl_->current = session;
    if (session.empty()) return {};
    Impl::Asset* selected = nullptr;
    try { selected = &impl_->get(session); }
    catch (const std::length_error&) { impl_->current.clear(); ++impl_->dropped; return {}; }
    auto& asset = *selected;
    asset.touched = ++impl_->clock;
    std::error_code error;
    std::filesystem::last_write_time(impl_->path(session), std::filesystem::file_time_type::clock::now(), error);
    impl_->trim(impl_->hot_limit);
    return asset.active;
}

std::shared_ptr<const MtpLoraState> MtpLoraSessions::boundary() {
    std::lock_guard lock(impl_->mutex);
    if (impl_->current.empty()) return {};
    auto& asset = impl_->assets.at(impl_->current);
    if (asset.ready) {
        asset.active = std::move(asset.ready);
        asset.dirty = true;
        ++impl_->applied;
        impl_->condition.notify_one();
    }
    asset.touched = ++impl_->clock;
    return asset.active;
}

bool MtpLoraSessions::wants_batch() {
    std::lock_guard lock(impl_->mutex);
    impl_->round_contended = impl_->busy || impl_->work_bytes > 0;
    if (impl_->cooldown) { --impl_->cooldown; return false; }
    return !impl_->current.empty() && !impl_->failed &&
        ++impl_->assets.at(impl_->current).rounds % impl_->stride == 0 &&
        !impl_->pending && !impl_->busy && !impl_->assets.at(impl_->current).ready;
}

void MtpLoraSessions::observe(double ms, int tokens, int depth) {
    if (!std::isfinite(ms) || ms <= 0 || tokens <= 0 || depth <= 0 || depth > 5) return;
    std::lock_guard lock(impl_->mutex);
    const auto value = ms / tokens;
    auto& average = impl_->round_contended ? impl_->contended_ms[depth] : impl_->idle_ms[depth];
    auto& samples = impl_->round_contended ? impl_->contended_samples[depth] : impl_->idle_samples[depth];
    average = samples++ == 0 ? value : 0.875 * average + 0.125 * value;
    if (impl_->round_contended && samples >= 8 && impl_->idle_samples[depth] >= 8 &&
        average > impl_->idle_ms[depth] * 1.08) {
        impl_->cooldown = 128;
        impl_->stride = std::min<std::uint64_t>(128, impl_->stride * 2);
        impl_->contended_samples.fill(0);
        ++impl_->throttles;
    }
}

bool MtpLoraSessions::submit(MtpLoraBatch batch, std::uint64_t version) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->current.empty() || impl_->failed || impl_->pending || impl_->busy) { ++impl_->dropped; return false; }
    auto& asset = impl_->assets.at(impl_->current);
    if (asset.active->version != version || asset.ready) { ++impl_->dropped; return false; }
    impl_->work_bytes = batch.workspace_bytes + 2 * asset.active->bytes() +
        (batch.teacher_logits.size() + batch.position_weights.size()) * sizeof(float);
    for (const auto& example : batch.examples) if (example) impl_->work_bytes += example->bytes();
    impl_->trim(impl_->hot_limit);
    if (impl_->resident() > impl_->hot_limit) {
        impl_->work_bytes = 0;
        ++impl_->dropped;
        return false;
    }
    impl_->pending = Impl::Job{impl_->current, asset.epoch, asset.active, std::move(batch)};
    ++impl_->submitted;
    impl_->condition.notify_one();
    return true;
}

void MtpLoraSessions::end() {
    std::lock_guard lock(impl_->mutex);
    impl_->current.clear();
    impl_->trim(impl_->hot_limit);
}

bool MtpLoraSessions::close(const std::string& session) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->current == session) impl_->current.clear();
    if (impl_->pending && impl_->pending->session == session) {
        impl_->pending.reset();
        ++impl_->completed;
        ++impl_->dropped;
        if (!impl_->busy) impl_->work_bytes = 0;
    }
    const auto removed = impl_->assets.erase(session);
    return std::filesystem::remove(impl_->path(session)) || removed > 0;
}

bool MtpLoraSessions::fork(const std::string& source, const std::string& target) {
    if (source.empty() || target.empty() || source == target) throw std::invalid_argument("invalid MTP LoRA fork");
    std::lock_guard lock(impl_->mutex);
    if (!impl_->assets.contains(source) && !std::filesystem::is_regular_file(impl_->path(source))) return false;
    std::shared_ptr<const MtpLoraState> original;
    try { original = impl_->get(source).active; }
    catch (const std::length_error&) { ++impl_->dropped; return false; }
    impl_->trim(impl_->hot_limit - original->bytes());
    if (impl_->resident() > impl_->hot_limit - original->bytes()) { ++impl_->dropped; return false; }
    auto state = std::make_shared<MtpLoraState>(*original);
    impl_->assets[target] = Impl::Asset{state, {}, ++impl_->clock, impl_->clock, true};
    impl_->condition.notify_one();
    impl_->trim(impl_->hot_limit);
    return true;
}

void MtpLoraSessions::clear() {
    std::lock_guard lock(impl_->mutex);
    impl_->assets.clear();
    if (impl_->pending) { ++impl_->completed; ++impl_->dropped; }
    impl_->pending.reset();
    impl_->current.clear();
    if (!impl_->busy) impl_->work_bytes = 0;
    for (const auto& file : std::filesystem::directory_iterator(impl_->directory))
        if (file.path().extension() == ".lora" && std::filesystem::is_regular_file(file.symlink_status()))
            std::filesystem::remove(file.path());
}

std::size_t MtpLoraSessions::trim(std::size_t bytes) {
    std::lock_guard lock(impl_->mutex);
    return impl_->trim(bytes);
}

std::size_t MtpLoraSessions::bytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->resident();
}

std::vector<std::pair<std::string, double>> MtpLoraSessions::metrics() const {
    std::lock_guard lock(impl_->mutex);
    return {{"mtp_ttt_enabled", 1}, {"mtp_ttt_failed", impl_->failed},
        {"mtp_ttt_resident_bytes", double(impl_->resident())}, {"mtp_ttt_sessions", double(impl_->assets.size())},
        {"mtp_ttt_submitted", double(impl_->submitted)}, {"mtp_ttt_updates", double(impl_->updates)},
        {"mtp_ttt_applied", double(impl_->applied)},
        {"mtp_ttt_completed", double(impl_->completed)},
        {"mtp_ttt_stride", double(impl_->stride)}, {"mtp_ttt_throttles", double(impl_->throttles)},
        {"mtp_ttt_cooldown_rounds", double(impl_->cooldown)},
        {"mtp_ttt_rejected", double(impl_->rejected)}, {"mtp_ttt_dropped", double(impl_->dropped)},
        {"mtp_ttt_loss_before", impl_->loss_before}, {"mtp_ttt_loss_after", impl_->loss_after},
        {"mtp_ttt_trust_kl", impl_->trust_kl},
        {"mtp_ttt_corrupt_snapshots", double(impl_->corrupt)}, {"mtp_ttt_busy", impl_->busy}};
}

} // namespace mfq::metal
