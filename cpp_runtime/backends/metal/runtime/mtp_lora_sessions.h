#pragma once

#include "mtp_lora.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mfq::metal {

std::vector<std::uint8_t> encode_mtp_lora(const MtpLoraState&, const std::string& fingerprint);
MtpLoraState decode_mtp_lora(std::span<const std::uint8_t>, const std::string& fingerprint,
    int inputs, int outputs, int rank);

class MtpLoraSessions {
public:
    using Train = std::function<MtpLoraUpdate(const MtpLoraState&, const MtpLoraBatch&)>;
    MtpLoraSessions(int inputs, int outputs, int rank, std::string fingerprint,
        std::filesystem::path directory, std::size_t hot_bytes, std::size_t disk_bytes,
        Train train, std::size_t trainer_bytes = 0,
        std::function<void(MtpLoraState&)> initialize = {});
    void set_trainer_bytes(std::size_t bytes);
    ~MtpLoraSessions();
    MtpLoraSessions(const MtpLoraSessions&) = delete;
    MtpLoraSessions& operator=(const MtpLoraSessions&) = delete;

    std::shared_ptr<const MtpLoraState> begin(const std::string& session);
    std::shared_ptr<const MtpLoraState> boundary();
    bool wants_batch();
    void observe(double cycle_ms, int tokens, int depth);
    bool submit(MtpLoraBatch, std::uint64_t version);
    void end();
    bool close(const std::string& session);
    bool fork(const std::string& source, const std::string& target);
    void clear();
    std::size_t trim(std::size_t target_bytes);
    std::size_t bytes() const;
    std::vector<std::pair<std::string, double>> metrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
