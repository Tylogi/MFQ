#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::cuda {

class MfeMxfp4Unsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct MfqRecordRange {
    std::string name;
    std::string dtype;
    std::string source_path;
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
    std::function<void(std::uint64_t, std::span<std::uint8_t>)> read_range;
};

struct MfeMxfp4ExpertPart {
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
};

// Exact-range view over one canonical MFE routed projection. The store
// retains only metadata; expert payloads remain in the MFQ container until a
// cache miss asks for their MXFP4 values and scales.
class MfeMxfp4ExpertStore {
public:
    enum Field : std::size_t {
        values = 0,
        scales = 1,
        field_count = 2,
    };

    explicit MfeMxfp4ExpertStore(MfqRecordRange record);

    int num_experts() const noexcept;
    int out_per_expert() const noexcept;
    int neuron_len() const noexcept;
    std::uint64_t values_bytes_per_expert() const noexcept;
    std::uint64_t scales_bytes_per_expert() const noexcept;
    std::uint64_t payload_bytes() const noexcept;
    const MfqRecordRange& record() const noexcept;

    const MfeMxfp4ExpertPart& part(
        int expert,
        std::size_t field) const;
    void read_part_into(
        const MfeMxfp4ExpertPart& part,
        std::span<std::uint8_t> destination) const;
    std::vector<std::uint8_t> read_blob() const;

private:
    std::vector<std::uint8_t> read_range(
        std::uint64_t offset,
        std::uint64_t nbytes) const;
    void read_range_into(
        std::uint64_t offset,
        std::span<std::uint8_t> destination) const;

    MfqRecordRange record_;
    int num_experts_ = 0;
    int out_per_expert_ = 0;
    int neuron_len_ = 0;
    std::uint64_t values_bytes_per_expert_ = 0;
    std::uint64_t scales_bytes_per_expert_ = 0;
    std::uint64_t payload_bytes_ = 0;
    std::vector<std::array<MfeMxfp4ExpertPart, field_count>> experts_;
};

struct MfeMxfp4ReadRequest {
    const MfeMxfp4ExpertStore* store = nullptr;
    const MfeMxfp4ExpertPart* part = nullptr;
    std::span<std::uint8_t> destination;
};

struct MfeMxfp4ReadBatchStats {
    std::uint64_t bytes = 0;
    std::uint64_t calls = 0;
    std::uint64_t file_opens = 0;
    std::uint64_t wall_nanoseconds = 0;
};

struct MfeMxfp4ReadState;

class MfeMxfp4ReadTicket {
public:
    MfeMxfp4ReadTicket() = default;
    MfeMxfp4ReadTicket(MfeMxfp4ReadTicket&&) noexcept;
    MfeMxfp4ReadTicket& operator=(MfeMxfp4ReadTicket&&) noexcept;
    ~MfeMxfp4ReadTicket();

    MfeMxfp4ReadTicket(const MfeMxfp4ReadTicket&) = delete;
    MfeMxfp4ReadTicket& operator=(const MfeMxfp4ReadTicket&) = delete;

    bool valid() const noexcept;
    MfeMxfp4ReadBatchStats wait();

private:
    explicit MfeMxfp4ReadTicket(
        std::shared_ptr<MfeMxfp4ReadState> state);

    std::shared_ptr<MfeMxfp4ReadState> state_;

    friend class MfeMxfp4ReadPool;
};

class MfeMxfp4ReadPool {
public:
    explicit MfeMxfp4ReadPool(std::size_t workers);
    ~MfeMxfp4ReadPool();

    MfeMxfp4ReadPool(const MfeMxfp4ReadPool&) = delete;
    MfeMxfp4ReadPool& operator=(const MfeMxfp4ReadPool&) = delete;

    std::size_t workers() const noexcept;
    MfeMxfp4ReadTicket submit(
        std::span<const MfeMxfp4ReadRequest> requests);
    MfeMxfp4ReadBatchStats read(
        std::span<const MfeMxfp4ReadRequest> requests);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::cuda
