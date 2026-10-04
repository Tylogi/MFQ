#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mfq::engine {

struct ResourceExhausted : std::runtime_error { using std::runtime_error::runtime_error; };

// A lease follows the physical request, including preparation and cleanup.
// The Engine thread owns all mutations; leases may outlive the budget owner.
class AdmissionBudget {
    struct State { std::vector<std::size_t> capacity, used; };
    std::shared_ptr<State> state_;
  public:
    class Lease {
        friend class AdmissionBudget;
        std::shared_ptr<State> state_;
        std::vector<std::size_t> bytes_;
        Lease(std::shared_ptr<State> state, std::vector<std::size_t> bytes)
            : state_(std::move(state)), bytes_(std::move(bytes)) {}
      public:
        Lease(Lease&&) noexcept = default;
        Lease& operator=(Lease&&) = delete;
        Lease(const Lease&) = delete;
        ~Lease() {
            if (state_) for (std::size_t i = 0; i < bytes_.size(); ++i) state_->used[i] -= bytes_[i];
        }
    };
    explicit AdmissionBudget(std::vector<std::size_t> capacity = {})
        : state_(std::make_shared<State>(State{capacity, std::vector<std::size_t>(capacity.size())})) {}
    const auto& capacity() const { return state_->capacity; }
    const auto& used() const { return state_->used; }
    std::size_t available(const std::vector<std::size_t>& cost) const {
        if (cost.size() != state_->capacity.size()) throw std::invalid_argument("memory budget topology changed");
        auto result = std::size_t(-1);
        for (std::size_t i = 0; i < cost.size(); ++i) if (cost[i])
            result = std::min(result, (state_->capacity[i] - state_->used[i]) / cost[i]);
        return result;
    }
    std::optional<Lease> reserve(std::vector<std::size_t> cost) {
        if (cost.size() != state_->capacity.size()) throw std::invalid_argument("memory budget topology changed");
        for (std::size_t i = 0; i < cost.size(); ++i)
            if (cost[i] > state_->capacity[i]) throw ResourceExhausted("request exceeds the Engine memory budget");
        if (!available(cost)) return std::nullopt;
        for (std::size_t i = 0; i < cost.size(); ++i) state_->used[i] += cost[i];
        return Lease(state_, std::move(cost));
    }
};

} // namespace mfq::engine
