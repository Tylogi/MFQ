#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace mfq::engine { class GrammarConstraint; }

// Concrete, engine-owned grammar cursor; no caller-provided behavior.
class MfqTokenConstraint {
public:
    explicit MfqTokenConstraint(std::shared_ptr<mfq::engine::GrammarConstraint> implementation);
    bool allows(std::int64_t token);
    void apply(float* logits, std::size_t count);
    void accept(std::int64_t token);
    std::shared_ptr<MfqTokenConstraint> clone() const;

private:
    std::shared_ptr<mfq::engine::GrammarConstraint> implementation_;
};

using MfqTokenConstraintPtr = std::shared_ptr<MfqTokenConstraint>;
