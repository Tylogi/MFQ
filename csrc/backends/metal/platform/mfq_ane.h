#pragma once

#include <memory>
#include <span>
#include <vector>

namespace mfq::metal {

class AneMatmul {
public:
    AneMatmul(int rows, int inner, int columns);
    ~AneMatmul();
    AneMatmul(const AneMatmul&) = delete;
    AneMatmul& operator=(const AneMatmul&) = delete;
    std::vector<float> operator()(std::span<const float> left, std::span<const float> right);
    std::size_t bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
