#pragma once

#include <cstdint>

int run_backend_bf16_add_check(std::int64_t elements, int repetitions);
int run_backend_argmax_check(std::int64_t elements, int repetitions);
