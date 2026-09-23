#include "mfq_cuda_runtime.h"

#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc < 2 || std::string_view(argv[1]) != "kl") {
        std::cerr << "usage: mfq-eval kl --model MODEL_PATH --kl-base REFERENCE\n";
        return 2;
    }
    return mfq::cuda::run_eval(argc - 1, argv + 1);
}
