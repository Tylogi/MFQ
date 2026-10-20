#include <string_view>

namespace mfq::cuda::commands {
int run_runtime(int argc, char** argv);
int run_reference(int argc, char** argv);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--wt2-reference")
        return mfq::cuda::commands::run_reference(argc, argv);
    return mfq::cuda::commands::run_runtime(argc, argv);
}
