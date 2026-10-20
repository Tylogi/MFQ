#include "runtime/mlx_builtin_kernels.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: mfq-metal-source-export OUTPUT_DIRECTORY");
        const std::filesystem::path root(argv[1]);
        std::filesystem::create_directories(root);
        const std::pair<const char*, std::string (*)()> sources[]{
            {"mfq_qsa_prefill_indexer_v1", mfq::metal::qsa_prefill_indexer_source},
            {"mfq_sparse_attention_v1", mfq::metal::sparse_attention_source},
            {"mfq_dsv4_mxfp4_blocks_v5", mfq::metal::dsv4_mxfp4_blocks_source},
        };
        for (const auto& [name, source] : sources) {
            std::ofstream output(root / (std::string(name) + ".metal"), std::ios::binary);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output << source();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
