#include "mlx_builtin_kernels.h"
#include "mlx_nint_metadata.h"
#include "mfq_mfe_prefill_embedded.h"
#include <utility>

namespace mfq::metal {
namespace {
#include "../kernels/mfq_qsa_prefill.inc"
#include "../kernels/mfq_sdpa_decode.inc"
}

std::string qsa_prefill_indexer_source() {
    std::string source = "#include <metal_stdlib>\n"
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n";
    source += detail::kSteelMmaSource;
    source += "\n#define STEEL_PRAGMA_UNROLL _Pragma(\"clang loop unroll(full)\")\n";
    source += kQsaPrefillIndexerSource;
    return source;
}

std::string sparse_attention_source() {
    std::string source = "#include <metal_stdlib>\n"
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\nusing bfloat16_t = bfloat;\n";
    source += detail::kSteelAttentionSource;
    source += detail::kDsaSparsePrefillSource;
    source += detail::kSparseBlockGqaSource;
    return source;
}

std::string dsv4_mxfp4_blocks_source() {
    std::string source = "#include <metal_stdlib>\n"
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"
        "#define MFQ_ENABLE_DSV4_MXFP4_BLOCKS 1\nusing namespace metal;\n";
    source += detail::kNintMetadataSource;
    source += detail::kSteelMmaSource;
    source += detail::kMfePrefillSource;
    return source;
}

std::string sdpa_decode_source() {
    std::string source = sparse_attention_source();
    source += kSdpaDecodeSource;
    for (const auto& type : {std::pair{"f16", "half"}, std::pair{"bf16", "bfloat"}}) {
        const std::string reduction = "mfq_sdpa_decode_2pass_2<" + std::string(type.second) + ", 256>";
        source += "\ntemplate [[host_name(\"mfq_sdpa_decode_2pass_2_" + std::string(type.first) + "_256\")]] "
            "[[kernel]] decltype(" + reduction + ") " + reduction + ";\n";
        const std::string arguments = std::string(type.second) + ", 256, 256, 384";
        source += "\ntemplate [[host_name(\"mfq_sdpa_decode_sparse_" + std::string(type.first) + "_256\")]] "
            "[[kernel]] decltype(mfq_sdpa_decode_2pass_1<" + arguments + ">) "
            "mfq_sdpa_decode_2pass_1<" + arguments + ">;\n";
    }
    return source;
}

}
