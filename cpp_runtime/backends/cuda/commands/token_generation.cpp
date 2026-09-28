#include "cli.h"
#include "token_generation.h"

#include "models/loader.h"
#include "mfq_tensor_backend.h"

#include <utility>

namespace mfq::cuda::internal {

int run_cuda_token_generation(
        mfq::cuda::CudaLoadOptions& load_options,
        const mfq::cuda::TokenInputOptions& token_options) {
    mfq_tensor_backend::NoGradGuard no_grad;
    return with_loaded_cuda_model(
        load_options, false,
        [&]<mfq::cuda::CudaBackbone>(auto& model, auto&, auto t0, auto t1) {
            const auto ids_vector = token_options.ids_file.empty()
                ? parse_ids(token_options.ids_arg)
                : load_ids_file(token_options.ids_file);
            auto ids = mfq_tensor_backend::tensor(
                ids_vector,
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA)).unsqueeze(0);
            return generate_cli_tokens(
                model, std::move(ids), token_options.gen, false, t0, t1);
        });
}

} // namespace mfq::cuda::internal
