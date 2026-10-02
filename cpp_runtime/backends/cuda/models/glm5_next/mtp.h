#pragma once

#include "../mtp.h"
#include "core/attention.h"
#include "models/glm5_next/config.h"

namespace mfq::cuda::glm5_next {
using attention_ops::Linear;
class SparseMla;

struct Glm5NextMtp final : MtpModule {
    using Tensor = mfq_tensor_backend::Tensor;
    struct Layer {
        Tensor attention_norm, ffn_norm;
        std::unique_ptr<SparseMla> attention;
        Linear ffn;
    };

    mfq::models::glm5_next::Config config;
    Tensor embedding_norm, hidden_norm, output_norm;
    Linear fusion;
    std::vector<Layer> layers;
    std::vector<int64_t> lengths;
    CudaExecutionContext *execution = nullptr;
    int64_t batch = 0;

    Glm5NextMtp();
    ~Glm5NextMtp() override;
    Glm5NextMtp(Glm5NextMtp&&) noexcept;
    Glm5NextMtp& operator=(Glm5NextMtp&&) noexcept;

    static std::optional<Glm5NextMtp> load_if_present(CudaExecutionContext &execution,
                                                      const mfq::ModelSource &file,
                                                      const mfq::models::glm5_next::Config &main);

    void reset(int64_t next_batch = 1) override;

    std::pair<Tensor, Tensor> evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth = 0, bool cache = true,
                                       const Tensor &supplied_positions = {},
                                       const Tensor &supplied_embeddings = {});

    Tensor forward(const MtpTarget &target, Tensor hidden, Tensor ids) override;

    MtpStep step(const MtpTarget &target, Tensor hidden, Tensor ids) override;

    int64_t cache_position() const noexcept override;

    void trim_cache_to(int64_t position) override;

    bool teacher_forced_prompt_prime() const noexcept override;

    bool target_bootstrap_decode() const noexcept override;

    bool preserve_output_dtype() const noexcept override;
};

} // namespace mfq::cuda::glm5_next
