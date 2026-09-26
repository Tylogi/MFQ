#pragma once

namespace mfq::cuda {
struct MiniCPMO45CausalLm;
struct MiniCPMOTtsCausalLm;
}

template <typename Model>
struct RuntimeComponents;

template <>
RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMO45CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMOTtsCausalLm& model,
    bool load_optional_components);
