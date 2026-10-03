#pragma once

#include "mfq_model_graph.h"

namespace mfq {

using ModelGraph = MfqModelGraph;
using ModelGraphComponent = MfqGraphComponent;
using ModelGraphTopology = MfqGraphTopology;

inline constexpr auto kModelGraphAsset = kMfqModelGraphAsset;
inline constexpr auto kHfSourceMapAsset = kMfqHfSourceMapAsset;
inline constexpr auto kCanonicalTensorNamespace = kMfqCanonicalTensorNamespace;

} // namespace mfq
