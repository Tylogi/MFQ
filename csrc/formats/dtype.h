#pragma once

#include <string_view>

namespace mfq {

inline constexpr std::string_view kNintDtype = "NINT";
inline constexpr std::string_view kMfeDtype = "MFE";
inline constexpr std::string_view kMfeDeltaDtype = "MFED";
inline constexpr std::string_view kNvqDtype = "NVQ";
inline constexpr std::string_view kNpqDtype = "NPQ";
inline constexpr std::string_view kNepqDtype = "NEPQ";
inline constexpr std::string_view kMxfp4SqDtype = "MXFP4-SQ";
inline constexpr std::string_view kMxfp8SqDtype = "MXFP8-SQ";
inline constexpr std::string_view kFp8Block128SqDtype = "FP8-128SQ";

inline bool is_current_mfe_magic(std::string_view magic) noexcept {
    return magic == "MFE1";
}

inline bool is_current_mfe_delta_magic(std::string_view magic) noexcept {
    return magic == "MFD1";
}

} // namespace mfq
