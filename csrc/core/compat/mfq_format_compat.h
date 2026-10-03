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

inline bool is_legacy_nint_dtype(std::string_view dtype) noexcept {
    return dtype == "NINTv2" ||
        (dtype.size() == 5 && dtype.substr(0, 4) == "NINT" &&
         dtype[4] >= '1' && dtype[4] <= '8');
}

inline std::string_view canonical_format_dtype(
    std::string_view dtype) noexcept {
    if (dtype == kNintDtype || is_legacy_nint_dtype(dtype)) {
        return kNintDtype;
    }
    if (dtype == "NINTM") {
        return kMfeDtype;
    }
    if (dtype == "NINTMD") {
        return kMfeDeltaDtype;
    }
    if (dtype == kNvqDtype || dtype == "NIQ2" ||
        dtype == "NIQ2J" || dtype == "NIQ3" ||
        dtype == "NVQ1-L" || dtype == "NVQ1-S" ||
        dtype == "NVQ2" || dtype == "NVQ2J" ||
        dtype == "NVQ2J-L" || dtype == "NVQ2J-XL" ||
        dtype == "NVQ3" || dtype == "NVQ3J" ||
        dtype == "NVQ3J-512" || dtype == "NVQ3J-L") {
        return kNvqDtype;
    }
    if (dtype == kNpqDtype || dtype == "NPQ0-L" ||
        dtype == "NPQ0-S") {
        return kNpqDtype;
    }
    if (dtype == kNepqDtype || dtype == "NEPQ0-L" ||
        dtype == "NEPQ0-S" || dtype == "NEPQ1-L" ||
        dtype == "NEPQ1-S" || dtype == "NEPQ0-A" ||
        dtype == "NEPQ1-A") {
        return kNepqDtype;
    }
    if (dtype == kMxfp4SqDtype || dtype == "MXFP4-SQ2" ||
        dtype == "MXFP4-SQ3") {
        return kMxfp4SqDtype;
    }
    return dtype;
}

inline bool is_current_mfe_magic(std::string_view magic) noexcept {
    return magic == "MFE1";
}

inline bool is_current_mfe_delta_magic(std::string_view magic) noexcept {
    return magic == "MFD1";
}

inline bool is_legacy_mfe_magic(std::string_view magic) noexcept {
    return magic == "NIM1" || magic == "NIM2";
}

inline bool is_legacy_mfe_delta_magic(std::string_view magic) noexcept {
    return magic == "NID2";
}

} // namespace mfq
