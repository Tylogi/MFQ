#pragma once
#include <memory>

struct NintPrefillDecodeCacheState;
// Reuse the existing temporary dense projection within one prefill layer.
// No decoded weights survive the scope or affect decode/MTP dispatch.
class NintPrefillDecodeCacheScope {
    std::unique_ptr<NintPrefillDecodeCacheState> state_;
public:
    explicit NintPrefillDecodeCacheScope(bool enabled);
    ~NintPrefillDecodeCacheScope();
    void clear();
    NintPrefillDecodeCacheScope(const NintPrefillDecodeCacheScope&)=delete;
    NintPrefillDecodeCacheScope& operator=(const NintPrefillDecodeCacheScope&)=delete;
};
