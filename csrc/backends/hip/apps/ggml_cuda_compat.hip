#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime_api.h>

// MFQ's custom kernels use the integrated GGML CUDA helper headers without
// linking the complete ggml-cuda backend. Provide the host symbols they need.
void ggml_cuda_error(
    const char * statement,
    const char * function,
    const char * file,
    int line,
    const char * message) {
    std::fprintf(
        stderr,
        "CUDA error: %s\n  statement: %s\n  function: %s\n  at %s:%d\n",
        message != nullptr ? message : "unknown",
        statement != nullptr ? statement : "unknown",
        function != nullptr ? function : "unknown",
        file != nullptr ? file : "unknown",
        line);
    std::abort();
}

int ggml_cuda_get_device() {
    int device = 0;
    const hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) {
        ggml_cuda_error(
            "hipGetDevice(&device)",
            __func__,
            __FILE__,
            __LINE__,
            hipGetErrorString(status));
    }
    return device;
}
