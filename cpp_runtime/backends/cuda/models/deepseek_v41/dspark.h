#pragma once

#include <memory>

namespace mfq {
class ModelSource;
namespace models::deepseek_v41 { struct Config; }
}
struct CudaExecutionContext;
struct MtpModule;

namespace mfq::cuda::deepseek_v41_runtime {

std::unique_ptr<::MtpModule> load_dspark_if_present(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const mfq::models::deepseek_v41::Config& config);
void run_dspark_self_check();

} // namespace mfq::cuda::deepseek_v41_runtime
