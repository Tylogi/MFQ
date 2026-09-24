# Native CUDA backend.

if(NOT TARGET mfq-runtime-communication)
    message(FATAL_ERROR
        "The CUDA runtime requires MFQ_BUILD_RUNTIME_COMMUNICATION=ON")
endif()

set(MFQ_CUDA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(MFQ_CUDA_KERNEL_ROOT "${MFQ_REPOSITORY_ROOT}/mfq/kernels/cuda")

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(MFQ_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures compiled into MFQ CUDA targets")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(mfq-cuda-storage STATIC
    ${MFQ_CUDA_ROOT}/storage/mfe_expert_store.cpp
)
add_library(mfq::cuda-storage ALIAS mfq-cuda-storage)
target_include_directories(mfq-cuda-storage PUBLIC
    ${MFQ_CUDA_ROOT}/storage
)
target_link_libraries(mfq-cuda-storage PUBLIC Threads::Threads)
target_compile_features(mfq-cuda-storage PUBLIC cxx_std_20)
set_target_properties(mfq-cuda-storage PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(mfq-cuda-storage PRIVATE /utf-8 /EHsc)
endif()

add_library(mfq-cuda-core STATIC
    ${MFQ_CUDA_ROOT}/src/mfq_cuda_context.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor.cpp
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_ops.cu
)
add_library(mfq::cuda-core ALIAS mfq-cuda-core)
target_include_directories(mfq-cuda-core PUBLIC
    ${MFQ_CUDA_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mfq-cuda-core PUBLIC CUDA::cudart CUDA::cublas)
target_compile_features(mfq-cuda-core PUBLIC cxx_std_20)
target_compile_options(mfq-cuda-core PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)
set_target_properties(mfq-cuda-core PROPERTIES
    CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

set(MFQ_CUDA_KERNEL_SOURCES
    ${MFQ_CUDA_KERNEL_ROOT}/acc.cu
    ${MFQ_CUDA_KERNEL_ROOT}/activation.cu
    ${MFQ_CUDA_KERNEL_ROOT}/attention.cu
    ${MFQ_CUDA_KERNEL_ROOT}/attention_mma.cu
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v4_attention.cu
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v4_hc.cu
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v41.cu
    ${MFQ_CUDA_KERNEL_ROOT}/embedding.cu
    ${MFQ_CUDA_KERNEL_ROOT}/flash_next.cu
    ${MFQ_CUDA_KERNEL_ROOT}/gated_delta_net.cu
    ${MFQ_CUDA_KERNEL_ROOT}/glm_dsa.cu
    ${MFQ_CUDA_KERNEL_ROOT}/kv_cache.cu
    ${MFQ_CUDA_KERNEL_ROOT}/paged_kv.cu
    ${MFQ_CUDA_KERNEL_ROOT}/moe.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mx_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/fp8_sq.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mxfp4_sq.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nepq.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nepq_residual.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nint_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/norm.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nvq_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/rope.cu
    ${MFQ_CUDA_KERNEL_ROOT}/sampling.cu
    ${MFQ_CUDA_KERNEL_ROOT}/ssm_conv.cu
)

add_library(mfq-cuda-native-kernels STATIC ${MFQ_CUDA_KERNEL_SOURCES})
add_library(mfq::cuda-native-kernels ALIAS mfq-cuda-native-kernels)
target_compile_definitions(mfq-cuda-native-kernels PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1
)
target_include_directories(mfq-cuda-native-kernels PRIVATE
    ${MFQ_REPOSITORY_ROOT}
    ${MFQ_CUDA_ROOT}/include
    ${MFQ_CUDA_KERNEL_ROOT}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mfq-cuda-native-kernels
    PUBLIC mfq-cuda-core CUDA::cuda_driver
    PRIVATE mfq-core mfq-ggml-internal-headers
)
target_compile_options(mfq-cuda-native-kernels PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
)
set_target_properties(mfq-cuda-native-kernels PROPERTIES
    CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
    foreach(mfq_cuda_target mfq-cuda-core mfq-cuda-native-kernels)
        target_compile_options(${mfq_cuda_target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        )
    endforeach()
endif()

if(BUILD_TESTING)
    function(mfq_add_cuda_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE mfq-cuda-core ${ARGN})
        set_target_properties(${target} PROPERTIES
            CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
            CUDA_RUNTIME_LIBRARY Shared
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    add_executable(mfq-cuda-mfe-expert-store-test
        ${MFQ_CUDA_ROOT}/tests/mfq_mfe_expert_store_test.cpp)
    target_link_libraries(mfq-cuda-mfe-expert-store-test PRIVATE
        mfq-cuda-storage)
    target_compile_features(mfq-cuda-mfe-expert-store-test PRIVATE cxx_std_20)
    add_test(NAME mfq-cuda-mfe-expert-store-test
        COMMAND mfq-cuda-mfe-expert-store-test)

    add_executable(mfq-mxfp4-sq-test ${MFQ_CUDA_ROOT}/tests/mfq_mxfp4_sq_test.cu)
    target_compile_definitions(mfq-mxfp4-sq-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-mxfp4-sq-test PRIVATE ${MFQ_REPOSITORY_ROOT})
    target_link_libraries(mfq-mxfp4-sq-test PRIVATE
        mfq-core mfq-cuda-core mfq-cuda-native-kernels)
    set_target_properties(mfq-mxfp4-sq-test PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    add_executable(mfq-paged-kv-attention-bench
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_paged_kv_attention_bench.cu)
    target_compile_definitions(mfq-paged-kv-attention-bench PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1)
    target_link_libraries(mfq-paged-kv-attention-bench PRIVATE
        mfq-cuda-core mfq-cuda-native-kernels)
    set_target_properties(mfq-paged-kv-attention-bench PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    mfq_add_cuda_test(mfq-cuda-context-test
        ${MFQ_CUDA_ROOT}/tests/mfq_cuda_context_test.cu)
    mfq_add_cuda_test(mfq-cuda-activation-test
        ${MFQ_CUDA_ROOT}/tests/mfq_cuda_activation_test.cu
        mfq-cuda-native-kernels)
    mfq_add_cuda_test(mfq-paged-kv-test
        ${MFQ_CUDA_ROOT}/tests/mfq_paged_kv_test.cu
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-paged-kv-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1)
    mfq_add_cuda_test(mfq-deepseek-v41-hc-test
        ${MFQ_CUDA_ROOT}/tests/mfq_deepseek_v41_hc_test.cu
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-deepseek-v41-hc-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-deepseek-v41-hc-test PRIVATE
        ${MFQ_REPOSITORY_ROOT})
    mfq_add_cuda_test(mfq-native-tensor-cuda-test
        ${MFQ_CUDA_ROOT}/tests/mfq_native_tensor_cuda_test.cu)
    mfq_add_cuda_test(mfq-flash-next-test
        ${MFQ_CUDA_ROOT}/tests/mfq_flash_next_test.cu
        mfq-cuda-native-kernels mfq-json)
    target_compile_definitions(mfq-flash-next-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-flash-next-test PRIVATE
        ${MFQ_REPOSITORY_ROOT}
        ${MFQ_CUDA_ROOT}/models)
endif()

set(MFQ_NCCL_ROOT "" CACHE PATH
    "Optional NCCL installation root for tensor-parallel inference")
if(UNIX)
    find_library(MFQ_NCCL_LIBRARY
        NAMES nccl libnccl.so.2
        HINTS
            "${MFQ_NCCL_ROOT}/lib"
            "${MFQ_NCCL_ROOT}/lib64"
            "$ENV{NCCL_ROOT}/lib"
            "$ENV{NCCL_ROOT}/lib64")
    find_path(MFQ_NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS
            "${MFQ_NCCL_ROOT}/include"
            "$ENV{NCCL_ROOT}/include")
endif()

find_package(OpenMP QUIET)

function(mfq_configure_cuda_runtime_target target)
    target_include_directories(${target} PRIVATE
        ${MFQ_REPOSITORY_ROOT}
        ${MFQ_CUDA_ROOT}
        ${MFQ_CUDA_ROOT}/include
        ${MFQ_CUDA_ROOT}/models
        ${MFQ_CUDA_ROOT}/ops/include
        ${MFQ_CUDA_ROOT}/commands
        ${MFQ_CUDA_ROOT}/runtime
        ${MFQ_CUDA_KERNEL_ROOT}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        mfq-core
        mfq-json
        mfq-runtime-communication
        mfq-ggml-internal-headers
    )
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MFQ_NCCL_LIBRARY AND MFQ_NCCL_INCLUDE_DIR)
        target_include_directories(${target} PRIVATE "${MFQ_NCCL_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE "${MFQ_NCCL_LIBRARY}")
        target_compile_definitions(${target} PRIVATE MFQ_HAVE_NCCL=1)
    endif()
endfunction()

set(MFQ_CUDA_MODEL_SOURCES
    ${MFQ_CUDA_ROOT}/models/deepseek_v4/deepseek_v4_causal_lm.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v41/deepseek_v41_causal_lm.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v41/deepseek_v41_dspark.cpp
    ${MFQ_CUDA_ROOT}/models/qwen4_exp/causal_lm.cpp
    ${MFQ_CUDA_ROOT}/models/glm5_next/causal_lm.cpp
    ${MFQ_CUDA_ROOT}/models/gemma4/gemma4_causal_lm.cpp
    ${MFQ_CUDA_ROOT}/models/glm_dsa/glm_dsa_causal_lm.cpp
    ${MFQ_CUDA_ROOT}/models/minicpmo45/minicpmo45_runtime.cpp
    ${MFQ_CUDA_ROOT}/models/qwen35/qwen35_causal_lm.cpp
    ${MFQ_CUDA_ROOT}/models/registry.cpp
)

set(MFQ_CUDA_RUNTIME_OP_SOURCES
    ${MFQ_CUDA_ROOT}/ops/format.cpp
    ${MFQ_CUDA_ROOT}/ops/fp8_sq.cpp
    ${MFQ_CUDA_ROOT}/ops/mx.cpp
    ${MFQ_CUDA_ROOT}/ops/mxfp4_sq.cpp
    ${MFQ_CUDA_ROOT}/ops/moe.cpp
    ${MFQ_CUDA_ROOT}/ops/nint.cpp
    ${MFQ_CUDA_ROOT}/ops/quant_linear.cpp
    ${MFQ_CUDA_ROOT}/ops/vq.cpp
)

add_library(mfq-cuda-runtime STATIC
    ${MFQ_CUDA_RUNTIME_OP_SOURCES}
    ${MFQ_CUDA_ROOT}/runtime/cuda_runtime.cpp
    ${MFQ_CUDA_ROOT}/runtime/runner.cpp
    ${MFQ_CUDA_ROOT}/runtime/decode_graph.cpp
    ${MFQ_CUDA_ROOT}/runtime/cuda_execution.cpp
    ${MFQ_CUDA_ROOT}/runtime/causal_lm.cpp
    ${MFQ_CUDA_ROOT}/runtime/causal_lm_loader.cpp
    ${MFQ_CUDA_ROOT}/runtime/cuda_transformer.cpp
    ${MFQ_CUDA_ROOT}/runtime/cuda_transformer_loader.cpp
    ${MFQ_CUDA_ROOT}/runtime/mtp.cpp
    ${MFQ_CUDA_ROOT}/runtime/moe_expert_cache.cpp
    ${MFQ_CUDA_ROOT}/runtime/runtime_components.cpp
    ${MFQ_CUDA_ROOT}/apps/ggml_cuda_compat.cu
    ${MFQ_CUDA_MODEL_SOURCES}
)
add_library(mfq::cuda-runtime ALIAS mfq-cuda-runtime)
mfq_configure_cuda_runtime_target(mfq-cuda-runtime)
target_link_libraries(mfq-cuda-runtime PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    mfq-cuda-core
    mfq-cuda-native-kernels
    mfq-cuda-storage
)
target_compile_definitions(mfq-cuda-runtime PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1
    NOMINMAX
)
if(MSVC)
    target_compile_options(mfq-cuda-runtime PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
        "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
    )
endif()
set_target_properties(mfq-cuda-runtime PROPERTIES
    CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

add_library(mfq-cuda-eval STATIC ${MFQ_CUDA_ROOT}/eval/kl.cpp)
mfq_configure_cuda_runtime_target(mfq-cuda-eval)
target_include_directories(mfq-cuda-eval PRIVATE ${MFQ_CUDA_ROOT}/storage)
target_compile_definitions(mfq-cuda-eval PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-cuda-eval PUBLIC mfq-cuda-runtime)

add_library(mfq-cuda-diagnostics STATIC
    ${MFQ_CUDA_ROOT}/diagnostics/backend_checks.cpp
    ${MFQ_CUDA_ROOT}/diagnostics/model_checks.cpp
)
mfq_configure_cuda_runtime_target(mfq-cuda-diagnostics)
target_include_directories(mfq-cuda-diagnostics PRIVATE ${MFQ_CUDA_ROOT}/storage)
target_compile_definitions(mfq-cuda-diagnostics PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-cuda-diagnostics PUBLIC mfq-cuda-runtime)
if(MSVC)
    foreach(target mfq-cuda-diagnostics mfq-cuda-eval)
        target_compile_options(${target} PRIVATE /utf-8 /EHsc /bigobj)
    endforeach()
endif()

add_executable(mfq-runtime
    ${MFQ_CUDA_ROOT}/apps/runtime_main.cpp
    ${MFQ_CUDA_ROOT}/commands/runtime.cpp
)
target_link_libraries(mfq-runtime PRIVATE
    mfq-cuda-runtime mfq-runtime-communication)
target_include_directories(mfq-runtime PRIVATE
    ${MFQ_CUDA_ROOT}
    ${MFQ_CUDA_ROOT}/runtime
)
target_compile_features(mfq-runtime PRIVATE cxx_std_20)
set_target_properties(mfq-runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-diagnostics
    ${MFQ_CUDA_ROOT}/apps/diagnostics_main.cpp
    ${MFQ_CUDA_ROOT}/commands/diagnostics.cpp
)
mfq_configure_cuda_runtime_target(mfq-diagnostics)
target_include_directories(mfq-diagnostics PRIVATE ${MFQ_CUDA_ROOT}/storage)
target_compile_definitions(mfq-diagnostics PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-diagnostics PRIVATE mfq-cuda-diagnostics)
target_compile_features(mfq-diagnostics PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(mfq-diagnostics PRIVATE /utf-8 /EHsc /bigobj)
endif()
set_target_properties(mfq-diagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-eval
    ${MFQ_CUDA_ROOT}/apps/eval_main.cpp
    ${MFQ_CUDA_ROOT}/commands/eval.cpp
)
mfq_configure_cuda_runtime_target(mfq-eval)
target_include_directories(mfq-eval PRIVATE ${MFQ_CUDA_ROOT}/storage)
target_compile_definitions(mfq-eval PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-eval PRIVATE
    mfq-cuda-eval mfq-cuda-runtime mfq-cuda-native-kernels mfq-ggml)
target_compile_features(mfq-eval PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(mfq-eval PRIVATE /utf-8 /EHsc /bigobj)
endif()
set_target_properties(mfq-eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(BUILD_TESTING)
    add_test(
        NAME mfq-deepseek-v41-runtime-check
        COMMAND mfq-diagnostics --check-deepseek-v41)
    add_test(NAME mfq-diagnostics-help COMMAND mfq-diagnostics --help)
    add_test(NAME mfq-eval-help COMMAND mfq-eval --help)
    add_test(NAME mfq-eval-kl-help COMMAND mfq-eval kl --help)
    add_test(NAME mfq-runtime-cli-options COMMAND mfq-runtime --ids 1,2 --gen 3 --help)
    add_test(NAME mfq-diagnostics-cli-options COMMAND mfq-diagnostics --check-linear x --kl-mmq default --help)
    add_test(NAME mfq-eval-cli-options COMMAND mfq-eval kl --kl-base x --kl-evaluator optimized --help)
    add_test(NAME mfq-runtime-rejects-check
        COMMAND mfq-runtime --check-mfq-container)
    add_test(NAME mfq-runtime-rejects-kl
        COMMAND mfq-runtime --kl-base reference.bin)
    add_test(NAME mfq-runtime-rejects-bad-number
        COMMAND mfq-runtime --gen 3x --help)
    add_test(NAME mfq-runtime-rejects-missing-value
        COMMAND mfq-runtime --model --help)
    add_test(NAME mfq-runtime-rejects-bad-ids
        COMMAND mfq-runtime --ids 1,x --help)
    add_test(NAME mfq-diagnostics-rejects-kl
        COMMAND mfq-diagnostics --kl-base reference.bin)
    add_test(NAME mfq-diagnostics-rejects-missing-value
        COMMAND mfq-diagnostics --check-linear --check-backend-argmax)
    add_test(NAME mfq-eval-rejects-check
        COMMAND mfq-eval kl --check-mfq-container)
    add_test(NAME mfq-eval-rejects-bad-number
        COMMAND mfq-eval kl --kl-chunks 1x --help)
    set_tests_properties(
        mfq-runtime-rejects-check mfq-runtime-rejects-kl
        mfq-runtime-rejects-bad-number mfq-runtime-rejects-missing-value
        mfq-runtime-rejects-bad-ids mfq-diagnostics-rejects-kl mfq-diagnostics-rejects-missing-value
        mfq-eval-rejects-check mfq-eval-rejects-bad-number
        PROPERTIES WILL_FAIL TRUE)
endif()

option(MFQ_BUILD_TORCH_REFERENCE_RUNTIME
    "Build the optional LibTorch CUDA runtime used only for A/B validation"
    OFF)
if(MFQ_BUILD_TORCH_REFERENCE_RUNTIME)
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    if(NOT TARGET CUDA::nvToolsExt)
        find_library(NVTOOLSEXT_LIBRARY
            NAMES nvToolsExt nvToolsExt64_1
            PATHS "${CUDAToolkit_LIBRARY_DIR}" "$ENV{CUDA_PATH}/lib/x64"
            NO_DEFAULT_PATH)
        if(NVTOOLSEXT_LIBRARY)
            add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
            set_target_properties(CUDA::nvToolsExt PROPERTIES
                IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}")
        else()
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        endif()
    endif()
    find_package(Torch REQUIRED)
    add_executable(mfq-runtime-torch
        ${MFQ_CUDA_ROOT}/apps/runtime_main.cpp
        ${MFQ_CUDA_ROOT}/commands/runtime.cpp
        ${MFQ_CUDA_RUNTIME_OP_SOURCES}
        ${MFQ_CUDA_ROOT}/runtime/cuda_runtime.cpp
        ${MFQ_CUDA_ROOT}/runtime/runner.cpp
        ${MFQ_CUDA_ROOT}/runtime/decode_graph.cpp
        ${MFQ_CUDA_ROOT}/runtime/cuda_execution.cpp
        ${MFQ_CUDA_ROOT}/runtime/causal_lm.cpp
        ${MFQ_CUDA_ROOT}/runtime/causal_lm_loader.cpp
        ${MFQ_CUDA_ROOT}/runtime/cuda_transformer.cpp
        ${MFQ_CUDA_ROOT}/runtime/cuda_transformer_loader.cpp
        ${MFQ_CUDA_ROOT}/runtime/mtp.cpp
        ${MFQ_CUDA_ROOT}/runtime/moe_expert_cache.cpp
        ${MFQ_CUDA_ROOT}/runtime/runtime_components.cpp
        ${MFQ_CUDA_ROOT}/apps/ggml_cuda_compat.cu
        ${MFQ_CUDA_MODEL_SOURCES}
        ${MFQ_CUDA_KERNEL_SOURCES}
    )
    mfq_configure_cuda_runtime_target(mfq-runtime-torch)
    target_include_directories(mfq-runtime-torch PRIVATE ${Python_INCLUDE_DIRS})
    target_link_libraries(mfq-runtime-torch PRIVATE
        ${TORCH_LIBRARIES}
        Python::Python
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::cublas
        mfq-cuda-storage
    )
    target_compile_definitions(mfq-runtime-torch PRIVATE NOMINMAX)
    target_compile_options(mfq-runtime-torch PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
    )
    set_target_properties(mfq-runtime-torch PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(NOT WIN32)
    set(MFQ_SYSTEM_RUNTIME_DIR
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
        "directory containing the host libstdc++ runtime")
    set_property(TARGET mfq-runtime PROPERTY
        BUILD_RPATH "${MFQ_SYSTEM_RUNTIME_DIR}")
endif()

set_source_files_properties(
    ${MFQ_CUDA_KERNEL_ROOT}/attention_mma.cu
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v4_attention.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math;--extended-lambda"
)
set_source_files_properties(
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v4_hc.cu
    PROPERTIES COMPILE_OPTIONS "--extended-lambda"
)
