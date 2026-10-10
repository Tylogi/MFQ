cmake_minimum_required(VERSION 3.26)
project(mfq_llama_format_bench LANGUAGES C CXX CUDA)

set(LLAMA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../references/llama.cpp" CACHE PATH
    "Independent llama.cpp checkout used only by the comparison executable")
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_FA OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_SOURCE}/ggml" ggml)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama_format_bench.cu")
    find_package(CUDAToolkit REQUIRED)
    add_executable(llama-format-bench llama_format_bench.cu)
    target_include_directories(llama-format-bench PRIVATE
        "${LLAMA_SOURCE}/ggml/src" "${LLAMA_SOURCE}/ggml/src/ggml-cuda" ..)
    target_link_libraries(llama-format-bench PRIVATE ggml ggml-cuda CUDA::cudart CUDA::cublas)
    set_target_properties(llama-format-bench PROPERTIES CUDA_STANDARD 17)
endif()
