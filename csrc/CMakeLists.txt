cmake_minimum_required(VERSION 3.26)
project(mfq_cpp_runtime LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MFQ_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(MFQ_CORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/core/include")

# Shared engine module headers. They are PRIVATE to mfq-engine; backends, apps
# and tests add these directories to their include path to drive the shared
# generation loop, which tests/test_engine_include_graph.py keeps bounded.
set(MFQ_ENGINE_PRIVATE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/requests
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/batch
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/state
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/weights
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/text
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/src
)

option(
    MFQ_BUILD_METAL_RUNTIME
    "Build the native MLX/Metal runtime on Apple Silicon"
    ${APPLE}
)

# Only this backend enables the CUDA language or requires the CUDA toolkit.
if(APPLE)
    set(_MFQ_BUILD_CUDA_RUNTIME_DEFAULT OFF)
else()
    set(_MFQ_BUILD_CUDA_RUNTIME_DEFAULT ON)
endif()
option(
    MFQ_BUILD_CUDA_RUNTIME
    "Build the native CUDA runtime, which requires the CUDA toolkit"
    ${_MFQ_BUILD_CUDA_RUNTIME_DEFAULT}
)
unset(_MFQ_BUILD_CUDA_RUNTIME_DEFAULT)

if(APPLE)
    set(_MFQ_BUILD_RUNTIME_COMMUNICATION_DEFAULT OFF)
else()
    set(_MFQ_BUILD_RUNTIME_COMMUNICATION_DEFAULT ON)
endif()
option(
    MFQ_BUILD_RUNTIME_COMMUNICATION
    "Build private HTTP/stdio communication modules for native runtimes"
    ${_MFQ_BUILD_RUNTIME_COMMUNICATION_DEFAULT}
)
unset(_MFQ_BUILD_RUNTIME_COMMUNICATION_DEFAULT)

if(MFQ_BUILD_METAL_RUNTIME AND NOT APPLE)
    message(FATAL_ERROR "MFQ_BUILD_METAL_RUNTIME requires macOS")
endif()

if(MFQ_BUILD_CUDA_RUNTIME AND APPLE)
    message(FATAL_ERROR
        "MFQ_BUILD_CUDA_RUNTIME requires the CUDA toolkit, which macOS does not provide")
endif()

# Common code is split by responsibility. JSON is defined first because the
# canonical model graph and several public policy headers consume it.
add_subdirectory(components/json)
add_subdirectory(core)
add_subdirectory(backends/cpu)
add_subdirectory(models)

add_subdirectory(components/ggml)
add_subdirectory(components/tokenizer)

if(MFQ_BUILD_RUNTIME_COMMUNICATION)
    add_subdirectory(components/http)
endif()

add_subdirectory(engine)
add_subdirectory(scheduler)

if(MFQ_BUILD_RUNTIME_COMMUNICATION)
    add_subdirectory(transport)
endif()

if(MFQ_BUILD_METAL_RUNTIME)
    enable_language(OBJCXX)
    # Keep the established build output path and executable locations.
    add_subdirectory(backends/metal metal)
endif()

if(MFQ_BUILD_CUDA_RUNTIME)
    add_subdirectory(backends/cuda cuda)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
