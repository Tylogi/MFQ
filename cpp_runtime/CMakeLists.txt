cmake_minimum_required(VERSION 3.26)
project(mfq_cpp_runtime LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MFQ_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

option(
    MFQ_BUILD_METAL_RUNTIME
    "Build the native MLX/Metal runtime on Apple Silicon"
    ${APPLE}
)

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

# Common code is split by responsibility. JSON is defined first because the
# canonical model graph and several public policy headers consume it.
add_subdirectory(components/json)
add_subdirectory(core)
add_subdirectory(scheduler)

if(MFQ_BUILD_RUNTIME_COMMUNICATION)
    add_subdirectory(components/ggml)
    add_subdirectory(components/tokenizer)
    add_subdirectory(components/http)
    add_subdirectory(transport)
endif()

if(MFQ_BUILD_METAL_RUNTIME)
    enable_language(OBJCXX)
    # Keep the established build output path and executable locations.
    add_subdirectory(backends/metal metal)
endif()

if(NOT APPLE)
    add_subdirectory(backends/cuda cuda)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
