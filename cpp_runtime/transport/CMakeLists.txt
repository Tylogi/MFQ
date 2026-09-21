# Private HTTP/stdio communication modules for native runtime processes.

find_package(Threads REQUIRED)

add_library(mfq-runtime-communication STATIC
    include/mfq/communication.h
    src/transport.cpp
)
add_library(mfq::runtime-communication ALIAS mfq-runtime-communication)
target_include_directories(mfq-runtime-communication
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/components/tokenizer/src
)
target_link_libraries(mfq-runtime-communication
    PUBLIC
        mfq-runtime-scheduler
    PRIVATE
        mfq-http
        mfq-tokenizer
        mfq-json
        mfq-paged-prefix-cache
        Threads::Threads
)
target_compile_definitions(mfq-runtime-communication PRIVATE NOMINMAX)
target_compile_features(mfq-runtime-communication PUBLIC cxx_std_17)
set_target_properties(mfq-runtime-communication PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(mfq-runtime-communication PRIVATE /utf-8 /EHsc)
endif()
