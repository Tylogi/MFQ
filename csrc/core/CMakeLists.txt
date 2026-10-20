# Transitional aggregation of runtime contracts, placement rules and state storage.

find_package(Threads REQUIRED)
add_library(mfq-core INTERFACE)
add_library(mfq-qsa-kv-store STATIC qsa_kv_store.cpp include/mfq/qsa_kv_store.h)
target_include_directories(mfq-qsa-kv-store PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mfq-qsa-kv-store PUBLIC Threads::Threads)
target_compile_features(mfq-qsa-kv-store PUBLIC cxx_std_17)
set_target_properties(mfq-qsa-kv-store PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-qsa-kv-store PRIVATE /utf-8 /EHsc)
endif()
add_library(mfq::core ALIAS mfq-core)
target_include_directories(mfq-core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(mfq-core INTERFACE
    mfq-formats
    mfq-quant-readers
    mfq-host-parallel
    mfq-model-graph
    mfq-model-source
    mfq-json
    mfq-qsa-kv-store
)

find_package(Threads REQUIRED)

add_library(mfq-paged-prefix-cache STATIC
    mfq_paged_prefix_cache.cpp
    mfq_paged_prefix_cache.h
)
target_include_directories(mfq-paged-prefix-cache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(mfq-paged-prefix-cache PUBLIC Threads::Threads)
target_compile_features(mfq-paged-prefix-cache PUBLIC cxx_std_17)
set_target_properties(mfq-paged-prefix-cache PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
