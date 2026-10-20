# Backend-neutral graph, policies, interfaces, and generated tables.

find_package(Threads REQUIRED)
add_library(mfq-model-graph STATIC
    grid_vision.cpp
    grid_vision.h
    mfq_model_graph.cpp
    mfq_model_graph.h
    compat/mfq_legacy_model_graph.cpp
    compat/mfq_legacy_model_graph.h
    compat/mfq_legacy_tensor_names.cpp
    compat/mfq_legacy_tensor_names.h
    compat/mfq_format_compat.h
)
add_library(mfq::model-graph ALIAS mfq-model-graph)
target_include_directories(mfq-model-graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
)
target_link_libraries(mfq-model-graph PUBLIC mfq-json)
target_compile_features(mfq-model-graph PUBLIC cxx_std_17)
set_target_properties(mfq-model-graph PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(mfq-model-source STATIC
    io/file_range_reader.cpp
    io/async_range_reader.cpp
    include/mfq/io/file_range_reader.h
    quant/mfe_quant_expert_store.cpp
    include/mfq/quant/mfe_expert_store.h
    quant/nint_rows.cpp
    include/mfq/quant/nint_rows.h
    quant/nvq_rows.cpp
    include/mfq/quant/nvq_rows.h
    include/mfq/quant/packed_row_range.h
    model_source/hf_model_source.cpp
    model_source/hf_safetensors_source.cpp
    include/mfq/model_source/hf_model_source.h
    include/mfq/model_source/hf_safetensors_source.h
    include/mfq/model_graph.h
    include/mfq/model_source/model_source.h
    include/mfq/model_source/mfq_model_source.h
    model_source/mfq_model_source.cpp
    model_source/model_source.cpp
)
add_library(mfq::model-source ALIAS mfq-model-source)
target_include_directories(mfq-model-source PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
)
target_link_libraries(mfq-model-source PUBLIC mfq-model-graph)
target_link_libraries(mfq-model-source PRIVATE mfq-host-parallel)
target_link_libraries(mfq-model-source PRIVATE mfq-json)
target_compile_features(mfq-model-source PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(mfq-model-source PRIVATE /EHsc)
endif()
set_target_properties(mfq-model-source PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
)
target_link_libraries(mfq-core INTERFACE
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
