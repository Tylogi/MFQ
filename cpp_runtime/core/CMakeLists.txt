# Backend-neutral graph, policies, interfaces, and generated tables.

add_library(mfq-model-graph STATIC
    grid_vision.cpp
    grid_vision.h
    mfq_model_graph.cpp
    mfq_model_graph.h
    models/model_config.cpp
    models/model_config.h
    models/deepseek_v4.cpp
    models/deepseek_v4.h
    models/deepseek_v41.cpp
    models/deepseek_v41.h
    models/flash_next.cpp
    models/flash_next.h
    models/gemma4.cpp
    models/gemma4.h
    models/glm_dsa.cpp
    models/glm_dsa.h
    models/minicpmo45.cpp
    models/minicpmo45.h
    models/qwen35.cpp
    models/qwen35.h
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
    hf_model_source.cpp
    hf_safetensors_source.cpp
    include/mfq/hf_model_source.h
    include/mfq/hf_safetensors_source.h
    include/mfq/model_graph.h
    include/mfq/model_source.h
    include/mfq/mfq_model_source.h
    mfq_model_source.cpp
    model_source.cpp
)
add_library(mfq::model-source ALIAS mfq-model-source)
target_include_directories(mfq-model-source PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
)
target_link_libraries(mfq-model-source PUBLIC mfq-model-graph)
target_link_libraries(mfq-model-source PRIVATE mfq-json)
target_compile_features(mfq-model-source PUBLIC cxx_std_17)
set_target_properties(mfq-model-source PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(mfq-core INTERFACE)
add_library(mfq::core ALIAS mfq-core)
target_include_directories(mfq-core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
)
target_link_libraries(mfq-core INTERFACE
    mfq-model-graph
    mfq-model-source
    mfq-json
)

if(MFQ_BUILD_RUNTIME_COMMUNICATION)
    find_package(Threads REQUIRED)

    add_library(mfq-paged-prefix-cache STATIC
        mfq_paged_prefix_cache.cpp
        mfq_paged_prefix_cache.h
    )
    target_include_directories(mfq-paged-prefix-cache PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(mfq-paged-prefix-cache PUBLIC Threads::Threads)
    target_compile_features(mfq-paged-prefix-cache PUBLIC cxx_std_17)
    set_target_properties(mfq-paged-prefix-cache PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()
