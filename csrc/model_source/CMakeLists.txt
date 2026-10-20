add_library(mfq-model-source STATIC
    model_source.cpp
    mfq_model_source.cpp
    hf_model_source.cpp
    hf_safetensors_source.cpp
    compat/mfq_legacy_model_graph.cpp
    compat/mfq_legacy_tensor_names.cpp
)
add_library(mfq::model-source ALIAS mfq-model-source)
target_include_directories(mfq-model-source PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mfq-model-source PUBLIC mfq-model-graph mfq-io PRIVATE mfq-json)
target_compile_features(mfq-model-source PUBLIC cxx_std_17)
set_target_properties(mfq-model-source PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-model-source PRIVATE /EHsc)
endif()
