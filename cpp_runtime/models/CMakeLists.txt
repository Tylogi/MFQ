add_library(mfq-models STATIC
    model_config.cpp
    deepseek_v4.cpp
    deepseek_v41.cpp
    glm5_next.cpp
    qwen4_exp.cpp
    gemma4.cpp
    glm_dsa.cpp
    minicpmo45.cpp
    qwen35.cpp
)
add_library(mfq::models ALIAS mfq-models)
target_include_directories(mfq-models PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-models PUBLIC mfq-core)
target_compile_features(mfq-models PUBLIC cxx_std_17)
set_target_properties(mfq-models PROPERTIES POSITION_INDEPENDENT_CODE ON)
