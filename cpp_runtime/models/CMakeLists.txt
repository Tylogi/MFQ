add_library(mfq-models STATIC
    common/model_config.cpp
    deepseek_v4/config.cpp
    deepseek_v41/config.cpp
    glm5_next/config.cpp
    qwen4_exp/config.cpp
    gemma4/config.cpp
    glm_dsa/config.cpp
    minicpmo45/config.cpp
    qwen35/config.cpp
)
add_library(mfq::models ALIAS mfq-models)
target_include_directories(mfq-models PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-models PUBLIC mfq-core)
target_compile_features(mfq-models PUBLIC cxx_std_17)
set_target_properties(mfq-models PROPERTIES POSITION_INDEPENDENT_CODE ON)
