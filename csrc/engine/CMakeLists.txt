add_library(mfq-engine STATIC
    model/common/vision/grid_vision.cpp
    model/common/model_config.cpp
    model/deepseek_v4/config.cpp
    model/deepseek_v41/config.cpp
    model/gemma4/config.cpp
    model/glm5_next/config.cpp
    model/glm_dsa/config.cpp
    model/minicpmo45/config.cpp
    model/qwen35/config.cpp
    model/qwen4_exp/config.cpp
    weights/slots.cpp
    src/settings.cpp
    src/policy.cpp
    src/mtp_metrics.cpp
    src/emitter.cpp
)
add_library(mfq::engine ALIAS mfq-engine)
target_include_directories(mfq-engine
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${MFQ_ENGINE_PRIVATE_DIRS}
)
target_compile_features(mfq-engine PUBLIC cxx_std_20)
target_link_libraries(mfq-engine PRIVATE mfq-core)
set_target_properties(mfq-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET mfq-paged-prefix-cache)
    target_sources(mfq-engine PRIVATE state/paged.cpp)
    target_link_libraries(mfq-engine PRIVATE mfq-paged-prefix-cache)
endif()

if(TARGET mfq-tokenizer)
    target_sources(mfq-engine PRIVATE
        src/inference.cpp
        src/media.cpp
        requests/table.cpp
        src/constraint.cpp
        src/tokenizer.cpp
        src/processor.cpp
    )
    target_include_directories(mfq-engine PRIVATE
        ${PROJECT_SOURCE_DIR}/components/tokenizer
        ${PROJECT_SOURCE_DIR}/components/tokenizer/src
    )
    target_link_libraries(mfq-engine PRIVATE mfq-tokenizer)
    target_compile_definitions(mfq-engine PRIVATE MFQ_ENGINE_TEXT=1)
endif()
