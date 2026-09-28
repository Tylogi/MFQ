add_library(mfq-engine STATIC
    src/continuous_batching.cpp
    src/generation_policy.cpp
    src/mtp_metrics.cpp
    src/text_emitter.cpp
)
add_library(mfq::engine ALIAS mfq-engine)
target_include_directories(mfq-engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mfq-engine PUBLIC cxx_std_20)
target_link_libraries(mfq-engine PUBLIC mfq-core)
set_target_properties(mfq-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET mfq-paged-prefix-cache)
    target_sources(mfq-engine PRIVATE src/paged_session_bindings.cpp)
    target_link_libraries(mfq-engine PUBLIC mfq-paged-prefix-cache)
endif()

if(TARGET mfq-tokenizer)
    target_sources(mfq-engine PRIVATE
        src/inference.cpp
        src/token_constraint.cpp
        src/tokenizer.cpp
    )
    target_include_directories(mfq-engine PRIVATE
        ${PROJECT_SOURCE_DIR}/components/tokenizer
        ${PROJECT_SOURCE_DIR}/components/tokenizer/src
    )
    target_link_libraries(mfq-engine PUBLIC mfq-tokenizer)
endif()
