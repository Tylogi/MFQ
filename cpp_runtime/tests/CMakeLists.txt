# Backend-independent native tests.

add_executable(mfq-cuda-tensor-view-test
    ${PROJECT_SOURCE_DIR}/backends/cuda/tests/mfq_cuda_tensor_view_test.cpp
)
target_include_directories(mfq-cuda-tensor-view-test PRIVATE
    ${PROJECT_SOURCE_DIR}/backends/cuda/include
)
target_compile_features(mfq-cuda-tensor-view-test PRIVATE cxx_std_20)
add_test(NAME mfq-cuda-tensor-view-test COMMAND mfq-cuda-tensor-view-test)

add_executable(mfq-native-tensor-test
    ${PROJECT_SOURCE_DIR}/backends/cuda/src/mfq_native_tensor.cpp
    ${PROJECT_SOURCE_DIR}/backends/cuda/tests/mfq_native_tensor_cpu_test_stubs.cpp
    ${PROJECT_SOURCE_DIR}/backends/cuda/tests/mfq_native_tensor_test.cpp
)
target_include_directories(mfq-native-tensor-test PRIVATE
    ${PROJECT_SOURCE_DIR}/backends/cuda/include
)
target_compile_features(mfq-native-tensor-test PRIVATE cxx_std_20)
add_test(NAME mfq-native-tensor-test COMMAND mfq-native-tensor-test)

add_executable(mfq-paged-kv-allocator-test
    mfq_paged_kv_allocator_test.cpp
)
target_link_libraries(mfq-paged-kv-allocator-test PRIVATE mfq-engine)
target_compile_features(mfq-paged-kv-allocator-test PRIVATE cxx_std_17)
add_test(NAME mfq-paged-kv-allocator-test COMMAND mfq-paged-kv-allocator-test)

add_executable(mfq-session-snapshot-cache-test
    mfq_session_snapshot_cache_test.cpp
)
target_link_libraries(mfq-session-snapshot-cache-test PRIVATE mfq-engine)
target_compile_features(mfq-session-snapshot-cache-test PRIVATE cxx_std_17)
add_test(NAME mfq-session-snapshot-cache-test
    COMMAND mfq-session-snapshot-cache-test)

function(mfq_add_core_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE mfq-core)
    target_compile_features(${target} PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endfunction()

add_executable(mfq-text-emitter-test mfq_text_emitter_test.cpp)
target_link_libraries(mfq-text-emitter-test PRIVATE mfq-engine)
target_compile_features(mfq-text-emitter-test PRIVATE cxx_std_17)
add_test(NAME mfq-text-emitter-test COMMAND mfq-text-emitter-test)

mfq_add_core_test(mfq-model-graph-test mfq_model_graph_test.cpp)
mfq_add_core_test(mfq-model-source-test model_source_test.cpp)
mfq_add_core_test(mfq-grid-vision-test grid_vision_test.cpp)
mfq_add_core_test(mfq-deepseek-v41-config-test deepseek_v41_config_test.cpp)
target_link_libraries(mfq-deepseek-v41-config-test PRIVATE mfq-models)
mfq_add_core_test(mfq-model-config-test model_config_test.cpp)
target_link_libraries(mfq-model-config-test PRIVATE mfq-models)
add_executable(mfq-sampling-test mfq_sampling_test.cpp)
target_link_libraries(mfq-sampling-test PRIVATE mfq-engine)
target_compile_features(mfq-sampling-test PRIVATE cxx_std_17)
add_test(NAME mfq-sampling-test COMMAND mfq-sampling-test)
add_executable(mfq-moe-cache-policy-test moe_cache_policy_test.cpp)
target_link_libraries(mfq-moe-cache-policy-test PRIVATE mfq-engine)
target_compile_features(mfq-moe-cache-policy-test PRIVATE cxx_std_17)
add_test(NAME mfq-moe-cache-policy-test COMMAND mfq-moe-cache-policy-test)
mfq_add_core_test(mfq-tensor-parallel-test tensor_parallel_test.cpp)

add_executable(mfq-cuda-model-plan-test
    ${PROJECT_SOURCE_DIR}/backends/cuda/tests/cuda_model_plan_test.cpp
)
target_include_directories(mfq-cuda-model-plan-test PRIVATE
    ${PROJECT_SOURCE_DIR}/backends/cuda/include
)
target_link_libraries(mfq-cuda-model-plan-test PRIVATE mfq-core)
target_compile_features(mfq-cuda-model-plan-test PRIVATE cxx_std_20)
add_test(NAME mfq-cuda-model-plan-test COMMAND mfq-cuda-model-plan-test)

add_executable(mfq-inference-test inference_test.cpp)
target_link_libraries(mfq-inference-test PRIVATE mfq-engine)
add_test(NAME mfq-inference-test COMMAND mfq-inference-test)

add_executable(mfq-mtp-policy-test mtp_policy_test.cpp)
target_link_libraries(mfq-mtp-policy-test PRIVATE mfq-engine)
target_compile_features(mfq-mtp-policy-test PRIVATE cxx_std_20)
add_test(NAME mfq-mtp-policy-test COMMAND mfq-mtp-policy-test)

add_executable(mfq-moe-cache-profile-test moe_cache_profile_test.cpp)
target_link_libraries(mfq-moe-cache-profile-test PRIVATE mfq-core mfq-json)
target_compile_features(mfq-moe-cache-profile-test PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(mfq-moe-cache-profile-test PRIVATE /utf-8 /EHsc)
endif()
add_test(NAME mfq-moe-cache-profile-test COMMAND mfq-moe-cache-profile-test)

if(TARGET mfq-paged-prefix-cache)
    add_executable(mfq-paged-prefix-cache-test
        mfq_paged_prefix_cache_test.cpp
    )
    target_link_libraries(mfq-paged-prefix-cache-test PRIVATE
        mfq-paged-prefix-cache
    )
    target_compile_features(mfq-paged-prefix-cache-test PRIVATE cxx_std_17)
    add_test(NAME mfq-paged-prefix-cache-test COMMAND mfq-paged-prefix-cache-test)
endif()

if(TARGET mfq-runtime-communication)
    mfq_add_core_test(mfq-runtime-transport-test mfq_runtime_transport_test.cpp)
    target_link_libraries(mfq-runtime-transport-test PRIVATE mfq-runtime-communication)

    add_executable(mfq-runtime-profile-test mfq_runtime_profile_test.cpp)
    target_link_libraries(mfq-runtime-profile-test PRIVATE mfq-runtime-communication)
    target_compile_features(mfq-runtime-profile-test PRIVATE cxx_std_17)
    add_test(NAME mfq-runtime-profile-test COMMAND mfq-runtime-profile-test)
endif()
