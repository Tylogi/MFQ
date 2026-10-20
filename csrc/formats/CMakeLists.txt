add_library(mfq-formats INTERFACE)
target_include_directories(mfq-formats INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(mfq-quant-readers STATIC
    quant/nint_rows.cpp
    quant/nvq_rows.cpp
    quant/mfe_quant_expert_store.cpp
)
target_link_libraries(mfq-quant-readers PUBLIC mfq-formats mfq-io PRIVATE mfq-host-parallel)
target_compile_features(mfq-quant-readers PUBLIC cxx_std_17)
set_target_properties(mfq-quant-readers PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-quant-readers PRIVATE /EHsc)
endif()
