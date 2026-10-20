add_library(mfq-io STATIC
    file_range_reader.cpp
    async_range_reader.cpp
    file_range_reader.h
    async_range_reader.h
    read_span.h
)
target_include_directories(mfq-io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mfq-io PRIVATE mfq-host-parallel)
target_compile_features(mfq-io PUBLIC cxx_std_17)
set_target_properties(mfq-io PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-io PRIVATE /EHsc)
endif()
