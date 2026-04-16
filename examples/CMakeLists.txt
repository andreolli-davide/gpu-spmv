cmake_minimum_required(VERSION 3.18)

add_executable(example_spmv_basics example_spmv_basics.cpp)

target_include_directories(example_spmv_basics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/cpu
)

target_link_libraries(example_spmv_basics PRIVATE common spmv_cpu)

# Copy binary to build root for convenience
add_custom_command(TARGET example_spmv_basics POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:example_spmv_basics>
        ${CMAKE_BINARY_DIR}/example_spmv_basics
)
