cmake_minimum_required(VERSION 3.18)

add_executable(test_spmv_cpu
    test_spmv_cpu.cpp
)

target_include_directories(test_spmv_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/common
)

target_link_libraries(test_spmv_cpu PRIVATE common)

# Copy test binary into build root for convenience
add_custom_command(TARGET test_spmv_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:test_spmv_cpu>
        ${CMAKE_BINARY_DIR}/test_spmv_cpu
)
