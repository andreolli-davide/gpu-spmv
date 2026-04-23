cmake_minimum_required(VERSION 3.18)

add_executable(test_spmv_cpu
    test_spmv_cpu.cpp
)

target_include_directories(test_spmv_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/cpu
)

target_link_libraries(test_spmv_cpu PRIVATE common spmv_cpu)

# Copy test binary into build root for convenience
add_custom_command(TARGET test_spmv_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:test_spmv_cpu>
        ${CMAKE_BINARY_DIR}/test_spmv_cpu
)

if(CUDAToolkit_FOUND)
    add_executable(test_spmv_gpu
        test_spmv_gpu.cpp
    )

    target_include_directories(test_spmv_gpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src/common
        ${PROJECT_SOURCE_DIR}/src/cpu
        ${PROJECT_SOURCE_DIR}/src/gpu
    )

    target_link_libraries(test_spmv_gpu PRIVATE common spmv_cpu spmv_gpu_v1)

    add_custom_command(TARGET test_spmv_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:test_spmv_gpu>
            ${CMAKE_BINARY_DIR}/test_spmv_gpu
    )
endif()
