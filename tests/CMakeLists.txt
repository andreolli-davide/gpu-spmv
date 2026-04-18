cmake_minimum_required(VERSION 3.18)

# -----------------------------------------------------------------------------
# CPU SpMV correctness test
# -----------------------------------------------------------------------------
add_executable(test_spmv_cpu
    test_spmv_cpu.cpp
)

target_include_directories(test_spmv_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/cpu
)

target_link_libraries(test_spmv_cpu PRIVATE common spmv_cpu)

add_custom_command(TARGET test_spmv_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:test_spmv_cpu>
        ${CMAKE_BINARY_DIR}/test_spmv_cpu
)

# -----------------------------------------------------------------------------
# GPU SpMV correctness test (requires CUDA)
# -----------------------------------------------------------------------------
if(CUDAToolkit_FOUND)
    enable_language(CUDA)

    add_executable(test_spmv_gpu_v1
        gpu/test_spmv_gpu_v1.cpp
    )

    target_include_directories(test_spmv_gpu_v1 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src/common
        ${PROJECT_SOURCE_DIR}/src/cpu
        ${PROJECT_SOURCE_DIR}/src/gpu
    )

    target_link_libraries(test_spmv_gpu_v1 PRIVATE
        common
        spmv_cpu
        gpu_lib
    )

    target_compile_options(test_spmv_gpu_v1 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
        >
    )

    add_custom_command(TARGET test_spmv_gpu_v1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:test_spmv_gpu_v1>
            ${CMAKE_BINARY_DIR}/test_spmv_gpu_v1
    )

    # -----------------------------------------------------------------------------
    # GPU SpMV benchmark executable
    # -----------------------------------------------------------------------------
    add_executable(benchmark_spmv_gpu
        gpu/benchmark_spmv_gpu.cpp
    )

    target_include_directories(benchmark_spmv_gpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src/common
        ${PROJECT_SOURCE_DIR}/src/cpu
        ${PROJECT_SOURCE_DIR}/src/gpu
    )

    target_link_libraries(benchmark_spmv_gpu PRIVATE
        common
        spmv_cpu
        gpu_lib
    )

    target_compile_options(benchmark_spmv_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
        >
    )

    add_custom_command(TARGET benchmark_spmv_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:benchmark_spmv_gpu>
            ${CMAKE_BINARY_DIR}/benchmark_spmv_gpu
    )
endif()