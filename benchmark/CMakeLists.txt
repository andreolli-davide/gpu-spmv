add_executable(benchmark_spmv benchmark_spmv.cpp)

target_include_directories(benchmark_spmv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
)

target_link_libraries(benchmark_spmv PRIVATE
    common
    spmv_cpu
    spmv_gpu_v1
)
