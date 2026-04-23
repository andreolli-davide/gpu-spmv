add_library(spmv_gpu_v1 STATIC
    spmv_gpu_v1.cu
)

target_include_directories(spmv_gpu_v1 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(spmv_gpu_v1 PUBLIC
    common
    ${CUDAToolkit_LIBRARIES}
)
