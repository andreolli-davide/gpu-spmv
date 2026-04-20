add_library(common
    matrix_market.cpp
    sparse_matrix.cpp
    timer.cu
)

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(common PUBLIC OpenMP::OpenMP_CXX)
if(CUDAToolkit_FOUND)
    target_compile_definitions(common PUBLIC SPMV_CUDA_ENABLED=1)
    target_link_libraries(common PUBLIC CUDA::cudart)
endif()
