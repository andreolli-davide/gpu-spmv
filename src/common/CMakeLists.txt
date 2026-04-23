add_library(common
    matrix_market.cpp
    sparse_matrix.cpp
    timer.cpp
)

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(common PUBLIC OpenMP::OpenMP_CXX)
if(CUDAToolkit_FOUND)
    target_include_directories(common SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(common PUBLIC /opt/shares/cuda/software/CUDA/12.5.0/lib64/libcudart.so)
endif()
