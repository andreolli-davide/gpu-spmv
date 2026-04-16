add_library(common
    matrix_market.cpp
    sparse_matrix.cpp
    timer.cpp
)

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(common PUBLIC OpenMP::OpenMP_CXX)
if(CUDAToolkit_FOUND)
    target_link_libraries(common PUBLIC CUDAToolkit::CUDAToolkit)
endif()
