# spmv_cpu static library
add_library(spmv_cpu STATIC spmv_cpu.cpp)
target_include_directories(spmv_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spmv_cpu PUBLIC mtx_parser)

# test executable
add_executable(test_spmv_cpu test_spmv_cpu.cpp)
target_link_libraries(test_spmv_cpu spmv_cpu mtx_parser)
