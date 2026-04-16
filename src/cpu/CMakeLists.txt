add_executable(spmv_cpu
    spmv_cpu.cpp
)

target_include_directories(spmv_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(spmv_cpu PRIVATE common)
