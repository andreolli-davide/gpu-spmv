add_library(spmv_cpu STATIC
    spmv_cpu.cpp
)

target_include_directories(spmv_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(spmv_cpu PUBLIC common)
