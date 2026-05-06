cuda_add_library(mtx_parser STATIC
  mtx_parser.cpp
  mtx_parser_gpu.cu
)
target_include_directories(mtx_parser PUBLIC ${CMAKE_SOURCE_DIR}/include)