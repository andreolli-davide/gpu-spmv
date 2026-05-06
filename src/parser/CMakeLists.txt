# MTX Parser library
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  cuda_add_library(mtx_parser STATIC
    mtx_parser.cpp
    mtx_parser_gpu.cu
  )
else()
  # Build without GPU code
  add_library(mtx_parser STATIC mtx_parser.cpp)
endif()
target_include_directories(mtx_parser PUBLIC ${CMAKE_SOURCE_DIR}/include)
