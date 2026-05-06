# MTX Parser library
if(CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
  add_library(mtx_parser STATIC mtx_parser.cpp mtx_parser_gpu.cu)
  set_target_properties(mtx_parser PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(mtx_parser CUDA::cudart)
else()
  # Build without GPU code
  add_library(mtx_parser STATIC mtx_parser.cpp)
endif()
target_include_directories(mtx_parser PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Test executable
if(CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
  add_executable(test_parser test_parser.cu)
  set_target_properties(test_parser PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(test_parser mtx_parser CUDA::cudart)
else()
  add_executable(test_parser test_parser.cu)
  target_link_libraries(test_parser mtx_parser)
endif()

# Small diagnostic executable (removed - used for debugging)
# Bone010 diagnostic executable (removed - used for debugging)
# Webbase diagnostic executable (removed - used for debugging)
