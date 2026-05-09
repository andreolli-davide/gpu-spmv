if(NOT CUDAToolkit_FOUND OR NOT CMAKE_CUDA_COMPILER)
  message(STATUS "spmv_gpu: CUDA not available, skipping")
  return()
endif()

# Static library: CSR-Scalar kernel
add_library(spmv_gpu STATIC csr_scalar.cu)
set_target_properties(spmv_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(spmv_gpu PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spmv_gpu PUBLIC mtx_parser CUDA::cudart)

# Validation binary
add_executable(validate_csr_scalar validate_csr_scalar.cu)
target_link_libraries(validate_csr_scalar spmv_gpu spmv_cpu CUDA::cudart)
