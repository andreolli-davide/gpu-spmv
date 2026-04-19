# GPU SpMV library — Phase 2
# Requires CUDA Toolkit to be found at parent scope

if(NOT CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit not found — GPU targets will be skipped")
    return()
endif()

# CUDA architectures for Ampere (sm_80, sm_86, sm_89)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -arch=sm_80")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# GPU library — compiled from .cu source files
# Contains GPU SpMV kernels (v1, v2, ELL, CSR-Adaptive) and memory utility functions
add_library(gpu_lib STATIC
    spmv_gpu_v1.cu
    spmv_gpu_v2.cu
    spmv_ell.cu
    spmv_csr_adaptive.cu
    gpu_persistent_buffers.cpp
    gpu_utils.cu
)

target_include_directories(gpu_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIR}
)

target_link_libraries(gpu_lib PUBLIC
    common
    CUDA::cudart_static
)

target_compile_options(gpu_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_80>
)

# Convenience INTERFACE library for header-only access
# Tests can link to this to get include directories
add_library(gpu_utils INTERFACE)
target_include_directories(gpu_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIR}
)
target_link_libraries(gpu_utils INTERFACE gpu_lib)

# Phase 2 message
message(STATUS "GPU SpMV library configured for Ampere (sm_80, sm_86, sm_89)")
