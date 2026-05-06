cmake_minimum_required(VERSION 3.18)
project(gpu_spmv VERSION 1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Enable CUDA architecture (Volta V100 = 70, RTX 2080 = 75)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Project sources
add_subdirectory(src)