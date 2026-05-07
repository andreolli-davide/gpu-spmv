cmake_minimum_required(VERSION 3.18)

project(gpu_spmv VERSION 1.0 LANGUAGES C CXX)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
endif()

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Project sources
add_subdirectory(src)