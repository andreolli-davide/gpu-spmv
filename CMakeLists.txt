cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SpMV_GPU VERSION 1.0 LANGUAGES CXX)

# --------------------------------------------------------------------------
# Build type
# --------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Defaulting to Release build")
    set(CMAKE_BUILD_TYPE Release)
endif()

# --------------------------------------------------------------------------
# Compiler options
# --------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g)
else()
    add_compile_options(-O3)
endif()

# --------------------------------------------------------------------------
# CUDA Toolkit (optional — enable CUDA language only when found)
# --------------------------------------------------------------------------
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA Runtime version: ${CUDAToolkit_VERSION}")
    enable_language(CUDA)
else()
    message(STATUS "CUDA Toolkit not found — GPU targets will be skipped")
endif()

# --------------------------------------------------------------------------
# OpenMP (optional)
# --------------------------------------------------------------------------
find_package(OpenMP OPTIONAL COMPONENT CXX)

# --------------------------------------------------------------------------
# Matrix directory (required)
# --------------------------------------------------------------------------
option(MATRIX_DIR "Directory containing Matrix Market (.mtx) files" "")

if(NOT MATRIX_DIR)
    message(FATAL_ERROR "MATRIX_DIR must be set via -DMATRIX_DIR=/path/to/matrices")
endif()

if(NOT EXISTS "${MATRIX_DIR}")
    message(FATAL_ERROR "MATRIX_DIR does not exist: ${MATRIX_DIR}")
endif()

message(STATUS "Matrix directory: ${MATRIX_DIR}")

# --------------------------------------------------------------------------
# Subdirectories
# --------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
