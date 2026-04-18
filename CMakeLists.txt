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
    # Re-find after enable_language to populate variables
    find_package(CUDAToolkit REQUIRED)
else()
    message(STATUS "CUDA Toolkit not found — GPU targets will be skipped")
    enable_language(CUDA)
endif()

# --------------------------------------------------------------------------
# OpenMP (optional — required for CPU parallel baseline)
# --------------------------------------------------------------------------
# Try the standard CMake module first.
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND)
    # ------------------------------------------------------------------
    # Workaround for macOS with AppleClang/Homebrew LLVM:
    # Homebrew installs libomp as a dylib without providing the CMake
    # FindOpenMP module with IMPORTED targets.  We create the target
    # manually so target_link_libraries(... OpenMP::OpenMP_CXX) works.
    # ------------------------------------------------------------------
    find_library(OPENMP_LIB NAMES omp PATHS /opt/homebrew/lib NO_DEFAULT_PATH)
    if(OPENMP_LIB)
        # Find omp.h in the Homebrew LLVM include path
        file(GLOB _llvm_omp_glob "/opt/homebrew/Cellar/llvm/*/lib/clang/*/include/omp.h")
        if(_llvm_omp_glob)
            list(GET _llvm_omp_glob 0 OPENMP_INCLUDE_DIR)
            get_filename_component(OPENMP_INCLUDE_DIR "${OPENMP_INCLUDE_DIR}" DIRECTORY)
        endif()

        if(OPENMP_INCLUDE_DIR AND OPENMP_LIB)
            message(STATUS "OpenMP found via Homebrew libomp: ${OPENMP_LIB}")
            add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${OPENMP_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${OPENMP_LIB}"
            )
            set(OpenMP_CXX_FOUND TRUE)
        endif()
    endif()
endif()

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
# Documentation (optional)
# --------------------------------------------------------------------------
option(BUILD_DOCS "Build documentation with Doxygen/Sphinx" OFF)
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    add_subdirectory(docs)
endif()

# --------------------------------------------------------------------------
# Subdirectories
# --------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
