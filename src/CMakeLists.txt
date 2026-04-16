add_subdirectory(common)
add_subdirectory(cpu)

# GPU targets are built only when CUDA Toolkit is available (Phase 2+)
if(CUDAToolkit_FOUND)
    add_subdirectory(gpu)
else()
    message(STATUS "Skipping gpu/ — CUDA Toolkit not found")
endif()
