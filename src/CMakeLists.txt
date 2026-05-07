# Build parser library
add_subdirectory(parser)

# Build spmv_cpu library
add_subdirectory(spmv_cpu)

# Build hellocuda (only if CUDA is available)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  add_executable(hellocuda hellocuda.cu)
  target_link_libraries(hellocuda CUDA::cudart)
else()
  # Create a dummy target so the project can still build
  add_custom_target(hellocuda ALL
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA not available - hellocuda skipped"
    COMMENT "hellocuda requires CUDA toolkit"
  )
endif()
