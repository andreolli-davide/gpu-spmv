add_subdirectory(parser)
add_executable(hellocuda hellocuda.cu)
target_link_libraries(hellocuda CUDA::cudart)