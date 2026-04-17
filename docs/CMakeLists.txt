cmake_minimum_required(VERSION 3.16)
project(docs)

find_package(Doxygen REQUIRED)

# Sphinx is installed in a venv — use the venv python explicitly
set(SPHINX_EXECUTABLE "${PROJECT_SOURCE_DIR}/.docs-venv/bin/sphinx-build")

# Doxygen configuration is processed from Doxyfile.in with @variables@
configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

# Generate Sphinx HTML from conf.py + index.rst
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sphinx-build/index.html
  COMMAND ${SPHINX_EXECUTABLE} -b html
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/sphinx-build
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/conf.py
           ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
           ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
           ${PROJECT_SOURCE_DIR}/src/common
           ${PROJECT_SOURCE_DIR}/src/cpu
           ${PROJECT_SOURCE_DIR}/examples
  COMMENT "Building HTML documentation with Sphinx"
)

add_custom_target(docs ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sphinx-build/index.html)