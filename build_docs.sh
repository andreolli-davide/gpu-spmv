#!/bin/bash
# Script to build SpMV GPU Optimization documentation
# Usage: ./build_docs.sh

set -e  # Exit on any error

echo "Building SpMV GPU Optimization documentation..."

# Ensure we're in the project root
cd "$(dirname "$0")"

# Create temporary directory for processed Doxyfile
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Process Doxyfile.in to replace CMake variables
echo "Processing Doxyfile configuration..."
sed -e "s|@PROJECT_SOURCE_DIR@|$(pwd)|g" \
    -e "s|@CMAKE_CURRENT_BINARY_DIR@|$TEMP_DIR|g" \
    -e "s|@PROJECT_VERSION@|1.0|g" \
    docs/Doxyfile.in > "$TEMP_DIR/Doxyfile"

# Run Doxygen to generate XML from source comments
echo "Generating Doxygen XML..."
doxygen "$TEMP_DIR/Doxyfile"

# Build HTML documentation with Sphinx (using venv)
echo "Building HTML documentation with Sphinx..."
./.docs-venv/bin/sphinx-build -b html docs/ docs/_build/html

echo "Documentation build complete!"
echo "Open docs/_build/html/index.html in your browser to view."