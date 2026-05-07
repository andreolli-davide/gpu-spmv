# GPU-SPMV-2

Sparse Matrix-Vector Multiplication (SpMV) on GPU — research project.

## Overview

This project implements and benchmarks GPU kernels for sparse matrix-vector multiplication,
focused on CSR and related formats with CUDA.

## Project Structure

```
gpu-spmv-2/
├── src/                  # Source code
├── include/              # Headers
├── data/                 # Matrix market files, test data
├── scripts/              # Build and run scripts
├── resources/            # Documentation (matrix mappings, dataset guides, etc.)
├── build/                # CMake build output (gitignored)
├── outputs/              # Test outputs and logs (gitignored)
├── docs/                 # Development notes (plans, specs, design docs)
├── CLAUDE.md             # This file
└── AGENTS.md             # Agent instructions
```

## Documentation

**All project documentation is in `resources/`** for easy discovery and context injection.

### Core References (Start Here)

- `resources/README.md` — Complete overview and quick start guide
- `resources/PAPER_TO_SUITESPARSE_GUIDE.md` — Quick lookup by paper name with download instructions
- `resources/MATRIX_MAPPING.md` — Comprehensive reference with all 29 matrices and metadata
- `resources/DATASET_SELECTION.md` — Subset of 10 matrices selected for the project with rationale

### Additional Resources

- `resources/MATRIX_DETAILED_PROPERTIES.md` — Detailed matrix properties and characteristics
- `resources/matrix_download_links.md` — Quick reference for matrix download links
- `resources/cited_matrices_in_benchmarks.md` — Original extraction from papers with citations
- `resources/MATRIX_RESOURCES.md` — Workflows, examples, and troubleshooting guide
- `docs/superpowers/` — Development notes, design docs, and implementation plans

### Context Injection

When injecting context into Claude prompts, include the `resources/` folder to provide comprehensive documentation without token overhead:

```bash
# All documentation is consolidated and cleaned
resources/  # Include this for complete matrix documentation
```

## Scripts

All utility scripts are in `scripts/` for matrix downloading and data fetching:

- `scripts/download_all_29_matrices.sh` — Download all 29 GPU SpMV benchmark matrices from SuiteSparse Collection (parallel with fallback retry)
- `scripts/fetch_matrix_download_links.py` — Fetch actual download links by parsing SuiteSparse collection pages
- `scripts/fetch_matrix_details.py` — Fetch detailed matrix properties (structure, symmetry, etc.)

## Building

```bash
# Create build directory and configure
mkdir -p build && cd build
cmake ..

# Compile
cmake --build .
```

## Cache Behavior Verification

Use **Valgrind/Cachegrind** to analyze CPU cache performance of host code (memory accesses, cache misses, branch prediction):

```bash
# Run cachegrind on a CPU-bound program
valgrind --tool=cachegrind ./build/src/<target> [args]

# View results (cached from previous run)
cg_annotate cachegrind.out.<pid>

# Compare two runs
cg_diff vgout1 vgout2
```

**What to check:**
- **D1 miss rate**: Data cache L1 misses — high misses indicate poor spatial locality
- **LL miss rate**: Last-level cache misses — indicates working set too large for L3
- **I1 miss rate**: Instruction cache — usually a problem with tight loops

**Typical SpMV concerns:**
- Row-index traversal: sequential → good L1 cache utilization
- Value access: strided → depends on row width
- Re-using row offsets within a thread block

**Note:** Cachegrind simulates cache, not GPU. GPU memory access patterns are analyzed with `cuda-memcheck` or `compute-sanitizer`.

See AGENTS.md for remote execution instructions.