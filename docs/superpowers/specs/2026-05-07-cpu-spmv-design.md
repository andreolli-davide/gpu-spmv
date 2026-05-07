# CPU SpMV Reference Implementation

## Purpose

Standalone CPU implementation of sparse matrix-vector multiplication (SpMV) in CSR format. Serves as correctness reference for future GPU kernel implementations. GPU test code calls this directly and compares results in-memory.

## Module: `src/spmv_cpu/`

Static library (`spmv_cpu`) + test executable (`test_spmv_cpu`).

### Files

- `spmv_cpu.h` — public API
- `spmv_cpu.cpp` — implementation
- `test_spmv_cpu.cpp` — standalone test
- `CMakeLists.txt` — build config

## API

```cpp
#pragma once
#include "parser/mtx_parser.h"

// y = A * x, where A is CSR. Caller allocates x (size num_cols) and y (size num_rows).
void spmv_csr_cpu(const MtxCsr& A, const double* x, double* y);

// Compare two vectors element-wise. Returns max absolute difference.
// Prints per-element errors exceeding tolerance.
double compare_vectors(const double* a, const double* b, int n, double tolerance = 1e-6);
```

## Algorithm

Standard row-by-row CSR SpMV:

```
for each row i in [0, num_rows):
    y[i] = 0
    for j in [row_ptr[i], row_ptr[i+1]):
        y[i] += values[j] * x[col_indices[j]]
```

No parallelism, no SIMD. Simplicity and correctness over performance.

## Test Executable (`test_spmv_cpu`)

Takes matrix file path as CLI argument. Uses existing parser library (`parse_mtx_csr()`) to load matrix. If input is COO, parser's `coo_to_csr()` handles conversion.

### Test sequence

1. **All-ones vector**: `x[i] = 1.0` for all i. Run SpMV, store result.
2. **Random vector**: `x[i]` from `std::mt19937` with seed 42, uniform distribution [0, 1). Run SpMV, store result.
3. **Determinism check**: Run both inputs again. Compare with tolerance 0. Results must be bit-identical.

### Output

Prints pass/fail per test stage. Returns 0 on all-pass, 1 on any failure.

## Build Integration

### `src/spmv_cpu/CMakeLists.txt`

- `spmv_cpu`: static library from `spmv_cpu.cpp`
  - Links against: `mtx_parser` (for MtxCsr struct)
  - Include dirs: project `include/`
- `test_spmv_cpu`: executable from `test_spmv_cpu.cpp`
  - Links against: `spmv_cpu`, `mtx_parser`

### Root `CMakeLists.txt`

Add `add_subdirectory(src/spmv_cpu)`.

## Comparison Tolerance

Absolute difference: 1e-6. Matches existing project convention from `test_parser.cu`.

## Dependencies

- Existing `mtx_parser` library (MtxCsr, parse_mtx_csr, coo_to_csr)
- Standard C++ only — no CUDA dependency
