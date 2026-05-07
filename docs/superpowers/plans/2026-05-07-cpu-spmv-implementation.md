# CPU SpMV Reference Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone `spmv_cpu` static library exposing `spmv_csr_cpu()` and `compare_vectors()` as a correctness reference for GPU SpMV implementations.

**Architecture:** Pure C++ static library in `src/spmv_cpu/`, header in `include/spmv_cpu/spmv_cpu.h`, no CUDA dependency. A test executable `test_spmv_cpu` exercises two input vectors (all-ones and random seed-42) with determinism checks. GPU test code will link against this library for in-memory result comparison.

**Tech Stack:** C++17, CMake 3.18+, existing `mtx_parser` static library (for `MtxCsr`, `parse_mtx_csr()`).

---

### Task 1: Create the public header

**Files:**
- Create: `include/spmv_cpu/spmv_cpu.h`

- [ ] **Step 1: Write the header**

```cpp
// include/spmv_cpu/spmv_cpu.h
#pragma once
#include "parser/mtx_parser.h"

/// y = A * x using CSR layout.
/// Caller allocates x (size A.num_cols) and y (size A.num_rows).
void spmv_csr_cpu(const MtxCsr& A, const double* x, double* y);

/// Compare two length-n vectors element-wise using absolute difference.
/// Prints each element whose |a[i] - b[i]| exceeds tolerance.
/// Returns the maximum absolute difference found.
double compare_vectors(const double* a, const double* b, int n, double tolerance = 1e-6);
```

- [ ] **Step 2: Commit**

```bash
git add include/spmv_cpu/spmv_cpu.h
git commit -m "feat(spmv_cpu): add public header with spmv_csr_cpu and compare_vectors"
```

---

### Task 2: Implement the library

**Files:**
- Create: `src/spmv_cpu/spmv_cpu.cpp`

- [ ] **Step 1: Write the implementation**

```cpp
// src/spmv_cpu/spmv_cpu.cpp
#include "spmv_cpu/spmv_cpu.h"
#include <cmath>
#include <cstdio>

void spmv_csr_cpu(const MtxCsr& A, const double* x, double* y) {
    for (int i = 0; i < A.num_rows; ++i) {
        double acc = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            acc += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = acc;
    }
}

double compare_vectors(const double* a, const double* b, int n, double tolerance) {
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = std::fabs(a[i] - b[i]);
        if (diff > tolerance) {
            std::printf("  mismatch at [%d]: a=%.17g b=%.17g diff=%.3e\n", i, a[i], b[i], diff);
        }
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/spmv_cpu/spmv_cpu.cpp
git commit -m "feat(spmv_cpu): implement spmv_csr_cpu and compare_vectors"
```

---

### Task 3: Write the CMakeLists for the module

**Files:**
- Create: `src/spmv_cpu/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write `src/spmv_cpu/CMakeLists.txt`**

```cmake
# spmv_cpu static library
add_library(spmv_cpu STATIC spmv_cpu.cpp)
target_include_directories(spmv_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spmv_cpu PUBLIC mtx_parser)

# test executable
add_executable(test_spmv_cpu test_spmv_cpu.cpp)
target_link_libraries(test_spmv_cpu spmv_cpu mtx_parser)
```

- [ ] **Step 2: Wire into `src/CMakeLists.txt`**

Add `add_subdirectory(spmv_cpu)` after the existing `add_subdirectory(parser)` line:

```cmake
# Build parser library
add_subdirectory(parser)

# Build CPU SpMV reference library
add_subdirectory(spmv_cpu)
```

- [ ] **Step 3: Commit**

```bash
git add src/spmv_cpu/CMakeLists.txt src/CMakeLists.txt
git commit -m "build(spmv_cpu): add CMake target for spmv_cpu library and test executable"
```

---

### Task 4: Write the test executable

**Files:**
- Create: `src/spmv_cpu/test_spmv_cpu.cpp`

Note: `test_spmv_cpu.cpp` is what the `CMakeLists.txt` already refers to. Write it now.

- [ ] **Step 1: Write the test**

```cpp
// src/spmv_cpu/test_spmv_cpu.cpp
#include "spmv_cpu/spmv_cpu.h"
#include "parser/mtx_parser.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::vector<double> make_ones(int n) {
    return std::vector<double>(n, 1.0);
}

static std::vector<double> make_random(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> v(n);
    for (auto& val : v) val = dist(rng);
    return v;
}

// Run SpMV and return result vector.
static std::vector<double> run(const MtxCsr& A, const std::vector<double>& x) {
    std::vector<double> y(A.num_rows, 0.0);
    spmv_csr_cpu(A, x.data(), y.data());
    return y;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: test_spmv_cpu <matrix.mtx>\n");
        return 1;
    }

    MtxCsr A = parse_mtx_csr(argv[1]);
    std::printf("Matrix: %d rows, %d cols, %d nnz\n", A.num_rows, A.num_cols, A.num_nonzeros);

    bool all_pass = true;

    // Stage 1: all-ones vector
    {
        auto x = make_ones(A.num_cols);
        auto y1 = run(A, x);
        auto y2 = run(A, x);
        double max_diff = compare_vectors(y1.data(), y2.data(), A.num_rows, 0.0);
        bool pass = (max_diff == 0.0);
        std::printf("[%s] Stage 1 (all-ones determinism): max_diff=%.3e\n",
                    pass ? "PASS" : "FAIL", max_diff);
        all_pass &= pass;
    }

    // Stage 2: random vector (seed 42)
    {
        auto x = make_random(A.num_cols, 42);
        auto y1 = run(A, x);
        auto y2 = run(A, x);
        double max_diff = compare_vectors(y1.data(), y2.data(), A.num_rows, 0.0);
        bool pass = (max_diff == 0.0);
        std::printf("[%s] Stage 2 (random-seed-42 determinism): max_diff=%.3e\n",
                    pass ? "PASS" : "FAIL", max_diff);
        all_pass &= pass;
    }

    return all_pass ? 0 : 1;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/spmv_cpu/test_spmv_cpu.cpp
git commit -m "feat(spmv_cpu): add test executable with all-ones and random-seed-42 determinism checks"
```

---

### Task 5: Build and verify

**Files:** none created — verification only

- [ ] **Step 1: Configure and build**

```bash
mkdir -p build && cd build && cmake .. && cmake --build . --target test_spmv_cpu 2>&1
```

Expected: build succeeds with no errors. `test_spmv_cpu` binary appears in `build/src/spmv_cpu/`.

- [ ] **Step 2: Run against a small matrix**

Pick any `.mtx` file from `matrices/` (e.g. the smallest available):

```bash
./build/src/spmv_cpu/test_spmv_cpu matrices/<any>.mtx
```

Expected output format:
```
Matrix: NNNN rows, MMMM cols, KKKK nnz
[PASS] Stage 1 (all-ones determinism): max_diff=0.000e+00
[PASS] Stage 2 (random-seed-42 determinism): max_diff=0.000e+00
```

Exit code: 0

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "chore(spmv_cpu): verify build and test pass"
```

If there are no leftover changes, skip this commit — the build itself produces no tracked files.

---

## File Map Summary

| Path | Action | Purpose |
|------|--------|---------|
| `include/spmv_cpu/spmv_cpu.h` | Create | Public API: `spmv_csr_cpu`, `compare_vectors` |
| `src/spmv_cpu/spmv_cpu.cpp` | Create | Row-by-row CSR SpMV + vector comparison |
| `src/spmv_cpu/test_spmv_cpu.cpp` | Create | Determinism tests: all-ones + random seed-42 |
| `src/spmv_cpu/CMakeLists.txt` | Create | `spmv_cpu` static lib + `test_spmv_cpu` executable |
| `src/CMakeLists.txt` | Modify | Add `add_subdirectory(spmv_cpu)` |
