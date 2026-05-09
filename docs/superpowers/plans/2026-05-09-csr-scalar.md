# CSR-Scalar GPU SpMV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the CSR-Scalar CUDA kernel for SpMV (one thread per row) and a standalone validation binary that compares GPU output against the CPU reference.

**Architecture:** New `spmv_gpu` module under `src/spmv_gpu/` mirrors the `spmv_cpu` layout — a static library (`csr_scalar.cu`) exposing a single host wrapper, plus a standalone `validate_csr_scalar` binary that links both `spmv_gpu` and `spmv_cpu`. The kernel uses `float` throughout, consistent with the project-wide float migration.

**Tech Stack:** CUDA C++ (float), CMake 3.18+, existing `mtx_parser` + `spmv_cpu` libraries.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `include/spmv_gpu/csr_scalar.cuh` | Public header — `spmv_csr_scalar()` declaration |
| Create | `src/spmv_gpu/csr_scalar.cu` | `__global__` kernel + host wrapper |
| Create | `src/spmv_gpu/validate_csr_scalar.cu` | Standalone validation binary |
| Create | `src/spmv_gpu/CMakeLists.txt` | Build rules for lib + binary |
| Modify | `src/CMakeLists.txt` | Add `add_subdirectory(spmv_gpu)` inside CUDA guard |

---

### Task 1: Create the public header

**Files:**
- Create: `include/spmv_gpu/csr_scalar.cuh`

- [ ] **Step 1: Create the header file**

```cpp
// include/spmv_gpu/csr_scalar.cuh
#pragma once
#include "parser/mtx_parser.h"

/// GPU SpMV: y = A*x using CSR-Scalar (one CUDA thread per matrix row).
/// d_x must be a device pointer of size A.num_cols.
/// d_y must be a device pointer of size A.num_rows.
/// Caller is responsible for allocation, H2D transfer of d_x, and D2H
/// transfer of d_y after the call.
void spmv_csr_scalar(const DeviceMatrix& A, const float* d_x, float* d_y);
```

- [ ] **Step 2: Commit**

```bash
git add include/spmv_gpu/csr_scalar.cuh
git commit -m "feat(spmv_gpu): add CSR-Scalar public header"
```

---

### Task 2: Implement the kernel and host wrapper

**Files:**
- Create: `src/spmv_gpu/csr_scalar.cu`

- [ ] **Step 1: Write the kernel file**

```cuda
// src/spmv_gpu/csr_scalar.cu
#include "spmv_gpu/csr_scalar.cuh"
#include <cuda_runtime.h>
#include <cstdint>

/// CSR-Scalar kernel: one thread per matrix row.
/// Each thread independently accumulates its row's dot product.
/// Load imbalance is expected on irregular matrices — this is intentional
/// (it's the baseline to beat with CSR-Vector / warp-level kernels).
__global__
static void spmv_csr_scalar_kernel(
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_indices,
    const float*   __restrict__ values,
    int32_t        num_rows,
    const float*   __restrict__ d_x,
    float*                      d_y)
{
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float sum = 0.0f;
    for (int32_t i = row_ptr[row]; i < row_ptr[row + 1]; ++i)
        sum += values[i] * d_x[col_indices[i]];
    d_y[row] = sum;
}

void spmv_csr_scalar(const DeviceMatrix& A, const float* d_x, float* d_y) {
    constexpr int32_t block_size = 256;
    int32_t num_blocks = (A.num_rows + block_size - 1) / block_size;
    spmv_csr_scalar_kernel<<<num_blocks, block_size>>>(
        A.d_row_ptr, A.d_col_indices, A.d_values,
        A.num_rows, d_x, d_y);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/spmv_gpu/csr_scalar.cu
git commit -m "feat(spmv_gpu): implement CSR-Scalar kernel"
```

---

### Task 3: Write the CMakeLists for spmv_gpu

**Files:**
- Create: `src/spmv_gpu/CMakeLists.txt`

- [ ] **Step 1: Write the CMakeLists**

```cmake
# src/spmv_gpu/CMakeLists.txt
if(NOT CUDAToolkit_FOUND OR NOT CMAKE_CUDA_COMPILER)
  message(STATUS "spmv_gpu: CUDA not available, skipping")
  return()
endif()

# Static library: CSR-Scalar kernel
add_library(spmv_gpu STATIC csr_scalar.cu)
set_target_properties(spmv_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(spmv_gpu PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spmv_gpu PUBLIC mtx_parser CUDA::cudart)

# Validation binary
add_executable(validate_csr_scalar validate_csr_scalar.cu)
set_target_properties(validate_csr_scalar PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(validate_csr_scalar spmv_gpu spmv_cpu)
```

- [ ] **Step 2: Wire into src/CMakeLists.txt**

Open `src/CMakeLists.txt`. The current file:

```cmake
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
```

Add `add_subdirectory(spmv_gpu)` inside the `if(CUDAToolkit_FOUND)` block, after the parser and spmv_cpu lines, before the hellocuda block:

```cmake
# Build parser library
add_subdirectory(parser)

# Build spmv_cpu library
add_subdirectory(spmv_cpu)

# GPU kernels and validation (only if CUDA is available)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  add_subdirectory(spmv_gpu)
  add_executable(hellocuda hellocuda.cu)
  target_link_libraries(hellocuda CUDA::cudart)
else()
  # Create a dummy target so the project can still build
  add_custom_target(hellocuda ALL
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA not available - hellocuda skipped"
    COMMENT "hellocuda requires CUDA toolkit"
  )
endif()
```

- [ ] **Step 3: Verify build succeeds (kernel compiles, no linker errors)**

```bash
cmake -B build -S . && cmake --build build --target spmv_gpu 2>&1 | tail -20
```

Expected: no errors, `libspmv_gpu.a` produced.

- [ ] **Step 4: Commit**

```bash
git add src/spmv_gpu/CMakeLists.txt src/CMakeLists.txt
git commit -m "build(spmv_gpu): add CMakeLists for CSR-Scalar library"
```

---

### Task 4: Write the validation binary

**Files:**
- Create: `src/spmv_gpu/validate_csr_scalar.cu`

- [ ] **Step 1: Write the validation binary**

```cuda
// src/spmv_gpu/validate_csr_scalar.cu
#include "spmv_gpu/csr_scalar.cuh"
#include "spmv_cpu/spmv_cpu.h"
#include "parser/mtx_parser.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::vector<float> make_random(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& val : v) val = dist(rng);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: validate_csr_scalar <matrix.mtx>\n");
        return 1;
    }
    const char* path = argv[1];

    // ── Load matrix ──────────────────────────────────────────────────────────
    MtxCsr host_A;
    DeviceMatrix dev_A{};
    try {
        host_A = parse_mtx_csr(path);
        parse_mtx_gpu(path, &dev_A);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error loading '%s': %s\n", path, e.what());
        return 1;
    }
    std::printf("Matrix:   %s\n", path);
    std::printf("Size:     %d x %d, %d nnz\n",
                host_A.num_rows, host_A.num_cols, host_A.num_nonzeros);

    // ── Input vector x (same for CPU and GPU) ────────────────────────────────
    auto x = make_random(host_A.num_cols, 42);

    // ── CPU SpMV ─────────────────────────────────────────────────────────────
    std::vector<float> y_cpu(host_A.num_rows, 0.0f);
    auto t0 = std::chrono::high_resolution_clock::now();
    spmv_csr_cpu(host_A, x.data(), y_cpu.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── GPU SpMV ─────────────────────────────────────────────────────────────
    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaMalloc(&d_x, host_A.num_cols * sizeof(float));
    cudaMalloc(&d_y, host_A.num_rows * sizeof(float));

    cudaMemcpy(d_x, x.data(), host_A.num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up launch (not timed) to avoid first-launch overhead
    spmv_csr_scalar(dev_A, d_x, d_y);
    cudaDeviceSynchronize();

    // Timed launch
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    spmv_csr_scalar(dev_A, d_x, d_y);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // Copy result back
    std::vector<float> y_gpu(host_A.num_rows);
    cudaMemcpy(y_gpu.data(), d_y, host_A.num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    free_gpu(&dev_A);

    // ── Comparison ───────────────────────────────────────────────────────────
    std::printf("CPU time: %.3f ms\n", cpu_ms);
    std::printf("GPU time: %.3f ms  (kernel only, excludes H2D/D2H)\n", gpu_ms);

    float max_diff = compare_vectors(y_cpu.data(), y_gpu.data(), host_A.num_rows, 1e-5f);
    std::printf("Max diff: %.3e\n", max_diff);

    bool pass = (max_diff <= 1e-5f);
    std::printf("[%s]\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
```

- [ ] **Step 2: Build the validation binary**

```bash
cmake --build build --target validate_csr_scalar 2>&1 | tail -20
```

Expected: no errors, `build/src/spmv_gpu/validate_csr_scalar` produced.

- [ ] **Step 3: Commit**

```bash
git add src/spmv_gpu/validate_csr_scalar.cu
git commit -m "feat(spmv_gpu): add CSR-Scalar validation binary"
```

---

### Task 5: Run validation and verify correctness

**Files:**
- No file changes — this task runs the binary.

- [ ] **Step 1: Download a small test matrix if none available**

Check `data/` for any `.mtx` files:
```bash
ls data/
```

If empty, use the built-in parser test matrix or download one. The project has `scripts/download_all_29_matrices.sh` — run it for a single small matrix, e.g. `bcspwr01` (~39 rows):

```bash
# Only if data/ is empty:
bash scripts/download_all_29_matrices.sh 2>&1 | head -30
```

Alternatively create a tiny hand-crafted test matrix `data/test_tiny.mtx`:
```
%%MatrixMarket matrix coordinate real general
4 4 7
1 1 1.0
1 2 2.0
2 2 3.0
2 3 4.0
3 3 5.0
3 4 6.0
4 4 7.0
```

- [ ] **Step 2: Run validation binary**

```bash
./build/src/spmv_gpu/validate_csr_scalar data/test_tiny.mtx
```

Expected output (values will vary):
```
Matrix:   data/test_tiny.mtx
Size:     4 x 4, 7 nnz
CPU time: 0.xxx ms
GPU time: 0.xxx ms  (kernel only, excludes H2D/D2H)
Max diff: 0.000e+00
[PASS]
```

Exit code must be 0: `echo $?` → `0`

- [ ] **Step 3: Run on a larger matrix if available**

```bash
./build/src/spmv_gpu/validate_csr_scalar data/<some_larger_matrix>.mtx
```

Expected: `[PASS]`, `Max diff` below `1e-5`.

- [ ] **Step 4: Commit test matrix if hand-crafted**

```bash
# Only if you created data/test_tiny.mtx:
git add data/test_tiny.mtx
git commit -m "test(data): add tiny 4x4 hand-crafted test matrix"
```

---

## Self-Review

**Spec coverage:**
- [x] CSR-Scalar kernel (one thread per row) — Task 2
- [x] `float` throughout — Task 1, 2
- [x] `block_size=256` — Task 2
- [x] Validation binary with verbose output — Task 4
- [x] CPU timing via `std::chrono` — Task 4
- [x] GPU timing via `cudaEvent_t` (kernel only) — Task 4
- [x] Tolerance `1e-5f` — Task 4
- [x] Random x vector seed 42 — Task 4
- [x] Exit 0/1 — Task 4
- [x] CMake build integration — Task 3
- [x] Module layout mirrors `spmv_cpu` — Tasks 1–3

**No placeholders:** confirmed — every step has actual code.

**Type consistency:**
- `spmv_csr_scalar` declared in Task 1, defined in Task 2, called in Task 4 ✓
- `DeviceMatrix.d_values` is `float*` (from `mtx_parser.h`) — kernel uses `float*` ✓
- `spmv_csr_cpu` signature: `(const MtxCsr&, const float*, float*)` — matches Task 4 usage ✓
- `compare_vectors` signature: `(const float*, const float*, int, float)` — matches Task 4 usage ✓
