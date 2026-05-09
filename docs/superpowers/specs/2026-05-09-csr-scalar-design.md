# CSR-Scalar GPU SpMV — Design Spec

**Date:** 2026-05-09  
**Status:** Approved

## Overview

Implement `CSR-Scalar`: a CUDA kernel for Sparse Matrix-Vector Multiplication (SpMV)
where each GPU thread processes exactly one matrix row. Accompanied by a standalone
validation binary that compares GPU output against the CPU reference.

## Motivation

Simplest possible GPU SpMV baseline. One thread per row is predictable and easy to
reason about. Suffers from load imbalance on irregular matrices (rows with very
different nnz counts), making it useful as a performance floor to compare against
more sophisticated kernels (CSR-Vector, CSR-Warp) later.

## File Layout

### New files

```
include/spmv_gpu/csr_scalar.cuh         # Public header — kernel host wrapper
src/spmv_gpu/csr_scalar.cu              # Kernel implementation + host wrapper
src/spmv_gpu/validate_csr_scalar.cu     # Standalone validation binary
src/spmv_gpu/CMakeLists.txt             # Build rules
```

### Modified files

```
src/CMakeLists.txt                      # add_subdirectory(spmv_gpu)
```

## Kernel Design

**Name:** `CSR-Scalar`  
**Strategy:** One CUDA thread per matrix row. Each thread iterates over its row's
nonzeros sequentially, accumulating into a local `float` register.

```cuda
__global__ void spmv_csr_scalar_kernel(
    const int32_t* row_ptr,
    const int32_t* col_indices,
    const float*   values,
    int32_t        num_rows,
    const float*   d_x,
    float*         d_y)
{
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float sum = 0.0f;
    for (int32_t i = row_ptr[row]; i < row_ptr[row + 1]; ++i)
        sum += values[i] * d_x[col_indices[i]];
    d_y[row] = sum;
}
```

**Launch config:** `block_size=256`, `num_blocks=ceil(num_rows / 256)`.

**Type:** `float` throughout — consistent with project-wide float migration.

**No zeroing required:** kernel writes every `d_y[row]` unconditionally.

## Public Interface

`include/spmv_gpu/csr_scalar.cuh`:

```cpp
#pragma once
#include "parser/mtx_parser.h"

/// GPU SpMV: y = A*x using CSR-Scalar (one thread per row).
/// d_x and d_y must be device pointers of size A.num_cols and A.num_rows.
/// Caller is responsible for allocation and H2D/D2H transfers.
void spmv_csr_scalar(const DeviceMatrix& A, const float* d_x, float* d_y);
```

## Validation Binary

**Target:** `validate_csr_scalar`  
**Usage:** `./validate_csr_scalar <matrix.mtx>`  
**Exit codes:** 0 = PASS, 1 = FAIL

### Flow

1. Parse `.mtx` → `MtxCsr` (host) + `DeviceMatrix` (GPU via `parse_mtx_gpu`)
2. Generate random `x` vector: uniform [0, 1], seed 42
3. **CPU:** run `spmv_csr_cpu`, measure wall-clock time with `std::chrono`
4. **GPU:**
   - `cudaMalloc` `d_x` and `d_y`
   - H2D copy `x → d_x`
   - Wrap kernel launch in `cudaEvent_t` pair for timing
   - D2H copy `d_y → y_gpu`
5. Compare `y_cpu` vs `y_gpu` using `compare_vectors` (tolerance `1e-5f`)
6. Print verbose report; exit 0/1

### Output Format

```
Matrix:   <path>
Size:     <rows> x <cols>, <nnz> nnz
CPU time: <ms> ms
GPU time: <ms> ms  (kernel only, excludes H2D/D2H)
Max diff: <val>
[PASS] or [FAIL]
```

### Timing note

GPU time measured with `cudaEvent_t` around kernel launch only — excludes H2D/D2H
transfers. This isolates pure compute performance.

## Build Integration

`src/spmv_gpu/CMakeLists.txt`:
- Static library `spmv_gpu` from `csr_scalar.cu`, links `mtx_parser`, `CUDA::cudart`
- Executable `validate_csr_scalar` from `validate_csr_scalar.cu`, links `spmv_gpu` + `spmv_cpu`
- Guard with `if(CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)`

`src/CMakeLists.txt`:
- Add `add_subdirectory(spmv_gpu)` inside the CUDA guard

## Known Issues / Notes

- `src/parser/mtx_parser_gpu.cu` contains an older `spmv_gpu_kernel_impl` using
  `double` — a leftover from before the float migration. This new module supersedes
  it. The old kernel is not removed here (out of scope), but it should be cleaned up.
- FP accumulation order differs between CPU and GPU — tolerance `1e-5f` accounts for this.
