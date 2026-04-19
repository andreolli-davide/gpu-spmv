// =============================================================================
// spmv_gpu_v2.cu
// =============================================================================
// GPU SpMV kernel v2 — Shared memory tiling for input vector x
//
// Based on Bell & Garland '09 shared memory tiling approach:
//
//   "Hot data is cached in shared memory to reduce global memory traffic:
//    shared_mem[i] = values[i] (loaded per warp)"
//   "Shared memory latency: ~1–10 ns vs. global memory: ~100–400 ns"
//
// Key optimizations over v1:
//   - Each block loads a CONTIGUOUS chunk of x into shared memory
//   - Threads access x from shared memory instead of global memory
//   - Shared memory size: configurable 16KB-48KB (within Ampere limits)
//
// Ampere (compute capability 8.0) shared memory limits:
//   - Max shared memory per block: 48 KB (49152 bytes)
//   - Default: 32 KB (4096 double elements)
//
// Shared Memory Layout:
//   - extern __shared__ double smem_x[];
//   - Block b loads x elements [b * SHARED_ELEMENTS, (b+1) * SHARED_ELEMENTS)
//   - Thread checks if col_index is in shared range, accesses accordingly
//
// Thread Mapping:
//   thread_id = block_idx * block_dim + thread_idx (one thread per row)
//
// Persistent Buffer Support:
//   For repeated SpMV calls with the same matrix, see gpu_persistent_buffers.h
//   and spmv_gpu_v2_persistent() to eliminate per-call cudaMalloc/cudaFree overhead.
//
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

#include "gpu_utils.h"
#include "spmv_selector.h"

namespace spmv {

// =============================================================================
// Kernel Configuration
// =============================================================================

// Default shared memory size: 32 KB (4096 double elements)
// Must not exceed 48 KB (Ampere limit)
#ifndef SHARED_MEM_SIZE
#define SHARED_MEM_SIZE (32 * 1024)  // 32 KB default
#endif

// Number of double elements that fit in shared memory
constexpr int SHARED_ELEMENTS = SHARED_MEM_SIZE / sizeof(double);

// Block dimensions
constexpr int BLOCK_DIM = 256;

// =============================================================================
// spmv_gpu_v2_kernel — Shared memory tiled CSR SpMV kernel
// =============================================================================
// Each block loads a contiguous chunk of x into shared memory, then threads
// access x from shared memory instead of global memory.
//
// Algorithm:
//   1. Block computes row range: [block_start, block_end)
//   2. Block cooperatively loads x[block_start : block_start + SHARED_ELEMENTS)
//      into shared memory (each thread loads several elements in strided fashion)
//   3. __syncthreads() ensures all loads complete
//   4. Each thread computes its row's dot product, accessing x from shared
//      memory when col_index falls within the loaded range, otherwise using
//      __ldg for global memory access
//
// @param d_values    CSR values array (size nnz)
// @param d_col_index CSR column index array (size nnz)
// @param d_row_ptr   CSR row pointer array (size rows+1)
// @param d_x         Input vector (size cols)
// @param d_y         Output vector (size rows)
// @param rows        Number of matrix rows
//
// =============================================================================

template <int SHARED_ELEMENTS>
__global__ void spmv_gpu_v2_kernel(const double* __restrict__ d_values,
                                   const int64_t* __restrict__ d_col_index,
                                   const int64_t* __restrict__ d_row_ptr,
                                   const double* __restrict__ d_x,
                                   double* __restrict__ d_y,
                                   int64_t rows,
                                   int64_t x_size) {
    // Shared memory for input vector x (tiled)
    // Size: SHARED_ELEMENTS doubles (default 32KB = 4096 elements)
    extern __shared__ double smem_x[];

    // Thread mapping: one thread per row
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Bounds check
    if (row >= rows) {
        return;
    }

    // Each block loads a contiguous chunk of x into shared memory
    // Block b loads elements [b * SHARED_ELEMENTS, (b+1) * SHARED_ELEMENTS)
    const int64_t block_start = static_cast<int64_t>(blockIdx.x) * SHARED_ELEMENTS;

    // Thread loads elements in strided fashion: thread t loads element at
    // index block_start + threadIdx.x, block_start + threadIdx.x + blockDim.x, etc.
    for (int64_t i = threadIdx.x; i < SHARED_ELEMENTS; i += blockDim.x) {
        const int64_t x_idx = block_start + i;
        if (x_idx < x_size) {  // bounds check for last block
            smem_x[i] = __ldg(&d_x[x_idx]);
        }
    }

    // Ensure all shared memory loads complete before computation
    __syncthreads();

    // Compute row range
    const int64_t row_start = d_row_ptr[row];
    const int64_t row_end   = d_row_ptr[row + 1];

    // Accumulate dot product for this row
    double sum = 0.0;
    for (int64_t j = row_start; j < row_end; ++j) {
        const int64_t col = d_col_index[j];
        const double val = d_values[j];

        // Access x from shared memory if col is in this block's loaded range,
        // otherwise fall back to __ldg for global memory access
        const int64_t smem_offset = col - block_start;
        if (smem_offset >= 0 && smem_offset < SHARED_ELEMENTS) {
            sum += val * smem_x[smem_offset];
        } else {
            sum += val * __ldg(&d_x[col]);
        }
    }

    d_y[row] = sum;
}

// =============================================================================
// spmv_gpu_v2 — High-level wrapper with shared memory tiling
// =============================================================================
// Wrapper function for GPU SpMV v2 with shared memory tiling for input vector.
//
// Handles:
//   1. GPU memory allocation
//   2. Host-to-device transfer
//   3. Kernel launch with shared memory configuration
//   4. Device-to-host transfer
//   5. GPU memory deallocation
//
// Shared Memory Configuration:
//   - Default: 32 KB (configurable at compile time via SHARED_MEM_SIZE)
//   - Must not exceed 48 KB (Ampere limit)
//
// @param A   Input matrix in CSR format, size rows × cols
// @param x   Input dense vector, size cols
// @param y   Output dense vector, size rows. Resized automatically.
//
// =============================================================================

void spmv_gpu_v2(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    // Validate shared memory size (48 KB limit for Ampere)
    static_assert(SHARED_MEM_SIZE <= 48 * 1024,
                  "Shared memory size exceeds Ampere limit of 48 KB");

    // Allocate device memory for matrix
    DeviceMatrix d_matrix = allocate_device_matrix(A);

    // Allocate device vectors
    DeviceVector d_x = copy_vector_to_device(x);
    DeviceVector d_y;
    d_y.size = A.rows;
    CUDA_CHECK(cudaMalloc(&d_y.d_data, A.rows * sizeof(double)));

    // Zero-initialize output vector
    CUDA_CHECK(cudaMemset(d_y.d_data, 0, A.rows * sizeof(double)));

    // Kernel configuration
    const int grid_dim = static_cast<int>((A.rows + BLOCK_DIM - 1) / BLOCK_DIM);

    // Calculate shared memory size in bytes
    const size_t shared_mem_bytes = SHARED_MEM_SIZE;

    // Launch the kernel with shared memory tiling for x
    spmv_gpu_v2_kernel<SHARED_ELEMENTS><<<grid_dim, BLOCK_DIM, shared_mem_bytes>>>(
        d_matrix.d_values,
        d_matrix.d_col_index,
        d_matrix.d_row_ptr,
        d_x.d_data,
        d_y.d_data,
        A.rows,
        A.cols
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));

    // Copy result back to host
    y.resize(A.rows);
    CUDA_CHECK(cudaMemcpy(y.data.data(), d_y.d_data,
                          A.rows * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Free device memory
    free_device_matrix(d_matrix);
    free_device_vector(d_x);
    free_device_vector(d_y);
}

// =============================================================================
// spmv_gpu_v2_custom_smem — Wrapper with custom shared memory size
// =============================================================================
// Variant that allows specifying shared memory size at runtime (within limits).
//
// @param A             Input matrix in CSR format
// @param x             Input dense vector
// @param y             Output dense vector (resized automatically)
// @param shared_kb    Shared memory size in KB (16-48, clamped to limit)
//
// =============================================================================

void spmv_gpu_v2_custom_smem(const SparseMatrix& A, const DenseVector& x,
                             DenseVector& y, int shared_kb) {
    // Clamp to Ampere limit (48 KB)
    const size_t shared_mem_bytes = std::min(shared_kb, 48) * 1024;

    // Allocate device memory
    DeviceMatrix d_matrix = allocate_device_matrix(A);
    DeviceVector d_x = copy_vector_to_device(x);
    DeviceVector d_y;
    d_y.size = A.rows;
    CUDA_CHECK(cudaMalloc(&d_y.d_data, A.rows * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_y.d_data, 0, A.rows * sizeof(double)));

    // Kernel launch with custom shared memory
    constexpr int BLOCK_DIM = 256;
    const int grid_dim = static_cast<int>((A.rows + BLOCK_DIM - 1) / BLOCK_DIM);

    // For custom sizes, use template instantiation based on actual size
    if (shared_mem_bytes <= 16 * 1024) {
        constexpr int ELEMENTS_16K = (16 * 1024) / sizeof(double);
        spmv_gpu_v2_kernel<ELEMENTS_16K><<<grid_dim, BLOCK_DIM, shared_mem_bytes>>>(
            d_matrix.d_values, d_matrix.d_col_index, d_matrix.d_row_ptr,
            d_x.d_data, d_y.d_data, A.rows, A.cols);
    } else if (shared_mem_bytes <= 32 * 1024) {
        constexpr int ELEMENTS_32K = (32 * 1024) / sizeof(double);
        spmv_gpu_v2_kernel<ELEMENTS_32K><<<grid_dim, BLOCK_DIM, shared_mem_bytes>>>(
            d_matrix.d_values, d_matrix.d_col_index, d_matrix.d_row_ptr,
            d_x.d_data, d_y.d_data, A.rows, A.cols);
    } else {
        constexpr int ELEMENTS_48K = (48 * 1024) / sizeof(double);
        spmv_gpu_v2_kernel<ELEMENTS_48K><<<grid_dim, BLOCK_DIM, shared_mem_bytes>>>(
            d_matrix.d_values, d_matrix.d_col_index, d_matrix.d_row_ptr,
            d_x.d_data, d_y.d_data, A.rows, A.cols);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));

    // Copy result back
    y.resize(A.rows);
    CUDA_CHECK(cudaMemcpy(y.data.data(), d_y.d_data,
                          A.rows * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    free_device_matrix(d_matrix);
    free_device_vector(d_x);
    free_device_vector(d_y);
}

// =============================================================================
// spmv_gpu_v2_autotuned — Auto-tuned block size SpMV
// =============================================================================

void spmv_gpu_v2_autotuned(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    const int64_t avg_nnz_per_row = (A.rows > 0) ? (A.nnz / A.rows) : 1;

    BlockSizeTuning tuning = auto_select_block_size(A.nnz, A.rows, avg_nnz_per_row);

    spmv_gpu_v2_custom_smem(A, x, y, tuning.block_size);
}

} // namespace spmv
// =============================================================================
// spmv_gpu_v2_auto — Auto-selecting format SpMV
// =============================================================================

void spmv_gpu_v2_auto(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    FormatSelection sel = select_format(A);

    switch (sel.format) {
        case SpMVFormat::CSR_ADAPTIVE: {
            auto meta = compute_adaptive_meta(A);
            spmv_csr_adaptive(A, x, y, meta);
            break;
        }
        case SpMVFormat::ELL: {
            auto ell = csr_to_ell(A);
            spmv_ell(ell, x, y);
            break;
        }
        case SpMVFormat::CSR_TILED:
        default: {
            spmv_gpu_v2(A, x, y);
            break;
        }
    }
}
