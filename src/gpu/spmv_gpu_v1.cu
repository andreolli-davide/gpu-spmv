// =============================================================================
// spmv_gpu_v1.cu
// =============================================================================
// GPU SpMV kernel v1 — Basic CSR row-parallel kernel following Bell & Garland '09
//
// Algorithm:
//   - One thread per matrix row (thread_id = blockIdx.x * blockDim.x + threadIdx.x)
//   - Each thread computes: y[i] = sum_{k=row_ptr[i]}^{row_ptr[i+1]-1} values[k] * x[col_index[k]]
//   - Coalesced memory access for values and col_index (adjacent threads access adjacent entries)
//   - __ldg for reading input vector x (non-coherent global load via texture cache)
//
// Thread Mapping (Bell & Garland '09):
//   thread_id = block_idx * block_dim + thread_idx
//
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>

#include "gpu_utils.h"

namespace spmv {

// =============================================================================
// spmv_gpu_v1_kernel
// =============================================================================
// Basic CSR row-parallel SpMV kernel.
//
// Each CUDA thread handles one matrix row. Threads access the values and
// col_index arrays in a coalesced pattern since row i accesses values at
// indices row_ptr[i] through row_ptr[i+1]-1.
//
// @param d_values    CSR values array (size nnz)
// @param d_col_index CSR column index array (size nnz)
// @param d_row_ptr   CSR row pointer array (size rows+1)
// @param d_x         Input vector (size cols)
// @param d_y         Output vector (size rows)
// @param rows        Number of matrix rows
//
__global__ void spmv_gpu_v1_kernel(const double* __restrict__ d_values,
                                   const int64_t* __restrict__ d_col_index,
                                   const int64_t* __restrict__ d_row_ptr,
                                   const double* __restrict__ d_x,
                                   double* __restrict__ d_y,
                                   int64_t rows) {
    // Thread mapping: one thread per row
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Bounds check: some threads may be idle if rows is not divisible by block size
    if (row >= rows) {
        return;
    }

    // Compute the range of non-zeros for this row
    const int64_t row_start = d_row_ptr[row];
    const int64_t row_end   = d_row_ptr[row + 1];

    // Accumulate the dot product for this row
    double sum = 0.0;
    for (int64_t j = row_start; j < row_end; ++j) {
        const int64_t col = d_col_index[j];           // Which column of A does this entry belong to?
        sum += d_values[j] * __ldg(&d_x[col]);        // __ldg for read-only global memory (non-coherent)
    }

    d_y[row] = sum;
}

// =============================================================================
// spmv_gpu_v1 — High-level wrapper
// =============================================================================
// Top-level wrapper for GPU SpMV v1. Handles:
//   1. GPU memory allocation (cudaMalloc)
//   2. Host-to-device transfer (cudaMemcpy H2D)
//   3. Kernel launch (calls the actual GPU kernel)
//   4. Device-to-host transfer (cudaMemcpy D2H)
//   5. GPU memory deallocation
//
// The actual kernel is spmv_gpu_v1_kernel.
//
// @param A   Input matrix in CSR format, size rows × cols
// @param x   Input dense vector, size cols
// @param y   Output dense vector, size rows. Resized automatically.
//
void spmv_gpu_v1(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    // Allocate device memory for matrix
    DeviceMatrix d_matrix = allocate_device_matrix(A);

    // Allocate device vectors
    DeviceVector d_x = copy_vector_to_device(x);
    DeviceVector d_y;
    d_y.size = A.rows;
    CUDA_CHECK(cudaMalloc(&d_y.d_data, A.rows * sizeof(double)));

    // Zero-initialize output vector (required by the algorithm)
    CUDA_CHECK(cudaMemset(d_y.d_data, 0, A.rows * sizeof(double)));

    // Kernel configuration: one thread per row
    constexpr int block_dim = 256;
    const int grid_dim = static_cast<int>((A.rows + block_dim - 1) / block_dim);

    // Launch the kernel
    spmv_gpu_v1_kernel<<<grid_dim, block_dim>>>(
        d_matrix.d_values,
        d_matrix.d_col_index,
        d_matrix.d_row_ptr,
        d_x.d_data,
        d_y.d_data,
        A.rows
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

} // namespace spmv
