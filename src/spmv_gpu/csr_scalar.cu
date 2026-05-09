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
    float* __restrict__         d_y)
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
