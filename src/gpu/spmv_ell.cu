// =============================================================================
// spmv_ell.cu
// =============================================================================
// GPU SpMV kernel using ELLPACK format — fixed-length row representation
//
// ELLPACK (Equal-Length List) pads all rows to max_row_length, enabling:
//   - Coalesced memory access: consecutive threads access consecutive memory
//   - No row-boundary checks: fixed-length inner loop
//   - Simple indexing: row_offset = row * max_row_length
//
// Thread Mapping:
//   One thread per matrix row. Each thread iterates exactly MAX_ROW_LENGTH
//   times using #pragma unroll for efficiency.
//
// Performance Characteristics:
//   - Best for: Regular matrices with similar row lengths (FEM, structured grids)
//   - Warning: Can be wasteful for irregular matrices (e.g., webbase)
//
// Reference: Bell & Garland '09 "Efficient Sparse Matrix-Vector Multiplication on GPUs"
//
// =============================================================================

#include "spmv_ell.h"
#include "gpu_utils.h"

namespace spmv {

template <int MAX_ROW_LENGTH>
__global__ void spmv_ell_kernel(const double* __restrict__ d_values,
                                 const int64_t* __restrict__ d_col_index,
                                 const double* __restrict__ d_x,
                                 double* __restrict__ d_y,
                                 int64_t rows) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const int64_t row_offset = row * MAX_ROW_LENGTH;
    double sum = 0.0;

    #pragma unroll
    for (int j = 0; j < MAX_ROW_LENGTH; ++j) {
        const int64_t col = d_col_index[row_offset + j];
        if (col >= 0) {
            sum += d_values[row_offset + j] * __ldg(&d_x[col]);
        }
    }

    d_y[row] = sum;
}

void spmv_ell(const ELL_SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    DeviceMatrix d_matrix = allocate_device_matrix_ell(A);
    DeviceVector d_x = copy_vector_to_device(x);
    DeviceVector d_y;
    d_y.size = A.rows;
    CUDA_CHECK(cudaMalloc(&d_y.d_data, A.rows * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_y.d_data, 0, A.rows * sizeof(double)));

    const int BLOCK_DIM = 256;
    const int grid_dim = static_cast<int>((A.rows + BLOCK_DIM - 1) / BLOCK_DIM);

    if (A.max_row_length <= 4) {
        spmv_ell_kernel<4><<<grid_dim, BLOCK_DIM>>>(
            d_matrix.d_values_ell, d_matrix.d_col_index_ell,
            d_x.d_data, d_y.d_data, A.rows);
    } else if (A.max_row_length <= 8) {
        spmv_ell_kernel<8><<<grid_dim, BLOCK_DIM>>>(
            d_matrix.d_values_ell, d_matrix.d_col_index_ell,
            d_x.d_data, d_y.d_data, A.rows);
    } else if (A.max_row_length <= 16) {
        spmv_ell_kernel<16><<<grid_dim, BLOCK_DIM>>>(
            d_matrix.d_values_ell, d_matrix.d_col_index_ell,
            d_x.d_data, d_y.d_data, A.rows);
    } else if (A.max_row_length <= 32) {
        spmv_ell_kernel<32><<<grid_dim, BLOCK_DIM>>>(
            d_matrix.d_values_ell, d_matrix.d_col_index_ell,
            d_x.d_data, d_y.d_data, A.rows);
    } else if (A.max_row_length <= 64) {
        spmv_ell_kernel<64><<<grid_dim, BLOCK_DIM>>>(
            d_matrix.d_values_ell, d_matrix.d_col_index_ell,
            d_x.d_data, d_y.d_data, A.rows);
    } else if (A.max_row_length <= 128) {
        spmv_ell_kernel<128><<<grid_dim, BLOCK_DIM>>>(
            d_matrix.d_values_ell, d_matrix.d_col_index_ell,
            d_x.d_data, d_y.d_data, A.rows);
    } else {
        spmv_ell_kernel<256><<<grid_dim, BLOCK_DIM>>>(
            d_matrix.d_values_ell, d_matrix.d_col_index_ell,
            d_x.d_data, d_y.d_data, A.rows);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));

    y.resize(A.rows);
    CUDA_CHECK(cudaMemcpy(y.data.data(), d_y.d_data,
                          A.rows * sizeof(double), cudaMemcpyDeviceToHost));

    free_device_matrix_ell(d_matrix);
    free_device_vector(d_x);
    free_device_vector(d_y);
}

} // namespace spmv