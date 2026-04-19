#include "spmv_csr_adaptive.h"
#include <algorithm>

namespace spmv {

CSRAdaptiveMeta compute_adaptive_meta(const SparseMatrix& A, int warp_size) {
    CSRAdaptiveMeta meta;

    std::vector<int64_t> row_lengths(A.rows);
    for (int64_t i = 0; i < A.rows; ++i) {
        row_lengths[i] = A.row_ptr[i + 1] - A.row_ptr[i];
    }

    std::vector<std::pair<int64_t, int64_t>> row_pairs(A.rows);
    for (int64_t i = 0; i < A.rows; ++i) {
        row_pairs[i] = {i, row_lengths[i]};
    }
    std::sort(row_pairs.begin(), row_pairs.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    const int64_t num_blocks = (A.rows + warp_size - 1) / warp_size;
    meta.num_row_blocks = num_blocks;
    meta.row_block_ptr.resize(num_blocks + 1);
    meta.row_block_nnz.resize(num_blocks);
    meta.row_to_block.resize(A.rows);
    meta.sorted_row_order.resize(A.rows);

    int64_t current_block = 0;
    int64_t block_start = 0;
    int64_t block_nnz = 0;

    for (int64_t i = 0; i < A.rows; ++i) {
        const int64_t row = row_pairs[i].first;
        const int64_t len = row_pairs[i].second;

        meta.sorted_row_order[i] = row;
        meta.row_to_block[row] = current_block;
        block_nnz += len;

        const bool last_in_block = ((i + 1) % warp_size == 0) || (i == A.rows - 1);
        if (last_in_block) {
            meta.row_block_ptr[current_block] = block_start;
            meta.row_block_nnz[current_block] = block_nnz;
            block_start = i + 1;
            block_nnz = 0;
            current_block++;
        }
    }
    meta.row_block_ptr[num_blocks] = A.rows;

    return meta;
}

template <int WARP_SIZE>
__global__ void spmv_csr_adaptive_kernel(
    const double* __restrict__ d_values,
    const int64_t* __restrict__ d_col_index,
    const int64_t* __restrict__ d_row_ptr,
    const int64_t* __restrict__ d_row_block_ptr,
    const double* __restrict__ d_x,
    double* __restrict__ d_y,
    int64_t num_row_blocks) {

    const int64_t block_id = blockIdx.x;
    if (block_id >= num_row_blocks) return;

    const int lane_id = threadIdx.x % WARP_SIZE;

    const int64_t row_start = d_row_block_ptr[block_id];
    const int64_t row_end = d_row_block_ptr[block_id + 1];

    const int64_t local_row = lane_id;
    const int64_t global_row = row_start + local_row;

    if (global_row >= row_end) return;

    double sum = 0.0;
    const int64_t row_len = d_row_ptr[global_row + 1] - d_row_ptr[global_row];
    for (int64_t j = 0; j < row_len; ++j) {
        const int64_t idx = d_row_ptr[global_row] + j;
        const int64_t col = d_col_index[idx];
        sum += d_values[idx] * __ldg(&d_x[col]);
    }

    d_y[global_row] = sum;
}

void spmv_csr_adaptive(const SparseMatrix& A, const DenseVector& x, DenseVector& y,
                       const CSRAdaptiveMeta& meta) {
    DeviceMatrix d_matrix = allocate_device_matrix(A);
    DeviceVector d_x = copy_vector_to_device(x);
    DeviceVector d_y;
    d_y.size = A.rows;
    CUDA_CHECK(cudaMalloc(&d_y.d_data, A.rows * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_y.d_data, 0, A.rows * sizeof(double)));

    DeviceVector d_row_block_ptr;
    d_row_block_ptr.size = meta.row_block_ptr.size();
    CUDA_CHECK(cudaMalloc(&d_row_block_ptr.d_data, meta.row_block_ptr.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_row_block_ptr.d_data, meta.row_block_ptr.data(),
                         meta.row_block_ptr.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    const int warp_size = 32;
    const int grid_dim = static_cast<int>(meta.num_row_blocks);
    constexpr int BLOCK_DIM = 32;

    spmv_csr_adaptive_kernel<warp_size><<<grid_dim, BLOCK_DIM>>>(
        d_matrix.d_values, d_matrix.d_col_index, d_matrix.d_row_ptr,
        d_row_block_ptr.d_data,
        d_x.d_data, d_y.d_data, meta.num_row_blocks);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));

    y.resize(A.rows);
    CUDA_CHECK(cudaMemcpy(y.data.data(), d_y.d_data,
                          A.rows * sizeof(double), cudaMemcpyDeviceToHost));

    free_device_matrix(d_matrix);
    free_device_vector(d_x);
    free_device_vector(d_y);
    free_device_vector(d_row_block_ptr);
}

} // namespace spmv
