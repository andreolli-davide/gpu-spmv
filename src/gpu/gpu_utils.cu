// =============================================================================
// gpu_utils.cu
// =============================================================================
// Implementation of GPU memory management utilities.
// =============================================================================

#include "gpu_utils.h"
#include <cstdio>   // fprintf, stderr
#include <cstdlib>  // exit, EXIT_FAILURE

namespace spmv {

// =============================================================================
// DeviceMatrix
// =============================================================================

bool DeviceMatrix::is_allocated() const {
    return d_values != nullptr && d_col_index != nullptr && d_row_ptr != nullptr;
}

// =============================================================================
// DeviceVector
// =============================================================================

bool DeviceVector::is_allocated() const {
    return d_data != nullptr;
}

// =============================================================================
// allocate_device_matrix
// =============================================================================

DeviceMatrix allocate_device_matrix(const SparseMatrix& h_matrix) {
    DeviceMatrix d_matrix;
    d_matrix.rows = h_matrix.rows;
    d_matrix.cols = h_matrix.cols;
    d_matrix.nnz  = h_matrix.nnz;

    CUDA_CHECK(cudaMalloc(&d_matrix.d_values,    h_matrix.nnz  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_matrix.d_col_index, h_matrix.nnz  * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_matrix.d_row_ptr,    (h_matrix.rows + 1) * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_matrix.d_values,
                          h_matrix.values.data(),
                          h_matrix.nnz * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_matrix.d_col_index,
                          h_matrix.col_index.data(),
                          h_matrix.nnz * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_matrix.d_row_ptr,
                          h_matrix.row_ptr.data(),
                          (h_matrix.rows + 1) * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    return d_matrix;
}

// =============================================================================
// copy_matrix_to_device
// =============================================================================

void copy_matrix_to_device(const SparseMatrix& h_matrix,
                           DeviceMatrix& d_matrix,
                           bool async,
                           cudaStream_t stream) {
    if (async) {
        CUDA_CHECK(cudaMemcpyAsync(d_matrix.d_values,
                                   h_matrix.values.data(),
                                   h_matrix.nnz * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));

        CUDA_CHECK(cudaMemcpyAsync(d_matrix.d_col_index,
                                   h_matrix.col_index.data(),
                                   h_matrix.nnz * sizeof(int64_t),
                                   cudaMemcpyHostToDevice,
                                   stream));

        CUDA_CHECK(cudaMemcpyAsync(d_matrix.d_row_ptr,
                                   h_matrix.row_ptr.data(),
                                   (h_matrix.rows + 1) * sizeof(int64_t),
                                   cudaMemcpyHostToDevice,
                                   stream));
    } else {
        CUDA_CHECK(cudaMemcpy(d_matrix.d_values,
                              h_matrix.values.data(),
                              h_matrix.nnz * sizeof(double),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_matrix.d_col_index,
                              h_matrix.col_index.data(),
                              h_matrix.nnz * sizeof(int64_t),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_matrix.d_row_ptr,
                              h_matrix.row_ptr.data(),
                              (h_matrix.rows + 1) * sizeof(int64_t),
                              cudaMemcpyHostToDevice));
    }
}

// =============================================================================
// copy_vector_to_device
// =============================================================================

DeviceVector copy_vector_to_device(const DenseVector& h_vec) {
    DeviceVector d_vec;
    d_vec.size = h_vec.size;

    CUDA_CHECK(cudaMalloc(&d_vec.d_data, h_vec.size * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_vec.d_data,
                          h_vec.data.data(),
                          h_vec.size * sizeof(double),
                          cudaMemcpyHostToDevice));

    return d_vec;
}

// =============================================================================
// copy_vector_to_device_async
// =============================================================================

void copy_vector_to_device_async(const DenseVector& h_vec,
                                 DeviceVector& d_vec,
                                 cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(d_vec.d_data,
                               h_vec.data.data(),
                               h_vec.size * sizeof(double),
                               cudaMemcpyHostToDevice,
                               stream));
}

// =============================================================================
// copy_vector_from_device
// =============================================================================

void copy_vector_from_device(const DeviceVector& d_vec,
                             DenseVector& h_vec,
                             bool async,
                             cudaStream_t stream) {
    if (h_vec.size == 0) {
        h_vec.resize(d_vec.size);
    }

    if (async) {
        CUDA_CHECK(cudaMemcpyAsync(h_vec.data.data(),
                                   d_vec.d_data,
                                   d_vec.size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    } else {
        CUDA_CHECK(cudaMemcpy(h_vec.data.data(),
                              d_vec.d_data,
                              d_vec.size * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
}

// =============================================================================
// free_device_matrix
// =============================================================================

void free_device_matrix(DeviceMatrix& d_matrix) {
    if (d_matrix.d_values) {
        CUDA_CHECK(cudaFree(d_matrix.d_values));
        d_matrix.d_values = nullptr;
    }
    if (d_matrix.d_col_index) {
        CUDA_CHECK(cudaFree(d_matrix.d_col_index));
        d_matrix.d_col_index = nullptr;
    }
    if (d_matrix.d_row_ptr) {
        CUDA_CHECK(cudaFree(d_matrix.d_row_ptr));
        d_matrix.d_row_ptr = nullptr;
    }
    d_matrix.rows = 0;
    d_matrix.cols = 0;
    d_matrix.nnz  = 0;
}

// =============================================================================
// free_device_vector
// =============================================================================

void free_device_vector(DeviceVector& d_vec) {
    if (d_vec.d_data) {
        CUDA_CHECK(cudaFree(d_vec.d_data));
        d_vec.d_data = nullptr;
    }
    d_vec.size = 0;
}

} // namespace spmv