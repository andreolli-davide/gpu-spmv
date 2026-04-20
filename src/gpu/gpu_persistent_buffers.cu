// =============================================================================
// gpu_persistent_buffers.cu
// =============================================================================

#include "gpu_persistent_buffers.h"
#include "gpu_utils.h"
#include "spmv_gpu_v2.h"
#include <stdexcept>

namespace spmv {

PersistentBufferManager::PersistentBufferManager()
    : is_initialized(false) {
    vector_x.d_data = nullptr;
    vector_x.size = 0;
    vector_y.d_data = nullptr;
    vector_y.size = 0;
}

PersistentBufferManager::~PersistentBufferManager() {
    free_all();
}

PersistentBufferManager::PersistentBufferManager(PersistentBufferManager&& other) noexcept
    : matrix(other.matrix)
    , vector_x(other.vector_x)
    , vector_y(other.vector_y)
    , is_initialized(other.is_initialized) {
    other.matrix.d_values = nullptr;
    other.matrix.d_col_index = nullptr;
    other.matrix.d_row_ptr = nullptr;
    other.vector_x.d_data = nullptr;
    other.vector_x.size = 0;
    other.vector_y.d_data = nullptr;
    other.vector_y.size = 0;
    other.is_initialized = false;
}

PersistentBufferManager& PersistentBufferManager::operator=(PersistentBufferManager&& other) noexcept {
    if (this != &other) {
        free_all();
        matrix = other.matrix;
        vector_x = other.vector_x;
        vector_y = other.vector_y;
        is_initialized = other.is_initialized;
        other.matrix.d_values = nullptr;
        other.matrix.d_col_index = nullptr;
        other.matrix.d_row_ptr = nullptr;
        other.vector_x.d_data = nullptr;
        other.vector_x.size = 0;
        other.vector_y.d_data = nullptr;
        other.vector_y.size = 0;
        other.is_initialized = false;
    }
    return *this;
}

void PersistentBufferManager::upload_matrix(const SparseMatrix& A) {
    if (is_initialized) {
        if (matrix.rows == A.rows && matrix.nnz == A.nnz) {
            return;
        }
        free_device_matrix(matrix);
    }
    matrix = allocate_device_matrix(A);
    is_initialized = true;
}

void PersistentBufferManager::upload_vector_x(const DenseVector& x) {
    if (vector_x.size != x.size) {
        if (vector_x.d_data) {
            free_device_vector(vector_x);
        }
        vector_x = copy_vector_to_device(x);
    } else {
        CUDA_CHECK(cudaMemcpy(vector_x.d_data, x.data.data(),
                             x.size * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
}

void PersistentBufferManager::allocate_output(int64_t rows) {
    if (vector_y.size != rows) {
        if (vector_y.d_data) {
            free_device_vector(vector_y);
        }
        vector_y.size = rows;
        CUDA_CHECK(cudaMalloc(&vector_y.d_data, rows * sizeof(double)));
    }
    CUDA_CHECK(cudaMemset(vector_y.d_data, 0, rows * sizeof(double)));
}

void PersistentBufferManager::download_vector_y(DenseVector& y) {
    y.resize(vector_y.size);
    CUDA_CHECK(cudaMemcpy(y.data.data(), vector_y.d_data,
                         vector_y.size * sizeof(double),
                         cudaMemcpyDeviceToHost));
}

void PersistentBufferManager::free_all() {
    if (is_initialized) {
        free_device_matrix(matrix);
        is_initialized = false;
    }
    if (vector_x.d_data) {
        free_device_vector(vector_x);
        vector_x.d_data = nullptr;
        vector_x.size = 0;
    }
    if (vector_y.d_data) {
        free_device_vector(vector_y);
        vector_y.d_data = nullptr;
        vector_y.size = 0;
    }
}

void spmv_gpu_v2_persistent(PersistentBufferManager& buf,
                             const DenseVector& x, DenseVector& y) {
    (void)x;
    buf.allocate_output(buf.matrix.rows);

    constexpr int BLOCK_DIM = 256;
    constexpr int SHARED_ELEMENTS = (32 * 1024) / sizeof(double);
    const int grid_dim = static_cast<int>((buf.matrix.rows + BLOCK_DIM - 1) / BLOCK_DIM);

    spmv_gpu_v2_kernel<SHARED_ELEMENTS><<<grid_dim, BLOCK_DIM, 32 * 1024>>>(
        buf.matrix.d_values,
        buf.matrix.d_col_index,
        buf.matrix.d_row_ptr,
        buf.vector_x.d_data,
        buf.vector_y.d_data,
        buf.matrix.rows,
        buf.matrix.rows);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));

    buf.download_vector_y(y);
}

} // namespace spmv
