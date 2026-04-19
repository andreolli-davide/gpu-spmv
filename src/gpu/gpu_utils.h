// =============================================================================
// gpu_utils.h
// =============================================================================
// GPU memory management utilities for SpMV.
//
// Provides CUDA memory allocation, transfer, and cleanup for CSR sparse matrix
// and dense vector data. All functions use the existing SparseMatrix and
// DenseVector types from sparse_matrix.h.
//
// CUDA Error Handling
// -------------------
// All CUDA calls are wrapped with CUDA_CHECK() which aborts on failure.
// For kernel launches, use the cudaGetLastError() check after synchronization.
//
// Memory Layout
// -------------
// Device matrices store CSR data in separate arrays:
//   - d_values:  nnz doubles (non-zero values)
//   - d_col_index: nnz int64_t (column indices)
//   - d_row_ptr:  (rows+1) int64_t (row pointers)
//
// =============================================================================

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cstdint>   // int64_t
#include <cstdio>    // fprintf, stderr
#ifdef __CUDACC__
#include <cuda_runtime.h>  // CUDA runtime API
#endif

#include "../common/sparse_matrix.h"  // SparseMatrix, DenseVector

namespace spmv {

// =============================================================================
// DeviceMatrix — GPU storage for a CSR sparse matrix
// =============================================================================
// Holds pointers to device memory for the three CSR arrays.
// Use allocate_device_matrix() to allocate and populate.
// Use free_device_matrix() to release.
//
struct DeviceMatrix {
    double*   d_values    = nullptr;  // non-zero values, size nnz
    int64_t*  d_col_index = nullptr;  // column indices, size nnz
    int64_t*  d_row_ptr   = nullptr;  // row pointers, size rows+1

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz  = 0;

    // Returns true if all pointers are non-null (allocated).
    bool is_allocated() const;
};

// =============================================================================
// DeviceVector — GPU storage for a dense vector
// =============================================================================
struct DeviceVector {
    double*  d_data = nullptr;  // elements, size size
    int64_t   size  = 0;        // number of elements

    bool is_allocated() const;
};

// =============================================================================
// allocate_device_matrix
// =============================================================================
// Allocates GPU memory for a CSR matrix and copies host data to device.
//
// @param h_matrix  Host sparse matrix in CSR format (source)
// @return          DeviceMatrix with allocated GPU pointers
//
// Uses cudaMalloc for device allocation and cudaMemcpy for H2D transfer.
// All three arrays are allocated and copied atomically.
//
DeviceMatrix allocate_device_matrix(const SparseMatrix& h_matrix);

// =============================================================================
// copy_matrix_to_device
// =============================================================================
// Copies matrix data from host to already-allocated device storage.
// Use after allocate_device_matrix() or to update an existing device matrix.
//
// @param h_matrix    Host sparse matrix (source)
// @param d_matrix    DeviceMatrix with pre-allocated storage (destination)
// @param async      If true, uses cudaMemcpyAsync for overlapped transfer
// @param stream     CUDA stream for async transfers (ignored if async=false)
//
void copy_matrix_to_device(const SparseMatrix& h_matrix,
                           DeviceMatrix& d_matrix,
                           bool async = false,
                           cudaStream_t stream = 0);

// =============================================================================
// copy_vector_to_device
// =============================================================================
// Allocates GPU memory for a dense vector and copies data from host to device.
//
// @param h_vec     Host dense vector (source)
// @return          DeviceVector with allocated GPU pointer
//
DeviceVector copy_vector_to_device(const DenseVector& h_vec);

// =============================================================================
// copy_vector_to_device_async
// =============================================================================
// Async version: copies data to pre-allocated device vector.
// Use when you have an existing DeviceVector and want pipelined transfers.
//
// @param h_vec     Host dense vector (source)
// @param d_vec     DeviceVector with pre-allocated storage (destination)
// @param stream   CUDA stream for the transfer
//
void copy_vector_to_device_async(const DenseVector& h_vec,
                                 DeviceVector& d_vec,
                                 cudaStream_t stream);

// =============================================================================
// copy_vector_from_device
// =============================================================================
// Copies data from device vector back to host.
//
// @param d_vec     Device vector (source)
// @param h_vec    Host dense vector with matching size (destination)
// @param async    If true, uses cudaMemcpyAsync
// @param stream   CUDA stream for async transfers
//
void copy_vector_from_device(const DeviceVector& d_vec,
                             DenseVector& h_vec,
                             bool async = false,
                             cudaStream_t stream = 0);

// =============================================================================
// free_device_matrix
// =============================================================================
// Frees all GPU memory associated with a DeviceMatrix.
// Safe to call on null pointers (no-op).
//
void free_device_matrix(DeviceMatrix& d_matrix);

// =============================================================================
// free_device_vector
// =============================================================================
// Frees GPU memory for a DeviceVector.
// Safe to call on null pointers (no-op).
//
void free_device_vector(DeviceVector& d_vec);

// =============================================================================
// CUDA_CHECK macro
// =============================================================================
// Error checking wrapper for synchronous CUDA calls.
//
// Usage:
//   CUDA_CHECK(cudaMalloc(&ptr, size));    // aborts on error
//   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
//
// For async operations (kernel launches), use:
//   kernel<<<...>>>(...);
//   CUDA_CHECK(cudaGetLastError());        // check kernel launch errors
//
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA ERROR] %s (%s:%d)\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =============================================================================
// CUDA_TIMER macros — scoped kernel timing
// =============================================================================
// Lightweight timing helper that integrates with the existing GPUTimer.
//
// Usage:
//   GPUTimer timer;
//   timer.start();
//   kernel<<<blocks, threads, 0, stream>>>(...);
//   cudaStreamSynchronize(stream);
//   timer.stop();
//   double ms = timer.elapsed_ms();
//
// Note: GPUTimer is defined in timer.h and handles CUDA event creation/
// destruction automatically via RAII.
//

} // namespace spmv

#endif // GPU_UTILS_H