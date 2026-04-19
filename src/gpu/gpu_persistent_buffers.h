// =============================================================================
// gpu_persistent_buffers.h
// =============================================================================
// GPU Persistent Buffer Management for SpMV
//
// Problem: Per-call cudaMalloc/cudaFree overhead can be 50%+ of total SpMV time
// for small/medium matrices. Each spmv_gpu_v2() call:
//   1. cudaMalloc for matrix, x vector, y vector
//   2. cudaMemcpy H2D for matrix and x
//   3. Kernel execution (microseconds)
//   4. cudaMemcpy D2H for y
//   5. cudaFree for all three
//
// Solution: PersistentBufferManager maintains GPU buffers across SpMV calls.
//   - Matrix uploaded once, reused for all subsequent SpMV calls
//   - Vector x uploaded each call (necessary - it's the input)
//   - Vector y downloaded each call (necessary - it's the output)
//
// Usage pattern:
//   PersistentBufferManager buf;
//   buf.upload_matrix(A);              // Once per matrix
//   for (int iter = 0; iter < max_iter; ++iter) {
//       buf.upload_vector_x(x);         // Each SpMV
//       spmv_gpu_v2_persistent(buf, x, y);
//       buf.download_vector_y(y);       // Each SpMV
//   }
//
// Based on: Greathouse & Daga '14, Liu & Vinter '15
//
// =============================================================================

#ifndef GPU_PERSISTENT_BUFFERS_H
#define GPU_PERSISTENT_BUFFERS_H

#include "sparse_matrix.h"  // SparseMatrix, DenseVector
#include "gpu_utils.h"      // DeviceMatrix, DeviceVector

namespace spmv {

// =============================================================================
// PersistentBufferManager — Maintains GPU buffers across SpMV calls
// =============================================================================
// Eliminates per-call malloc/free overhead by reusing GPU allocations.
//
// Thread-safety: Not thread-safe. Each buffer manager should be owned by
// a single thread, or synchronized externally.
//
// Memory: Allocations persist until free_all() or destructor is called.
//         Call free_all() before destructing to avoid memory leaks.
//
struct PersistentBufferManager {
    DeviceMatrix matrix;      // Matrix data (reused across calls)
    DeviceVector vector_x;    // Input vector buffer
    DeviceVector vector_y;    // Output vector buffer
    bool is_initialized;      // Tracks if matrix buffer is valid

    PersistentBufferManager();
    ~PersistentBufferManager();

    // Disable copy (GPU pointers can't be copied)
    PersistentBufferManager(const PersistentBufferManager&) = delete;
    PersistentBufferManager& operator=(const PersistentBufferManager&) = delete;

    // Enable move (for use in containers)
    PersistentBufferManager(PersistentBufferManager&& other) noexcept;
    PersistentBufferManager& operator=(PersistentBufferManager&& other) noexcept;

    // Initialize/refresh matrix buffers (only when matrix changes)
    // If already initialized with same dimensions, this is a no-op.
    // If dimensions differ, reallocates.
    void upload_matrix(const SparseMatrix& A);

    // Transfer vector x to GPU (called each SpMV)
    // Automatically reallocates if size changed.
    void upload_vector_x(const DenseVector& x);

    // Pre-allocate output vector y buffer (call once before benchmarks)
    void allocate_output(int64_t rows);

    // Transfer result vector y back to host (called each SpMV)
    void download_vector_y(DenseVector& y);

    // Get pointer to output vector data on GPU (for direct kernel use)
    double* get_output_data_ptr() { return vector_y.d_data; }

    // Get pointer to matrix device structure (for direct kernel use)
    const DeviceMatrix& get_matrix() const { return matrix; }

    // Get pointer to input vector data on GPU (for direct kernel use)
    const DeviceVector& get_vector_x() const { return vector_x; }

    // Cleanup all GPU memory
    void free_all();

    // Check if manager has valid matrix buffer
    bool has_matrix() const { return is_initialized; }
};

// =============================================================================
// spmv_gpu_v2_persistent — SpMV using persistent buffers
// =============================================================================
// High-performance SpMV that uses pre-allocated GPU buffers.
// Eliminates allocation overhead for repeated SpMV calls.
//
// @param buf     Persistent buffer manager with uploaded matrix
// @param x       Input vector (uploaded to GPU via buf.upload_vector_x)
// @param y       Output vector (downloaded from GPU via buf.download_vector_y)
//
// Usage:
//   PersistentBufferManager buf;
//   buf.upload_matrix(A);
//   for (...) {
//       buf.upload_vector_x(x);
//       spmv_gpu_v2_persistent(buf, x, y);
//       buf.download_vector_y(y);
//   }
//
void spmv_gpu_v2_persistent(PersistentBufferManager& buf,
                           const DenseVector& x, DenseVector& y);

} // namespace spmv

#endif // GPU_PERSISTENT_BUFFERS_H