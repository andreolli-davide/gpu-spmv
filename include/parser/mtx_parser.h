#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

/// Sparse matrix in Coordinate (COO) format — main entry point for parsing.
/// Data is stored as three parallel arrays of equal length: one row index
/// per nonzero, one column index per nonzero, and one value per nonzero.
/// This format is simple but not memory-efficient for SpMV because row
/// indices are stored explicitly rather than implied by structure.
struct MtxCoo {
    int32_t num_rows;
    int32_t num_cols;
    int32_t num_nonzeros;
    std::vector<int32_t> row_indices;  ///< 0-indexed row of each nonzero
    std::vector<int32_t> col_indices;  ///< 0-indexed column of each nonzero
    std::vector<float> values;         ///< numeric value of each nonzero
};

/// Sparse matrix in Compressed Sparse Row (CSR) format — the workhorse format
/// for efficient SpMV.  Row offsets are stored in a "row pointer" array of
/// length num_rows+1; nonzero K belongs to row R where
///   row_ptr[R] <= K < row_ptr[R+1]
/// Column index and value for nonzero K are at col_indices[K] and values[K].
/// This compression eliminates per-row row-index storage, saving memory and
/// enabling stride-1 access patterns in SpMV.
struct MtxCsr {
    int32_t num_rows;
    int32_t num_cols;
    int32_t num_nonzeros;
    std::vector<int32_t> row_ptr;       ///< size = num_rows + 1; row_ptr[i] is start of row i
    std::vector<int32_t> col_indices;   ///< column index for each nonzero (0-indexed)
    std::vector<float> values;         ///< numeric value for each nonzero
};

/// GPU-resident sparse matrix in CSR format.
/// Unlike the host structs above, this version stores raw device pointers
/// (CUDA device memory) instead of std::vector — no implicit allocation or
/// managed migration.  Caller is responsible for allocating and freeing via
/// parse_mtx_gpu() / free_gpu().
struct DeviceMatrix {
    int32_t num_rows;
    int32_t num_cols;
    int32_t num_nonzeros;
    int32_t* d_row_ptr;     ///< device pointer, size = num_rows + 1
    int32_t* d_col_indices; ///< device pointer, size = num_nonzeros
    float*   d_values;      ///< device pointer, size = num_nonzeros
};

// ─────────────────────────────────────────────────────────────────────────────
// Parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a Matrix Market (.mtx) file into a host-side COO structure.
/// Throws std::runtime_error if the file cannot be opened or has an invalid
/// format.  MTX 1-based indices are converted to 0-based internally.
/// Only the "coordinate" variant (real/complex pattern) is supported.
MtxCoo parse_mtx(const std::string& filepath);

/// Convenience: parse directly to CSR by chaining parse_mtx + coo_to_csr.
MtxCsr parse_mtx_csr(const std::string& filepath);

// ─────────────────────────────────────────────────────────────────────────────
// Format conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert COO → CSR via counting sort + prefix sum.
/// Runtime: O(num_nonzeros + num_rows).  No temporary storage beyond a
/// working copy of row_ptr used during the fill phase.
MtxCsr coo_to_csr(const MtxCoo& coo);

/// CSR → COO by expanding each row's run of nonzeros into individual entries.
/// Runtime: O(num_nonzeros).  Mostly useful for validation against parse_mtx.
MtxCoo csr_to_coo(const MtxCsr& csr);

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference SpMV
// ─────────────────────────────────────────────────────────────────────────────

/// CPU SpMV: y = A*x  using CSR layout.
/// Straightforward row-by-row accumulation.  Serves as the reference
/// implementation for correctness verification against the GPU kernel.
void spmv_cpu(const MtxCsr& csr, const float* x, float* y);

/// CPU SpMV: y = A*x  using COO layout.
/// Accumulator is zeroed once upfront, then each nonzero adds to its row.
/// Note: this accumulates row-by-row but the inner loop order differs from
/// CSR (no guarantee on traversal order), which means floating-point
/// accumulation order varies between COO and CSR — intentional, to catch
/// FP-associativity bugs in GPU kernels.
void spmv_cpu(const MtxCoo& coo, const float* x, float* y);

// ─────────────────────────────────────────────────────────────────────────────
// GPU parse + SpMV
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an MTX file and transfer it directly into GPU device memory.
/// Internally: parse_mtx (host COO) → coo_to_csr (host CSR) → cudaMalloc
/// for each array → cudaMemcpy H2D.  The result holds raw CUDA pointers;
/// free them by calling free_gpu().
void parse_mtx_gpu(const std::string& filepath, DeviceMatrix* result);

/// Free all three device arrays in a DeviceMatrix and null the pointers.
/// Safe to call on a partially-constructed DeviceMatrix (skips null pointers).
void free_gpu(DeviceMatrix* mat);

/// GPU SpMV kernel — one thread per matrix row, each thread accumulates
/// its row's nonzeros independently.  Launch configuration is chosen inside
/// the function (block_size=256, enough blocks to cover all rows).
void spmv_gpu_kernel(const DeviceMatrix& csr, const float* d_x, float* d_y);
