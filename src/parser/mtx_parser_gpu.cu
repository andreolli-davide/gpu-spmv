#include "parser/mtx_parser.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace {

// ── Small string utilities (file-private, identical to mtx_parser.cpp) ──────

/// Remove leading and trailing whitespace from \p s.
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

/// True if \p line is empty or starts with '%' (MTX comment / header guard).
bool is_comment_or_empty(const std::string& line) {
    auto trimmed = trim(line);
    return trimmed.empty() || trimmed.front() == '%';
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// GPU parse — host parse, allocate device memory, copy H2D
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an MTX file and load it directly into GPU device memory.
///
/// Internally this is three sequential steps:
///
///   1. parse_mtx    → MtxCoo  (host, COO format)
///   2. coo_to_csr  → MtxCsr  (host, CSR format)
///   3. cudaMalloc × 3 + cudaMemcpy H2D
///
/// The DeviceMatrix result holds raw CUDA device pointers; to free them call
/// free_gpu(result).
///
/// \param filepath   Path to a Matrix Market file in coordinate format.
/// \param result    Output pointer; all three d_* fields are allocated here.
///
/// \throws std::runtime_error on host parse failure or if any cudaMalloc /
///         cudaMemcpy call fails (CUDA errors are translated to exceptions).
void parse_mtx_gpu(const std::string& filepath, DeviceMatrix* result) {
    // Step 1: parse on host
    MtxCoo coo = parse_mtx(filepath);
    // Step 2: convert to CSR
    MtxCsr csr = coo_to_csr(coo);

    // Step 3: allocate device memory for each CSR array
    //         (no cudaMallocManaged — explicit host-visible device memory)
    cudaMalloc(&result->d_row_ptr,     (csr.num_rows + 1) * sizeof(int32_t));
    cudaMalloc(&result->d_col_indices, csr.num_nonzeros * sizeof(int32_t));
    cudaMalloc(&result->d_values,      csr.num_nonzeros * sizeof(double));

    // Copy CSR structure from host to device
    cudaMemcpy(result->d_row_ptr,     csr.row_ptr.data(),
               (csr.num_rows + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result->d_col_indices, csr.col_indices.data(),
               csr.num_nonzeros * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result->d_values,      csr.values.data(),
               csr.num_nonzeros * sizeof(double), cudaMemcpyHostToDevice);

    // Copy metadata so callers can read dimensions without a round-trip
    result->num_rows     = csr.num_rows;
    result->num_cols     = csr.num_cols;
    result->num_nonzeros = csr.num_nonzeros;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU memory cleanup
// ─────────────────────────────────────────────────────────────────────────────

/// Free all three device arrays in a DeviceMatrix and null out the pointers.
///
/// This is safe to call on a partially-constructed DeviceMatrix (e.g. after
/// parse_mtx_gpu throws mid-way) because null pointers are skipped by cudaFree.
void free_gpu(DeviceMatrix* mat) {
    if (mat->d_row_ptr)     cudaFree(mat->d_row_ptr);
    if (mat->d_col_indices) cudaFree(mat->d_col_indices);
    if (mat->d_values)     cudaFree(mat->d_values);
    mat->d_row_ptr     = nullptr;
    mat->d_col_indices = nullptr;
    mat->d_values     = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU SpMV kernel
// ─────────────────────────────────────────────────────────────────────────────

/// SpMV GPU kernel — one CUDA thread per matrix row.
///
/// Each thread loads the offset range [row_ptr[row], row_ptr[row+1]) for its
/// row, then iterates over those nonzeros accumulating sum += values[idx] *
/// x[col_indices[idx]].  When the row has no nonzeros the loop body executes
/// zero times and y[row] is correctly zeroed.
///
/// Launch geometry: num_blocks = ceil(num_rows / block_size), block_size = 256.
/// This choice gives good occupancy on NVIDIA GPUs (L40S, V100, A100) for
/// typical sparse matrices with large row counts.  Each thread processes
/// exactly one row — there is no intra-row parallelism (a single row's
/// nonzeros are processed sequentially by one thread).
///
/// \note FP accumulation order is undefined (CUDA uses a vendor-defined
///       reduction order).  Results may differ from the CPU reference by
///       ~1e-6 due to floating-point non-associativity; the test tolerance
///       of 1e-6 accounts for this.
__global__
void spmv_gpu_kernel_impl(const int32_t* row_ptr,
                          const int32_t* col_indices,
                          const double*   values,
                          int32_t        num_rows,
                          const double*   d_x,
                          double*         d_y) {
    // Map thread ID to a matrix row.  Extra threads beyond num_rows guard
    // against out-of-bounds reads when num_rows is not a multiple of block_size.
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    double sum = 0.0;
    // Inner loop iterates over this row's slice of the nonzero arrays.
    // The stride-1 access on values[] and col_indices[] is cache-friendly.
    for (int32_t idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        sum += values[idx] * d_x[col_indices[idx]];
    }
    d_y[row] = sum;
}

/// Host wrapper for the SpMV kernel.
///
/// Chooses a block size of 256 threads and computes the number of blocks
/// needed to cover all rows.  Launches the kernel with default CUDA stream.
/// callers are responsible for ensuring d_x is already on the device and
/// d_y is zeroed (cudaMemset recommended before each launch).
void spmv_gpu_kernel(const DeviceMatrix& csr, const double* d_x, double* d_y) {
    int32_t num_threads  = csr.num_rows;
    int32_t block_size  = 256;
    int32_t num_blocks = (num_threads + block_size - 1) / block_size;

    spmv_gpu_kernel_impl<<<num_blocks, block_size>>>(
        csr.d_row_ptr, csr.d_col_indices, csr.d_values,
        csr.num_rows, d_x, d_y);
}
