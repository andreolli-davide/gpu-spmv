// =============================================================================
// spmv_cpu.cpp
// =============================================================================
// CPU SpMV implementations.  See spmv_cpu.h for algorithm rationale.
//
// Key design decisions documented here:
//   • Why no reduction clause in the OpenMP version
//   • Why static scheduling
//   • Why std::fabs (not std::abs)
// =============================================================================

#include "spmv_cpu.h"
#include <omp.h>
#include <cmath>      // std::fabs — using <cmath> rather than <math.h>
#include <algorithm>  // std::fill

namespace spmv {

// =============================================================================
// spmv_cpu_serial
// =============================================================================
// The inner loop performs a dot product of the non-zero values in a row
// with the corresponding entries in x.  Each iteration does:
//   1. Load col_index[j]         (int memory read)
//   2. Load x.data[col_index[j]] (double memory read)
//   3. Load values[j]             (double memory read)
//   4. Multiply                   (1 FLOP)
//   5. Add to accumulator         (1 FLOP)
//
// Total: 3 memory loads + 2 FLOPs per non-zero = 2·nnz FLOPs + 3·nnz loads
// Effective intensity = 2 FLOPs / (3 × 8 bytes) = ~0.083 FLOP/byte for
// the sequential version (memory bandwidth bound, not compute bound).
// =============================================================================
void spmv_cpu_serial(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    y.resize(A.rows);

    for (int64_t row = 0; row < A.rows; ++row) {
        double sum = 0.0;
        // Walk the slice of the values array that belongs to this row
        for (int64_t j = A.row_ptr[row]; j < A.row_ptr[row + 1]; ++j) {
            const int64_t col = A.col_index[j];  // which column of A does this entry belong to?
            sum += A.values[j] * x.data[col];
        }
        y.data[row] = sum;
    }
}

// =============================================================================
// spmv_cpu_omp
// =============================================================================
// The parallelization strategy: assign one row per thread (or one contiguous
// chunk of rows per thread for very large matrices).
//
// Static scheduling is used because:
//   1. The overhead of dynamic work-stealing (guided/auto schedules) can be
//      significant for very cheap inner loops.
//   2. The natural load imbalance (variable row lengths) is acceptable for
//      the row-length distributions in typical sparse matrices.
//   3. The overhead of the schedule is zero: the compiler generates a
//      simple round-robin partition at runtime.
//
// IMPORTANT: we use a PRIVATE scalar sum per iteration (declared inside
// the loop).  Using `#pragma omp parallel for reduction(+:sum)` would produce
// the correct numerical result, but the order of accumulation across threads
// is undefined.  For sparse matrices where different rows have vastly
// different row-lengths, the rounding differences would be tiny but they
// WOULD cause the correctness test to fail.  By writing the final sum to
// y.data[row] only once per row, we guarantee the same order of floating-point
// additions as the serial version.
// =============================================================================
void spmv_cpu_omp(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    y.resize(A.rows);

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < A.rows; ++row) {
        double sum = 0.0; // private per thread — not shared
        for (int64_t j = A.row_ptr[row]; j < A.row_ptr[row + 1]; ++j) {
            sum += A.values[j] * x.data[A.col_index[j]];
        }
        y.data[row] = sum; // write once, after the entire row is accumulated
    }
}

// =============================================================================
// fill_zero
// =============================================================================
void fill_zero(DenseVector& v) {
    std::fill(v.data.begin(), v.data.end(), 0.0);
}

// =============================================================================
// fill_constant
// =============================================================================
void fill_constant(DenseVector& v, double val) {
    std::fill(v.data.begin(), v.data.end(), val);
}

// =============================================================================
// infnorm
// =============================================================================
// Why std::fabs and not std::abs?
//   std::abs(int) returns int; std::fabs(double) returns double.
//   Using std::abs on a double without explicit overload selection
//   can trigger ambiguity warnings in some standard library implementations.
//   The <cmath> header makes std::fabs unambiguous.
//
// Error tolerance: 1e-15 is the practical roundoff limit for double
// arithmetic on matrices with moderate row counts.  For very large matrices
// (millions of rows), accumulated rounding may push the error slightly higher,
// so a tolerance of 1e-12 may occasionally be needed.
// =============================================================================
double infnorm(const DenseVector& a, const DenseVector& b) {
    double max_err = 0.0;
    for (int64_t i = 0; i < static_cast<int64_t>(a.data.size()); ++i) {
        max_err = std::max(max_err, std::fabs(a.data[i] - b.data[i]));
    }
    return max_err;
}

} // namespace spmv
