// =============================================================================
// example_spmv_basics.cpp
// =============================================================================
// Introductory example demonstrating the complete SpMV pipeline:
//
//   1. Build a sparse matrix in COO format (or load from .mtx)
//   2. Convert to CSR format
//   3. Create input vector x
//   4. Run serial and OpenMP SpMV
//   5. Verify both implementations agree
//   6. Print timing and output
//
// This example requires NO external .mtx files — it uses a built-in
// 5×5 symmetric tridiagonal matrix with known properties.
//
// How to build and run
// ---------------------
//   cmake --build build
//   ./build/example_spmv_basics
//
// Optional: load a real Matrix Market file instead of the built-in matrix:
//   // In main(), replace the inline matrix construction with:
//   auto A = spmv::parse_matrix_market("/path/to/your/matrix.mtx");
//
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>     // std::fabs

#include "spmv_cpu.h"
#include "timer.h"
#include "matrix_market.h"
#include "sparse_matrix.h"

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(6);

    // =========================================================================
    // Step 1 — Build (or load) a sparse matrix
    // =========================================================================
    //
    // Two ways to get a SparseMatrix:
    //
    //   (A) Load from a Matrix Market .mtx file:
    //         SparseMatrix A = spmv::parse_matrix_market("/path/to/file.mtx");
    //
    //   (B) Build inline (what we do here for the self-contained example):
    //         Step 1a: create a COO matrix (coordinate format, row/col/value triples)
    //         Step 1b: convert to CSR via coo_to_csr()
    //
    // CSR is the native format for SpMV — all subsequent steps use CSR.
    // =========================================================================

    // The 5×5 symmetric tridiagonal test matrix:
    //
    //   | 2  1   .   .   . |
    //   | 1  3   1   .   . |
    //   | .  1   2   1   . |
    //   | .   .  1   2   1 |
    //   | .   .   .   1   1 |
    //
    // This is small enough to verify by hand and is a realistic "structured
    // sparse" pattern (a tridiagonal arises in 1D PDE discretizations).
    spmv::COO_SparseMatrix coo(5, 5, 13);
    coo.allocate();

    double vals[]   = {2, 1,   1, 3, 1,   1, 2, 1,   1, 2, 1,   1, 1};
    int64_t rows[]  = {0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4};
    int64_t cols[]  = {0, 1,   0, 1, 2,   1, 2, 3,   2, 3, 4,   3, 4};

    for (int64_t i = 0; i < 13; ++i) {
        coo.values[i] = vals[i];
        coo.row[i]    = rows[i];
        coo.col[i]    = cols[i];
    }

    // Convert from COO (what we just built) to CSR (what SpMV needs)
    spmv::SparseMatrix A = spmv::coo_to_csr(coo);

    std::cout << "=== Sparse Matrix Basics ===\n\n";
    std::cout << "Matrix: " << A.rows << " × " << A.cols
              << ", nnz = " << A.nnz << "\n";
    std::cout << "Host memory: " << A.memory_bytes() << " bytes\n\n";

    std::cout << "CSR layout:\n";
    for (int64_t i = 0; i < A.rows; ++i) {
        const int64_t start = A.row_ptr[i];
        const int64_t end   = A.row_ptr[i + 1];
        std::cout << "  row " << i << ": entries at indices ["
                  << start << ", " << end << ")\n    values: ";
        for (int64_t j = start; j < end; ++j) {
            std::cout << A.values[j] << " ";
        }
        std::cout << "\n    cols: ";
        for (int64_t j = start; j < end; ++j) {
            std::cout << A.col_index[j] << " ";
        }
        std::cout << "\n";
    }

    // =========================================================================
    // Step 2 — Create the input vector x
    // =========================================================================
    //
    // x = all-ones:  chosen because A·ones = row-sum of A, which is easy to
    // verify by inspection.  For the matrix above:
    //
    //   y[0] = 2 + 1 = 3
    //   y[1] = 1 + 3 + 1 = 5
    //   y[2] = 1 + 2 + 1 = 4
    //   y[3] = 1 + 2 + 1 = 4
    //   y[4] = 1 + 1 = 2
    // =========================================================================
    spmv::DenseVector x(A.cols);
    spmv::fill_constant(x, 1.0);
    std::cout << "\nInput vector x: all-ones\n";

    // =========================================================================
    // Step 3 — Serial SpMV
    // =========================================================================
    spmv::DenseVector y_serial(A.rows);
    spmv::CPUTimer t;
    t.start();
    spmv::spmv_cpu_serial(A, x, y_serial);
    t.stop();
    std::cout << "\nSerial SpMV: " << t.elapsed_ms() << " ms\n";

    std::cout << "\nResult y = A * x:\n";
    for (int64_t i = 0; i < A.rows; ++i) {
        std::cout << "  y[" << i << "] = " << y_serial.data[i] << "\n";
    }

    // =========================================================================
    // Step 4 — OpenMP SpMV
    // =========================================================================
    spmv::DenseVector y_omp(A.rows);
    t.start();
    spmv::spmv_cpu_omp(A, x, y_omp);
    t.stop();
    std::cout << "\nOpenMP SpMV: " << t.elapsed_ms() << " ms\n";

    // =========================================================================
    // Step 5 — Correctness verification
    // =========================================================================
    const double err = spmv::infnorm(y_serial, y_omp);
    std::cout << "\nCorrectness check (serial vs OpenMP):\n";
    std::cout << "  L-inf error: " << err << "\n";

    if (err == 0.0) {
        std::cout << "\nPASS — serial and OpenMP results match exactly.\n";
    } else {
        std::cout << "\nFAIL — results differ.\n";
        return 1;
    }

    // =========================================================================
    // Bonus: verify against known expected values (for this specific matrix)
    // =========================================================================
    std::cout << "\nManual verification (x = ones, expect y = row sums):\n";
    double expected[] = {3.0, 5.0, 4.0, 4.0, 2.0};
    for (int64_t i = 0; i < A.rows; ++i) {
        std::cout << "  y[" << i << "] = " << y_serial.data[i]
                  << " (expected " << expected[i] << ") — "
                  << (std::fabs(y_serial.data[i] - expected[i]) < 1e-15 ? "OK" : "WRONG")
                  << "\n";
    }

    return 0;
}
