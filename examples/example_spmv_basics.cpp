// examples/example_spmv_basics.cpp
// Run with: cmake --build build && ./build/examples/example_spmv_basics
// (No .mtx file required — uses the built-in 5x5 test matrix)

#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "spmv_cpu.h"
#include "timer.h"
#include "matrix_market.h"
#include "sparse_matrix.h"

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(6);

    // ------------------------------------------------------------------
    // Load matrix — parses an .mtx file if a path is given,
    // otherwise uses the built-in 5×5 symmetric test matrix.
    // ------------------------------------------------------------------
    spmv::SparseMatrix A;  // empty CSR matrix

    // Force the built-in path by passing no args; in real usage:
    //   A = spmv::parse_matrix_market("/path/to/matrix.mtx");
    auto coo = spmv::COO_SparseMatrix(5, 5, 13);
    coo.allocate();

    // 5×5 symmetric tridiagonal:
    // [2, 1, 0, 0, 0]
    // [1, 3, 1, 0, 0]
    // [0, 1, 2, 1, 0]
    // [0, 0, 1, 2, 1]
    // [0, 0, 0, 1, 1]
    double vals[]   = {2, 1,  1, 3, 1,  1, 2, 1,  1, 2, 1,  1, 1};
    int64_t rows[]  = {0, 0,  1, 1, 1,  2, 2, 2,  3, 3, 3,  4, 4};
    int64_t cols[]  = {0, 1,  0, 1, 2,  1, 2, 3,  2, 3, 4,  3, 4};
    for (int64_t i = 0; i < 13; ++i) {
        coo.values[i] = vals[i];
        coo.row[i]    = rows[i];
        coo.col[i]    = cols[i];
    }

    A = spmv::coo_to_csr(coo);

    std::cout << "Matrix: " << A.rows << " x " << A.cols
              << ", nnz = " << A.nnz << "\n";
    std::cout << "Memory: " << A.memory_bytes() << " bytes (host)\n\n";

    // ------------------------------------------------------------------
    // Input vector x = all ones  (A's diagonal dominance makes result easy
    // to cross-check: row 0 → 2+1=3, row 1 → 1+3+1=5, etc.)
    // ------------------------------------------------------------------
    spmv::DenseVector x(A.cols);
    spmv::fill_constant(x, 1.0);

    // ------------------------------------------------------------------
    // Serial SpMV
    // ------------------------------------------------------------------
    spmv::DenseVector y(A.rows);
    spmv::CPUTimer t;
    t.start();
    spmv::spmv_cpu_serial(A, x, y);
    t.stop();
    std::cout << "Serial time: " << t.elapsed_ms() << " ms\n";

    std::cout << "\nResult y = A * x:\n";
    for (int64_t i = 0; i < A.rows; ++i) {
        std::cout << "  y[" << i << "] = " << y.data[i] << "\n";
    }

    // ------------------------------------------------------------------
    // OpenMP SpMV
    // ------------------------------------------------------------------
    spmv::DenseVector y_omp(A.rows);
    t.start();
    spmv::spmv_cpu_omp(A, x, y_omp);
    t.stop();
    std::cout << "\nOpenMP time: " << t.elapsed_ms() << " ms\n";

    // ------------------------------------------------------------------
    // Sanity check: both should be bitwise identical
    // ------------------------------------------------------------------
    double err = spmv::infnorm(y, y_omp);
    std::cout << "L-inf error (serial vs OMP): " << err << "\n";
    if (err == 0.0) {
        std::cout << "\nPASS — serial and OpenMP results match exactly.\n";
    } else {
        std::cout << "\nFAIL — results differ.\n";
        return 1;
    }

    return 0;
}
