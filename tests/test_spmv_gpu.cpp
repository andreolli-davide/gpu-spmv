// =============================================================================
// test_spmv_gpu.cpp
// =============================================================================
// Correctness test for the GPU SpMV implementation.
//
// What this test verifies
// -----------------------
// GPU SpMV produces BITWISE identical results to the CPU reference
// (infnorm < 1e-15).
//
// The test compares spmv_gpu_v1() against spmv_cpu_serial() as the
// golden reference implementation.
//
// Why this is sufficient
// -----------------------
// For a known test matrix (the 5×5 tridiagonal), we can manually verify
// that A·ones = [3, 5, 4, 4, 2]^T.  Running the test with x = all-ones
// confirms this without needing any external reference.
//
// Usage
// -----
//   ./test_spmv_gpu [path/to/matrix.mtx]
//
//   Without arguments: uses the built-in 5×5 symmetric tridiagonal matrix.
//                      The all-ones input vector makes the result trivially
//                      verifiable: each y[i] = sum of row i of A.
//
//   path/to/matrix.mtx: loads and tests with a real Matrix Market file.
//
//   Exit codes:  0 = all checks passed, 1 = check failed
// =============================================================================

#include <cstdio>    // std::printf, std::fprintf
#include <cstdlib>   // std::exit
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "spmv_gpu.h"
#include "spmv_cpu.h"
#include "timer.h"
#include "matrix_market.h"

namespace {

// =============================================================================
// make_tiny_test_matrix — built-in 5×5 symmetric tridiagonal
// =============================================================================
// This matrix is small enough to verify by hand, and its properties make
// the all-ones input case especially easy to check:
//
//   A = | 2  1  .  .  . |
//       | 1  3  1  .  . |
//       | .  1  2  1  . |
//       | .  .  1  2  1 |
//       | .  .  .  1  1 |
//
// With x = [1, 1, 1, 1, 1]:
//   y[0] = 2·1 + 1·1           = 3
//   y[1] = 1·1 + 3·1 + 1·1     = 5
//   y[2] = 1·1 + 2·1 + 1·1     = 4   (symmetric row 2 = row 1)
//   y[3] = 1·1 + 2·1 + 1·1     = 4
//   y[4] = 1·1 + 1·1           = 2
//
// This gives us a known-good result even with no .mtx file present.
// =============================================================================
spmv::COO_SparseMatrix make_tiny_test_matrix() {
    // 5×5 symmetric tridiagonal with 13 non-zeros:
    //
    //   Row 0: (0,0)=2, (0,1)=1       Row 3: (3,2)=1, (3,3)=2, (3,4)=1
    //   Row 1: (1,0)=1, (1,1)=3, (1,2)=1  Row 4: (4,3)=1, (4,4)=1
    //   Row 2: (2,1)=1, (2,2)=2, (2,3)=1
    spmv::COO_SparseMatrix coo(5, 5, 13);
    coo.allocate();

    double vals[] = {2, 1,   1, 3, 1,   1, 2, 1,   1, 2, 1,   1, 1};
    int64_t rows[] = {0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4};
    int64_t cols[] = {0, 1,   0, 1, 2,   1, 2, 3,   2, 3, 4,   3, 4};

    for (int64_t i = 0; i < 13; ++i) {
        coo.values[i] = vals[i];
        coo.row[i]    = rows[i];
        coo.col[i]    = cols[i];
    }
    return coo;
}

// =============================================================================
// print_vector — helper: print up to N entries of a vector
// =============================================================================
void print_vector(const spmv::DenseVector& v, int64_t max_entries = 8) {
    const int64_t n = std::min<int64_t>(v.size, max_entries);
    std::cout << "  [ ";
    for (int64_t i = 0; i < n; ++i) {
        std::cout << v.data[i];
        if (i < n - 1) std::cout << ", ";
    }
    if (v.size > n) std::cout << ", ...";
    std::cout << " ]  (" << v.size << " entries)\n";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(6);

    // =========================================================================
    // Step 1 — Load the matrix
    // =========================================================================
    spmv::SparseMatrix A; // CSR matrix

    if (argc > 1) {
        const std::string path = argv[1];
        std::cout << "Loading matrix from: " << path << "\n";
        try {
            A = spmv::parse_matrix_market(path);
        } catch (const std::exception& e) {
            std::cerr << "ERROR parsing matrix: " << e.what() << "\n";
            std::exit(1);
        }
    } else {
        std::cout << "No .mtx file provided — using built-in 5x5 test matrix\n";
        A = spmv::coo_to_csr(make_tiny_test_matrix());
    }

    std::cout << "Matrix: " << A.rows << " × " << A.cols
              << ", nnz = " << A.nnz
              << ", memory = " << A.memory_bytes() << " bytes\n";

    // =========================================================================
    // Step 2 — Build the input vector x (all-ones)
    // =========================================================================
    spmv::DenseVector x(A.cols);
    spmv::fill_constant(x, 1.0);
    std::cout << "x: all-ones vector\n";

    // =========================================================================
    // Step 3 — Run CPU serial SpMV (golden reference)
    // =========================================================================
    spmv::DenseVector y_cpu(A.rows);
    spmv::CPUTimer t_cpu;
    t_cpu.start();
    spmv::spmv_cpu_serial(A, x, y_cpu);
    t_cpu.stop();
    std::cout << "CPU serial time: " << t_cpu.elapsed_ms() << " ms\n";

    // =========================================================================
    // Step 4 — Run GPU SpMV
    // =========================================================================
    spmv::DenseVector y_gpu(A.rows);
    spmv::GPUTimer t_gpu;
    t_gpu.start();
    bool gpu_ok = spmv::spmv_gpu_v1(A, x, y_gpu);
    t_gpu.stop();
    if (!gpu_ok) {
        std::cerr << "FAIL: GPU SpMV returned error\n";
        std::exit(1);
    }
    std::cout << "GPU time:       " << t_gpu.elapsed_ms() << " ms\n";

    // =========================================================================
    // Step 5 — Correctness check (infnorm)
    // =========================================================================
    const double err = spmv::infnorm(y_cpu, y_gpu);
    std::cout << "\n--- Correctness ---\n";
    std::cout << "CPU-GPU L-inf error: " << err << "\n";

    if (err > 1e-15) {
        std::cerr << "FAIL: CPU and GPU results differ (err = " << err << ")\n";
        std::cerr << "This indicates a bug in the GPU implementation.\n";
        std::exit(1);
    }
    std::cout << "PASS: CPU and GPU results match (infnorm < 1e-15)\n";

    // =========================================================================
    // Step 6 — Optional: manual verification for the 5×5 test matrix
    // =========================================================================
    if (A.rows == 5) {
        // Expected y = A * ones = row-sums of A
        double expected[] = {3.0, 5.0, 4.0, 4.0, 2.0};
        std::cout << "\n--- Manual verification (5×5 all-ones input) ---\n";
        bool manual_ok = true;
        for (int64_t i = 0; i < 5; ++i) {
            const double diff = std::fabs(y_cpu.data[i] - expected[i]);
            std::cout << "  y[" << i << "] = " << y_cpu.data[i]
                      << " (expected " << expected[i] << ")";
            if (diff > 1e-15) {
                std::cout << "  MISMATCH";
                manual_ok = false;
            }
            std::cout << "\n";
        }
        if (!manual_ok) {
            std::cerr << "Manual verification FAILED — check the matrix data.\n";
            std::exit(1);
        }
        std::cout << "Manual verification PASSED\n";
    }

    // =========================================================================
    // Step 7 — Print result summary
    // =========================================================================
    std::cout << "\n--- Result summary ---\n";
    std::cout << "CPU output (first " << std::min<int64_t>(8, A.rows) << " entries):\n  ";
    print_vector(y_cpu);

    std::cout << "\nAll tests PASSED.\n";
    return 0;
}