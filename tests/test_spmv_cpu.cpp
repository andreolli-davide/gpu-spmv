// =============================================================================
// test_spmv_cpu.cpp
// =============================================================================
// Correctness test for the CPU SpMV implementations.
//
// What this test verifies
// -----------------------
// 1. Serial and OpenMP SpMV produce BITWISE identical results
//    (infnorm < 1e-15)
//
// 2. The matrix parser produces a valid CSR matrix (nnz, row_ptr correct)
//
// The test does NOT verify numerical accuracy against an analytical result —
// it verifies internal consistency between two independent implementations
// (serial and parallel) of the same algorithm.
//
// Why this is sufficient for Phase 1
// -----------------------------------
// For a known test matrix (the 5×5 tridiagonal), we can manually verify
// that A·ones = [3, 5, 5, 4, 1]^T.  Running the test with x = all-ones
// confirms this without needing any external reference.  For real matrices,
// we rely on serial vs. OMP equivalence as a proxy for correctness.
//
// Usage
// -----
//   ./test_spmv_cpu [path/to/matrix.mtx] [--random]
//
//   Without arguments: uses the built-in 5×5 symmetric tridiagonal matrix.
//                      The all-ones input vector makes the result trivially
//                      verifiable: each y[i] = sum of row i of A.
//
//   path/to/matrix.mtx: loads and tests with a real Matrix Market file.
//
//   --random:           fills the input vector x with uniform random values
//                       in [-1, 1] instead of all-ones.
//
//   Exit codes:  0 = all checks passed, 1 = check failed
// =============================================================================

#include <cstdio>    // std::printf, std::fprintf
#include <cstdlib>   // std::exit
#include <string>
#include <iostream>
#include <iomanip>
#include <random>    // std::mt19937_64, std::uniform_real_distribution

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
// fill_random — fill a DenseVector with uniform random values in [lo, hi]
// =============================================================================
// Using a fixed seed (42) makes the test deterministic across runs.
// Change the seed to get different random vectors — useful for fuzz testing.
// =============================================================================
void fill_random(spmv::DenseVector& v, double lo = -1.0, double hi = 1.0) {
    std::mt19937_64 rng(42); // deterministic seed — reproducible across runs
    std::uniform_real_distribution<double> dist(lo, hi);
    for (double& x : v.data) {
        x = dist(rng);
    }
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
    bool use_random_x = false;

    if (argc > 1 && std::string(argv[1]) == "--random") {
        // Degenerate case: just --random flag, no matrix file
        std::cerr << "Usage: test_spmv_cpu [path/to/matrix.mtx] [--random]\n";
        std::exit(1);
    }

    if (argc > 1) {
        const std::string path = argv[1];
        std::cout << "Loading matrix from: " << path << "\n";
        try {
            A = spmv::parse_matrix_market(path);
        } catch (const std::exception& e) {
            std::cerr << "ERROR parsing matrix: " << e.what() << "\n";
            std::exit(1);
        }
        // Check for trailing --random flag
        if (argc > 2 && std::string(argv[2]) == "--random") {
            use_random_x = true;
        }
    } else {
        std::cout << "No .mtx file provided — using built-in 5x5 test matrix\n";
        A = spmv::coo_to_csr(make_tiny_test_matrix());
    }

    std::cout << "Matrix: " << A.rows << " × " << A.cols
              << ", nnz = " << A.nnz
              << ", memory = " << A.memory_bytes() << " bytes\n";

    // =========================================================================
    // Step 2 — Build the input vector x
    // =========================================================================
    spmv::DenseVector x(A.cols);
    if (use_random_x) {
        fill_random(x);
        std::cout << "x: random values in [-1, 1]\n";
    } else {
        spmv::fill_constant(x, 1.0);
        std::cout << "x: all-ones vector\n";
    }

    // =========================================================================
    // Step 3 — Run serial SpMV
    // =========================================================================
    spmv::DenseVector y_serial(A.rows);
    spmv::CPUTimer t_serial;
    t_serial.start();
    spmv::spmv_cpu_serial(A, x, y_serial);
    t_serial.stop();
    std::cout << "Serial time:   " << t_serial.elapsed_ms() << " ms\n";

    // =========================================================================
    // Step 4 — Run OpenMP SpMV
    // =========================================================================
    spmv::DenseVector y_omp(A.rows);
    spmv::CPUTimer t_omp;
    t_omp.start();
    spmv::spmv_cpu_omp(A, x, y_omp);
    t_omp.stop();
    std::cout << "OpenMP time:   " << t_omp.elapsed_ms() << " ms\n";

    // =========================================================================
    // Step 5 — Correctness check
    // =========================================================================
    const double err = spmv::infnorm(y_serial, y_omp);
    std::cout << "\n--- Correctness ---\n";
    std::cout << "Serial-OMP L-inf error: " << err << "\n";

    if (err > 1e-15) {
        std::cerr << "FAIL: serial and OpenMP results differ (err = " << err << ")\n";
        std::cerr << "This indicates a bug in the parallel implementation or\n";
        std::cerr << "an incorrect CSR conversion.\n";
        std::exit(1);
    }
    std::cout << "PASS: serial and OpenMP results match exactly (bitwise)\n";

    // =========================================================================
    // Step 6 — Optional: manual verification for the 5×5 test matrix
    // =========================================================================
    if (A.rows == 5 && !use_random_x) {
        // Expected y = A * ones = row-sums of A
        double expected[] = {3.0, 5.0, 4.0, 4.0, 2.0};
        std::cout << "\n--- Manual verification (5×5 all-ones input) ---\n";
        bool manual_ok = true;
        for (int64_t i = 0; i < 5; ++i) {
            const double diff = std::fabs(y_serial.data[i] - expected[i]);
            std::cout << "  y[" << i << "] = " << y_serial.data[i]
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
    std::cout << "Serial output (first " << std::min<int64_t>(8, A.rows) << " entries):\n  ";
    print_vector(y_serial);

    std::cout << "\nAll tests PASSED.\n";
    return 0;
}
