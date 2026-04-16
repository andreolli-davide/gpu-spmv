#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

#include "spmv_cpu.h"
#include "timer.h"
#include "matrix_market.h"

namespace {

// --------------------------------------------------------------------------
// Tiny 5x5 test matrix embedded as a fallback when no .mtx arg is given
// --------------------------------------------------------------------------
spmv::COO_SparseMatrix make_tiny_test_matrix() {
    // 5x5 with 13 non-zeros:
    // [2, 1, 0, 0, 0]
    // [1, 3, 1, 0, 0]
    // [0, 1, 2, 1, 0]
    // [0, 0, 1, 2, 1]
    // [0, 0, 0, 1, 1]
    spmv::COO_SparseMatrix coo(5, 5, 13);
    coo.allocate();
    double vals[] = {2, 1, 1, 3, 1, 1, 2, 1, 1, 2, 1, 1, 1};
    int64_t rows[] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4};
    int64_t cols[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    for (int64_t i = 0; i < 13; ++i) {
        coo.values[i] = vals[i];
        coo.row[i] = rows[i];
        coo.col[i] = cols[i];
    }
    return coo;
}

void fill_random(DenseVector& v, double lo = -1.0, double hi = 1.0) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (auto& x : v.data) x = dist(rng);
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(6);

    // ------------------------------------------------------------------
    // Load matrix from .mtx file or use embedded fallback
    // ------------------------------------------------------------------
    spmv::SparseMatrix A;
    if (argc > 1) {
        std::string path = argv[1];
        std::cout << "Loading matrix from: " << path << "\n";
        A = spmv::parse_matrix_market(path);
    } else {
        std::cout << "No .mtx file provided — using built-in 5x5 test matrix\n";
        auto coo = make_tiny_test_matrix();
        A = spmv::coo_to_csr(coo);
    }

    std::cout << "Matrix: " << A.rows << " x " << A.cols
              << ", nnz = " << A.nnz << "\n";

    // ------------------------------------------------------------------
    // Build input vector x (dense, all ones for easy manual verification)
    // ------------------------------------------------------------------
    spmv::DenseVector x(A.cols);
    if (argc > 2 && std::string(argv[2]) == "--random") {
        fill_random(x);
        std::cout << "x: random values\n";
    } else {
        spmv::fill_constant(x, 1.0);
        std::cout << "x: all-ones vector\n";
    }

    // ------------------------------------------------------------------
    // Serial baseline
    // ------------------------------------------------------------------
    spmv::DenseVector y_serial(A.rows);
    CPUTimer t_serial;
    t_serial.start();
    spmv::spmv_cpu_serial(A, x, y_serial);
    t_serial.stop();
    std::cout << "Serial:   " << t_serial.elapsed_ms() << " ms\n";

    // ------------------------------------------------------------------
    // OpenMP parallel
    // ------------------------------------------------------------------
    spmv::DenseVector y_omp(A.rows);
    CPUTimer t_omp;
    t_omp.start();
    spmv::spmv_cpu_omp(A, x, y_omp);
    t_omp.stop();
    std::cout << "OpenMP:   " << t_omp.elapsed_ms() << " ms\n";

    // ------------------------------------------------------------------
    // Correctness check: serial vs OMP should be identical
    // ------------------------------------------------------------------
    double err = spmv::infnorm(y_serial, y_omp);
    std::cout << "Serial-OMP L-inf error: " << err << "\n";
    if (err > 1e-15) {
        std::cerr << "FAIL: serial and OpenMP results differ (err = " << err << ")\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Print first few results for sanity check
    // ------------------------------------------------------------------
    int64_t show = std::min<int64_t>(8, A.rows);
    std::cout << "\nFirst " << show << " entries of y (serial):\n";
    for (int64_t i = 0; i < show; ++i) {
        std::cout << "  y[" << i << "] = " << y_serial.data[i] << "\n";
    }

    std::cout << "\nPASS\n";
    return 0;
}
