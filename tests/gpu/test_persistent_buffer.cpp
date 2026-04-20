// =============================================================================
// test_persistent_buffer.cpp
// =============================================================================
// Correctness test for spmv_gpu_v2_persistent() against CPU baseline.
//
// What this test verifies
// -----------------------
// 1. spmv_gpu_v2_persistent produces numerically correct results compared to
//    CPU serial implementation (infinity norm < 1e-10)
//
// 2. PersistentBufferManager correctly manages GPU memory lifecycle
//
// Usage
// -----
//   ./test_persistent_buffer --matrix path/to/matrix.mtx [--verify]
//
//   --matrix:   Path to Matrix Market file (required)
//   --verify:   Enable correctness verification (default: on)
//
// Exit codes:  0 = all checks passed, 1 = check failed
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include <cuda_runtime.h>

#include "spmv_cpu.h"
#include "timer.h"
#include "matrix_market.h"
#include "gpu_utils.h"
#include "spmv_gpu_v2.h"
#include "gpu_persistent_buffers.h"

namespace {

// =============================================================================
// fill_random — fill a DenseVector with uniform random values in [lo, hi]
// =============================================================================
void fill_random(spmv::DenseVector& v, double lo = -1.0, double hi = 1.0) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (double& x : v.data) {
        x = dist(rng);
    }
}

// =============================================================================
// print_vector — helper: print up to N entries of a vector

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

// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(6);

    // -------------------------------------------------------------------------
    // Parse command line arguments
    // -------------------------------------------------------------------------
    std::string matrix_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--matrix" && i + 1 < argc) {
            matrix_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " --matrix <path> [--verify]\n";
            std::cout << "  --matrix <path>  Path to Matrix Market file (.mtx)\n";
            std::cout << "  --verify         Run correctness verification (default)\n";
            std::cout << "\nExample:\n";
            std::cout << "  " << argv[0] << " --matrix ../data/ash219.mtx\n";
            return 0;
        }
    }

    if (matrix_path.empty()) {
        std::cerr << "Error: --matrix <path> is required\n";
        std::cerr << "Usage: " << argv[0] << " --matrix <path> [--verify]\n";
        return 1;
    }

    std::cout << "=======================================================\n";
    std::cout << "   GPU SpMV v2 Persistent Buffer Correctness Test\n";
    std::cout << "=======================================================\n\n";

    // -------------------------------------------------------------------------
    // Step 1 — Load the matrix
    // -------------------------------------------------------------------------
    std::cout << "Loading matrix from: " << matrix_path << "\n";
    spmv::SparseMatrix A;
    try {
        A = spmv::parse_matrix_market(matrix_path);
    } catch (const std::exception& e) {
        std::cerr << "ERROR parsing matrix: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Matrix: " << A.rows << " x " << A.cols
              << ", nnz = " << A.nnz << "\n\n";

    // -------------------------------------------------------------------------
    // Step 2 — Build input vector x (deterministic random values)
    // -------------------------------------------------------------------------
    spmv::DenseVector x(A.cols);
    fill_random(x, -1.0, 1.0);
    std::cout << "Input vector x: random values in [-1, 1], seed=42\n";

    // -------------------------------------------------------------------------
    // Step 3 — Run CPU reference (spmv_cpu_serial)
    // -------------------------------------------------------------------------
    spmv::DenseVector y_cpu(A.rows);
    spmv::CPUTimer t_cpu;
    t_cpu.start();
    spmv::spmv_cpu_serial(A, x, y_cpu);
    t_cpu.stop();
    std::cout << "CPU time:   " << t_cpu.elapsed_ms() << " ms\n";

    // -------------------------------------------------------------------------
    // Step 4 — Run GPU SpMV with persistent buffers
    // -------------------------------------------------------------------------
    spmv::PersistentBufferManager buf;
    spmv::DenseVector y_gpu(A.rows);
    spmv::GPUTimer t_gpu;

    // Upload matrix once (this is the key advantage of persistent buffers)
    std::cout << "\nUploading matrix to GPU (persistent buffer)...\n";
    buf.upload_matrix(A);

    // Upload vector x
    buf.upload_vector_x(x);

    // Allocate output
    buf.allocate_output(A.rows);

    // Run the kernel
    t_gpu.start();
    spmv::spmv_gpu_v2_persistent(buf, x, y_gpu);
    CUDA_CHECK(cudaStreamSynchronize(0));
    t_gpu.stop();

    std::cout << "GPU time:   " << t_gpu.elapsed_ms() << " ms\n";

    // -------------------------------------------------------------------------
    // Step 5 — Correctness verification
    // -------------------------------------------------------------------------
    std::cout << "\n=======================================================\n";
    std::cout << "              Correctness Verification\n";
    std::cout << "=======================================================\n\n";

    double max_err = spmv::infnorm(y_gpu, y_cpu);

    std::cout << "Infinity norm |y_gpu - y_cpu|_inf: " << max_err << "\n";
    std::cout << "Tolerance: 1e-10\n\n";

    const double tolerance = 1e-10;
    if (max_err < tolerance) {
        std::cout << "PASS: GPU output matches CPU reference within tolerance\n";
    } else {
        std::cerr << "FAIL: GPU output differs from CPU reference\n";
        std::cerr << "  Maximum error: " << max_err << "\n";
        std::cerr << "  Tolerance:     " << tolerance << "\n";

        // Print first few differences for debugging
        std::cout << "\nFirst few differences (y_gpu vs y_cpu):\n";
        int64_t print_count = std::min<int64_t>(10, A.rows);
        for (int64_t i = 0; i < print_count; ++i) {
            double diff = y_gpu.data[i] - y_cpu.data[i];
            std::cout << "  [" << i << "] GPU=" << y_gpu.data[i]
                      << " CPU=" << y_cpu.data[i]
                      << " diff=" << diff << "\n";
        }
        return 1;
    }

    // -------------------------------------------------------------------------
    // Step 6 — Print result summary
    // -------------------------------------------------------------------------
    std::cout << "\n=======================================================\n";
    std::cout << "                  Result Summary\n";
    std::cout << "=======================================================\n\n";

    std::cout << "CPU output (first " << std::min<int64_t>(8, A.rows) << " entries):\n";
    print_vector(y_cpu);

    std::cout << "\nGPU output (first " << std::min<int64_t>(8, A.rows) << " entries):\n";
    print_vector(y_gpu);

    std::cout << "\n=======================================================\n";
    std::cout << "          All Tests PASSED\n";
    std::cout << "=======================================================\n";

    return 0;
}
