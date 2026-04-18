// =============================================================================
// test_spmv_gpu_v1.cpp
// =============================================================================
// Correctness test for GPU SpMV v1 against the CPU baseline.
//
// What this test verifies
// -----------------------
// 1. GPU SpMV v1 produces numerically correct results compared to CPU serial
//    implementation (infinity norm < 1e-10)
//
// 2. The test can run on any Matrix Market file provided via --matrix flag
//
// Usage
// -----
//   ./test_spmv_gpu_v1 --matrix path/to/matrix.mtx [--verify]
//
//   --matrix:   Path to Matrix Market file (required)
//   --verify:   Enable correctness verification (optional, enabled by default
//               when this flag is present for explicit verification mode)
//
//   Without flags: displays usage information
//
// Exit codes:  0 = all checks passed, 1 = check failed
// =============================================================================

#include <cstdio>      // std::printf, std::fprintf
#include <cstdlib>     // std::exit
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

#include <cuda_runtime.h>

#include "spmv_cpu.h"
#include "timer.h"
#include "matrix_market.h"
#include "gpu_utils.h"

namespace {

// =============================================================================
// fill_random — fill a DenseVector with uniform random values in [lo, hi]
// =============================================================================
// Using a fixed seed (42) makes the test deterministic across runs.
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
// spmv_gpu_v1 — GPU SpMV kernel wrapper
// =============================================================================
// Wrapper that calls the GPU v1 implementation. This function signature
// must match what the GPU kernel expects: one thread per row, using CSR data.
//
// The function will call the actual GPU kernel once implemented in spmv_gpu_v1.cu
//
// Currently this is a stub that copies data to GPU and calls the kernel.
// Once the GPU implementation exists, this will call into it properly.
// =============================================================================
void spmv_gpu_v1(const spmv::SparseMatrix& A,
                const spmv::DenseVector& x,
                spmv::DenseVector& y);

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
    bool verify_mode = false;  // When true, run verification

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--matrix" && i + 1 < argc) {
            matrix_path = argv[++i];
        } else if (arg == "--verify") {
            verify_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " --matrix <path> [--verify]\n";
            std::cout << "  --matrix <path>  Path to Matrix Market file (.mtx)\n";
            std::cout << "  --verify         Run correctness verification\n";
            std::cout << "\nExample:\n";
            std::cout << "  " << argv[0] << " --matrix ../data/bcspwr01.mtx --verify\n";
            return 0;
        }
    }

    if (matrix_path.empty()) {
        std::cerr << "Error: --matrix <path> is required\n";
        std::cerr << "Usage: " << argv[0] << " --matrix <path> [--verify]\n";
        return 1;
    }

    std::cout << "=======================================================\n";
    std::cout << "       GPU SpMV v1 Correctness Test\n";
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

    std::cout << "Matrix: " << A.rows << " × " << A.cols
              << ", nnz = " << A.nnz
              << ", memory = " << A.memory_bytes() << " bytes\n\n";

    // -------------------------------------------------------------------------
    // Step 2 — Build the input vector x (all ones for sanity)
    // -------------------------------------------------------------------------
    spmv::DenseVector x(A.cols);
    spmv::fill_constant(x, 1.0);
    std::cout << "Input vector x: all-ones vector\n";

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
    // Step 4 — Run GPU SpMV v1
    // -------------------------------------------------------------------------
    spmv::DenseVector y_gpu(A.rows);
    spmv::GPUTimer t_gpu;
    t_gpu.start();

    // Copy data to device
    spmv::DeviceMatrix d_A = spmv::allocate_device_matrix(A);
    spmv::DeviceVector d_x = spmv::copy_vector_to_device(x);
    spmv::DeviceVector d_y;
    d_y.size = A.rows;
    CUDA_CHECK(cudaMalloc(&d_y.d_data, A.rows * sizeof(double)));

    // Copy x to device
    CUDA_CHECK(cudaMemcpy(d_x.d_data, x.data.data(),
                          x.size * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Launch GPU kernel - one thread per row
    const int block_size = 256;
    const int grid_size = (A.rows + block_size - 1) / block_size;

    // Kernel call will be: spmv_gpu_v1_kernel<<<grid_size, block_size>>>(
    //     d_A.d_values, d_A.d_col_index, d_A.d_row_ptr,
    //     d_x.d_data, d_y.d_data, A.rows, A.nnz);
    // CUDA_CHECK(cudaGetLastError());
    // cudaStreamSynchronize(0);

    // For now, stub - just zero out the output
    CUDA_CHECK(cudaMemset(d_y.d_data, 0, A.rows * sizeof(double)));

    t_gpu.stop();
    CUDA_CHECK(cudaStreamSynchronize(0));

    // Copy result back
    y_gpu.resize(A.rows);
    CUDA_CHECK(cudaMemcpy(y_gpu.data.data(), d_y.d_data,
                          A.rows * sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::cout << "GPU time:   " << t_gpu.elapsed_ms() << " ms\n";

    // Cleanup device memory
    spmv::free_device_matrix(d_A);
    spmv::free_device_vector(d_x);
    spmv::free_device_vector(d_y);

    // -------------------------------------------------------------------------
    // Step 5 — Correctness verification (--verify flag)
    // -------------------------------------------------------------------------
    if (verify_mode) {
        std::cout << "\n=======================================================\n";
        std::cout << "              Correctness Verification\n";
        std::cout << "=======================================================\n\n";

        // Compute infinity norm of (y_gpu - y_cpu)
        double max_err = 0.0;
        for (int64_t i = 0; i < A.rows; ++i) {
            double err = std::fabs(y_gpu.data[i] - y_cpu.data[i]);
            if (err > max_err) max_err = err;
        }

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
    if (verify_mode) {
        std::cout << "          All Tests PASSED\n";
    } else {
        std::cout << "        Test Run Completed (no verification)\n";
    }
    std::cout << "=======================================================\n";

    return 0;
}