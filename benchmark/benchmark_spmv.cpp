// =============================================================================
// benchmark_spmv.cpp
// =============================================================================
// Benchmark CPU vs GPU SpMV performance
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

#include "spmv_gpu.h"
#include "spmv_cpu.h"
#include "timer.h"
#include "matrix_market.h"

namespace {

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

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <matrix.mtx> [--runs N] [--warmup N]\n";
    std::cerr << "  --runs N    Number of benchmark iterations (default: 100)\n";
    std::cerr << "  --warmup N  Number of warmup iterations (default: 10)\n";
    std::exit(1);
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(6);

    if (argc < 2) {
        print_usage(argv[0]);
    }

    std::string matrix_path;
    int runs = 100;
    int warmup = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--runs" && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
        } else {
            matrix_path = arg;
        }
    }

    // =========================================================================
    // Step 1 — Load the matrix
    // =========================================================================
    spmv::SparseMatrix A;
    std::cout << "Loading matrix: " << matrix_path << "\n";
    try {
        A = spmv::parse_matrix_market(matrix_path);
    } catch (const std::exception& e) {
        std::cerr << "ERROR parsing matrix: " << e.what() << "\n";
        std::exit(1);
    }

    std::cout << "Matrix: " << A.rows << " x " << A.cols
              << ", nnz = " << A.nnz << "\n";

    const double gflop = 2.0 * static_cast<double>(A.nnz) / 1e9;

    // =========================================================================
    // Step 2 — Build the input vector x (all-ones)
    // =========================================================================
    spmv::DenseVector x(A.cols);
    spmv::fill_constant(x, 1.0);

    // =========================================================================
    // Step 3 — Warmup runs
    // =========================================================================
    spmv::DenseVector y_cpu(A.rows);
    spmv::DenseVector y_gpu(A.rows);

    std::cout << "\nWarmup (" << warmup << " iterations)...\n";
    for (int i = 0; i < warmup; ++i) {
        spmv::spmv_cpu_serial(A, x, y_cpu);
        spmv::spmv_gpu_v1(A, x, y_gpu);
    }

    // =========================================================================
    // Step 4 — CPU Benchmark
    // =========================================================================
    std::cout << "\n=== CPU Benchmark (" << runs << " iterations) ===\n";

    std::vector<double> cpu_times;
    cpu_times.reserve(runs);

    spmv::CPUTimer t_cpu;
    for (int i = 0; i < runs; ++i) {
        t_cpu.start();
        spmv::spmv_cpu_serial(A, x, y_cpu);
        t_cpu.stop();
        cpu_times.push_back(t_cpu.elapsed_ms());
    }

    double cpu_avg = 0, cpu_min = cpu_times[0], cpu_max = cpu_times[0];
    for (double t : cpu_times) {
        cpu_avg += t;
        cpu_min = std::min(cpu_min, t);
        cpu_max = std::max(cpu_max, t);
    }
    cpu_avg /= runs;

    double cpu_var = 0;
    for (double t : cpu_times) {
        double d = t - cpu_avg;
        cpu_var += d * d;
    }
    cpu_var = std::sqrt(cpu_var / runs);

    std::cout << "  Time:     " << cpu_avg << " ms (avg)\n";
    std::cout << "  Time:     " << cpu_min << " ms (min)\n";
    std::cout << "  Time:     " << cpu_max << " ms (max)\n";
    std::cout << "  Time:     " << cpu_var << " ms (stddev)\n";
    std::cout << "  GFLOP/s:  " << gflop / (cpu_min / 1000.0) << "\n";

    // =========================================================================
    // Step 5 — GPU Benchmark
    // =========================================================================
    std::cout << "\n=== GPU Benchmark (" << runs << " iterations) ===\n";

    std::vector<double> gpu_times;
    gpu_times.reserve(runs);

    spmv::GPUTimer t_gpu;
    bool gpu_ok = true;
    for (int i = 0; i < runs; ++i) {
        t_gpu.start();
        gpu_ok = spmv::spmv_gpu_v1(A, x, y_gpu);
        t_gpu.stop();
        if (!gpu_ok) {
            std::cerr << "FAIL: GPU SpMV returned error at iteration " << i << "\n";
            std::exit(1);
        }
        gpu_times.push_back(t_gpu.elapsed_ms());
    }

    double gpu_avg = 0, gpu_min = gpu_times[0], gpu_max = gpu_times[0];
    for (double t : gpu_times) {
        gpu_avg += t;
        gpu_min = std::min(gpu_min, t);
        gpu_max = std::max(gpu_max, t);
    }
    gpu_avg /= runs;

    double gpu_var = 0;
    for (double t : gpu_times) {
        double d = t - gpu_avg;
        gpu_var += d * d;
    }
    gpu_var = std::sqrt(gpu_var / runs);

    std::cout << "  Time:     " << gpu_avg << " ms (avg)\n";
    std::cout << "  Time:     " << gpu_min << " ms (min)\n";
    std::cout << "  Time:     " << gpu_max << " ms (max)\n";
    std::cout << "  Time:     " << gpu_var << " ms (stddev)\n";
    std::cout << "  GFLOP/s:  " << gflop / (gpu_min / 1000.0) << "\n";

    // =========================================================================
    // Step 6 — Correctness check
    // =========================================================================
    std::cout << "\n=== Correctness Check ===\n";
    const double err = spmv::infnorm(y_cpu, y_gpu);
    std::cout << "  L-inf error: " << err << "\n";
    if (err > 1e-15) {
        std::cerr << "FAIL: CPU and GPU results differ!\n";
        std::exit(1);
    }
    std::cout << "  PASS: Results match\n";

    // =========================================================================
    // Step 7 — Summary
    // =========================================================================
    std::cout << "\n=== Summary ===\n";
    std::cout << "matrix," << A.rows << "," << A.cols << "," << A.nnz << ","
              << gflop / (gpu_min / 1000.0) << ","
              << gflop / (cpu_min / 1000.0) << ","
              << cpu_min << "," << gpu_min << "\n";

    std::cout << "\nCSV: matrix,rows,cols,nnz,gflops_gpu,gflops_cpu,time_cpu_ms,time_gpu_ms\n";

    return 0;
}