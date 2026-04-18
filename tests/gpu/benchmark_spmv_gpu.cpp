// =============================================================================
// benchmark_spmv_gpu.cpp
// =============================================================================
// Performance benchmarking suite for GPU SpMV kernels.
//
// Measures kernel execution time, GFLOP/s, and memory bandwidth for sparse
// matrix-vector multiplication on NVIDIA GPU. Each matrix is run multiple
// times (default: 5) to collect statistics (average, min, max).
//
// Formulas (Bell & Garland '09):
//   GFLOP/s = (2 * nnz) / (t_kernel * 1e9)
//   B_eff   = (16 * nnz) / t_ms  [GB/s]
//
// Usage
// -----
//   ./benchmark_spmv_gpu --matrix <path> [--runs N] [--output <csv_path>]
//   ./benchmark_spmv_gpu --matrix ../data/matrix.mtx --runs 5
//   ./benchmark_spmv_gpu --matrix ../data/ --runs 5  (batch mode, all .mtx files)
//
// Output
// ------
//   CSV file: matrix_name, rows, cols, nnz, time_avg_ms, time_min_ms,
//             time_max_ms, gflops, gb_s
//
// Notes
// -----
//   - Kernel timing excludes I/O (matrix loading and vector setup)
//   - PCIe transfer time is included in kernel wrapper time, not measured
//     separately
//   - Uses GPU events for accurate kernel timing
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <random>

#include <cuda_runtime.h>

#include <dirent.h>
#include <sys/stat.h>

#include "spmv_cpu.h"
#include "spmv_gpu_v2.h"
#include "timer.h"
#include "matrix_market.h"
#include "gpu_utils.h"

namespace {

// =============================================================================
// Benchmark configuration
// =============================================================================
constexpr int DEFAULT_RUNS = 5;
constexpr const char* DEFAULT_OUTPUT = "results/benchmark_results.csv";

// =============================================================================
// print_usage
// =============================================================================
void print_usage(const char* prog) {
    std::printf("Usage: %s --matrix <path> [--runs N] [--output <csv_path>]\n", prog);
    std::printf("  --matrix <path>   Path to Matrix Market file (.mtx) or directory\n");
    std::printf("  --runs N          Number of runs per matrix (default: %d)\n", DEFAULT_RUNS);
    std::printf("  --output <path>   Output CSV file (default: %s)\n", DEFAULT_OUTPUT);
    std::printf("\nExamples:\n");
    std::printf("  %s --matrix ../data/bcsstk01.mtx --runs 5\n", prog);
    std::printf("  %s --matrix ../data/ --runs 10 --output results/my_bench.csv\n", prog);
}

// =============================================================================
// parse_args — parse command line arguments
// =============================================================================
struct Args {
    std::string matrix_path;
    int runs = DEFAULT_RUNS;
    std::string output_path = DEFAULT_OUTPUT;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--matrix" && i + 1 < argc) {
            args.matrix_path = argv[++i];
        } else if (arg == "--runs" && i + 1 < argc) {
            args.runs = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    return args;
}

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
// benchmark_spmv — run SpMV multiple times and collect statistics
// =============================================================================
struct BenchmarkResult {
    std::string matrix_name;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    double time_avg_ms = 0.0;
    double time_min_ms = 0.0;
    double time_max_ms = 0.0;
    double gflops = 0.0;
    double gb_s = 0.0;
};

BenchmarkResult benchmark_spmv(const spmv::SparseMatrix& A,
                              const std::string& matrix_name,
                              int runs) {
    // Allocate input and output vectors
    spmv::DenseVector x(A.cols);
    spmv::DenseVector y(A.rows);
    fill_random(x, -1.0, 1.0);

    // Timing setup
    std::vector<double> times;
    times.reserve(runs);

    // Warm-up run (not counted)
    spmv::DenseVector temp_y(A.rows);

    // Warm-up call
    spmv::spmv_gpu_v2(A, x, temp_y);
    CUDA_CHECK(cudaStreamSynchronize(0));

    // Benchmark runs
    for (int r = 0; r < runs; ++r) {
        // Reset output vector
        temp_y.resize(A.rows);
        std::fill(temp_y.data.begin(), temp_y.data.end(), 0.0);

        // Create GPU timer for accurate kernel timing
        spmv::GPUTimer timer;
        timer.start();

        // Call the wrapper function
        spmv::spmv_gpu_v2(A, x, temp_y);

        CUDA_CHECK(cudaGetLastError());
        timer.stop();
        CUDA_CHECK(cudaStreamSynchronize(0));

        double elapsed_ms = timer.elapsed_ms();
        times.push_back(elapsed_ms);
    }

    // Copy result back (for correctness verification, not timing)
    y = temp_y;

    // Compute statistics
    double sum = 0.0;
    double min_time = times[0];
    double max_time = times[0];
    for (double t : times) {
        sum += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    double avg_time = sum / runs;

    // Compute GFLOP/s and memory bandwidth
    // GFLOP/s = (2 * nnz) / (t_kernel * 1e9) where t_kernel is in seconds
    // B_eff = (16 * nnz) / t_ms [GB/s] where t_ms is in milliseconds
    double gflops = (2.0 * static_cast<double>(A.nnz)) / (avg_time * 1e-3 * 1e9);
    double gb_s = (16.0 * static_cast<double>(A.nnz)) / avg_time;

    BenchmarkResult result;
    result.matrix_name = matrix_name;
    result.rows = A.rows;
    result.cols = A.cols;
    result.nnz = A.nnz;
    result.time_avg_ms = avg_time;
    result.time_min_ms = min_time;
    result.time_max_ms = max_time;
    result.gflops = gflops;
    result.gb_s = gb_s;

    return result;
}

// =============================================================================
// write_csv_header — write CSV header
// =============================================================================
void write_csv_header(std::ofstream& out) {
    out << "matrix_name,rows,cols,nnz,time_avg_ms,time_min_ms,time_max_ms,"
        << "gflops,gb_s\n";
}

// =============================================================================
// write_csv_row — write a single benchmark result as CSV row
// =============================================================================
void write_csv_row(std::ofstream& out, const BenchmarkResult& r) {
    out << std::quoted(r.matrix_name) << ","
        << r.rows << ","
        << r.cols << ","
        << r.nnz << ","
        << std::scientific << std::setprecision(6)
        << r.time_avg_ms << ","
        << r.time_min_ms << ","
        << r.time_max_ms << ","
        << r.gflops << ","
        << r.gb_s << "\n";
}

// =============================================================================
// print_result — print benchmark result to console
// =============================================================================
void print_result(const BenchmarkResult& r) {
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "  " << std::setw(40) << std::left << r.matrix_name
              << " rows=" << std::setw(10) << r.rows
              << " cols=" << std::setw(10) << r.cols
              << " nnz=" << std::setw(12) << r.nnz << "\n";
    std::cout << "    time: " << r.time_avg_ms << " ms (avg), "
              << r.time_min_ms << " ms (min), "
              << r.time_max_ms << " ms (max)\n";
    std::cout << "    GFLOP/s: " << r.gflops << "\n";
    std::cout << "    GB/s: " << r.gb_s << "\n";
}

// =============================================================================
// list_mtx_files — recursively find all .mtx files in a directory
// =============================================================================
std::vector<std::string> list_mtx_files(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        return files;
    }

    struct dirent* entry;
    while ((entry = readdir(dp)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        std::string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                // Recurse into subdirectory
                auto subfiles = list_mtx_files(path);
                files.insert(files.end(), subfiles.begin(), subfiles.end());
            } else if (name.size() > 4 &&
                       name.substr(name.size() - 4) == ".mtx") {
                files.push_back(path);
            }
        }
    }
    closedir(dp);

    std::sort(files.begin(), files.end());
    return files;
}

}  // anonymous namespace

// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(3);

    // Parse command line arguments
    Args args = parse_args(argc, argv);

    if (args.matrix_path.empty()) {
        std::cerr << "Error: --matrix <path> is required\n";
        print_usage(argv[0]);
        return 1;
    }

    if (args.runs < 1) {
        std::cerr << "Error: --runs must be >= 1\n";
        return 1;
    }

    std::cout << "=======================================================\n";
    std::cout << "       GPU SpMV Performance Benchmark\n";
    std::cout << "=======================================================\n";
    std::cout << "Runs per matrix: " << args.runs << "\n";
    std::cout << "Output file: " << args.output_path << "\n\n";

    // Collect list of matrices to benchmark
    std::vector<std::string> matrix_paths;
    struct stat st;
    if (stat(args.matrix_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        matrix_paths = list_mtx_files(args.matrix_path);
    } else {
        matrix_paths.push_back(args.matrix_path);
    }

    if (matrix_paths.empty()) {
        std::cerr << "Error: No .mtx files found at path: " << args.matrix_path << "\n";
        return 1;
    }

    std::cout << "Found " << matrix_paths.size() << " matrix(ices) to benchmark\n\n";

    // Open output CSV file
    std::ofstream csv_out(args.output_path);
    if (!csv_out.is_open()) {
        std::cerr << "Error: Cannot open output file: " << args.output_path << "\n";
        return 1;
    }
    write_csv_header(csv_out);

    // Benchmark each matrix
    std::cout << "=======================================================\n";
    std::cout << "              Benchmarking Results\n";
    std::cout << "=======================================================\n\n";

    for (const auto& matrix_path : matrix_paths) {
        // Extract matrix name from path
        std::string matrix_name = matrix_path;
        size_t last_slash = matrix_name.find_last_of('/');
        if (last_slash != std::string::npos) {
            matrix_name = matrix_name.substr(last_slash + 1);
        }

        std::cout << "Benchmarking: " << matrix_name << "\n";

        // Load matrix
        spmv::SparseMatrix A;
        try {
            A = spmv::parse_matrix_market(matrix_path);
        } catch (const std::exception& e) {
            std::cerr << "  ERROR loading matrix: " << e.what() << "\n";
            continue;
        }

        std::cout << "  Matrix: " << A.rows << " x " << A.cols
                  << ", nnz = " << A.nnz << "\n";

        // Run benchmark
        BenchmarkResult result = benchmark_spmv(A, matrix_name, args.runs);

        // Write to CSV
        write_csv_row(csv_out, result);

        // Print to console
        print_result(result);
        std::cout << "\n";
    }

    csv_out.close();

    std::cout << "=======================================================\n";
    std::cout << "         Benchmark Complete\n";
    std::cout << "=======================================================\n";
    std::cout << "Results written to: " << args.output_path << "\n";

    return 0;
}