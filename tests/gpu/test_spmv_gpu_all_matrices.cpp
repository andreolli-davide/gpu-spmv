// =============================================================================
// test_spmv_gpu_all_matrices.cpp
// =============================================================================
// Correctness test for GPU SpMV v1 and v2 against CPU baseline for all
// 10 SuiteSparse matrices.
//
// What this test verifies
// -----------------------
// 1. GPU SpMV v1 produces numerically correct results (|y_gpu - y_cpu|_inf < 1e-10)
// 2. GPU SpMV v2 produces numerically correct results (|y_gpu - y_cpu|_inf < 1e-10)
//
// Matrices tested:
//   1. bcspwr01.mtx  — Power system network (small, structured)
//   2. arrow.mtx     — Arrowhead matrix
//   3. LFAT5.mtx    — Lower rank matrix (irregular)
//   4. lp_e226.mtx   — Linear programming problem
//   5. Ragusa16.mtx  — Sicily matrix
//   6. two.mtx
//   7. one.mtx
//   8. impcol_a.mtx
//   9. GD99_cc.mtx
//  10. arrowc.mtx
//
// Usage
// -----
//   ./test_spmv_gpu_all_matrices [--csv]
//
//   --csv:   Output results in CSV format to results/correctness_results.csv
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
#include <fstream>

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
// infnorm — compute infinity norm of (a - b)
// =============================================================================
double infnorm(const spmv::DenseVector& a, const spmv::DenseVector& b) {
    double max_err = 0.0;
    for (int64_t i = 0; i < static_cast<int64_t>(a.data.size()); ++i) {
        max_err = std::max(max_err, std::fabs(a.data[i] - b.data[i]));
    }
    return max_err;
}

// =============================================================================
// compute_density — compute matrix density (nnz / (rows * cols))
// =============================================================================
double compute_density(int64_t nnz, int64_t rows, int64_t cols) {
    return static_cast<double>(nnz) / (static_cast<double>(rows) * static_cast<double>(cols));
}

// =============================================================================
// Matrix test result structure
// =============================================================================
struct MatrixTestResult {
    std::string name;
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    double density;
    bool v1_passed;
    bool v2_passed;
    double v1_error;
    double v2_error;
    double v1_time_ms;
    double v2_time_ms;
    double cpu_time_ms;
};

const double TOLERANCE = 1e-10;

// List of test matrices
const std::vector<std::string> TEST_MATRICES = {
    "bcspwr01.mtx",
    "arrow.mtx",
    "LFAT5.mtx",
    "lp_e226.mtx",
    "Ragusa16.mtx",
    "two.mtx",
    "one.mtx",
    "impcol_a.mtx",
    "GD99_cc.mtx",
    "arrowc.mtx"
};

} // anonymous namespace

// =============================================================================
// test_single_matrix — run correctness test for one matrix
// =============================================================================
MatrixTestResult test_single_matrix(const std::string& matrix_path,
                                    const std::string& matrix_name) {
    MatrixTestResult result;
    result.name = matrix_name;

    std::cout << "=======================================================\n";
    std::cout << "  Testing: " << matrix_name << "\n";
    std::cout << "=======================================================\n";

    // Load the matrix
    spmv::SparseMatrix A;
    try {
        A = spmv::parse_matrix_market(matrix_path);
    } catch (const std::exception& e) {
        std::cerr << "ERROR parsing matrix: " << e.what() << "\n";
        result.v1_passed = false;
        result.v2_passed = false;
        return result;
    }

    result.rows = A.rows;
    result.cols = A.cols;
    result.nnz = A.nnz;
    result.density = compute_density(A.nnz, A.rows, A.cols);

    std::cout << "  Matrix: " << A.rows << " × " << A.cols
              << ", nnz = " << A.nnz
              << ", density = " << std::scientific << std::setprecision(3) << result.density
              << "\n";

    // Build input vector x (all ones for sanity)
    spmv::DenseVector x(A.cols);
    spmv::fill_constant(x, 1.0);

    // Run CPU reference
    spmv::DenseVector y_cpu(A.rows);
    spmv::CPUTimer t_cpu;
    t_cpu.start();
    spmv::spmv_cpu_serial(A, x, y_cpu);
    t_cpu.stop();
    result.cpu_time_ms = t_cpu.elapsed_ms();
    std::cout << "  CPU time:   " << std::fixed << std::setprecision(3) << result.cpu_time_ms << " ms\n";

    // =========================================================================
    // Test GPU v1
    // =========================================================================
    spmv::DenseVector y_gpu_v1(A.rows);
    spmv::GPUTimer t_gpu_v1;
    t_gpu_v1.start();
    spmv::spmv_gpu_v1(A, x, y_gpu_v1);
    t_gpu_v1.stop();
    result.v1_time_ms = t_gpu_v1.elapsed_ms();
    result.v1_error = infnorm(y_gpu_v1, y_cpu);
    result.v1_passed = (result.v1_error < TOLERANCE);

    std::cout << "  GPU v1 time: " << result.v1_time_ms << " ms, "
              << "error = " << std::scientific << std::setprecision(6) << result.v1_error << " "
              << (result.v1_passed ? "[PASS]" : "[FAIL]") << "\n";

    // =========================================================================
    // Test GPU v2
    // =========================================================================
    spmv::DenseVector y_gpu_v2(A.rows);
    spmv::GPUTimer t_gpu_v2;
    t_gpu_v2.start();
    spmv::spmv_gpu_v2(A, x, y_gpu_v2);
    t_gpu_v2.stop();
    result.v2_time_ms = t_gpu_v2.elapsed_ms();
    result.v2_error = infnorm(y_gpu_v2, y_cpu);
    result.v2_passed = (result.v2_error < TOLERANCE);

    std::cout << "  GPU v2 time: " << result.v2_time_ms << " ms, "
              << "error = " << std::scientific << std::setprecision(6) << result.v2_error << " "
              << (result.v2_passed ? "[PASS]" : "[FAIL]") << "\n";

    std::cout << "\n";
    return result;
}

// =============================================================================
// write_csv_results — write results to CSV file
// =============================================================================
void write_csv_results(const std::vector<MatrixTestResult>& results,
                       const std::string& filepath) {
    std::ofstream csv(filepath);
    if (!csv.is_open()) {
        std::cerr << "ERROR: Could not open CSV file for writing: " << filepath << "\n";
        return;
    }

    // Header
    csv << "matrix,rows,cols,nnz,density,v1_passed,v2_passed,"
        << "v1_error,v2_error,v1_time_ms,v2_time_ms,cpu_time_ms\n";

    // Data rows
    for (const auto& r : results) {
        csv << r.name << ","
            << r.rows << ","
            << r.cols << ","
            << r.nnz << ","
            << std::scientific << std::setprecision(6) << r.density << ","
            << (r.v1_passed ? "PASS" : "FAIL") << ","
            << (r.v2_passed ? "PASS" : "FAIL") << ","
            << std::scientific << std::setprecision(6) << r.v1_error << ","
            << std::scientific << std::setprecision(6) << r.v2_error << ","
            << std::fixed << std::setprecision(3) << r.v1_time_ms << ","
            << std::fixed << std::setprecision(3) << r.v2_time_ms << ","
            << std::fixed << std::setprecision(3) << r.cpu_time_ms << "\n";
    }

    csv.close();
    std::cout << "Results written to: " << filepath << "\n";
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(6);

    bool csv_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv") {
            csv_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--csv]\n";
            std::cout << "  --csv   Output results in CSV format\n";
            return 0;
        }
    }

    std::cout << "=======================================================\n";
    std::cout << "       GPU SpMV v1 & v2 Correctness Test\n";
    std::cout << "       All 10 SuiteSparse Matrices\n";
    std::cout << "=======================================================\n";
    std::cout << "Tolerance: " << TOLERANCE << " (infinity norm)\n\n";

    // Base path for matrices
    const std::string base_path = "../data/matrices/suiteSparse_matrices/";
    std::string results_dir = "./";

    std::vector<MatrixTestResult> all_results;
    int total_tests = 0;
    int passed_tests = 0;

    for (const auto& matrix_name : TEST_MATRICES) {
        std::string matrix_path = base_path + matrix_name;
        MatrixTestResult result = test_single_matrix(matrix_path, matrix_name);
        all_results.push_back(result);

        if (result.v1_passed && result.v2_passed) {
            passed_tests++;
        }
        total_tests += 2; // v1 and v2
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "=======================================================\n";
    std::cout << "                  Test Summary\n";
    std::cout << "=======================================================\n\n";

    std::cout << std::left << std::setw(15) << "Matrix"
              << std::right << std::setw(10) << "Rows"
              << std::setw(10) << "Cols"
              << std::setw(12) << "NNZ"
              << std::setw(12) << "v1 Error"
              << std::setw(10) << "v1 Pass"
              << std::setw(12) << "v2 Error"
              << std::setw(10) << "v2 Pass"
              << "\n";
    std::cout << std::string(91, '-') << "\n";

    for (const auto& r : all_results) {
        std::cout << std::left << std::setw(15) << r.name
                  << std::right << std::setw(10) << r.rows
                  << std::setw(10) << r.cols
                  << std::setw(12) << r.nnz
                  << std::setw(12) << std::scientific << std::setprecision(2) << r.v1_error
                  << std::setw(10) << (r.v1_passed ? "PASS" : "FAIL")
                  << std::setw(12) << std::scientific << std::setprecision(2) << r.v2_error
                  << std::setw(10) << (r.v2_passed ? "PASS" : "FAIL")
                  << "\n";
    }

    std::cout << "\n";
    std::cout << "Total: " << passed_tests << "/" << total_tests << " tests passed\n";

    // Write CSV if requested
    if (csv_mode) {
        std::string csv_path = results_dir + "correctness_results.csv";
        write_csv_results(all_results, csv_path);
    }

    // Determine exit code
    bool all_passed = true;
    for (const auto& r : all_results) {
        if (!r.v1_passed || !r.v2_passed) {
            all_passed = false;
            break;
        }
    }

    std::cout << "\n=======================================================\n";
    if (all_passed) {
        std::cout << "          All Tests PASSED\n";
    } else {
        std::cout << "          Some Tests FAILED\n";
    }
    std::cout << "=======================================================\n";

    return all_passed ? 0 : 1;
}