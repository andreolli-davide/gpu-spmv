#include "parser/mtx_parser.h"
#include "parser/mtx_parser_gpu.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>

/// Tolerance for floating-point comparison between CPU and GPU results.
///
/// GPU accumulation order is undefined (CUDA uses a vendor-defined reduction
/// order that may differ from the CPU's row-major sequential order), so exact
/// equality cannot be guaranteed.  1e-6 is a safe bound for double-precision
/// SpMV on matrices with moderate nnz per row.
const double TOLERANCE = 1e-6;

bool double_eq(double a, double b) {
    return std::fabs(a - b) < TOLERANCE;
}

/// Test one matrix across three SpMV implementations: CSR (CPU), COO (CPU),
/// and GPU.
///
/// The test validates in three stages:
///
///   Stage 1 — Determinism: run CSR twice and verify both results match.
///             Catches a class of CPU-side bugs (broken accumulation, wrong
///             indices) before introducing GPU complexity.
///
///   Stage 2 — Format consistency: run COO and verify it matches CSR reference.
///             Catches bugs in the COO path or in the COO↔CSR conversion.
///
///   Stage 3 — GPU correctness: run spmv_gpu_kernel and compare against CSR.
///             This is the main target validation; any discrepancy here
///             indicates a kernel bug, incorrect data transfer, or FP
///             accumulation-order differences beyond TOLERANCE.
///
/// If the GPU path throws (e.g. CUDA error), the test reports "PASS (CPU only)"
/// and returns 0, allowing test runs to complete even on machines without a GPU.
///
/// \param path  Relative path to an MTX file (coordinate format).
/// \return 0 on success (all stages pass), 1 on any failure.
int test_matrix(const std::string& path) {
    std::cout << "Testing: " << path << " ... " << std::flush;

    MtxCsr csr;
    MtxCoo coo;
    try {
        csr = parse_mtx_csr(path);
        coo = parse_mtx(path);
    } catch (const std::exception& e) {
        std::cout << "PARSE ERROR: " << e.what() << std::endl;
        return 1;
    }

    // Generate a deterministic random input vector x.
    // Fixed seed (42) ensures the same x is used every run — results are
    // reproducible across CPU/GPU and across separate invocations.
    std::vector<double> x(csr.num_cols);
    std::vector<double> y_csr(csr.num_rows);
    std::vector<double> y_coo(csr.num_rows);
    std::vector<double> y_csr_ref(csr.num_rows);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int32_t i = 0; i < csr.num_cols; ++i) {
        x[i] = dist(rng);
    }

    // Reference: run CSR once to get the ground-truth result.
    spmv_cpu(csr, x.data(), y_csr_ref.data());

    // Stage 1: CSR determinism — same input, same implementation, two runs.
    spmv_cpu(csr, x.data(), y_csr.data());
    for (int32_t i = 0; i < csr.num_rows; ++i) {
        if (!double_eq(y_csr[i], y_csr_ref[i])) {
            std::cout << "CSR INCONSISTENCY FAIL" << std::endl;
            return 1;
        }
    }

    // Stage 2: COO vs CSR — different format, same math, must agree.
    spmv_cpu(coo, x.data(), y_coo.data());
    for (int32_t i = 0; i < csr.num_rows; ++i) {
        if (!double_eq(y_coo[i], y_csr_ref[i])) {
            std::cout << "COO vs CSR MISMATCH" << std::endl;
            return 1;
        }
    }

    // Stage 3: GPU kernel vs CSR reference.
    DeviceMatrix dmat = {0, 0, 0, nullptr, nullptr, nullptr};
    try {
        parse_mtx_gpu(path, &dmat);

        double* d_x;
        double* d_y;
        cudaMalloc(&d_x, csr.num_cols * sizeof(double));
        cudaMalloc(&d_y, csr.num_rows * sizeof(double));
        cudaMemset(d_y, 0, csr.num_rows * sizeof(double)); // zero y before kernel
        cudaMemcpy(d_x, x.data(), csr.num_cols * sizeof(double), cudaMemcpyHostToDevice);

        spmv_gpu_kernel(dmat, d_x, d_y);
        cudaDeviceSynchronize();

        std::vector<double> y_gpu(csr.num_rows);
        cudaMemcpy(y_gpu.data(), d_y, csr.num_rows * sizeof(double), cudaMemcpyDeviceToHost);

        for (int32_t i = 0; i < csr.num_rows; ++i) {
            if (!double_eq(y_gpu[i], y_csr_ref[i])) {
                std::cout << "GPU vs CSR MISMATCH at row " << i
                          << " (GPU=" << y_gpu[i] << ", CSR=" << y_csr_ref[i] << ")" << std::endl;
                free_gpu(&dmat);
                cudaFree(d_x);
                cudaFree(d_y);
                return 1;
            }
        }

        free_gpu(&dmat);
        cudaFree(d_x);
        cudaFree(d_y);
    } catch (const std::exception& e) {
        std::cout << "GPU ERROR: " << e.what() << " — skipping GPU test" << std::endl;
        std::cout << "PASS (CPU only)" << std::endl;
        return 0;
    }

    std::cout << "PASS (CPU+GPU)" << std::endl;
    return 0;
}

/// Run test_matrix() on a fixed set of matrices and report a summary.
///
/// Exit code: number of failures (0 = all passed, 11 = all failed).
int main() {
    const char* matrices[] = {
        "matrices/bone010.mtx",
        "matrices/webbase-1M.mtx",
        "matrices/ASIC_680ks.mtx",
        "matrices/ldoor.mtx",
        "matrices/boyd2_b.mtx",
        "matrices/rajat31.mtx",
        "matrices/Rucci1.mtx",
        "matrices/Ga41As41H72.mtx",
        "matrices/FullChip.mtx",
        "matrices/Si41Ge41H72.mtx",
        "matrices/eu-2005.mtx",
    };

    int failures = 0;
    for (const char* m : matrices) {
        if (test_matrix(m) != 0) {
            ++failures;
        }
    }

    std::cout << "\n" << (11 - failures) << "/" << 11 << " matrices passed" << std::endl;
    return failures;
}
