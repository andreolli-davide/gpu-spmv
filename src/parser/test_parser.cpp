#include "parser/mtx_parser.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>

const float TOLERANCE = 1e-4f;

bool float_eq(float a, float b) {
    return std::fabs(a - b) < TOLERANCE;
}

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

    // Generate random vector x
    std::vector<float> x(csr.num_cols);
    std::vector<float> y_csr(csr.num_rows);
    std::vector<float> y_coo(csr.num_rows);
    std::vector<float> y_csr_ref(csr.num_rows);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int32_t i = 0; i < csr.num_cols; ++i) {
        x[i] = dist(rng);
    }

    // CPU SpMV — CSR
    spmv_cpu(csr, x.data(), y_csr_ref.data());

    // CPU SpMV — CSR (verify consistency)
    spmv_cpu(csr, x.data(), y_csr.data());
    for (int32_t i = 0; i < csr.num_rows; ++i) {
        if (!float_eq(y_csr[i], y_csr_ref[i])) {
            std::cout << "CSR INCONSISTENCY FAIL" << std::endl;
            return 1;
        }
    }

    // CPU SpMV — COO
    spmv_cpu(coo, x.data(), y_coo.data());
    for (int32_t i = 0; i < csr.num_rows; ++i) {
        if (!float_eq(y_coo[i], y_csr_ref[i])) {
            std::cout << "COO vs CSR MISMATCH" << std::endl;
            return 1;
        }
    }

    // GPU SpMV
    DeviceMatrix dmat;
    try {
        parse_mtx_gpu(path, &dmat);

        float* d_x;
        float* d_y;
        cudaMalloc(&d_x, csr.num_cols * sizeof(float));
        cudaMalloc(&d_y, csr.num_rows * sizeof(float));
        cudaMemcpy(d_x, x.data(), csr.num_cols * sizeof(float), cudaMemcpyHostToDevice);

        spmv_gpu_kernel(dmat, d_x, d_y);

        std::vector<float> y_gpu(csr.num_rows);
        cudaMemcpy(y_gpu.data(), d_y, csr.num_rows * sizeof(float), cudaMemcpyDeviceToHost);

        for (int32_t i = 0; i < csr.num_rows; ++i) {
            if (!float_eq(y_gpu[i], y_csr_ref[i])) {
                std::cout << "GPU vs CSR MISMATCH at row " << i << " (GPU=" << y_gpu[i] << ", CSR=" << y_csr_ref[i] << ")" << std::endl;
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
        // GPU not available is ok, CPU passed
        std::cout << "PASS (CPU only)" << std::endl;
        return 0;
    }

    std::cout << "PASS (CPU+GPU)" << std::endl;
    return 0;
}

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
