// src/spmv_gpu/validate_csr_scalar.cu
#include "spmv_gpu/csr_scalar.cuh"
#include "spmv_cpu/spmv_cpu.h"
#include "parser/mtx_parser.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::vector<float> make_random(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& val : v) val = dist(rng);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: validate_csr_scalar <matrix.mtx>\n");
        return 1;
    }
    const char* path = argv[1];

    // ── Load matrix ──────────────────────────────────────────────────────────
    MtxCsr host_A;
    DeviceMatrix dev_A{};
    try {
        host_A = parse_mtx_csr(path);
        parse_mtx_gpu(path, &dev_A);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error loading '%s': %s\n", path, e.what());
        return 1;
    }
    std::printf("Matrix:   %s\n", path);
    std::printf("Size:     %d x %d, %d nnz\n",
                host_A.num_rows, host_A.num_cols, host_A.num_nonzeros);

    // ── Input vector x (same for CPU and GPU) ────────────────────────────────
    auto x = make_random(host_A.num_cols, 42);

    // ── CPU SpMV ─────────────────────────────────────────────────────────────
    std::vector<float> y_cpu(host_A.num_rows, 0.0f);
    auto t0 = std::chrono::high_resolution_clock::now();
    spmv_csr_cpu(host_A, x.data(), y_cpu.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    constexpr float kTolerance = 1e-5f;

    // ── GPU SpMV ─────────────────────────────────────────────────────────────
    float* d_x = nullptr;
    float* d_y = nullptr;
    if (cudaMalloc(&d_x, host_A.num_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_y, host_A.num_rows * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        cudaFree(d_x); cudaFree(d_y); free_gpu(&dev_A);
        return 1;
    }

    if (cudaMemcpy(d_x, x.data(), host_A.num_cols * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy H2D failed\n");
        cudaFree(d_x); cudaFree(d_y); free_gpu(&dev_A);
        return 1;
    }

    // Warm-up launch (not timed) to avoid first-launch overhead
    spmv_csr_scalar(dev_A, d_x, d_y);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "Warm-up kernel failed\n");
        cudaFree(d_x); cudaFree(d_y); free_gpu(&dev_A);
        return 1;
    }

    // Timed launch
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    spmv_csr_scalar(dev_A, d_x, d_y);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "Timed kernel failed\n");
        cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
        cudaFree(d_x); cudaFree(d_y); free_gpu(&dev_A);
        return 1;
    }

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // Copy result back
    std::vector<float> y_gpu(host_A.num_rows);
    cudaMemcpy(y_gpu.data(), d_y, host_A.num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    free_gpu(&dev_A);

    // ── Comparison ───────────────────────────────────────────────────────────
    std::printf("CPU time: %.3f ms\n", cpu_ms);
    std::printf("GPU time: %.3f ms  (kernel only, excludes H2D/D2H)\n", gpu_ms);

    float max_diff = compare_vectors(y_cpu.data(), y_gpu.data(), host_A.num_rows, kTolerance);
    std::printf("Max diff: %.3e\n", max_diff);

    bool pass = (max_diff <= kTolerance);
    std::printf("[%s]\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
