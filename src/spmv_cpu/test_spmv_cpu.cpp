// src/spmv_cpu/test_spmv_cpu.cpp
#include "spmv_cpu/spmv_cpu.h"
#include "parser/mtx_parser.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::vector<float> make_ones(int n) {
    return std::vector<float>(n, 1.0f);
}

static std::vector<float> make_random(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& val : v) val = dist(rng);
    return v;
}

// Run SpMV and return result vector.
static std::vector<float> run(const MtxCsr& A, const std::vector<float>& x) {
    std::vector<float> y(A.num_rows, 0.0f);
    spmv_csr_cpu(A, x.data(), y.data());
    return y;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: test_spmv_cpu <matrix.mtx>\n");
        return 1;
    }

    MtxCsr A;
    try {
        A = parse_mtx_csr(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error loading matrix '%s': %s\n", argv[1], e.what());
        return 1;
    }
    std::printf("Matrix: %d rows, %d cols, %d nnz\n", A.num_rows, A.num_cols, A.num_nonzeros);

    bool all_pass = true;

    // Stage 1: all-ones vector
    {
        auto x = make_ones(A.num_cols);
        auto y1 = run(A, x);
        auto y2 = run(A, x);
        float max_diff = compare_vectors(y1.data(), y2.data(), A.num_rows, 0.0f);
        bool pass = (max_diff == 0.0f);
        std::printf("[%s] Stage 1 (all-ones determinism): max_diff=%.3e\n",
                    pass ? "PASS" : "FAIL", max_diff);
        all_pass &= pass;
    }

    // Stage 2: random vector (seed 42)
    {
        auto x = make_random(A.num_cols, 42);
        auto y1 = run(A, x);
        auto y2 = run(A, x);
        float max_diff = compare_vectors(y1.data(), y2.data(), A.num_rows, 0.0f);
        bool pass = (max_diff == 0.0f);
        std::printf("[%s] Stage 2 (random-seed-42 determinism): max_diff=%.3e\n",
                    pass ? "PASS" : "FAIL", max_diff);
        all_pass &= pass;
    }

    return all_pass ? 0 : 1;
}
