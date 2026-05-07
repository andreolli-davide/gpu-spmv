// src/spmv_cpu/test_spmv_cpu.cpp
#include "spmv_cpu/spmv_cpu.h"
#include "parser/mtx_parser.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::vector<double> make_ones(int n) {
    return std::vector<double>(n, 1.0);
}

static std::vector<double> make_random(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> v(n);
    for (auto& val : v) val = dist(rng);
    return v;
}

// Run SpMV and return result vector.
static std::vector<double> run(const MtxCsr& A, const std::vector<double>& x) {
    std::vector<double> y(A.num_rows, 0.0);
    spmv_csr_cpu(A, x.data(), y.data());
    return y;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: test_spmv_cpu <matrix.mtx>\n");
        return 1;
    }

    MtxCsr A = parse_mtx_csr(argv[1]);
    std::printf("Matrix: %d rows, %d cols, %d nnz\n", A.num_rows, A.num_cols, A.num_nonzeros);

    bool all_pass = true;

    // Stage 1: all-ones vector
    {
        auto x = make_ones(A.num_cols);
        auto y1 = run(A, x);
        auto y2 = run(A, x);
        double max_diff = compare_vectors(y1.data(), y2.data(), A.num_rows, 0.0);
        bool pass = (max_diff == 0.0);
        std::printf("[%s] Stage 1 (all-ones determinism): max_diff=%.3e\n",
                    pass ? "PASS" : "FAIL", max_diff);
        all_pass &= pass;
    }

    // Stage 2: random vector (seed 42)
    {
        auto x = make_random(A.num_cols, 42);
        auto y1 = run(A, x);
        auto y2 = run(A, x);
        double max_diff = compare_vectors(y1.data(), y2.data(), A.num_rows, 0.0);
        bool pass = (max_diff == 0.0);
        std::printf("[%s] Stage 2 (random-seed-42 determinism): max_diff=%.3e\n",
                    pass ? "PASS" : "FAIL", max_diff);
        all_pass &= pass;
    }

    return all_pass ? 0 : 1;
}
