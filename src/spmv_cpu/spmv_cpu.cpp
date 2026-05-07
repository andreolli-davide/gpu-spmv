// src/spmv_cpu/spmv_cpu.cpp
#include "spmv_cpu/spmv_cpu.h"
#include <cmath>
#include <cstdio>

void spmv_csr_cpu(const MtxCsr& A, const double* x, double* y) {
    for (int i = 0; i < A.num_rows; ++i) {
        double acc = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            acc += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = acc;
    }
}

double compare_vectors(const double* a, const double* b, int n, double tolerance) {
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = std::fabs(a[i] - b[i]);
        if (diff > tolerance) {
            std::printf("  mismatch at [%d]: a=%.17g b=%.17g diff=%.3e\n", i, a[i], b[i], diff);
        }
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}
