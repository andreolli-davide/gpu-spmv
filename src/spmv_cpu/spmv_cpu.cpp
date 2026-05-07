// src/spmv_cpu/spmv_cpu.cpp
#include "spmv_cpu/spmv_cpu.h"
#include <cmath>
#include <cstdio>

void spmv_csr_cpu(const MtxCsr& A, const float* x, float* y) {
    for (int i = 0; i < A.num_rows; ++i) {
        float acc = 0.0f;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            acc += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = acc;
    }
}

float compare_vectors(const float* a, const float* b, int n, float tolerance) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > tolerance) {
            std::printf("  mismatch at [%d]: a=%.9g b=%.9g diff=%.3e\n", i, a[i], b[i], diff);
        }
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}
