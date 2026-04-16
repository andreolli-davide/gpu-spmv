#include "spmv_cpu.h"
#include <omp.h>
#include <cmath>
#include <algorithm>

namespace spmv {

void spmv_cpu_serial(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    y.resize(A.rows);
    for (int64_t row = 0; row < A.rows; ++row) {
        double sum = 0.0;
        for (int64_t j = A.row_ptr[row]; j < A.row_ptr[row + 1]; ++j) {
            sum += A.values[j] * x.data[A.col_index[j]];
        }
        y.data[row] = sum;
    }
}

void spmv_cpu_omp(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    y.resize(A.rows);
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < A.rows; ++row) {
        double sum = 0.0;
        for (int64_t j = A.row_ptr[row]; j < A.row_ptr[row + 1]; ++j) {
            sum += A.values[j] * x.data[A.col_index[j]];
        }
        y.data[row] = sum;
    }
}

void fill_zero(DenseVector& v) {
    std::fill(v.data.begin(), v.data.end(), 0.0);
}

void fill_constant(DenseVector& v, double val) {
    std::fill(v.data.begin(), v.data.end(), val);
}

double infnorm(const DenseVector& a, const DenseVector& b) {
    double max_err = 0.0;
    for (int64_t i = 0; i < static_cast<int64_t>(a.data.size()); ++i) {
        max_err = std::max(max_err, std::fabs(a.data[i] - b.data[i]));
    }
    return max_err;
}

} // namespace spmv
