#include "spmv_selector.h"
#include <cmath>

namespace spmv {

FormatSelection select_format(const SparseMatrix& A) {
    FormatSelection sel;
    sel.format = SpMVFormat::CSR_TILED;
    sel.estimated_speedup_vs_csr = 1.0f;
    sel.reason = "default";

    if (A.rows == 0) return sel;

    double sum_len = 0.0, sum_sq_len = 0.0;
    int64_t max_len = 0;

    for (int64_t i = 0; i < A.rows; ++i) {
        const int64_t len = A.row_ptr[i + 1] - A.row_ptr[i];
        sum_len += len;
        sum_sq_len += len * len;
        max_len = std::max(max_len, len);
    }

    const double avg_len = sum_len / A.rows;
    const double variance = (sum_sq_len / A.rows) - (avg_len * avg_len);
    const double std_dev = std::sqrt(variance);
    const double coeff_var = (avg_len > 0) ? (std_dev / avg_len) : 0.0;

    // Power-law/irregular: high coefficient of variation
    // Greathouse '14: "For webbase, coefficient of variation = 2.1"
    if (coeff_var > 1.0 && avg_len < 10) {
        sel.format = SpMVFormat::CSR_ADAPTIVE;
        sel.estimated_speedup_vs_csr = 5.0f;
        sel.reason = "power-law distribution";
    }
    // Regular matrices: low variation
    // e.g., FEM/Spheres: cv=0.1, QCD: cv=0 (constant row length)
    else if (coeff_var < 0.5) {
        sel.format = SpMVFormat::ELL;
        sel.estimated_speedup_vs_csr = 2.0f;
        sel.reason = "regular structure";
    }

    return sel;
}

} // namespace spmv
