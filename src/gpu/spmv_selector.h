#ifndef SPMV_SELECTOR_H
#define SPMV_SELECTOR_H

#include "sparse_matrix.h"
#include "spmv_ell.h"
#include "spmv_csr_adaptive.h"

namespace spmv {

enum class SpMVFormat {
    CSR_ADAPTIVE,  // Greathouse & Daga '14 - for irregular matrices
    ELL,           // Bell & Garland '09 - for regular matrices
    CSR_TILED      // Your v2 - default
};

struct FormatSelection {
    SpMVFormat format;
    float estimated_speedup_vs_csr;
    const char* reason;
};

FormatSelection select_format(const SparseMatrix& A);

} // namespace spmv
#endif
