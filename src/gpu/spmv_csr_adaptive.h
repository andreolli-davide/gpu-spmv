#ifndef SPMV_CSR_ADAPTIVE_H
#define SPMV_CSR_ADAPTIVE_H

#include "sparse_matrix.h"
#include "gpu_utils.h"

namespace spmv {

struct CSRAdaptiveMeta {
    int64_t num_row_blocks;
    std::vector<int64_t> row_block_ptr;
    std::vector<int64_t> row_block_nnz;
    std::vector<int32_t> row_to_block;
    std::vector<int64_t> sorted_row_order;
};

CSRAdaptiveMeta compute_adaptive_meta(const SparseMatrix& A, int warp_size = 32);

void spmv_csr_adaptive(const SparseMatrix& A, const DenseVector& x, DenseVector& y,
                       const CSRAdaptiveMeta& meta);

} // namespace spmv
#endif
