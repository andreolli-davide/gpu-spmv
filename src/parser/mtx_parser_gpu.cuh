#pragma once

#include "parser/mtx_parser.h"

/// GPU SpMV kernel
void spmv_gpu_kernel(const DeviceMatrix& csr, const float* d_x, float* d_y);