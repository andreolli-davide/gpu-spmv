// =============================================================================
// timer.cpp
// =============================================================================

#include "timer.h"
#include <omp.h>
#include <cuda_runtime.h>

namespace spmv {

// =============================================================================
// CPUTimer
// =============================================================================
// omp_get_wtime() returns seconds as a double with nanosecond resolution.
// We multiply by 1000 to get milliseconds for consistency with GPUTimer's
// cudaEventElapsedTime(), which also returns milliseconds.
//
// Why wall-clock time and not CPU time?
//   CPU time would measure the aggregate CPU time consumed by the current
//   thread — it does NOT account for the fact that multiple threads run in
//   parallel.  For an 8-thread OMP region, CPU time would be ~8× wall time,
//   which is meaningless for performance comparison.  Wall-clock time is
//   the correct metric.
// =============================================================================
void CPUTimer::start() {
    start_time = omp_get_wtime();
}

void CPUTimer::stop() {
}

double CPUTimer::elapsed_ms() const {
    return (omp_get_wtime() - start_time) * 1000.0;
}

// =============================================================================
// GPUTimer — CUDA event-based timing
// =============================================================================
GPUTimer::GPUTimer() {
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&cuda_start_event));
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&cuda_stop_event));
}

GPUTimer::~GPUTimer() {
    if (cuda_start_event) cudaEventDestroy(static_cast<cudaEvent_t>(cuda_start_event));
    if (cuda_stop_event) cudaEventDestroy(static_cast<cudaEvent_t>(cuda_stop_event));
}

void GPUTimer::start() {
    cudaEventRecord(static_cast<cudaEvent_t>(cuda_start_event), 0);
}

void GPUTimer::stop() {
    cudaEventRecord(static_cast<cudaEvent_t>(cuda_stop_event), 0);
}

double GPUTimer::elapsed_ms() const {
    float ms = 0.0f;
    cudaEventSynchronize(static_cast<cudaEvent_t>(cuda_stop_event));
    cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(cuda_start_event), static_cast<cudaEvent_t>(cuda_stop_event));
    return static_cast<double>(ms);
}

} // namespace spmv
