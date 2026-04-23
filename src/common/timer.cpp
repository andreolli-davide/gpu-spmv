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
    // No-op: elapsed is computed lazily on each elapsed_ms() call.
    // This is intentional — you can call elapsed_ms() multiple times
    // to get a running total, or just call it once at the end.
}

double CPUTimer::elapsed_ms() const {
    return (omp_get_wtime() - start_time) * 1000.0;
}

// =============================================================================
// GPUTimer — Full implementation
// =============================================================================
// Uses CUDA events (RAII: created in constructor, destroyed in destructor).
// User must call cudaStreamSynchronize() before elapsed_ms().
// Gracefully degrades when CUDA is unavailable (returns 0.0).
// =============================================================================
GPUTimer::GPUTimer() {
    cudaEvent_t* start = reinterpret_cast<cudaEvent_t*>(&cuda_start_event);
    cudaEvent_t* stop = reinterpret_cast<cudaEvent_t*>(&cuda_stop_event);
    cudaError_t err = cudaEventCreate(start);
    if (err != cudaSuccess) {
        cuda_start_event = nullptr;
    }
    err = cudaEventCreate(stop);
    if (err != cudaSuccess) {
        cuda_stop_event = nullptr;
    }
}

GPUTimer::~GPUTimer() {
    if (cuda_start_event != nullptr) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cuda_start_event));
        cuda_start_event = nullptr;
    }
    if (cuda_stop_event != nullptr) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cuda_stop_event));
        cuda_stop_event = nullptr;
    }
}

void GPUTimer::start() {
    if (cuda_start_event == nullptr) {
        return;
    }
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(cuda_start_event), 0);
}

void GPUTimer::stop() {
    if (cuda_stop_event == nullptr) {
        return;
    }
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(cuda_stop_event), 0);
}

double GPUTimer::elapsed_ms() const {
    if (cuda_start_event == nullptr || cuda_stop_event == nullptr) {
        return 0.0;
    }
    cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(cuda_stop_event));

    float ms = 0.0f;
    cudaEventElapsedTime(&ms,
        reinterpret_cast<cudaEvent_t>(cuda_start_event),
        reinterpret_cast<cudaEvent_t>(cuda_stop_event));
    return static_cast<double>(ms);
}

} // namespace spmv
