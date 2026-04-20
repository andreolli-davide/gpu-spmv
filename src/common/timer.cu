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
// GPUTimer — CUDA event-based GPU timer (Phase 2)
// =============================================================================
// Uses cudaEvent_t for accurate GPU kernel timing on CUDA streams.
//
#if SPMV_CUDA_ENABLED

#include <cuda_runtime.h>

// Helper: check CUDA event error
static inline const char* cudaEventGetErrorString(cudaError_t err) {
    return cudaGetErrorString(err);
}

GPUTimer::GPUTimer() {
    cudaError_t err_start = cudaEventCreateWithFlags(
        reinterpret_cast<cudaEvent_t*>(&cuda_start_event),
        cudaEventDefault);
    cudaError_t err_stop = cudaEventCreateWithFlags(
        reinterpret_cast<cudaEvent_t*>(&cuda_stop_event),
        cudaEventDefault);
    if (err_start != cudaSuccess || err_stop != cudaSuccess) {
        cuda_start_event = nullptr;
        cuda_stop_event = nullptr;
    }
}

GPUTimer::~GPUTimer() {
    if (cuda_start_event) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cuda_start_event));
    }
    if (cuda_stop_event) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cuda_stop_event));
    }
}

void GPUTimer::start() {
    if (cuda_start_event) {
        cudaEventRecord(reinterpret_cast<cudaEvent_t>(cuda_start_event), 0);
    }
}

void GPUTimer::stop() {
    if (cuda_stop_event) {
        cudaEventRecord(reinterpret_cast<cudaEvent_t>(cuda_stop_event), 0);
    }
}

double GPUTimer::elapsed_ms() const {
    if (!cuda_start_event || !cuda_stop_event) {
        return 0.0;
    }
    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&ms,
        reinterpret_cast<cudaEvent_t>(cuda_start_event),
        reinterpret_cast<cudaEvent_t>(cuda_stop_event));
    if (err != cudaSuccess) {
        return 0.0f;
    }
    return static_cast<double>(ms);
}

#else // Non-CUDA fallback (Phase 1 stub)

GPUTimer::GPUTimer()  = default;
GPUTimer::~GPUTimer() = default;

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

#endif // SPMV_CUDA_ENABLED

} // namespace spmv
