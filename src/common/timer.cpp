// =============================================================================
// timer.cpp
// =============================================================================

#include "timer.h"

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
// GPUTimer — Phase 1 stub
// =============================================================================
// Full implementation in Phase 2 with CUDA headers.
// The stub is intentionally minimal so that:
//   • The code compiles and links without CUDA installed
//   • Any accidental use of GPUTimer in Phase 1 produces 0.0 rather than
//     silent garbage or a crash
// =============================================================================
GPUTimer::GPUTimer()  = default;
GPUTimer::~GPUTimer() = default;

void GPUTimer::start() {}
void GPUTimer::stop()  {}

double GPUTimer::elapsed_ms() const { return 0.0; }

} // namespace spmv
