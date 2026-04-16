#include "timer.h"
#include <omp.h>

// --------------------------------------------------------------------------
// CPUTimer
// --------------------------------------------------------------------------
void CPUTimer::start() {
    start_time = omp_get_wtime();
}

void CPUTimer::stop() {
    // no-op: elapsed is computed on the fly
}

double CPUTimer::elapsed_ms() const {
    return (omp_get_wtime() - start_time) * 1000.0;
}

// --------------------------------------------------------------------------
// GPUTimer
// --------------------------------------------------------------------------
// Stub implementation — full version in Phase 2 with CUDA headers
GPUTimer::GPUTimer()  = default;
GPUTimer::~GPUTimer() = default;

void GPUTimer::start() {}
void GPUTimer::stop()  {}

double GPUTimer::elapsed_ms() const { return 0.0; }
