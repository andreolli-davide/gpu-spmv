#pragma once

#include <chrono>
#include <cstdint>

// --------------------------------------------------------------------------
// CPU timer using omp_get_wtime (OpenMP wall time)
// --------------------------------------------------------------------------
struct CPUTimer {
    double start_time = 0.0;

    void start();
    void stop();

    // Elapsed time in milliseconds
    double elapsed_ms() const;
};

// --------------------------------------------------------------------------
// GPU timer using CUDA events (Phase 2+ — compiles to no-op without CUDA)
// --------------------------------------------------------------------------
struct GPUTimer {
    void* cuda_start_event = nullptr;
    void* cuda_stop_event  = nullptr;

    GPUTimer();
    ~GPUTimer();

    void start();
    void stop();

    double elapsed_ms() const;
};

