// =============================================================================
// timer.h
// =============================================================================
// CPU and GPU wall-clock timers with a consistent interface.
//
// Timer Design
// ------------
// Timing is essential for measuring kernel performance (GFLOP/s, throughput)
// and for comparing CPU vs GPU implementations fairly.  The interface is
// deliberately simple: start() / stop() / elapsed_ms().
//
// CPUTimer uses OpenMP's wall-clock timer (omp_get_wtime), which:
//
//   • Returns wall-clock time (not CPU time), so it is not affected by
//     frequency scaling, background processes, or other threads.
//   • Has nanosecond resolution on all major platforms.
//   • Is the standard timer for OpenMP parallel regions.
//
// GPUTimer uses CUDA events for GPU-side timing.  CUDA events are:
//   • Recorded on the GPU stream — they measure actual kernel execution time,
//     not including queuing or host-device transfer overhead.
//   • Synchronized automatically when elapsed_ms() is called.
//
// Using Two Timers vs. One
// ------------------------
// GPU execution is asynchronous — control returns to the host before the
// kernel finishes.  Therefore we CANNOT simply record omp_get_wtime() before
// and after cudaLaunchKernel().  We need CUDA events that are ordered by
// stream execution.
//
// Phase 2 will show the correct usage pattern:
//
//   GPUTimer t;
//   t.start();                          // records start event on stream
//   cudaLaunchKernel(...);               // returns immediately
//   t.stop();                           // records stop event on stream
//   cudaStreamSynchronize(stream);       // MUST wait before reading time
//   double ms = t.elapsed_ms();         // returns wall-clock elapsed time
//
// =============================================================================

#ifndef TIMER_H
#define TIMER_H

#include <cstdint> // int64_t (reserved for future use)

namespace spmv {

// =============================================================================
// CPUTimer — OpenMP wall-clock timer
// =============================================================================
// Accurate across threads and unaffected by CPU frequency scaling.
//
// Usage:
//   CPUTimer t;
//   t.start();
//   do_work();
//   t.stop();
//   double ms = t.elapsed_ms();  // milliseconds elapsed between start and stop
//
// Note: start() captures the time; stop() is a no-op because elapsed_ms()
// always computes the interval from the stored start_time to the current
// moment.  This means elapsed_ms() can be called multiple times after a
// single start() to measure cumulative time — just call start() again to
// reset.
//
struct CPUTimer {
    double start_time = 0.0; // captured by omp_get_wtime()

    // Record the current wall-clock time as the start point.
    void start();

    // No-op: CPUTimer computes elapsed on-the-fly in elapsed_ms().
    // Provided for API symmetry with GPUTimer.
    void stop();

    // Returns the time elapsed since start() was last called, in milliseconds.
    // Repeated calls without a preceding start() return time since the
    // previous start().
    double elapsed_ms() const;
};

// =============================================================================
// GPUTimer — CUDA event-based GPU timer
// =============================================================================
// Measures GPU kernel execution time on a CUDA stream.
//
// Phase 1: This is a no-op stub.  Full implementation arrives in Phase 2
// when CUDA headers are available and the build links against cuda_runtime.
//
// Design rationale for the full version:
//   • cudaEvent_t is an opaque handle — we store as void* to avoid a
//     public dependency on <cuda_runtime.h> in this header.
//   • Construction creates the event pair; destruction destroys them.
//     This follows RAII: the timer is always valid after construction.
//   • start() and stop() record events but do NOT synchronize — the user
//     must call cudaStreamSynchronize() before elapsed_ms() reads them.
//     This is because cudaEventElapsedTime() handles the synchronization.
//
struct GPUTimer {
    void* cuda_start_event = nullptr; // cudaEvent_t, stored as void*
    void* cuda_stop_event  = nullptr; // cudaEvent_t, stored as void*

    GPUTimer();
    ~GPUTimer();

    // Record the start event on the current CUDA stream.
    // After this call, any kernel launched on the same stream will
    // be ordered after this event.
    void start();

    // Record the stop event on the current CUDA stream.
    // All kernels launched before this call will be ordered before it.
    void stop();

    // Returns elapsed time in milliseconds between the most recent
    // start() and stop() calls.
    // @requires  cudaStreamSynchronize() has been called since stop()
    double elapsed_ms() const;
};

} // namespace spmv

#endif // TIMER_H
