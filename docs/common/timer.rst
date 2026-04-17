.. Timer Utilities
   Scientific documentation
   ========================

Timer Utilities
===============

Overview
--------

Accurate performance measurement is fundamental to scientific computing research.
This module provides two timer types with a consistent interface:

1. **CPUTimer**: OpenMP wall-clock timer for CPU code
2. **GPUTimer**: CUDA event-based timer for GPU code

Both share the same interface (:cpp:func:`start`, :cpp:func:`stop`,
:cpp:func:`elapsed_ms`) but measure time in fundamentally different execution
contexts.

CPUTimer: OpenMP Wall-Clock Timer
---------------------------------

The CPUTimer uses ``omp_get_wtime()``, the OpenMP wall-clock timer function.

Mathematical Basis
~~~~~~~~~~~~~~~~~~

The wall-clock time returned by ``omp_get_wtime()`` is defined as:

.. math::

   t_{\text{wall}} = \text{time since an epoch (typically Jan 1, 1970 UTC)}

The ``omp_get_wtime()`` function guarantees:

1. **Monotonicity**: :math:`t_{i+1} \geq t_i` for successive calls
2. **Consistency across threads**: All threads in an OpenMP program see the same
   time base
3. **Nanosecond resolution**: :math:`\text{resolution} = \Delta t_{\text{min}}`
   typically :math:`< 1` ns on modern hardware

For a code segment:

.. math::

   \Delta t = \text{omp\_get\_wtime()}_{\text{after}} - \text{omp\_get\_wtime()}_{\text{before}}

The reported time is **wall-clock time**, not CPU time. This is critical because:

.. math::

   \text{CPU time} &= \text{wall time} \times \text{number of active cores} \times \text{utilization} \\
   \text{For 4 cores at 100\% utilization:}&\ \text{CPU time} = 4 \times \text{wall time} \\
   \text{For 4 cores at 25\% utilization:}&\ \text{CPU time} = 1 \times \text{wall time}

Wall-clock time is not affected by:
- CPU frequency scaling (P-states, Turbo Boost)
- Background processes
- Context switches
- Other threads in the system

CPUTimer Design
~~~~~~~~~~~~~~~

The CPUTimer is deliberately simple:

.. code-block:: cpp

   struct CPUTimer {
       double start_time = 0.0;
       void start();      // start_time = omp_get_wtime()
       void stop();       // no-op
       double elapsed_ms() const; // (omp_get_wtime() - start_time) * 1000
   };

The ``stop()`` function is a no-op because ``elapsed_ms()`` computes the interval
on-the-fly:

.. math::

   t_{\text{elapsed}} = (\text{omp\_get\_wtime()}_{\text{now}} - \text{start\_time}) \times 1000

This design allows calling ``elapsed_ms()`` multiple times after a single
``start()`` to measure cumulative time.

GPUTimer: CUDA Event-Based Timer
--------------------------------

GPU timing requires a fundamentally different approach due to **asynchronous
execution**.

The Asynchronous Execution Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA kernels are launched asynchronously:

.. code-block:: cpp

   t.start();                      // records event E1 on stream
   cudaLaunchKernel(...);          // returns IMMEDIATELY
   t.stop();                       // records event E2 on stream
   // at this point, kernel may NOT have finished!
   double ms = t.elapsed_ms();     // WRONG if called here

Host code continues executing while the kernel runs on the GPU. Therefore,
host-side timing (even with ``omp_get_wtime()``) measures:

.. math::

   t_{\text{host}} = t_{\text{kernel}} + t_{\text{launch}} + t_{\text{queue}} + t_{\text{transfer}}

where :math:`t_{\text{kernel}}` is the actual GPU execution time.

CUDA Event Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA events provide a mechanism to measure actual kernel time:

1. Events are recorded on the CUDA **stream** in program order
2. ``cudaEventElapsedTime()`` synchronizes internally:
   
   .. math::

      t_{\text{kernel}} = \text{cudaEventElapsedTime}(E_{\text{start}}, E_{\text{stop}})

3. The user must call ``cudaStreamSynchronize()`` before reading the result
   (though ``cudaEventElapsedTime()`` handles this internally)

Correct Usage Pattern
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   GPUTimer t;
   t.start();                    // records start event on stream
   cudaLaunchKernel(...);       // returns immediately
   t.stop();                    // records stop event on stream
   cudaStreamSynchronize(stream); // WAIT for completion
   double ms = t.elapsed_ms();  // now correct

GPUTimer Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA events are created with:

.. code-block:: cpp

   cudaEventCreate(&cuda_start_event);  // opaque handle
   cudaEventCreate(&cuda_stop_event);

The ``void*`` storage avoids a public dependency on ``<cuda_runtime.h>`` in
the header, following the **pimpl idiom**.

The ``start()`` and ``stop()`` functions record events but do not synchronize:

.. code-block:: cpp

   void GPUTimer::start() {
       cudaEventRecord(cuda_start_event, 0);  // stream 0 (default)
   }

The ``elapsed_ms()`` function calls ``cudaEventElapsedTime()``:

.. math::

   \text{elapsed\_ms} = \text{cudaEventElapsedTime}(\text{start}, \text{stop}) \times 1000

Phase 1: Stub Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Phase 1, CUDA headers are not available. The GPUTimer is a no-op stub:

.. code-block:: cpp

   double GPUTimer::elapsed_ms() const { return 0.0; }

Full implementation arrives in Phase 2 when the build system links against
``cuda_runtime``.

API Reference
-------------

.. cpp:type:: struct CPUTimer

   OpenMP wall-clock timer. Accurate across threads and unaffected by
   CPU frequency scaling.

   .. cpp:func:: void start()

      Record the current wall-clock time as the start point.

      Calls ``omp_get_wtime()`` and stores the result in ``start_time``.

   .. cpp:func:: void stop()

      No-op. Provided for API symmetry with ``GPUTimer``.

      CPUTimer computes elapsed on-the-fly in ``elapsed_ms()``.

   .. cpp:func:: double elapsed_ms() const

      Returns the time elapsed since ``start()`` was last called, in milliseconds.

      .. math::

         t_{\text{ms}} = (\text{omp\_get\_wtime()}_{\text{now}} - \text{start\_time}) \times 1000

      Repeated calls without a preceding ``start()`` return time since the
      previous ``start()``.

.. cpp:type:: struct GPUTimer

   CUDA event-based GPU timer. Measures GPU kernel execution time on a CUDA stream.

   Phase 1: This is a no-op stub. Full implementation arrives in Phase 2
   when CUDA headers are available.

   .. cpp:func:: GPUTimer()

      Constructs the timer. Creates the underlying ``cudaEvent_t`` handles.

      RAII: the timer is always valid after construction.

   .. cpp:func:: ~GPUTimer()

      Destroys the underlying CUDA events.

   .. cpp:func:: void start()

      Record the start event on the current CUDA stream.
      After this call, any kernel launched on the same stream will
      be ordered after this event.

   .. cpp:func:: void stop()

      Record the stop event on the current CUDA stream.
      All kernels launched before this call will be ordered before it.

   .. cpp:func:: double elapsed_ms() const

      Returns elapsed time in milliseconds between the most recent
      ``start()`` and ``stop()`` calls.

      **Important**: ``cudaStreamSynchronize()`` must have been called since
      ``stop()`` before reading this value.

      .. math::

         t_{\text{ms}} = \text{cudaEventElapsedTime}(E_{\text{start}}, E_{\text{stop}}) \times 1000

Performance Measurement Best Practices
--------------------------------------

1. **Warm-up runs**: Discard the first 1-2 iterations to avoid cold-start effects
2. **Multiple samples**: Report the mean and standard deviation over 5-10 runs
3. **Steady state**: Ensure the system is in a stable state (no background processes)
4. **Synchronization**: Always synchronize GPU before timing (``cudaStreamSynchronize``)
5. **Memory transfer**: Time memory transfers separately from kernel execution
6. **Clock gating**: Be aware that GPU clock rates vary with workload (Thermal Throttling)

References
----------

* OpenMP Architecture Review Board. *OpenMP Application Programming Interface*, Version 5.0.
  https://www.openmp.org/specifications/
* NVIDIA Corporation. *CUDA Toolkit Documentation*. https://docs.nvidia.com/cuda/
* Intel Developer Zone. *Improving Performance through OpenMP* (application note).
