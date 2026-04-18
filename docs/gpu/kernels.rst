.. GPU Kernels — SpMV Implementation Documentation
   Implementation details with algorithm descriptions and paper references
   =======================================================================

GPU Kernels
===========

This section documents our GPU SpMV kernel implementations, including the
algorithm descriptions, mathematical frameworks, and paper references.

----

1. v1 — Basic CSR Row-Parallel Kernel
--------------------------------------

**Paper Reference**: Bell, N., & Garland, M. (2009). *Efficient Sparse-Matrix-Vector
Multiplication on GPUs*. SIAM SDM '09 / NVIDIA Technical Report NVR-2008-004.

Algorithm Description
~~~~~~~~~~~~~~~~~~~~

The v1 kernel implements the canonical **thread-per-row** mapping from
Bell & Garland '09. Each CUDA thread processes exactly one matrix row.

Thread Mapping
^^^^^^^^^^^^^^

.. math::

   \text{thread\_id} = \text{block\_idx} \times \text{block\_dim} + \text{thread\_idx}

Row :math:`i` is assigned to thread :math:`i`. The thread computes:

.. math::

   y_i = \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1}
         \text{values}[k] \cdot x_{\text{col\_index}[k]}

CSR Kernel Algorithm
^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   __global__ void spmv_csr_kernel(const int* row_ptr,
                                   const int* col_index,
                                   const double* values,
                                   const double* x,
                                   double* y,
                                   int num_rows) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < num_rows) {
           double sum = 0.0;
           for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
               sum += values[k] * x[col_index[k]];
           }
           y[i] = sum;
       }
   }

Memory Access Patterns
^^^^^^^^^^^^^^^^^^^^^^

**Coalesced access** to ``values`` and ``col_index``:

.. math::

   \text{thread}_j \text{ accesses} \rightarrow \text{values}[\text{base} + j]

Adjacent threads access adjacent memory locations, triggering **coalesced global
memory loads** on Ampere (128-byte cache lines, 32-byte transactions).

**Gather access** to ``x``:

.. math::

   x\_index = \text{col\_index}[k]

Accesses depend on column index distribution. Random distributions cause
uncoalesced reads.

Effective Memory Bandwidth
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   B_{\text{eff}} = \frac{2 \cdot \text{nnz} \cdot \sizeof(\text{double})}{t_{\text{kernel}}}}
                  = \frac{16 \cdot \text{nnz}}{t_{\text{ms}}} \quad \text{[GB/s]}

where :math:`t_{\text{kernel}}` is the kernel execution time in seconds.

Limitations
^^^^^^^^^^^

1. **Load imbalance**: Warp divergence when rows have vastly different lengths

   .. math::

      \text{row\_len}_i = \text{row\_ptr}[i+1] - \text{row\_ptr}[i]

   A warp processing 32 consecutive rows may encounter row lengths from 1 to 1000+.

2. **Low warp utilization**: For rows with :math:`\text{row\_len}_i < 32`,
   some threads in the warp are idle.

3. **No shared memory caching**: Every access to ``x`` goes to global memory.

Kernel Launch Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   int block_size = 256;  // multi-warp block
   int grid_size = (num_rows + block_size - 1) / block_size;
   spmv_csr_kernel<<<grid_size, block_size>>>(...);

Recommended block sizes for Ampere:

.. math::

   \text{block\_size} \in [128, 256, 512] \quad \text{(multiples of 32)}

Relevance
^^^^^^^^^

The v1 kernel serves as the **baseline implementation**. It demonstrates the
fundamental challenges in GPU SpMV (load imbalance, memory access patterns)
that our v2 kernel addresses through optimizations.

----

2. v2 — Optimized Kernel
------------------------

**Paper References**:

- Greathouse, J. L., & Daga, M. (2014). *Efficient Sparse Matrix-Vector Multiplication
  on GPUs Using a CSR-Adaptive Storage Format*. SC '14.
- Liu, W., & Vinter, B. (2015). *CSR5: An Efficient Storage Format for Cross-Platform
  Sparse Matrix-Vector Multiplication*. ICS '15.
- Chu, G., et al. (2023). *Efficient Algorithm Design of Optimizing SpMV on GPU*.
  HPDC '23.

Overview
~~~~~~~~

The v2 kernel incorporates multiple optimizations targeting the Ampere
architecture. It combines **adaptive row grouping** for load balancing,
**shared memory tiling** for the input vector, and **non-coherent loads**
for efficient memory access.

Key Optimizations
~~~~~~~~~~~~~~~~~

1. **Adaptive Row Grouping** (Greathouse & Daga '14)
2. **Shared Memory Tiling** (Bell & Garland '09)
3. **Non-Coherent Loads** via ``__ldg`` (Chu et al. '23)
4. **Tile-Based Load Balancing** (Liu & Vinter '15)

Adaptive Row Grouping
^^^^^^^^^^^^^^^^^^^^^

**Problem**: Warp divergence from variable row lengths.

.. math::

   \text{row\_len}_i = \text{row\_ptr}[i+1] - \text{row\_ptr}[i]

With simple row-to-thread mapping, a warp may process rows with lengths
varying from 1 to 1000+, causing severe load imbalance.

**Solution**: Group rows by workload into **warps of similar rows**:

.. math::

   \text{warp\_size} = 32 \quad \text{(one warp per group)}

Group construction:

1. Compute row lengths :math:`\text{row\_len}_i`
2. Sort rows by length (or use histogram-based binning)
3. Assign 32 consecutive rows of similar length to each warp

**Load Balancing Efficiency**:

.. math::

   \eta_{\text{load}} = \frac{\sum_{i} \text{row\_len}_i}{\text{num\_warps} \times \max_i \text{row\_len}_i}

For uniform row lengths, :math:`\eta_{\text{load}} \approx 1.0`. For power-law
distributions, naive CSR gives :math:`\eta_{\text{load}} \ll 1`.

Shared Memory Tiling
^^^^^^^^^^^^^^^^^^^^

**Problem**: Repeated accesses to the input vector ``x`` go to global memory.

**Solution**: Cache ``x`` in shared memory. Each block loads a contiguous chunk:

.. math::

   \text{shared\_mem}[i] = x[\text{block\_id} \times \text{SHARED\_ELEMENTS} + i]

**Memory latency comparison**:

=========== ==============
Level       Latency
=========== ==============
L1 cache    ~1–3 cycles
L2 cache    ~40 cycles
Global mem  ~400 cycles
Shared mem  ~1–10 ns
=========== ==============

Shared memory latency is ~1–10 ns vs. global memory ~100–400 ns.

Non-Coherent Loads (``__ldg``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``x`` elements outside the shared memory range, use the ``__ldg``
intrinsic:

.. code-block:: cpp

   double x_val = __ldg(&x[col_index[k]]);

This generates ``ld.global.nc`` (non-coherent global load), which:

- Bypasses the L1 cache (reduces coherency overhead)
- Still uses L2 cache for reuse
- Ideal for read-only data accessed multiple times

Tile-Based Load Balancing
^^^^^^^^^^^^^^^^^^^^^^^^^

Inspired by Liu & Vinter's CSR5, the v2 kernel uses **tiles** for dynamic
work distribution:

.. math::

   \mathbf{A} \rightarrow \text{tiles} \quad \text{(power-of-2 row groups)}

Tile size is chosen to balance:

1. Enough rows per tile for coalesced access
2. Few enough rows per tile for load balancing

.. math::

   \text{tile\_size} = 2^{\lceil \log_2(\sqrt{\text{nnz}}) \rceil}

Occupancy Tuning (Ampere)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Occupancy formula**:

.. math::

   \text{occupancy} = \frac{\text{active\_warps}}{\text{max\_warps\_per\_SM}}

Higher occupancy hides memory latency:

.. math::

   \text{latency\_hidden} = \text{occupancy} \times \text{latency}_{\text{mem}}

**Ampere memory hierarchy**:

=========== ============== ===============
Level       Latency        Bandwidth
=========== ============== ===============
L1 cache    ~1–3 cycles    ~12,288 GB/s
L2 cache    ~40 cycles     ~2,000 GB/s
Global mem  ~400 cycles    ~1,500 GB/s
=========== ============== ===============

Shared Memory Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default shared memory per block:

.. math::

   \text{SHARED\_MEM\_SIZE} = 32 \ \text{KB} = 4096 \ \text{doubles}

Ampere limit: 48 KB per block.

Algorithm Summary
~~~~~~~~~~~~~~~~~

The v2 kernel processes tiles with:

1. **Load balancing**: Rows grouped by similar nnz per warp
2. **Memory optimization**: x cached in shared memory, __ldg fallback
3. **Coalesced access**: Sequential access to values and col_index
4. **Occupancy tuning**: Block size selected for maximum occupancy

Relevance
^^^^^^^^^

The v2 kernel addresses the limitations of v1 through:

- **Greathouse & Daga '14**: Eliminates warp divergence via adaptive grouping
- **Liu & Vinter '15**: Tile-based load balancing without preprocessing
- **Chu et al. '23**: Efficient memory access via __ldg and shared memory

----

3. Performance Results
----------------------

Performance Summary
~~~~~~~~~~~~~~~~~~~

============  =======  ========  =======  =======
Kernel        Format   Load Bal.  Shared   __ldg
============  =======  ========  =======  =======
v1            CSR      —         —        —
v2            CSR      ✅        ✅       ✅
============  =======  ========  =======  =======

Key Metrics
~~~~~~~~~~~

**Effective Bandwidth**:

.. math::

   B_{\text{eff}} = \frac{2 \cdot \text{nnz}}{t_{\text{ms}}} \quad \text{[GB/s]}

**GFLOP/s** (2 FLOPs per non-zero):

.. math::

   \text{GFLOP/s} = \frac{2 \cdot \text{nnz}}{t_{\text{s}}} \times 10^{-9}

Expected Improvements
~~~~~~~~~~~~~~~~~~~~~

For **regular matrices** (low row-length variance):
- v2 ~1.2–1.5× faster than v1 (shared memory benefit)

For **irregular matrices** (power-law distribution):
- v2 ~2–4× faster than v1 (adaptive row grouping benefit)

References
----------

* Bell, N., & Garland, M. (2009). Efficient Sparse-Matrix-Vector Multiplication on
  GPUs. NVIDIA Technical Report NVR-2008-004 / SIAM SDM '09.

* Greathouse, J. L., & Daga, M. (2014). Efficient Sparse Matrix-Vector Multiplication
  on GPUs Using a CSR-Adaptive Storage Format. SC '14.

* Liu, W., & Vinter, B. (2015). CSR5: An Efficient Storage Format for Cross-Platform
  Sparse Matrix-Vector Multiplication. ICS '15.

* Chu, G., et al. (2023). Efficient Algorithm Design of Optimizing SpMV on GPU. HPDC '23.