.. CPU SpMV Implementations
   Scientific documentation with formulas
   =======================================

CPU SpMV Implementations
========================

Overview
--------

**Sparse Matrix-Vector Multiplication (SpMV)** is the fundamental kernel in
iterative linear solvers and eigenvalue methods. Given a sparse matrix
:math:`\mathbf{A} \in \mathbb{R}^{m \times n}`, an input dense vector
:math:`\mathbf{x} \in \mathbb{R}^{n}`, and an output dense vector
:math:`\mathbf{y} \in \mathbb{R}^{m}`, SpMV computes:

.. math::

   \mathbf{y} = \mathbf{A} \cdot \mathbf{x}

Element-wise, this is:

.. math::

   y_i = \sum_{j=0}^{n-1} a_{ij} \cdot x_j \quad \forall i \in [0, m)

However, for a sparse matrix where :math:`\text{nnz} \ll m \times n`,
we only iterate over non-zero entries.

SpMV in CSR Format
------------------

For a CSR-format matrix (see :ref:`csr-format` in sparse_matrix.rst),
the SpMV operation becomes:

.. math::

   y_i = \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} 
         \text{values}[k] \cdot x_{\text{col\_index}[k]}

This is the **canonical SpMV algorithm**: for each row :math:`i`, accumulate the
dot product of the row's non-zeros with the corresponding :math:`x` entries.

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

The time complexity of SpMV is:

.. math::

   T_{\text{SpMV}} = \Theta(\text{nnz})

This is optimal: every non-zero must be visited at least once.

Arithmetic Intensity
~~~~~~~~~~~~~~~~~~~~

**Arithmetic intensity** (FLOPs per byte of memory traffic) characterizes
the compute vs. memory balance:

For each non-zero, we perform:
- 1 multiplication: :math:`v \cdot x_j`
- 1 addition: accumulate into :math:`y_i`

Total: **2 FLOPs per non-zero**

Memory traffic (for CSR on CPU, streaming through once):

.. math::

   \text{reads} = \underbrace{\text{nnz} \cdot 8}_{\text{values}}
               + \underbrace{\text{nnz} \cdot 8}_{\text{col\_index}}
               + \underbrace{\text{nnz} \cdot 8}_{x\text{ (read once, cached)}}
               = 24 \cdot \text{nnz} \ \text{bytes}

   \text{writes} = \underbrace{m \cdot 8}_{y\text{ (output)}}
                 \approx 8 \cdot \text{nnz} \ \text{bytes for large } m

For typical sparse matrices with balanced rows:

.. math::

   \text{bytes per nnz} \approx \frac{24 \cdot \text{nnz}}{\text{nnz}} = 24 \ \text{B/nnz}

   \text{AI} = \frac{2 \cdot \text{nnz}}{24 \cdot \text{nnz}} = \frac{1}{12} \approx 0.083
            \ \text{FLOPs/byte}

This is **memory-bound** (AI << 1), not compute-bound. Optimization focus should
be on maximizing memory bandwidth utilization, not FLOP count.

Serial Implementation (spmv_cpu_serial)
----------------------------------------

The serial implementation uses a single thread with straightforward row-wise
iteration:

.. code-block:: cpp

   void spmv_cpu_serial(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
       y.resize(A.rows);
       for (int64_t i = 0; i < A.rows; ++i) {
           double sum = 0.0;
           for (int64_t k = A.row_ptr[i]; k < A.row_ptr[i+1]; ++k) {
               sum += A.values[k] * x.data[A.col_index[k]];
           }
           y.data[i] = sum;
       }
   }

Correctness Argument
~~~~~~~~~~~~~~~~~~~~~

**Theorem**: The serial implementation computes :math:`\mathbf{y} = \mathbf{A} \cdot \mathbf{x}` exactly.

**Proof**: For each row :math:`i`:

1. The inner loop iterates over :math:`k \in [\text{row\_ptr}[i], \text{row\_ptr}[i+1])`
2. For each :math:`k`, it adds :math:`\text{values}[k] \cdot x_{\text{col\_index}[k]}`
3. By the CSR invariant, this is precisely:
   
   .. math::

      \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} a_{i, \text{col\_index}[k]} \cdot x_{\text{col\_index}[k]}
    = \sum_{j=0}^{n-1} a_{ij} \cdot x_j

   since :math:`a_{ij} = 0` for all :math:`j \notin \{\text{col\_index}[k]\}` in the range.

4. The result is written to ``y[i]``.

By induction over all rows :math:`i = 0, \ldots, m-1`, we have :math:`y_i = (\mathbf{A} \cdot \mathbf{x})_i`
for all :math:`i`. ∎

IEEE-754 Determinism
~~~~~~~~~~~~~~~~~~~~~

The serial implementation is **deterministic** because:

1. The accumulation order is fixed: left-to-right within each row
2. IEEE-754 floating-point addition is deterministic
3. No data races (single thread)

**Lemma**: The accumulation order within a row is exactly the order of entries
in the CSR ``values`` array, which is file order (Matrix Market) preserved through
COO→CSR conversion.

OpenMP Parallel Implementation (spmv_cpu_omp)
----------------------------------------------

The parallel implementation uses OpenMP to distribute rows across CPU threads:

.. code-block:: cpp

   void spmv_cpu_omp(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
       y.resize(A.rows);
       #pragma omp parallel for schedule(static)
       for (int64_t i = 0; i < A.rows; ++i) {
           double sum = 0.0;
           for (int64_t k = A.row_ptr[i]; k < A.row_ptr[i+1]; ++k) {
               sum += A.values[k] * x.data[A.col_index[k]];
           }
           y.data[i] = sum;
       }
   }

Parallelization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~

**Row-level parallelism**: Each row :math:`i` is independent, so we distribute
rows across threads using static scheduling.

Why Static Schedule?
^^^^^^^^^^^^^^^^^^^^

The work per row varies: :math:`\text{row\_ptr}[i+1] - \text{row\_ptr}[i]` is the
number of non-zeros in row :math:`i`. For random matrices, this follows a
distribution (often Poisson or multinomial).

However, for the static scheduling of simple vector kernels like SpMV:

1. **Load imbalance is acceptable**: The variation in row lengths averages out
   over the thousands of rows typical in sparse matrices
2. **No scheduling overhead**: ``static`` scheduling has near-zero overhead
3. **Cache locality**: Consecutive iterations access consecutive memory,
   enabling hardware prefetching

Dynamic scheduling would introduce more overhead (work queue management,
atomic operations) than it would save for this memory-bound kernel.

Why Not a Reduction Clause?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A naive OpenMP reduction:

.. code-block:: cpp

   #pragma omp parallel for reduction(+:y[:A.rows])  // WRONG

The reduction clause accumulates into an **unspecified order** across threads.
For floating-point addition:

.. math::

   ((a + b) + c) + d \neq (a + b) + (c + d) \quad \text{in IEEE-754}

Different thread interleavings produce different rounding errors, resulting in
**bitwise non-deterministic output**.

Our solution: **private scalar accumulator per row**:

.. code-block:: cpp

   double sum = 0.0;  // private to each thread
   // ... accumulate into sum ...
   y.data[i] = sum;   // write once, atomically (but writes are to different indices)

Since each ``y[i]`` is written by exactly one thread (row :math:`i` belongs to
exactly one thread under static scheduling), there are no data races.

Correctness Argument
~~~~~~~~~~~~~~~~~~~~~

**Theorem**: ``spmv_cpu_omp`` produces **bitwise identical** output to ``spmv_cpu_serial``.

**Proof**: We show that for every row :math:`i`, the computation is identical:

1. Row :math:`i` is assigned to exactly one thread under static scheduling
2. That thread computes:

   .. math::

      \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} 
      \text{values}[k] \cdot x_{\text{col\_index}[k]}

   in exactly the same order as the serial version (sequential within thread)
3. The result is written to ``y[i]`` once, with no concurrent writes
4. No other thread reads or writes ``y[i]``

Therefore, :math:`y_i^{\text{serial}} = y_i^{\text{omp}}` for all :math:`i`. ∎

This bitwise equality is critical for the correctness test infrastructure.

Vector Utility Functions
------------------------

.. cpp:function:: void fill_zero(DenseVector& v)

   Sets all entries of a vector to 0.0.

   .. math::

      v_i = 0.0 \quad \forall i \in [0, v.\text{size})

.. cpp:function:: void fill_constant(DenseVector& v, double val)

   Sets all entries of a vector to a given value.

   .. math::

      v_i = \text{val} \quad \forall i \in [0, v.\text{size})

.. cpp:function:: double infnorm(const DenseVector& a, const DenseVector& b)

   Computes the **L-infinity norm** of the difference vector :math:`\mathbf{a} - \mathbf{b}`:

   .. math::

      \|\mathbf{a} - \mathbf{b}\|_\infty = \max_{i} |a_i - b_i|

   Used to compare two SpMV outputs for correctness.

   **Tolerance**: We consider two results "equal" if the L-inf error is below
   :math:`10^{-15}` for double-precision arithmetic. This accounts for rounding
   differences in accumulation order while catching real bugs.

   :param a: First vector (size :math:`N`)
   :param b: Second vector (size :math:`N`); must have same size as ``a``
   :return: :math:`\max_i |a_i - b_i|`

API Reference
-------------

.. cpp:function:: void spmv_cpu_serial(const SparseMatrix& A, const DenseVector& x, DenseVector& y)

   Sequential (single-threaded) CSR SpMV.

   The **canonical reference implementation**. Runs on one CPU core.
   Used as the golden reference for all correctness tests.

   **Performance**: :math:`O(\text{nnz})` memory bandwidth-bound operations.
   Typical throughput: 5–15 GB/s on a modern CPU (DDR4 or DDR5).

   :param A: Input matrix in CSR format, size :math:`m \times n`
   :param x: Input dense vector, size :math:`n`
   :param[out] y: Output dense vector, size :math:`m`. Resized automatically.

.. cpp:function:: void spmv_cpu_omp(const SparseMatrix& A, const DenseVector& x, DenseVector& y)

   OpenMP-parallel CSR SpMV.

   Parallel row-wise SpMV using OpenMP. One thread per row (or a contiguous
   chunk of rows for very large matrices).

   **Performance**: Near-linear speedup with core count for memory-bandwidth-bound
   workloads. Typical: 3.5–4× speedup on 4-core CPU.

   :param A: Input matrix in CSR format, size :math:`m \times n`
   :param x: Input dense vector, size :math:`n`
   :param[out] y: Output dense vector, size :math:`m`. Resized automatically.

Performance Characterization
----------------------------

Cache Behavior
~~~~~~~~~~~~~~

SpMV is a **streaming** operation with poor temporal locality:

1. ``values`` and ``col_index``: traversed once, in order (good spatial locality)
2. ``x.data``: read multiple times (once per non-zero in each column accessed)
3. ``y.data``: written once per row

For a matrix with :math:`\bar{n nz}` non-zeros per row (the row-length average):

.. math::

   \text{bytes per row processed} = 2 \cdot \bar{n nz} \cdot 8 \ \text{B (values + col\_index)}
                                  + \bar{n nz} \cdot 8 \ \text{B (reads of x)}
                                  + 8 \ \text{B (write of y)}

Modern CPUs can sustain ~50 GB/s bandwidth, limiting SpMV throughput to:

.. math::

   \text{throughput} \leq \frac{50 \ \text{GB/s}}{24 \ \text{B/nnz}} \approx 2 \ \text{GNNZ/s}

where "GNNZ/s" means billions of non-zeros per second.

SIMD Vectorization
~~~~~~~~~~~~~~~~~~

The inner loop is amenable to SIMD (AVX2/AVX-512) vectorization:

.. math::

   \text{AVX-512 width} = 8 \ \text{doubles per cycle}

The key is to vectorize the **multiplications** and use horizontal addition
for the reduction:

.. code-block:: text

   // Scalar
   sum += values[k] * x[col_index[k]];

   // SIMD concept (not actual code)
   vec_load values[0:7]
   vec_load x[col_index[0:7]]
   vec_mul
   vec_hadd  // horizontal add
   sum += result

This can provide 4–8× speedup on the compute portion, but memory bandwidth
remains the bottleneck.

Multithreading Roofline
~~~~~~~~~~~~~~~~~~~~~~~

For a 4-core CPU with 50 GB/s memory bandwidth:

.. code::

   Single-core throughput: ~12-15 GNNZ/s
   4-core throughput:      ~40-50 GNNZ/s (near roofline)

The parallel efficiency is high because there is no shared mutable state
in the hot loop.

References
----------

* Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
  Chapter 5: Basic Iterative Methods.
* Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An Insightful
  Visual Performance Model for Multi-Core Architectures. *Communications of the ACM*, 52(4), 65–76.
* Im, E., & Yelick, K. (2001). Optimizing Sparse Matrix Vector Multiplication on SMPs.
  *Lecture Notes in Computer Science*, 1175–1187.
* Gahvari, H., et al. (2006). Modeling the Performance of Sparse Matrix Vector
  Multiplication on Multicore Systems. *CLUSTER*, 1–11.
