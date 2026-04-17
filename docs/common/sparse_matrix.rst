.. Sparse Matrix Data Structures
   Scientific documentation with formulas
   =======================================

Sparse Matrix Data Structures
=============================

Introduction
------------

A sparse matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix in which
the majority of elements are zero.  The **sparsity** is defined as:

.. math::

   s = 1 - \frac{\text{nnz}}{m \times n}

where :math:`\text{nnz}` denotes the number of non-zero entries.  For matrices arising
in scientific computing applications (finite element methods, power grids, graph
analytics), sparsity values of :math:`s > 0.99` are typical, making dense storage
prohibitively expensive.

Memory Comparison
~~~~~~~~~~~~~~~~~

For a dense matrix:

.. math::

   \text{Mem}_{\text{dense}} = m \times n \times sizeof(\text{double})
                              = 8 \cdot m \cdot n \ \text{bytes}

For a CSR matrix (see below):

.. math::

   \text{Mem}_{\text{CSR}} = \underbrace{\text{nnz} \cdot 8}_{\text{values}}
                           + \underbrace{\text{nnz} \cdot 8}_{\text{col\_index}}
                           + \underbrace{(m+1) \cdot 8}_{\text{row\_ptr}}
                           = 8 \cdot (2 \cdot \text{nnz} + m + 1) \ \text{bytes}

The compression ratio is:

.. math::

   \rho = \frac{\text{Mem}_{\text{dense}}}{\text{Mem}_{\text{CSR}}}
        = \frac{m \cdot n}{2 \cdot \text{nnz} + m + 1}

For a typical sparse matrix from the SuiteSparse collection with
:math:`m = n = 10^6` and :math:`\text{nnz} = 10^7`:

.. math::

   \rho = \frac{10^{12}}{2 \cdot 10^7 + 10^6 + 1} \approx 47.5

meaning dense storage would require approximately **47× more memory**.

.. _csr-format:

CSR (Compressed Sparse Row) Format
----------------------------------

The **Compressed Sparse Row (CSR)** format, also known as the *Harwell-Boeing*
format, stores a sparse matrix using three arrays:

1. **values** :math:`\in \mathbb{R}^{\text{nnz}}` — the non-zero values in row-major order
2. **col\_index** :math:`\in \{0, 1, \ldots, n-1\}^{\text{nnz}}` — column indices
3. **row\_ptr** :math:`\in \{0, 1, \ldots, \text{nnz}\}^{m+1}` — row start offsets

The row :math:`i` occupies the index range :math:`[\text{row\_ptr}[i], \text{row\_ptr}[i+1])`.
This enables :math:`O(1)` access to any row.

CSR Array Layout
~~~~~~~~~~~~~~~~

For a matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` with :math:`\text{nnz}`
non-zeros:

.. math::

   \mathbf{A} = \begin{pmatrix}
       a_{00} & a_{01} &        & a_{0,n-1} \\
              & a_{11} &        &           \\
       a_{20} &        & \ddots &           \\
              &        &        & a_{m-1,n-1}
   \end{pmatrix}

The three arrays are:

.. math::

   \text{values}   &= [a_{00}, a_{01}, a_{0,n-1}, a_{11}, a_{20}, \ldots, a_{m-1,n-1}] \\
   \text{col\_index} &= [0, 1, n-1, 1, 0, \ldots, n-1] \\
   \text{row\_ptr}   &= [0,\ \underbrace{\text{nnz}_0}_{\text{row 0}},\ 
                           \text{nnz}_0 + \text{nnz}_1,\ \ldots,\ 
                           \sum_{i=0}^{m-1} \text{nnz}_i = \text{nnz}]

where :math:`\text{nnz}_i` is the number of non-zeros in row :math:`i`.

Example: 3×4 Matrix
~~~~~~~~~~~~~~~~~~~~

Consider:

.. math::

   \mathbf{A} = \begin{pmatrix}
       a & b & 0 & 0 \\
       0 & c & 0 & 0 \\
       d & 0 & e & 0
   \end{pmatrix}, \quad \text{nnz} = 5

CSR representation:

.. code-block:: text

   values    = [a, b, c, d, e]
   col_index = [0, 3, 1, 0, 2]  (0-based columns)
   row_ptr   = [0, 2, 3, 5]     (row 0 has indices [0,2), row 1 → [2,3), row 2 → [3,5))

Formal Invariants
~~~~~~~~~~~~~~~~~

The CSR format maintains the following invariants:

.. math::

   &\text{values.size()} = \text{col\_index.size()} = \text{nnz} \\
   &\text{row\_ptr.size()} = m + 1 \\
   &\text{row\_ptr}[0] = 0 \\
   &\text{row\_ptr}[m] = \text{nnz} \\
   &\text{row\_ptr}[i] \leq \text{row\_ptr}[i+1] \quad \forall i \in [0, m) \\
   &0 \leq \text{col\_index}[j] < n \quad \forall j \in [0, \text{nnz})

.. _coo-format:

COO (Coordinate) Format
-----------------------

The **Coordinate (COO)** format stores each non-zero as an explicit
:math:`(row, col, value)` triple:

1. **values** :math:`\in \mathbb{R}^{\text{nnz}}` — the non-zero values
2. **row** :math:`\in \{0, \ldots, m-1\}^{\text{nnz}}` — row indices
3. **col** :math:`\in \{0, \ldots, n-1\}^{\text{nnz}}` — column indices

All three arrays are **parallel** — entry at index :math:`k` represents
:math:`(\text{row}[k], \text{col}[k], \text{values}[k])`.

COO is the natural output of the Matrix Market parser and is easier to construct
incrementally, but does not support efficient row access (:math:`O(\text{nnz})`
scan required).

Formal Invariants
~~~~~~~~~~~~~~~~~

.. math::

   &\text{values.size()} = \text{row.size()} = \text{col.size()} = \text{nnz} \\
   &0 \leq \text{row}[k] < m \quad \forall k \in [0, \text{nnz}) \\
   &0 \leq \text{col}[k] < n \quad \forall k \in [0, \text{nnz})

COO does **not** require sorted entries.

COO to CSR Conversion
---------------------

The conversion from COO to CSR uses a **counting sort** algorithm with
:math:`O(\text{nnz})` time complexity and :math:`O(m)` auxiliary space.

Algorithm Steps
~~~~~~~~~~~~~~~

**Step 1: Count non-zeros per row**

.. math::

   \text{row\_count}[i] = \sum_{k=0}^{\text{nnz}-1} \mathbb{1}_{\{\text{row}[k] = i\}}

**Step 2: Prefix sum to compute row_ptr**

.. math::

   \text{row\_ptr}[0] &= 0 \\
   \text{row\_ptr}[i+1] &= \text{row\_ptr}[i] + \text{row\_count}[i] \quad i = 0, \ldots, m-1

Equivalently:

.. math::

   \text{row\_ptr}[i] = \sum_{j=0}^{i-1} \text{nnz}_j = \text{nnz in rows } [0, i)

**Step 3: Write entries using write counters**

.. code-block:: text

   write_offset[i] = row_ptr[i]  (copy)
   for k = 0 .. nnz-1:
       i = row[k]
       j = col[k]
       v = values[k]
       values_csr[write_offset[i]] = v
       col_index[write_offset[i]] = j
       write_offset[i]++

The write counters ensure each row's entries are contiguous in the output arrays.

Complexity Analysis
~~~~~~~~~~~~~~~~~~~

+----------+-------------------+------------------+
| Phase    | Time              | Extra Space      |
+==========+===================+==================+
| Count    | :math:`O(nnz)`    | :math:`O(m)`     |
+----------+-------------------+------------------+
| Prefix   | :math:`O(m)`      | —                |
+----------+-------------------+------------------+
| Write    | :math:`O(nnz)`    | :math:`O(m)`     |
+----------+-------------------+------------------+
| **Total**| :math:`O(nnz + m)`| :math:`O(m)`     |
+----------+-------------------+------------------+

The algorithm is cache-friendly because:
- Counting scans COO arrays once with sequential access
- Prefix sum is cache-backed for typical :math:`m \ll \text{nnz}`
- Write phase has strided (but predictable) access patterns

.. _dense-vector:

Dense Vector
------------

A **DenseVector** :math:`\mathbf{x} \in \mathbb{R}^n` stores all :math:`n`
elements explicitly:

.. math::

   \mathbf{x} = \begin{pmatrix} x_0 \\ x_1 \\ \vdots \\ x_{n-1} \end{pmatrix},
   \quad \text{data} = [x_0, x_1, \ldots, x_{n-1}]

Memory footprint:

.. math::

   \text{Mem}_{\text{vector}} = n \cdot sizeof(\text{double}) + sizeof(\text{int64_t})
                               = 8n + 8 \ \text{bytes}

API Reference
-------------

.. cpp:type:: struct SparseMatrix

   CSR-format sparse matrix.

   .. cpp:member:: int64_t rows

      Number of matrix rows (:math:`m`).

   .. cpp:member:: int64_t cols

      Number of matrix columns (:math:`n`).

   .. cpp:member:: int64_t nnz

      Number of non-zero entries.

   .. cpp:member:: std::vector<double> values

      Non-zero values, row-major order, size :math:`\text{nnz}`.

   .. cpp:member:: std::vector<int64_t> col_index

      Column index for each value, size :math:`\text{nnz}`, 0-based.

   .. cpp:member:: std::vector<int64_t> row_ptr

      Row start offsets, size :math:`m + 1`.

   .. cpp:func:: SparseMatrix(int64_t rows, int64_t cols, int64_t nnz)

      Constructs an empty matrix with given dimensions. Storage arrays are
      allocated but not filled; call :cpp:func:`allocate()` afterward.

   .. cpp:func:: void allocate()

      Resizes all three storage arrays to their required sizes.

   .. cpp:func:: int64_t memory_bytes() const

      Returns total memory footprint on host in bytes:

      .. math::

         \text{bytes} = 8 \cdot \text{nnz} + 8 \cdot \text{nnz} + 8 \cdot (m+1) + 24

.. cpp:type:: struct COO_SparseMatrix

   COO-format sparse matrix (intermediate representation).

   .. cpp:member:: int64_t rows, cols, nnz

      Matrix dimensions and non-zero count.

   .. cpp:member:: std::vector<double> values

   .. cpp:member:: std::vector<int64_t> row

   .. cpp:member:: std::vector<int64_t> col

      Parallel arrays of length :math:`\text{nnz}`.

   .. cpp:func:: COO_SparseMatrix(int64_t rows, int64_t cols, int64_t nnz)

   .. cpp:func:: void allocate()

   .. cpp:func:: int64_t memory_bytes() const

.. cpp:type:: struct DenseVector

   Dense vector for SpMV input/output.

   .. cpp:member:: int64_t size

      Number of elements (:math:`n`).

   .. cpp:member:: std::vector<double> data

      Element storage, size :math:`n`.

   .. cpp:func:: DenseVector(int64_t size)

      Constructs and allocates a vector of the given size, filled with zeros.

   .. cpp:func:: void resize(int64_t size)

      Resizes the vector, preserving existing data where possible.

   .. cpp:func:: int64_t memory_bytes() const

      .. math::

         \text{bytes} = 8 \cdot n + 8

References
----------

* Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
* Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
* Boisvert, R., Pozo, R., Remming, K. & Suzuki, J. (1997). The Matrix Market
  Exchange Formats. NIST Report NISTIR-6025.
