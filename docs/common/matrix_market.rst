.. Matrix Market Format and Parser
   Scientific documentation with formulas
   =======================================

Matrix Market Format and Parser
===============================

Overview
--------

The **Matrix Market Exchange Format** is a de facto standard for distributing
sparse matrices in scientific computing. It was developed by the National Institute
of Standards and Technology (NIST) and is documented in
Boisvert et al. (1997).

A Matrix Market file (extension ``.mtx``) consists of:

1. A header line identifying the format
2. Optional comment lines
3. Size line: ``m  n  nnz``
4. ``nnz`` data lines, each containing ``row  col  [value]``

Example Matrix Market File
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   %%MatrixMarket matrix coordinate real general
   %
   % This is a 5×5 symmetric matrix with 13 non-zeros
   %
        5     5    13
        1     1  2.0
        1     2  1.0
        2     1  1.0
        2     3  3.0
        3     2  3.0
        ...

Format Specification
--------------------

Header Banner
~~~~~~~~~~~~~

The first line has four space-separated fields:

.. math::

   \text{%%MatrixMarket} \quad \text{object} \quad \text{format} \quad \text{data\_type} \quad [\text{symmetry}]

Field Definitions
~~~~~~~~~~~~~~~~~

**Object** (second field):

.. list-table::
   :header-rows: 1

   * - Token
     - Meaning
   * - ``matrix``
     - Sparse matrix (supported)
   * - ``vector``
     - Dense vector (not supported)
   * - ``stiffness``
     - Element stiffness matrix (not supported)

**Format** (third field):

.. list-table::
   :header-rows: 1

   * - Token
     - Storage Format
     - Our Support
   * - ``coordinate``
     - COO format (row, col, value triples)
     - Supported
   * - ``array``
     - Dense row-major array
     - Not supported in Phase 1

**Data Type** (fourth field):

.. list-table::
   :header-rows: 1

   * - Token
     - Meaning
     - Our Treatment
   * - ``real``
     - IEEE-754 floating point
     - Stored as ``double``
   * - ``integer``
     - Integer values
     - Promoted to ``double``
   * - ``complex``
     - Complex number pair
     - Not supported in Phase 1
   * - ``pattern``
     - Value is implicitly 1.0
     - Treated as value = 1.0

**Symmetry** (optional fifth field):

.. list-table::
   :header-rows: 1

   * - Token
     - Meaning
     - Handling
   * - ``general``
     - All entries stored explicitly
     - Fully handled
   * - ``symmetric``
     - Only upper or lower triangle stored
     - Handled (duplicates to full matrix)
   * - ``skew-symmetric``
     - :math:`A_{ij} = -A_{ji}`
     - Not yet handled
   * - ``hermitian``
     - Complex conjugate symmetry
     - Not yet handled

Index Convention
~~~~~~~~~~~~~~~~

**Important**: Matrix Market uses **1-based indexing** (Fortran convention).
All indices are converted to **0-based** (C convention) upon reading.

.. math::

   \text{Matrix Market: } &1 \leq \text{row} \leq m, \quad 1 \leq \text{col} \leq n \\
   \text{Internal CSR: } &0 \leq i < m, \quad 0 \leq j < n

The conversion is simply:

.. math::

   i_{\text{CSR}} = i_{\text{MM}} - 1, \quad j_{\text{CSR}} = j_{\text{MM}} - 1

Parsing Algorithm
-----------------

The parser operates in two phases:

Phase 1: COO Parse
~~~~~~~~~~~~~~~~~~~

The file is read once, line by line:

1. **Skip comments**: lines starting with ``%``
2. **Read header**: parse the ``%%MatrixMarket`` banner
3. **Read size line**: extract :math:`m, n, \text{nnz}`
4. **Read data lines**: for each non-comment line:

   .. code-block:: text

      row_mm, col_mm, value = parse(line)
      row_csr = row_mm - 1
      col_csr = col_mm - 1
      append to COO_SparseMatrix

For ``pattern`` format, the value is implicitly 1.0:

.. math::

   \text{value} = \begin{cases}
       \text{parsed value} & \text{if format} = \text{real} \\
       \text{parsed value} & \text{if format} = \text{integer} \\
       1.0                 & \text{if format} = \text{pattern}
   \end{cases}

Phase 2: COO to CSR Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After parsing into COO format, the matrix is converted to CSR using the
algorithm described in :ref:`csr-format` (see sparse_matrix.rst).

Symmetry Expansion
~~~~~~~~~~~~~~~~~~

For ``symmetric`` matrices, the stored triangle represents the full matrix.
Each entry :math:`(i, j)` with :math:`i \neq j` implicitly defines
:math:`A_{ji} = A_{ij}`. The parser expands these entries:

.. math::

   \text{nnz}_{\text{full}} = 2 \cdot \text{nnz}_{\text{triangle}} - \text{nnz}_{\text{diagonal}}

Each off-diagonal entry is written twice (once for each position).

Error Handling
-------------

The parser throws ``std::runtime_error`` on:

.. list-table:: Parser Error Conditions
   :header-rows: 1
   :widths: 30, 70

   * - Error Condition
     - Description
   * - Missing banner
     - First line is not ``%%MatrixMarket``
   * - Unsupported format
     - ``array``, ``complex``, ``skew``, ``hermitian``
   * - Malformed size line
     - Non-integer values or wrong column count
   * - Index out of range
     - row or col outside :math:`[1, m]` or :math:`[1, n]`
   * - I/O error
     - File not found, read failure

The error message includes the problematic line number and content for debugging.

API Reference
-------------

**Note**: The C++ API reference below requires Doxygen to generate the XML
intermediate. Currently, these are documented for future integration with
Breathe. Run ``make docs`` after building the project with CMake to generate
the full API documentation.

.. cpp:type:: enum class MatrixFormat

   Storage layout as declared in the banner.

   .. cpp:enumerator:: COO

      Coordinate format — row/col/index triples (what we read).

   .. cpp:enumerator:: CSR

      Array format — dense row-major (not yet supported).

.. cpp:type:: enum class ScalarField

   Data type of each matrix entry.

   .. cpp:enumerator:: REAL

      IEEE-754 double (internal representation).

   .. cpp:enumerator:: INTEGER

      Integer values, promoted to double on read.

   .. cpp:enumerator:: COMPLEX

      Complex number pair (not supported in Phase 1).

   .. cpp:enumerator:: PATTERN

      Value is implicitly 1.0.

.. cpp:type:: enum class SymmetryField

   Symmetry of the matrix.

   .. cpp:enumerator:: GENERAL

      All entries stored explicitly.

   .. cpp:enumerator:: SYMMETRIC

      Only one triangle stored; doubles on expansion.

   .. cpp:enumerator:: SKEW_SYMMETRIC

      :math:`A_{ij} = -A_{ji}` (not yet handled).

   .. cpp:enumerator:: HERMITIAN

      Complex conjugate symmetry (not yet handled).

.. cpp:type:: struct MatrixMarketHeader

   Parsed header — the four tokens of the ``%%MatrixMarket`` banner.

   .. cpp:member:: MatrixFormat format

   .. cpp:member:: ScalarField scalar

   .. cpp:member:: SymmetryField symmetry

.. cpp:function:: SparseMatrix parse_matrix_market(const std::string& filepath)

   Parse a ``.mtx`` file and return a CSR matrix.

   This is the **primary entry point**. Internally it:

   1. Reads into ``COO_SparseMatrix`` via :cpp:func:`parse_matrix_market_coo`
   2. Converts to CSR via :cpp:func:`coo_to_csr`

   :param filepath: Absolute or relative path to the ``.mtx`` file
   :return: ``SparseMatrix`` in CSR format, ready for SpMV
   :throws: ``std::runtime_error`` on malformed input

.. cpp:function:: COO_SparseMatrix parse_matrix_market_coo(const std::string& filepath)

   Parse a ``.mtx`` file and return a COO matrix.

   Use this when you need the raw :math:`(row, col, value)` triples — for
   example to build other formats (ELL, HYB) or to inspect the original ordering.

   :param filepath: Absolute or relative path to the ``.mtx`` file
   :return: ``COO_SparseMatrix``; entries are in file order (not sorted)
   :throws: ``std::runtime_error`` on malformed input

.. cpp:function:: SparseMatrix coo_to_csr(const COO_SparseMatrix& coo)

   Convert a COO matrix to CSR format using counting sort.

   Algorithm complexity: :math:`O(m + \text{nnz})` time, :math:`O(m)` auxiliary space.

   :param coo: Input matrix in COO format; entries need NOT be sorted
   :return: ``SparseMatrix`` in CSR format

References
----------

* Boisvert, R., Pozo, R., Remming, K. & Suzuki, J. (1997).
  *The Matrix Market Exchange Formats*.
  NIST Report NISTIR-6025. https://math.nist.gov/MatrixMarket/
* Matrix Market website: https://math.nist.gov/MatrixMarket/
