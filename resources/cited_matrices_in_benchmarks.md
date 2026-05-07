# Sparse Matrices Cited in GPU SpMV Papers - Comprehensive List

## Overview
This document compiles all sparse matrices cited in benchmarks across the papers in the `/papers` folder. The matrices are primarily sourced from the **University of Florida Sparse Matrix Collection** (also known as SuiteSparse Matrix Collection).

---

## Papers and Their Benchmark Matrices

### 1. Bell et al. (2009) - "Efficient Sparse-Matrix-Vector Multiplication on GPUs" 
**Source**: `03_bell_2009_spmv_gpu.pdf`  
**Benchmark Suite**: Table 3 - Unstructured matrices for performance testing

**Matrices (14 total)**:
- Dense (2K × 2K, 4M nonzeros)
- Protein (36K × 36K, 4.3M nonzeros)
- FEM/Spheres (83K × 83K, 6M nonzeros)
- FEM/Cantilever (62K × 62K, 4M nonzeros)
- Wind Tunnel (218K × 218K, 11.6M nonzeros)
- FEM/Harbor (47K × 47K, 2.4M nonzeros)
- QCD (49K × 49K, 1.9M nonzeros)
- FEM/Ship (141K × 141K, 7.8M nonzeros)
- Economics (207K × 207K, 1.3M nonzeros)
- Epidemiology (526K × 526K, 2.1M nonzeros)
- FEM/Accelerator (121K × 121K, 2.6M nonzeros)
- Circuit (171K × 171K, 958K nonzeros)
- Webbase (1M × 1M, 3.1M nonzeros)
- LP (4K × 1.097M, 11.3M nonzeros)

---

### 2. Greathouse & Daga (2014) - "Efficient Sparse-Matrix-Vector Multiplication on GPUs Using the CSR Storage Format"
**Source**: `05_greathouse_2014_csr_adaptive.pdf`  
**Benchmark Suite**: Table I - Overview of sparse matrices used for evaluation

**Matrices (16 total)**:
**Regular/Structured Matrices**:
- Dense2 (2K × 2K, 4M nonzeros)
- Protein (36K × 36K, 4.3M nonzeros)
- FEM/Spheres (83K × 83K, 6M nonzeros)
- FEM/Cantilever (62K × 62K, 4M nonzeros)
- Wind Tunnel (218K × 218K, 11.6M nonzeros)
- FEM/Harbor (47K × 47K, 2.4M nonzeros)
- QCD (49K × 49K, 1.9M nonzeros)
- FEM/Ship (141K × 141K, 7.8M nonzeros)

**Irregular/Unstructured Matrices**:
- Economics (207K × 207K, 1.3M nonzeros)
- Epidemiology (526K × 526K, 2.1M nonzeros)
- FEM/Accelerator (121K × 121K, 2.6M nonzeros)
- Circuit (171K × 171K, 958K nonzeros)
- Webbase (1M × 1M, 3.1M nonzeros)
- LP (4K × 1.097M, 11.3M nonzeros)
- circuit5M (5.558M × 5.558M, 59.5M nonzeros)
- eu-2005 (863K × 863K, 19.2M nonzeros)

---

### 3. Liu & Vinter (2015) - "CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication"
**Source**: `06_liu_2015_csr5.pdf`  
**Benchmark Suite**: Table 2 - 24 matrices (14 regular, 10 irregular)

**Regular Matrices (r1-r14)**:
- Dense (2K × 2K)
- Protein (36K × 36K)
- FEM/Spheres (83K × 83K)
- FEM/Cantilever (62K × 62K)
- Wind Tunnel (218K × 218K)
- QCD (49K × 49K)
- Epidemiology (526K × 526K)
- FEM/Harbor (47K × 47K)
- FEM/Ship (141K × 141K)
- Economics (207K × 207K)
- FEM/Accelerator (121K × 121K)
- Circuit (171K × 171K)
- Ga41As41H72
- Si41Ge41H72

**Irregular Matrices (i1-i10)**:
- Webbase
- LP
- Circuit5M (circuit simulation, ~59.5M nonzeros)
- eu-2005 (European web crawl)
- in-2004 (Indochina web)
- mip1
- ASIC_680k (or ASIC 680k - circuit design)
- dc2 (representative irregular matrix with very long rows, up to 114K nonzeros per row)
- FullChip
- ins2

---

### 4. Merrill & Garland (2016) - "Merge-Based Parallel Sparse Matrix-Vector Multiplication"
**Source**: `07_merrill_2016_merge_based.pdf`  
**Benchmark Suite**: University of Florida Sparse Matrix Collection

**Named Matrices**:
- thermomech_dK (temperature deformation)
- cnr-2000 (web connectivity)
- ASIC_320k (circuit simulation)

**Referenced Dataset**: Mentions webbase-2001 (12% empty rows)

---

### 5. Niu et al. (2021) - "TileSpMV: A Tiled Algorithm for Sparse Matrix-Vector Multiplication on GPUs"
**Source**: `09_niu_2021_tiled_spmv.pdf`  
**Benchmark Suite**: Table II - 16 representative matrices

**Named Matrices**:
- webbase-1M
- ldoor

**Note**: Paper mentions 16 representative matrices but only 2 are explicitly named in extracted text.

---

### 6. Chu et al. (2023) - "Efficient Algorithm Design for Sparse Matrix-Vector Multiplication on GPUs"
**Source**: `02_chu_2023_efficient_algorithm_design.pdf`  
**Benchmark Suite**: SuiteSparse Matrix Collection (University of Florida Sparse Matrix Collection)

**Coverage**: 91% of the entire SuiteSparse Matrix Collection

---

### 7. Gao et al. (2024) - "A Systematic Literature Survey of Sparse Matrix-Vector Multiplication"
**Source**: `01_gao_2024_systematic_survey.pdf`  
**Benchmark Suite**: SuiteSparse Matrix Collection

**Coverage**: More than 2,800 sparse matrices from the collection

**Libraries Tested**:
- cuSPARSE
- CUSP
- MAGMA
- Ginkgo

---

## Summary Statistics

| Paper | Year | Number of Named Matrices | Collection Used |
|-------|------|--------------------------|-----------------|
| Bell et al. | 2009 | 14 | Matrix Market/Custom |
| Greathouse & Daga | 2014 | 16 | Custom (8 from Bell, 8 additional) |
| Liu & Vinter | 2015 | 24 (14 regular, 10 irregular) | University of Florida |
| Merrill & Garland | 2016 | 3 explicitly named | University of Florida |
| Niu et al. | 2021 | 2 explicitly named | SuiteSparse Collection |
| Chu et al. | 2023 | ~2,360+ matrices (91% of collection) | SuiteSparse Collection |
| Gao et al. | 2024 | ~2,800+ matrices | SuiteSparse Collection |

---

## Most Frequently Cited Matrices

The following matrices appear in multiple papers:

**Very Common (≥4 papers)**:
- Dense
- Protein
- FEM/Spheres
- FEM/Cantilever
- Wind Tunnel
- QCD
- FEM/Harbor
- FEM/Ship
- Economics
- Epidemiology
- FEM/Accelerator
- Circuit
- Webbase
- LP

**Moderately Common (2-3 papers)**:
- circuit5M
- eu-2005
- ASIC_320k / ASIC_680k
- thermomech_dK
- ldoor
- cnr-2000

---

## Matrix Classification

### By Structure
**Regular/Structured Matrices** (predictable, dense rows):
- Dense, Protein, FEM matrices (Spheres, Cantilever, Harbor, Ship, Accelerator), Wind Tunnel, QCD, Epidemiology, Economics

**Irregular/Unstructured Matrices** (variable row lengths):
- Circuit, Webbase, LP, Circuit5M, eu-2005, in-2004, mip1, ASIC variants, dc2, FullChip, ins2

### By Application Domain
- **Finite Element Methods (FEM)**: FEM/Spheres, FEM/Cantilever, FEM/Harbor, FEM/Ship, FEM/Accelerator
- **Chemistry/Physics**: QCD, Protein
- **Web Graphs/Networks**: Webbase, cnr-2000, eu-2005, in-2004
- **Circuit Simulation**: Circuit, Circuit5M, ASIC variants
- **Mathematical**: LP (linear programming), Economics, Epidemiology, Dense

---

## Data Source Information

### University of Florida Sparse Matrix Collection
- **Also known as**: SuiteSparse Matrix Collection
- **Website**: Reference [205] in Gao et al. (2024)
- **Total Matrices**: 2,800+ as of 2024
- **Format**: MatrixMarket format
- **Common use**: Standard benchmark suite for evaluating sparse matrix algorithms

### Matrix Market Format
- Standard text-based format for sparse matrices
- Contains: dimensions, number of nonzeros, and coordinate representation
- Most named matrices from pre-2020 papers sourced from this collection

---

## Notes

1. **Naming conventions**: Most matrices are from the University of Florida collection and follow naming patterns like `name_XXXK` (where K denotes thousands of rows/columns) or descriptive names like `FEM/Spheres`.

2. **Matrix reuse**: The core set of ~14 matrices (Dense, Protein, FEM variants, Wind Tunnel, QCD, etc.) from Bell (2009) became the standard benchmark set and appears in subsequent papers.

3. **Scale evolution**: 
   - 2009: 14 test matrices
   - 2014: 16 test matrices
   - 2015: 24 matrices (formalized regular/irregular split)
   - 2020s: Entire collection (2,000+ matrices) for comprehensive evaluation

4. **Sparsity patterns**: Papers specifically mention extreme cases like:
   - `dc2`: Single row with 114K nonzeros (15% of total matrix nonzeros) in a 117K-row matrix
   - `LP`: 2,632.9 nonzeros per row on average (very dense)
   - `Webbase`: 3.1 nonzeros per row (very sparse)

---

## References

All matrices are publicly available from:
- **University of Florida Sparse Matrix Collection**: https://sparse.tamu.edu/ (formerly at Florida)
- **SuiteSparse Matrix Collection**: Maintained by Tim Davis
