# SpMV Deliverable 1 - Selected Dataset (10 Matrices)

## Selection Strategy

The 10 matrices are selected to **stress different access patterns and memory behaviors**:

- **Group 1 (Structured, regular)**: Favor simple CSR kernels with predictable row patterns
- **Group 2 (Sparse, irregular)**: Expose load imbalance and irregular access
- **Group 3 (Graphs)**: Test memory coalescing on unstructured patterns
- **Group 4 (Large-scale)**: Validate scalability and memory bandwidth

---

## Selected Matrices

| # | Name | Rows | Nonzeros | nnz/row | Max/Min | Application | Category | URL |
|---|------|------|----------|---------|---------|-------------|----------|-----|
| 1 | **pdb1HYS** | 36,417 | 4,344,765 | 119 | Regular | Protein Structure | Structured | https://sparse.tamu.edu/Williams/pdb1HYS |
| 2 | **consph** | 83,334 | 6,010,480 | 72 | Regular | FEM / Spheres | Structured | https://sparse.tamu.edu/Williams/consph |
| 3 | **cop20k_A** | 121,192 | 2,624,331 | 22 | Moderate | FEM / Accelerator | Structured | https://sparse.tamu.edu/Williams/cop20k_A |
| 4 | **mac_econ_fwd500** | 206,500 | 1,273,389 | 6 | Highly Variable | Economics | Sparse Irregular | https://sparse.tamu.edu/Williams/mac_econ_fwd500 |
| 5 | **mc2depi** | 525,825 | 2,100,225 | 4 | Highly Variable | Epidemiology | Sparse Irregular | https://sparse.tamu.edu/Williams/mc2depi |
| 6 | **webbase-1M** | 1,000,005 | 3,105,536 | 3 | Extreme | Web Graph | Sparse Irregular | https://sparse.tamu.edu/Williams/webbase-1M |
| 7 | **cnr-2000** | 325,557 | 3,216,152 | 10 | Variable | Directed Graph | Graph | https://sparse.tamu.edu/LAW/cnr-2000 |
| 8 | **eu-2005** | 862,664 | 19,235,140 | 22 | Variable | Web Domain (.eu) | Graph Dense | https://sparse.tamu.edu/LAW/eu-2005 |
| 9 | **thermal2** | 1,228,045 | 8,580,313 | 7 | Low | Thermal Problem | Large Sparse | https://sparse.tamu.edu/Schmid/thermal2 |
| 10 | **Hook_1498** | 1,498,023 | 59,374,451 | 40 | Low | Structural (Hook) | Large Dense | https://sparse.tamu.edu/Janna/Hook_1498 |

---

## Rationale by Group

### **Group 1: Structured, Regular Row Patterns (1–3)**

**Matrices**: pdb1HYS, consph, cop20k_A

These matrices come from **FEM and physical simulations** with predictable row structures. They represent the "easy case" where:
- Row lengths are **regular and uniform** → simple CSR parallelization should work well
- Memory access is **predictable** → good cache utilization expected
- **Expected behavior**: Basic thread-per-row or block-per-row kernels should achieve high efficiency

**Size progression**: 36K → 83K → 121K rows (small to medium)

---

### **Group 2: Sparse, Highly Irregular (4–6)**

**Matrices**: mac_econ_fwd500, mc2depi, webbase-1M

These matrices expose **load imbalance and irregular row-length variation**:
- **Very low nnz/row** (3–6) → global memory is underutilized
- **High max/min ratio** → some threads do much more work than others → **bad load balance**
- **Unstructured access patterns** → poor memory coalescing

**Expected behavior**:
- Simple CSR parallelization will suffer from **warp divergence** and **idle threads**
- Requires **advanced techniques** (e.g., merge-based partitioning, CSR5) to redistribute work
- This group justifies format exploration and optimization

**Size progression**: 200K → 525K → 1M rows

---

### **Group 3: Directed Graphs (7–8)**

**Matrices**: cnr-2000, eu-2005

Graph sparsity patterns differ from FEM:
- **Column access** is unpredictable (not sequential)
- **Row structure** varies but is graph-natural (not FEM-like)
- cnr-2000 (small, sparse) vs eu-2005 (large, denser) show **scaling under different densities**

**Expected behavior**: Memory coalescing degradation; techniques like **ELL or HYB formats** may help

---

### **Group 4: Large-Scale Validation (9–10)**

**Matrices**: thermal2, Hook_1498

Final validation on **large, realistic problems** at scale:
- **thermal2**: Large but sparse (7 nnz/row, 1.2M rows) → memory-bound regime
- **Hook_1498**: Large and dense (40 nnz/row, 1.5M rows) → compute-bound regime

This pair stresses scalability and reveals whether optimizations hold at realistic problem sizes.

---

## Why This Selection?

1. **Diversity in size**: 36K → 1.49M rows (40× range)
2. **Diversity in sparsity**: 3–119 nnz/row (40× range)
3. **Diverse row regularity**: Regular (pdb, consph) → highly variable (graphs, economics)
4. **Different application domains**: Physics/FEM, economics, epidemiology, web graphs
5. **Stresses different bottlenecks**:
   - Groups 1–2: CSR parallelization efficiency
   - Group 3: Memory coalescing
   - Group 4: Scalability

---

## Implementation Notes

- All matrices are **square** and suitable for SpMV
- All are available from **SuiteSparse Collection** with single `.mtx` file download
- **Symmetric matrices** (e.g., pdb1HYS, consph): store full structure; only upper or lower triangle need be used
- **Asymmetric matrices** (e.g., graphs): use as-is
- Download all matrices locally for reproducible benchmarks

---

## Dataset Table (For Report)

| Matrix | Rows | nnz | nnz/row | Symmetry | Type | Domain | Rationale |
|--------|------|-----|---------|----------|------|--------|-----------|
| pdb1HYS | 36.4K | 4.3M | 119 | Symmetric | Real | Protein | Structured, dense—baseline for simple CSR |
| consph | 83.3K | 6.0M | 72 | Symmetric | Real | FEM Sphere | Regular FEM structure |
| cop20k_A | 121.2K | 2.6M | 22 | Symmetric | Real | FEM Accelerator | Structured, medium density |
| mac_econ_fwd500 | 206.5K | 1.3M | 6 | Asymmetric | Real | Economics | Very sparse, highly irregular |
| mc2depi | 525.8K | 2.1M | 4 | Asymmetric | Integer | Epidemiology | Sparse, medium-large scale |
| webbase-1M | 1.0M | 3.1M | 3 | Asymmetric | Binary | Web Graph | Extremely sparse, large scale |
| cnr-2000 | 325.6K | 3.2M | 10 | Asymmetric | Binary | Directed Graph | Graph-structured, uneven |
| eu-2005 | 862.7K | 19.2M | 22 | Asymmetric | Binary | Web Domain | Large graph, higher density |
| thermal2 | 1.23M | 8.6M | 7 | Symmetric | Real | Thermal | Large, sparse, symmetric |
| Hook_1498 | 1.50M | 59.4M | 40 | Symmetric | Real | Structural | Very large, dense—scalability test |

---

## Download Instructions

All matrices are available from **SuiteSparse Collection** (https://sparse.tamu.edu/):

```bash
# Example: Download pdb1HYS
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz
tar -xzf pdb1HYS.tar.gz
# Extract .mtx file
```

Or use Python + scipy to download:

```python
import scipy.io as sio
import urllib.request

matrices = [
    ('Williams', 'pdb1HYS'),
    ('Williams', 'consph'),
    # ... etc
]

for group, name in matrices:
    url = f"https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"
    urllib.request.urlretrieve(url, f"{name}.tar.gz")
    # Extract and process
```
