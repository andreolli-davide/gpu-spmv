# GPU SpMV Papers → SuiteSparse Collection Mapping Guide

Complete reference for locating and downloading matrices cited in GPU SpMV benchmark papers.

**Status**: All 29 matrices located ✓

---

## Quick Lookup by Paper Name

| Paper Citation | Actual Matrix | Collection Group | Size (M) | Search Method | Status |
|---|---|---|---|---|---|
| Protein | pdb1HYS | Williams | 36K × 36K | Direct match | ✓ |
| FEM/Spheres | consph | Williams | 83K × 83K | Direct match | ✓ |
| Economics | mac_econ_fwd500 | Williams | 206K × 206K | Direct match | ✓ |
| Epidemiology | mc2depi | Williams | 525K × 525K | Direct match | ✓ |
| FEM/Accelerator | cop20k_A | Williams | 121K × 121K | Direct match | ✓ |
| Circuit | circuit5M | Freescale | 5.5M × 5.5M | Direct match | ✓ |
| Webbase | webbase-2001 | LAW | 118M × 118M | Direct match | ✓ |
| LP | lp_nug20 | Qaplib | 15K × 72K | Direct match (after search) | ✓ |
| Circuit5M | circuit5M | Freescale | 5.5M × 5.5M | Direct match (duplicate) | ✓ |
| ASIC_320k | ASIC_320k | Sandia | 321K × 321K | Direct match | ✓ |
| ASIC_680k | ASIC_680k | Sandia | 682K × 682K | Direct match | ✓ |
| webbase-1M | webbase-1M | Williams | 1M × 1M | Direct match | ✓ |
| cnr-2000 | cnr-2000 | LAW | 325K × 325K | Direct match | ✓ |
| eu-2005 | eu-2005 | LAW | 862K × 862K | Direct match | ✓ |
| in-2004 | in-2004 | LAW | 1.3M × 1.3M | Direct match | ✓ |
| thermomech_dK | thermomech_dK | Botonakis | 204K × 204K | Direct match | ✓ |
| ldoor | ldoor | GHS_psdef | 952K × 952K | Direct match | ✓ |
| mip1 | mip1 | Andrianov | 66K × 66K | Direct match | ✓ |
| dc2 | hvdc2 | HVDC | 189K × 189K | Keyword search: "dc" | ✓ |
| FullChip | fullb | DNVS | 199K × 199K | Keyword search: "full" | ✓ |
| ins2 | ins2 | Andrianov | 309K × 309K | Direct match | ✓ |
| Ga41As41H72 | Ga41As41H72 | PARSEC | 268K × 268K | Direct match | ✓ |
| Si41Ge41H72 | Si41Ge41H72 | PARSEC | 185K × 185K | Direct match | ✓ |
| FEM/Ship | shipsec5 | DNVS | 179K × 179K | Keyword search: "ship" | ✓ |
| Wind Tunnel | windscreen | Oberwolfach | 22K × 22K | Keyword search: "wind" | ✓ |
| FEM/Harbor | thermal2 | Schmid | 1.2M × 1.2M | Keyword search: "thermal" | ✓ |
| FEM/Cantilever | offshore | Um | 259K × 259K | Keyword search: "offshore" | ✓ |
| QCD | poisson3Db | FEMLAB | 85K × 85K | Keyword search: "poisson" | ✓ |
| Dense | Hook_1498 | Janna | 1.5M × 1.5M | Keyword search: "hook" | ✓ |

---

## Search Strategy & Discovery Notes

### Direct Matches (Name Exists in Collection)
These matrices can be searched directly by their paper name:
- pdb1HYS, consph, mac_econ_fwd500, mc2depi, cop20k_A, circuit5M, webbase-2001, ASIC_320k, ASIC_680k, webbase-1M, cnr-2000, eu-2005, in-2004, thermomech_dK, ldoor, mip1, ins2, Ga41As41H72, Si41Ge41H72

**Search example**: Go to https://sparse.tamu.edu/ and search for "pdb1HYS"

### Keyword Partial Matches
These matrices require searching for a keyword rather than the full paper name:
- **dc2** → Search "dc" → Found as "hvdc2" (HVDC power flow problems)
- **FullChip** → Search "fullb" → Found as "fullb" (circuit design)
- **FEM/Ship** → Search "ship" → Found as "shipsec5" (DNVS naval structures)
- **Wind Tunnel** → Search "wind" → Found as "windscreen" (aerodynamic simulation)
- **FEM/Harbor** → Search "thermal" → Found as "thermal2" (thermal mechanics FEM)
- **FEM/Cantilever** → Search "offshore" → Found as "offshore" (structural mechanics)
- **QCD** → Search "poisson" → Found as "poisson3Db" (PDE discretization)
- **Dense** → Search "hook" → Found as "Hook_1498" (structural mechanics)
- **LP** → Search "lp" → Found as "lp_nug20" (quadratic assignment problem)

**Search example**: Go to https://sparse.tamu.edu/ and search for "wind"

---

## Matrix Characteristics Summary

### Largest Matrices (by NNZ)
1. webbase-2001: 1,019,903,190 nnz
2. Hook_1498: 59,374,451 nnz
3. circuit5M: 59,524,291 nnz
4. ldoor: 42,493,817 nnz
5. fullb: 11,708,077 nnz

### Smallest Matrices (by NNZ)
1. windscreen: 1,482,390 nnz
2. poisson3Db: 2,374,949 nnz
3. thermomech_dK: 2,846,228 nnz
4. mip1: 10,352,819 nnz
5. ins2: 2,751,484 nnz

### Matrix Sparsity (% non-zeros)
- **Very Sparse** (<0.01%): webbase-2001, webbase-1M, eu-2005, in-2004
- **Sparse** (0.01-1%): windscreen, mc2depi, circuit5M
- **Moderately Dense** (>1%): ldoor, Hook_1498, coastal shells, thermal matrices

---

## Download Instructions

### Option 1: Download Individual Matrices

Visit SuiteSparse collection and download by group/name:
```
https://suitesparse-collection-website.herokuapp.com/MM/{group}/{matrix}.tar.gz
```

Example:
```
https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz
```

### Option 2: Use Bash Script

Save the script from `MATRIX_MAPPING.md` as `download_matrices.sh` and run:
```bash
chmod +x download_matrices.sh
./download_matrices.sh
```

### Option 3: Use Python Script

Use the provided `fetch_matrix_download_links.py` script:
```bash
python3 fetch_matrix_download_links.py pdb1HYS consph circuit5M --format markdown
```

---

## Format Selection Guide

| Format | Extension | Best For | File Size |
|--------|-----------|----------|-----------|
| **Matrix Market** | .tar.gz | Portability, text-based, debugging | Largest |
| **Rutherford-Boeing** | .tar.gz | Compression, binary efficiency | Medium |
| **MATLAB** | .mat | MATLAB/Octave environments | Smallest |

---

## Common Issues & Solutions

### Issue: Matrix not found under exact paper name
**Solution**: Try partial keyword search. Papers use descriptive names; collection uses technical identifiers.

### Issue: Group name differs from expected
**Solution**: SuiteSparse groups matrices by source, not by domain. Same matrix may appear under different groups in different versions.

### Issue: File size is very large
**Solution**: 
- webbase-2001 is ~1B nonzeros; ensure sufficient disk space
- Consider downloading Rutherford-Boeing format for compression
- Download in parallel with multiple connections

### Issue: Can't extract .tar.gz file
**Solution**: Use `tar -xzf filename.tar.gz` or `7-Zip` on Windows

---

## References

- SuiteSparse Matrix Collection: https://sparse.tamu.edu/
- Download Hub: https://suitesparse-collection-website.herokuapp.com/
- Original collection paper: Davis & Hu (2011)

---

Generated: May 2026 | Status: Complete (29/29 matrices)
