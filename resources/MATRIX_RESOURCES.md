# Sparse Matrix Resources for GPU SpMV Research

Complete resource guide for obtaining and working with sparse matrices cited in GPU sparse matrix-vector multiplication research papers.

## Quick Start

### To Get All Matrix Download Links:

1. **Run the automated script**:
   ```bash
   python3 fetch_matrix_links.py --format markdown --output matrix_links.md
   ```

2. **Or download the bash script**:
   ```bash
   python3 fetch_matrix_links.py --format bash --output download_matrices.sh
   chmod +x download_matrices.sh
   ./download_matrices.sh
   ```

3. **Or manually search**: Visit https://sparse.tamu.edu/ and search for any matrix name

---

## Available Resources

### 1. **cited_matrices_in_benchmarks.md**
Comprehensive list of all 29 sparse matrices cited across 7 GPU SpMV research papers.

**Contents**:
- Complete matrix inventory from each paper
- Matrix classification by structure (regular vs. irregular)
- Matrix classification by application domain
- Summary statistics and frequency analysis
- Most commonly cited matrices

**Use this to**: Understand which matrices are used in benchmarks and their characteristics.

### 2. **matrix_download_links.md**
Complete guide to downloading matrices from SuiteSparse Collection.

**Contents**:
- Download URL patterns and examples
- Step-by-step search instructions
- Batch download scripts (Python and Bash)
- Known matrix groups and identifiers
- Important notes on file sizes and formats
- Code examples for loading matrices

**Use this to**: Understand how to get the matrices and their download formats.

### 3. **fetch_matrix_links.py**
Automated Python tool for generating download links and batch scripts.

**Features**:
- Generates download links in multiple formats
- Creates bash scripts for batch downloading
- Creates Python scripts for programmatic downloading
- Outputs in CSV, Markdown, or shell script formats

**Usage**:
```bash
# Generate markdown table
python3 fetch_matrix_links.py --format markdown --output links.md

# Generate bash download script
python3 fetch_matrix_links.py --format bash --output download.sh

# Generate Python download script
python3 fetch_matrix_links.py --format python --output download_matrices.py

# Generate CSV for spreadsheet
python3 fetch_matrix_links.py --format csv --output matrices.csv
```

---

## Matrix Categories

### 29 Total Cited Matrices

| Category | Count | Matrices |
|----------|-------|----------|
| FEM/Structural | 6 | FEM/Spheres, FEM/Cantilever, FEM/Harbor, FEM/Ship, FEM/Accelerator, thermomech_dK, ldoor |
| Web/Networks | 5 | Webbase, webbase-1M, cnr-2000, eu-2005, in-2004 |
| Circuit/ASIC | 6 | Circuit, Circuit5M, ASIC_320k, ASIC_680k, FullChip, dc2 |
| Scientific | 3 | QCD, Protein, Economics, Epidemiology |
| Specialized | 7 | LP, mip1, ins2, Ga41As41H72, Si41Ge41H72, Wind Tunnel, Dense |

---

## Collection Information

### SuiteSparse Matrix Collection (Formerly University of Florida)

**Website**: https://sparse.tamu.edu/

**Key Information**:
- 2,800+ matrices
- Multiple download formats available
- Free public access
- Matrices organized by group/collection
- Detailed metadata for each matrix

**Maintained by**: Tim Davis (Texas A&M University)

**Available Formats**:
1. **MATLAB (.mat)** - Proprietary, direct MATLAB/Octave compatibility
2. **Rutherford-Boeing (.tar.gz)** - Text format with binary headers
3. **Matrix Market (.tar.gz)** - Standard ASCII portable format

---

## Workflow Examples

### Example 1: Download Single Matrix

```bash
# Visit website and find "webbase-1M" in "Matrices" group
# Then download via:
wget https://suitesparse-collection-website.herokuapp.com/MM/Matrices/webbase-1M.tar.gz
tar -xzf webbase-1M.tar.gz

# Load in Python
from scipy.io import mmread
A = mmread('webbase-1M/webbase-1M.mtx')
```

### Example 2: Batch Download All Matrices

```bash
# Generate download script
python3 fetch_matrix_links.py --format bash --output batch_download.sh
chmod +x batch_download.sh

# Run to download all ~29 matrices
./batch_download.sh

# Or use Python
python3 fetch_matrix_links.py --format python --output bulk_download.py
python3 bulk_download.py
```

### Example 3: Generate For Your CI/CD Pipeline

```bash
# Generate links in CSV for your data pipeline
python3 fetch_matrix_links.py --format csv --output matrices.csv

# Generate markdown for documentation
python3 fetch_matrix_links.py --format markdown --output docs/matrices.md
```

---

## Important Notes

1. **Finding Unknown Matrices**:
   - Some matrices have descriptive names in papers (e.g., "Wind Tunnel")
   - Search these manually on https://sparse.tamu.edu/
   - Once found, note the GROUP identifier
   - Use URL pattern: `https://suitesparse-collection-website.herokuapp.com/FORMAT/GROUP/NAME.EXTENSION`

2. **File Sizes** (typical Matrix Market format):
   - Small matrices: 1-10 MB
   - Medium matrices: 10-100 MB
   - Large matrices: 100MB-1GB+
   - Consider disk space before bulk downloading

3. **Matrix Formats**:
   - **Matrix Market**: Most portable, ASCII format, standard in research
   - **Rutherford-Boeing**: Text-based, includes metadata headers
   - **MATLAB**: Largest files, proprietary, easiest if you use MATLAB/Octave

4. **Loading Matrices**:
   ```python
   # Python (SciPy)
   from scipy.io import mmread
   A = mmread('matrix_name.mtx')
   
   # MATLAB/Octave
   A = mmread('matrix_name.mtx');
   
   # C++ (example using Eigen)
   // Manual parsing or use mmread library
   ```

---

## Summary of Documents

### Purpose Mapping

| Need | Use This Document |
|------|-------------------|
| Understand which matrices are cited | `cited_matrices_in_benchmarks.md` |
| Download matrices | `matrix_download_links.md` |
| Generate download links automatically | `fetch_matrix_links.py` |
| Learn about matrix properties | `cited_matrices_in_benchmarks.md` |
| Get batch download script | Run `fetch_matrix_links.py --format bash` |
| Integrate into pipeline | Run `fetch_matrix_links.py --format python` |

---

## Quick Reference URLs

- **SuiteSparse Collection**: https://sparse.tamu.edu/
- **GitHub Repository**: https://github.com/ScottKolo/suitesparse-matrix-collection-website
- **Maintainer**: https://people.engr.tamu.edu/davis/

---

## Troubleshooting

### Issue: Matrix not found on SuiteSparse
**Solution**: Matrix name might be different in collection
- Try partial name search on https://sparse.tamu.edu/
- Check `cited_matrices_in_benchmarks.md` for alternate names
- May be in a different group than expected

### Issue: Download link broken
**Solution**: Website may have changed
- Verify GROUP name on SuiteSparse first
- Double-check URL format
- Try different format (MATLAB vs Matrix Market)

### Issue: Can't extract tar.gz
**Solution**: May not be tar.gz format
```bash
# Check file type
file matrix_name.tar.gz

# Try different extraction
gunzip matrix_name.tar.gz
tar -xf matrix_name.tar
```

### Issue: Matrix file format not recognized
**Solution**: Check file format
```bash
# View first few lines
head -20 matrix_name.mtx

# Try different loader
# Python: scipy, numpy-sparse, petsc4py
# C++: Eigen, Boost, custom parser
```

---

## Contributing

If you identify additional matrices or find download links that don't work:

1. Update the matrix group information in `fetch_matrix_links.py`
2. Test the download link
3. Update the documentation files
4. Consider submitting findings back to research community

---

## References

### Primary Papers
- Bell & Garland (2009) - "Efficient Sparse-Matrix-Vector Multiplication on GPUs"
- Greathouse & Daga (2014) - "Efficient Sparse-Matrix-Vector Multiplication on GPUs Using the CSR Storage Format"
- Liu & Vinter (2015) - "CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication"
- Merrill & Garland (2016) - "Merge-Based Parallel Sparse Matrix-Vector Multiplication"
- Chu et al. (2023) - "Efficient Algorithm Design for Sparse Matrix-Vector Multiplication on GPUs"
- Niu et al. (2021) - "TileSpMV: A Tiled Algorithm for Sparse Matrix-Vector Multiplication on GPUs"
- Gao et al. (2024) - "A Systematic Literature Survey of Sparse Matrix-Vector Multiplication"

### Data Sources
- SuiteSparse Matrix Collection: https://sparse.tamu.edu/
- Matrix Market: https://math.nist.gov/MatrixMarket/

---

## License & Attribution

These resources compile information from:
- Published research papers (open access)
- SuiteSparse Matrix Collection (public domain)
- Your GPU SpMV research project

Please cite the original papers when using these matrices in your research.
