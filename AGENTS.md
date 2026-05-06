# Agent Instructions for GPU-SPMV-2

## Remote HPC Access

All compilation and testing happens on the HPC cluster **baldo.disi.unitn.it**.

```
Server: davide.andreolli@baldo.disi.unitn.it
```

No credentials required — uses SSH keys if configured, otherwise interactive auth.

## Remote Execution Pattern

**For compilation, testing, and any code execution, you MUST:**

1. SSH to the remote server
2. Set up environment: `source /etc/profile.d/modules.sh`
3. Load CUDA: `module load CUDA/12.5.0` (latest available)
4. Run commands via `srun` (interactive) or `sbatch` (batch jobs)

## Building with CMake

Always use CMake for compilation:

```bash
# Clean build
rm -rf build && mkdir build && cd build

# Configure (enables CUDA language)
cmake ..

# Build
cmake --build .
```

The binary will be in `build/src/<target>`.

## SLURM Scripts

Use `sbatch` for all runs. The standard script pattern:

```bash
#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=HH:MM:SS
#SBATCH --job-name=<name>
#SBATCH --output=outputs/<name>-%j.out
#SBATCH --error=outputs/<name>-%j.err

source /etc/profile.d/modules.sh
module load CUDA/12.5.0
srun ./build/src/<target>
```

**All output files go in `outputs/`** — this directory is gitignored.

## Running Jobs with sbatch

**Inline sbatch (no script file needed):**

```bash
ssh <server> "source /etc/profile.d/modules.sh && module load CUDA/12.5.0 && sbatch << 'EOF'
#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=<name>
#SBATCH --output=outputs/<name>-%j.out
#SBATCH --error=outputs/<name>-%j.err
srun ./gpu-spmv-2/build/src/<target>
EOF"
```

This passes the script via stdin — no file creation needed on the remote.

**Quick test with srun (no sbatch needed):**
For fast iteration during development, use srun directly instead of sbatch:
```bash
ssh <server> "source /etc/profile.d/modules.sh && module load CUDA/12.5.0 && srun --partition=edu-short --account=gpu.computing26 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=1 --time=00:05:00 ./gpu-spmv-2/build/src/<target>"
```

**With job tracking:**
```bash
JOB_ID=$(ssh <server> "source /etc/profile.d/modules.sh && module load CUDA/12.5.0 && sbatch << 'EOF'
...
EOF")
echo "Submitted job: $JOB_ID"
# Poll for completion
sleep 10 && ssh <server> "cat outputs/<name>-$JOB_ID.out"
```

## Workflow for Testing

1. Ensure `outputs/` directory exists on remote: `ssh <server> "mkdir -p gpu-spmv-2/outputs"`
2. For quick tests: use srun directly (faster, no job queue)
3. For batch runs: submit inline via SSH with heredoc
4. Poll and retrieve output via SSH

## CUDA Version

Use **CUDA 12.5.0** — the latest available on the cluster. Always load with:
```bash
source /etc/profile.d/modules.sh && module load CUDA/12.5.0
```

## Important Notes

- Always use `srun` within SLURM scripts (not direct execution)
- The `edu-short` partition is appropriate for short test runs
- Check `module avail` on the remote to discover available CUDA versions
- For longer runs, adjust `--time` and `--partition` accordingly
- Output files go in `outputs/` directory (created by SLURM scripts)
- CMake puts binaries in `build/src/<target>` by default

## Cache Behavior Analysis (Valgrind/Cachegrind)

Use **cachegrind** to analyze CPU cache performance of host code:

```bash
# Run on remote (CUDA programs need module loaded)
ssh <server> "source /etc/profile.d/modules.sh && module load CUDA/12.5.0 && \
  valgrind --tool=cachegrind ./gpu-spmv-2/build/src/<target>"
```

**Reading results:**
```bash
# Annotate previous run (auto-generated cachegrind.out.<pid>)
ssh <server> "cg_annotate cachegrind.out.<pid>"

# Compare two runs
ssh <server> "cg_diff file1 file2"
```

**Key metrics:**
| Metric | Meaning |
|--------|---------|
| `D1 miss rate` | L1 data cache misses — aim for < 5% for sequential access |
| `LL miss rate` | Last-level cache misses — indicates working set size issues |
| `I1 miss rate` | Instruction cache — problematic in tight loops |

**SpMV-relevant patterns:**
- CSR row-pointer traversal: sequential, should have low D1 miss rate
- Column-index access: depends on matrix sparsity pattern
- Vector x/y access: contiguous → good cache behavior

**Typical workflow:**
1. Build with debug info: `cmake -DCMAKE_BUILD_TYPE=Debug ..`
2. Run cachegrind: `valgrind --tool=cachegrind ./build/src/spmv_test`
3. Annotate: `cg_annotate cachegrind.out.12345`
4. Look for hotspots in matrix loading or kernel launch overhead

**Note:** Cachegrind simulates CPU cache, not GPU. GPU cache behavior (L1/L2) requires `cuda-memcheck --tool=access` or NVIDIA compute-sanitizer.

**CUDA programs:** Dynamic linking adds overhead to cachegrind output (dl_lookup, dl_relocate_object). For isolated kernel analysis, link statically or focus on the non-CUDA portions.