#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=test-webbase-v2
#SBATCH --output=test-webbase-v2-%j.out
#SBATCH --error=test-webbase-v2-%j.err

export CUDA_HOME=/opt/shares/cuda/software/CUDA/12.5.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /home/davide.andreolli/uni/gpu/gpu-spmv/build
srun --gres=gpu:1 ./benchmark_spmv_gpu --matrix ../data/matrices/suiteSparse_matrices/webbase-1M.mtx --verify