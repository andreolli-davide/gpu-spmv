#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=test-webbase-v1
#SBATCH --output=test-webbase-v1-%j.out
#SBATCH --error=test-webbase-v1-%j.err

export CUDA_HOME=/opt/shares/cuda/software/CUDA/12.5.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /home/davide.andreolli/uni/gpu/gpu-spmv/build
mkdir -p results
srun --gres=gpu:1 ./test_spmv_gpu_v1 --matrix ../data/matrices/suiteSparse_matrices/webbase-1M.mtx --verify