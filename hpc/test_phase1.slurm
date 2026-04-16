#!/bin/bash
#============================================================================
# SLURM job script: Phase 1 Unit Test
#
# Isolated test for offline batch inference (Standard RAG + Vanilla).
# Tests that the TMA descriptor crash is resolved by reducing max_model_len.
#
# Usage:
#   sbatch hpc/test_phase1.slurm
#   sbatch hpc/test_phase1.slurm recipes/GLM-4.7-Flash.yaml
#============================================================================

#SBATCH --job-name=test-phase1
#SBATCH --output=logs/test-phase1-%j.out
#SBATCH --error=logs/test-phase1-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --partition=capella
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --licenses=cat

# Config
RAG_CONFIG="${1:-recipes/GLM-4.7-Flash.yaml}"

# Environment
module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
source .venv/bin/activate

mkdir -p logs

echo "============================================================"
echo " Phase 1 Unit Test"
echo "============================================================"
echo " Job ID:   ${SLURM_JOB_ID}"
echo " Node:     $(hostname)"
echo " GPUs:     ${SLURM_GPUS_PER_NODE}"
echo " Config:   ${RAG_CONFIG}"
echo "============================================================"

# Performance env vars
export OMP_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export TORCH_FLOAT32_MATMUL_PRECISION=high
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

srun python tests/test_phase1.py --config "${RAG_CONFIG}" --test 5
EXIT_CODE=$?

echo ""
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
