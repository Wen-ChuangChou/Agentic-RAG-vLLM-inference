#!/bin/bash
#============================================================================
# SLURM job script: Hybrid Agentic RAG Evaluation Pipeline
#
# Three-phase pipeline (managed by Python, not bash):
#   Phase 1: Offline batch (Standard RAG + Vanilla) via vllm.LLM
#   Phase 2: Concurrent Agentic RAG via vLLM server + asyncio
#   Phase 3: Offline batch Judge via vllm.LLM
#
# Usage:
#   sbatch hpc/run_agentic_rag.slurm
#   sbatch hpc/run_agentic_rag.slurm recipes/other-model.yaml
#   VLLM_MODEL_PATH=/path/to/model sbatch hpc/run_agentic_rag.slurm
#============================================================================

#SBATCH --job-name=agentic-rag
#SBATCH --output=logs/agentic-rag-%x-%j.out
#SBATCH --error=logs/agentic-rag-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --partition=capella
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --licenses=cat

# ==========================================================================
# Configuration (override via environment variables)
# ==========================================================================
# Handle optional config file as first argument
if [[ $# -gt 0 && ! $1 =~ ^- ]]; then
    RAG_CONFIG="$1"
    shift # Remove the config path from $@ so it's not passed twice
else
    RAG_CONFIG="recipes/GLM-4.7-Flash.yaml"
fi

# Extract model names from the YAML config (or fallback to env vars)
# This will be done in the Environment setup section after loading python!

# NVMe local staging — enabled by default for faster model loading
USE_NVME="${VLLM_USE_NVME:-true}"
NVME_MOUNT="/tmp"

# ==========================================================================
# Environment setup
# ==========================================================================
module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
source .venv/bin/activate

# Now that Python + PyYAML is available, extract models from config
MODEL_NAME="${VLLM_MODEL:-$(python -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['model']['model_id'])")}"
MODEL_PATH="${VLLM_MODEL_PATH:-$(python -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['model'].get('model_path', ''))")}"

JUDGE_MODEL="${VLLM_JUDGE_MODEL:-$(python -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['evaluation']['model_id'])")}"
JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-$(python -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['evaluation'].get('model_path', ''))")}"

mkdir -p logs

START_TIME=$(date +%s)
echo "============================================================"
echo " Hybrid Agentic RAG Pipeline"
echo "============================================================"
echo " Start time:        $(date)"
echo " Job ID:            ${SLURM_JOB_ID}"
echo " Node:              $(hostname)"
echo " GPUs:              ${SLURM_GPUS_PER_NODE}"
echo " Model:             ${MODEL_NAME}"
echo " Judge:             ${JUDGE_MODEL}"
echo " Config:            ${RAG_CONFIG}"
echo " NVMe staging:      ${USE_NVME}"
echo "============================================================"
echo ""

# --- Performance environment variables ---
export OMP_NUM_THREADS=1                                 # Avoid CPU thread contention
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1        # Better KV cache sizing
export NCCL_ASYNC_ERROR_HANDLING=1                       # Crash on NCCL hangs
export TOKENIZERS_PARALLELISM=false                      # Suppress tokenizer warnings
export TORCH_FLOAT32_MATMUL_PRECISION=high               # TF32 on H100s
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce GPU memory fragmentation
export VLLM_WORKER_MULTIPROC_METHOD=spawn                # Explicitly request spawn to quiet CUDA initialization warnings

# ==========================================================================
# NVMe Model Staging
# ==========================================================================

resolve_model_source() {
    # Find the actual model directory given a model name and optional path
    local MODEL_ID="$1"
    local EXPLICIT_PATH="$2"

    if [ -n "${EXPLICIT_PATH}" ] && [ -d "${EXPLICIT_PATH}" ]; then
        echo "${EXPLICIT_PATH}"
        return
    fi

    # Delegate to HuggingFace python wrapper to perfectly lookup the snapshot directory
    # Redirect stderr to avoid polluting the log if the model is somehow missing
    python -c "
import sys, warnings, logging
logging.disable(logging.CRITICAL); warnings.filterwarnings('ignore')
from huggingface_hub import snapshot_download
try:
    print(snapshot_download('${MODEL_ID}', local_files_only=True))
except Exception:
    pass
" 2>/dev/null
    echo ""
}

stage_to_nvme() {
    # Copy a model directory to NVMe SSD for faster I/O
    local SRC="$1"
    local BASENAME="$2"
    local VAR_NAME="$3"  # The environment variable to export (e.g. VLLM_MODEL_PATH)
    local DEST="${NVME_MOUNT}/vllm_models/${BASENAME}"

    if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
        echo "Source not found for ${BASENAME}, skipping NVMe staging."
        return
    fi

    echo "Staging ${BASENAME} to NVMe SSD ..."
    echo "  Source:  ${SRC}"
    echo "  Dest:    ${DEST}"
    local T0=$(date +%s)
    mkdir -p "${DEST}"
    rsync -aL --info=progress2 "${SRC}/" "${DEST}/"
    local T1=$(date +%s)
    echo "  Copied in $((T1 - T0))s  ($(du -sh "${DEST}" | cut -f1))"
    
    # Export the final path dynamically
    export ${VAR_NAME}="${DEST}"
    echo "  Set ${VAR_NAME}=${DEST}"
}

if [ "${USE_NVME}" = "true" ]; then
    echo ""
    echo "--- NVMe model staging ---"

    # Stage main model
    MAIN_SRC=$(resolve_model_source "${MODEL_NAME}" "${MODEL_PATH}")
    if [ -n "${MAIN_SRC}" ]; then
        stage_to_nvme "${MAIN_SRC}" "$(basename ${MODEL_NAME})" "VLLM_MODEL_PATH"
    fi

    # Stage judge model (if different from main)
    if [ "${JUDGE_MODEL}" != "${MODEL_NAME}" ]; then
        JUDGE_SRC=$(resolve_model_source "${JUDGE_MODEL}" "${JUDGE_MODEL_PATH}")
        if [ -n "${JUDGE_SRC}" ]; then
            stage_to_nvme "${JUDGE_SRC}" "$(basename ${JUDGE_MODEL})" "VLLM_JUDGE_MODEL_PATH"
        fi
    fi

    echo ""
else
    # Pass through explicit paths if set
    [ -n "${MODEL_PATH}" ] && export VLLM_MODEL_PATH="${MODEL_PATH}"
    [ -n "${JUDGE_MODEL_PATH}" ] && export VLLM_JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}"
fi

# ==========================================================================
# Run the hybrid pipeline
# ==========================================================================
echo "============================================================"
echo " Launching Python pipeline"
echo "============================================================"
echo ""

srun python agentic_rag.py --config "${RAG_CONFIG}" "$@"
EXIT_CODE=$?

# ==========================================================================
# Timing report
# ==========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo " Pipeline Complete"
echo "============================================================"
echo " Exit code:  ${EXIT_CODE}"
echo " End time:   $(date)"
echo " Duration:   ${HOURS}h ${MINUTES}m ${SECS}s"
echo "============================================================"

exit ${EXIT_CODE}
