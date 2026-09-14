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
#   sbatch hpc/run_agentic_rag.slurm --local-env recipes/Qwen3.6-35B-A3B-FP8.yaml
#   sbatch hpc/run_agentic_rag.slurm --test-ids 4 12 20 49 56
#   VLLM_MODEL_PATH=/path/to/model sbatch hpc/run_agentic_rag.slurm
#============================================================================

#SBATCH --job-name=agentic-rag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --partition=capella
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --licenses=cat

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
START_TIME=$(date +%s)

# ==========================================================================
# Configuration (override via environment variables)
# ==========================================================================
USE_LOCAL_ENV="${VLLM_LOCAL_ENV:-false}"
if [[ "${1:-}" == "--local-env" ]]; then
    USE_LOCAL_ENV=true
    shift
fi
# Handle optional config file as first argument
if [[ $# -gt 0 && ! $1 =~ ^- ]]; then
    RAG_CONFIG="$1"
    shift # Remove the config path from $@ so it's not passed twice
else
    RAG_CONFIG="recipes/GLM-4.7-Flash.yaml"
fi

# Extract model names from the YAML config (or fallback to env vars)
# This will be done in the Environment setup section after loading python!

# Model-weight staging is separate from Python-environment staging.
USE_NVME="${VLLM_USE_NVME:-false}"
RAG_JOB_TMP=$(mktemp -d "/tmp/agentic-rag-${SLURM_JOB_ID:?Submit with sbatch}-XXXXXX")
NVME_MOUNT="${RAG_JOB_TMP}"
cleanup_job_tmp() {
    if [[ -d "${RAG_JOB_TMP}" ]]; then
        echo "Removing job-local files: ${RAG_JOB_TMP}"
        rm -rf -- "${RAG_JOB_TMP}"
    fi
}
trap cleanup_job_tmp EXIT

# ==========================================================================
# Environment setup
# ==========================================================================
module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
RAG_PYTHON="${PWD}/.venv/bin/python"
if [[ "${USE_LOCAL_ENV}" == "true" ]]; then
    # Cached build files can contain an earlier job's /tmp source paths.
    # Isolate all compilation caches before importing vLLM or FlashInfer.
    export FLASHINFER_WORKSPACE_BASE="${RAG_JOB_TMP}"
    export VLLM_CACHE_ROOT="${RAG_JOB_TMP}/cache/vllm"
    export TORCHINDUCTOR_CACHE_DIR="${RAG_JOB_TMP}/cache/torchinductor"
    export TRITON_CACHE_DIR="${RAG_JOB_TMP}/cache/triton"
    mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    python -u hpc/stage_python_env.py .venv "${RAG_JOB_TMP}/venv"
    RAG_PYTHON="${RAG_JOB_TMP}/venv/bin/python"
fi
# Do not source the copied activate script: it contains the old absolute path.
export VIRTUAL_ENV="${RAG_PYTHON%/bin/python}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
echo "Pipeline Python: ${RAG_PYTHON}"
echo "Job-local files: ${RAG_JOB_TMP}"

# Now that Python + PyYAML is available, extract models from config
MODEL_NAME="${VLLM_MODEL:-$(python -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['model']['model_id'])")}"
MODEL_PATH="${VLLM_MODEL_PATH:-$(python -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['model'].get('model_path', ''))")}"

JUDGE_MODEL="${VLLM_JUDGE_MODEL:-$(python -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['evaluation']['model_id'])")}"
JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-$(python -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['evaluation'].get('model_path', ''))")}"

mkdir -p logs

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
echo " Local Python env:  ${USE_LOCAL_ENV}"
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

EXIT_CODE=0
srun "${RAG_PYTHON}" -u agentic_rag.py --config "${RAG_CONFIG}" "$@" || EXIT_CODE=$?

# ==========================================================================
# Cleanup & Timing report
# ==========================================================================

echo ""
echo "--- Cleanup ---"
# Remove only this job's environment, caches, and optionally staged models.
# Server diagnostics remain under logs/, and checkpoints/results stay persistent.
cleanup_job_tmp
trap - EXIT
echo "Cleanup complete."
echo ""

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
