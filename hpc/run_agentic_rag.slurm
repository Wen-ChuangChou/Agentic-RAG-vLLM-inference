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
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=capella
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --licenses=cat

# ==========================================================================
# Configuration (override via environment variables)
# ==========================================================================
MODEL_NAME="${VLLM_MODEL:-zai-org/GLM-4.7-Flash}"
MODEL_PATH="${VLLM_MODEL_PATH:-}"
JUDGE_MODEL="${VLLM_JUDGE_MODEL:-openai/gpt-oss-120b}"
JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-}"
RAG_CONFIG="${1:-recipes/GLM-4.7-Flash.yaml}"

# NVMe local staging — enabled by default for faster model loading
USE_NVME="${VLLM_USE_NVME:-true}"
NVME_MOUNT="/tmp"

# ==========================================================================
# Environment setup
# ==========================================================================
module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
source .venv/bin/activate

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

    # Try HF cache
    if [ -n "${HF_HOME}" ]; then
        local CACHE_DIR="${HF_HOME}/hub/models--$(echo ${MODEL_ID} | tr '/' '--')"
        if [ -d "${CACHE_DIR}" ]; then
            local SNAP=$(find "${CACHE_DIR}/snapshots" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)
            if [ -n "${SNAP}" ] && [ -d "${SNAP}" ]; then
                echo "${SNAP}"
                return
            fi
        fi
    fi
    echo ""
}

stage_to_nvme() {
    # Copy a model directory to NVMe SSD for faster I/O
    local SRC="$1"
    local BASENAME="$2"
    local DEST="${NVME_MOUNT}/vllm_models/${BASENAME}"

    if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
        echo ""
        return
    fi

    echo "Staging ${BASENAME} to NVMe SSD ..."
    echo "  Source:  ${SRC}"
    echo "  Dest:    ${DEST}"
    local T0=$(date +%s)
    mkdir -p "${DEST}"
    rsync -a --info=progress2 "${SRC}/" "${DEST}/"
    local T1=$(date +%s)
    echo "  Copied in $((T1 - T0))s  ($(du -sh "${DEST}" | cut -f1))"
    echo "${DEST}"
}

if [ "${USE_NVME}" = "true" ]; then
    echo ""
    echo "--- NVMe model staging ---"

    # Stage main model
    MAIN_SRC=$(resolve_model_source "${MODEL_NAME}" "${MODEL_PATH}")
    MAIN_NVME=$(stage_to_nvme "${MAIN_SRC}" "$(basename ${MODEL_NAME})")
    if [ -n "${MAIN_NVME}" ]; then
        export VLLM_MODEL_PATH="${MAIN_NVME}"
        echo "  VLLM_MODEL_PATH=${VLLM_MODEL_PATH}"
    fi

    # Stage judge model (if different from main)
    if [ "${JUDGE_MODEL}" != "${MODEL_NAME}" ]; then
        JUDGE_SRC=$(resolve_model_source "${JUDGE_MODEL}" "${JUDGE_MODEL_PATH}")
        JUDGE_NVME=$(stage_to_nvme "${JUDGE_SRC}" "$(basename ${JUDGE_MODEL})")
        if [ -n "${JUDGE_NVME}" ]; then
            export VLLM_JUDGE_MODEL_PATH="${JUDGE_NVME}"
            echo "  VLLM_JUDGE_MODEL_PATH=${VLLM_JUDGE_MODEL_PATH}"
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

python agentic_rag.py --config "${RAG_CONFIG}"
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
