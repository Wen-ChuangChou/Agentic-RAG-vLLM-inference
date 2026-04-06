#!/bin/bash
#============================================================================
# SLURM job script: Launch a vLLM OpenAI-compatible inference server
#
# Usage:
#   sbatch launch_vllm_server.sh
#
# After the server starts, connect from the client with:
#   export VLLM_API_BASE="http://$(hostname):${PORT}/v1"
#
# The server logs the connection URL to the output file for easy reference.
#============================================================================

#SBATCH --job-name=vllm-server
#SBATCH --output=logs/vllm-server-%j.out
#SBATCH --error=logs/vllm-server-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --partition=capella

# ==========================================================================
# Configuration — edit these variables for your model and environment
# ==========================================================================
MODEL_NAME="${VLLM_MODEL:-zai-org/GLM-4.7-FP8}"          # HuggingFace model ID
MODEL_PATH="${VLLM_MODEL_PATH:-}"                   # Local path (if pre-downloaded)
PORT="${VLLM_PORT:-8000}"
TP_SIZE="${VLLM_TP_SIZE:-4}"                        # tensor_parallel_size
EP="${VLLM_EP:-}"                                   # expert parallelism (MoE only)
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
DTYPE="${VLLM_DTYPE:-auto}"                         # auto, bfloat16, float16, fp8
API_KEY="${VLLM_API_KEY:-vllm-secret-key}"
ENABLE_PREFIX_CACHING="${VLLM_PREFIX_CACHE:-true}"

# === NVMe local staging (optional) ===
# Set USE_NVME=true to copy model weights from Lustre to local NVMe SSD
# before inference for faster model loading.
USE_NVME="${VLLM_USE_NVME:-false}"
NVME_MOUNT="/tmp"           # Local NVMe SSD mount point on compute nodes

# === HuggingFace cache ===
# HF_HOME should already be set in your environment profile.
# Uncomment and set if needed:
# export HF_HOME="/path/to/shared/hf_cache"

# ==========================================================================
# Environment setup — customize module loads for your HPC
# ==========================================================================
# module purge
# module load CUDA/12.x
# module load cuDNN/...
# source /path/to/your/vllm-env/bin/activate

# Create log directory
mkdir -p logs

# ==========================================================================
# NVMe local staging: copy model weights for faster loading
# ==========================================================================
SERVE_PATH=""

if [ "${USE_NVME}" = "true" ]; then
    # Determine source path: use MODEL_PATH if set, otherwise HF cache
    if [ -n "${MODEL_PATH}" ]; then
        SRC_PATH="${MODEL_PATH}"
    elif [ -n "${HF_HOME}" ]; then
        # HuggingFace cache layout: models--org--name/snapshots/<hash>/
        CACHE_DIR="${HF_HOME}/hub/models--$(echo ${MODEL_NAME} | tr '/' '--')"
        if [ -d "${CACHE_DIR}" ]; then
            # Find the latest snapshot
            SRC_PATH=$(find "${CACHE_DIR}/snapshots" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)
        fi
    fi

    if [ -n "${SRC_PATH}" ] && [ -d "${SRC_PATH}" ]; then
        LOCAL_MODEL_DIR="${NVME_MOUNT}/vllm_models/$(basename ${MODEL_NAME})"
        echo "============================================="
        echo "Staging model to NVMe SSD..."
        echo "  Source: ${SRC_PATH}"
        echo "  Destination: ${LOCAL_MODEL_DIR}"
        echo "============================================="

        COPY_START=$(date +%s)
        mkdir -p "${LOCAL_MODEL_DIR}"
        rsync -a --info=progress2 "${SRC_PATH}/" "${LOCAL_MODEL_DIR}/"
        COPY_END=$(date +%s)
        COPY_DURATION=$((COPY_END - COPY_START))

        echo "Model copy completed in ${COPY_DURATION} seconds"
        echo "Model size: $(du -sh ${LOCAL_MODEL_DIR} | cut -f1)"
        SERVE_PATH="${LOCAL_MODEL_DIR}"
    else
        echo "WARNING: Cannot find model at source path. Falling back to HF download."
        SERVE_PATH="${MODEL_NAME}"
    fi
else
    # Use MODEL_PATH if provided (Lustre), otherwise let vLLM download from HF
    if [ -n "${MODEL_PATH}" ]; then
        SERVE_PATH="${MODEL_PATH}"
    else
        SERVE_PATH="${MODEL_NAME}"
    fi
fi

# ==========================================================================
# Build vLLM launch command
# ==========================================================================
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model ${SERVE_PATH} \
    --served-model-name ${MODEL_NAME} \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --dtype ${DTYPE} \
    --port ${PORT} \
    --api-key ${API_KEY} \
    --host 0.0.0.0 \
    --trust-remote-code"

# Enable expert parallelism for MoE models
if [ -n "${EP}" ] && [ "${EP}" = "true" ]; then
    VLLM_CMD="${VLLM_CMD} --enable-expert-parallel"
fi

# Enable prefix caching (helps with repeated prompt prefixes in agentic use)
if [ "${ENABLE_PREFIX_CACHING}" = "true" ]; then
    VLLM_CMD="${VLLM_CMD} --enable-prefix-caching"
fi

# ==========================================================================
# Launch
# ==========================================================================
NODE_HOSTNAME=$(hostname)

echo ""
echo "============================================================"
echo " vLLM Inference Server"
echo "============================================================"
echo " Node:              ${NODE_HOSTNAME}"
echo " Model:             ${MODEL_NAME}"
echo " Serving from:      ${SERVE_PATH}"
echo " Port:              ${PORT}"
echo " Tensor Parallel:   ${TP_SIZE}"
echo " Expert Parallel:   ${EP:-disabled}"
echo " Dtype:             ${DTYPE}"
echo " Max Seq Len:       ${MAX_MODEL_LEN}"
echo " GPU Mem Util:      ${GPU_MEM_UTIL}"
echo " Prefix Caching:    ${ENABLE_PREFIX_CACHING}"
echo ""
echo " >>> Connect via: http://${NODE_HOSTNAME}:${PORT}/v1 <<<"
echo "============================================================"
echo ""

# Record start time for benchmarking model load
LOAD_START=$(date +%s)

eval ${VLLM_CMD}
