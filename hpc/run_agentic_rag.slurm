#!/bin/bash
#============================================================================
# SLURM job script: Launch vLLM server + run Agentic RAG evaluation
#
# Usage:
#   sbatch hpc/run_agentic_rag.slurm
#   sbatch hpc/run_agentic_rag.slurm --config config.yaml
#
# This script:
#   1. Starts a vLLM inference server in the background
#   2. Waits for the server to become healthy (polling /health endpoint)
#   3. Runs the agentic_rag.py evaluation pipeline
#   4. Cleans up the server process on exit
#============================================================================

#SBATCH --job-name=agentic-rag
#SBATCH --output=logs/agentic-rag-%x-%j.out
#SBATCH --error=logs/agentic-rag-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
##SBATCH --cpus-per-task=32
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=capella
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --licenses=cat

# ==========================================================================
# Configuration
# ==========================================================================

# --- vLLM server settings ---
MODEL_NAME="${VLLM_MODEL:-zai-org/GLM-4.7-Flash}"
MODEL_PATH="${VLLM_MODEL_PATH:-}"             # Local path if pre-downloaded
PORT="${VLLM_PORT:-8000}"
TP_SIZE="${VLLM_TP_SIZE:-2}"
EP="${VLLM_EP:-}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
DTYPE="${VLLM_DTYPE:-auto}"
API_KEY="${VLLM_API_KEY:-ai4all}"
ENABLE_PREFIX_CACHING="${VLLM_PREFIX_CACHE:-true}"

# --- NVMe local staging (set USE_NVME=true to copy model to local SSD) ---
USE_NVME="${VLLM_USE_NVME:-false}"
NVME_MOUNT="/tmp"

# --- Agentic RAG settings ---
RAG_CONFIG="${1:---config config.yaml}"       # Accepts --config <path> as first arg
HEALTH_TIMEOUT=600                            # Max seconds to wait for vLLM startup
HEALTH_POLL_INTERVAL=10                       # Seconds between health checks

# ==========================================================================
# Environment setup
# ==========================================================================
module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
source .venv/bin/activate

# Create log directory
mkdir -p logs

START_TIME=$(date +%s)
echo "============================================================"
echo " Agentic RAG with vLLM Inference"
echo "============================================================"
echo " Start time:        $(date)"
echo " Job ID:            ${SLURM_JOB_ID}"
echo " Node:              $(hostname)"
echo " GPUs:              ${SLURM_GPUS_PER_NODE}"
echo "============================================================"
echo ""

# ==========================================================================
# Performance environment variables (learned from server logs)
# ==========================================================================

# Avoid unnecessary CPU thread contention (vLLM log warning recommendation)
export OMP_NUM_THREADS=1

# Better CUDA graph memory accounting for KV cache allocation (vLLM v0.19+)
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

# Force crash on NCCL hangs instead of silent deadlocks
export NCCL_ASYNC_ERROR_HANDLING=1

# Suppress tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Enable TensorFloat32 for better matmul performance on H100s
export TORCH_FLOAT32_MATMUL_PRECISION=high

# ==========================================================================
# Resolve model path (NVMe staging or Lustre)
# ==========================================================================
SERVE_PATH=""

if [ "${USE_NVME}" = "true" ]; then
    if [ -n "${MODEL_PATH}" ]; then
        SRC_PATH="${MODEL_PATH}"
    elif [ -n "${HF_HOME}" ]; then
        CACHE_DIR="${HF_HOME}/hub/models--$(echo ${MODEL_NAME} | tr '/' '--')"
        if [ -d "${CACHE_DIR}" ]; then
            SRC_PATH=$(find "${CACHE_DIR}/snapshots" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)
        fi
    fi

    if [ -n "${SRC_PATH}" ] && [ -d "${SRC_PATH}" ]; then
        LOCAL_MODEL_DIR="${NVME_MOUNT}/vllm_models/$(basename ${MODEL_NAME})"
        echo "Staging model to NVMe SSD..."
        echo "  Source: ${SRC_PATH}"
        echo "  Destination: ${LOCAL_MODEL_DIR}"
        COPY_START=$(date +%s)
        mkdir -p "${LOCAL_MODEL_DIR}"
        rsync -a --info=progress2 "${SRC_PATH}/" "${LOCAL_MODEL_DIR}/"
        COPY_END=$(date +%s)
        echo "Model copy completed in $((COPY_END - COPY_START)) seconds"
        echo "Model size: $(du -sh ${LOCAL_MODEL_DIR} | cut -f1)"
        SERVE_PATH="${LOCAL_MODEL_DIR}"
    else
        echo "WARNING: Cannot find model at source path. Falling back to HF download."
        SERVE_PATH="${MODEL_NAME}"
    fi
else
    if [ -n "${MODEL_PATH}" ]; then
        SERVE_PATH="${MODEL_PATH}"
    else
        SERVE_PATH="${MODEL_NAME}"
    fi
fi

# ==========================================================================
# Build and launch vLLM server in background
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
    --trust-remote-code \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
    --reasoning-parser glm45"

# Expert parallelism for MoE models
if [ -n "${EP}" ] && [ "${EP}" = "true" ]; then
    VLLM_CMD="${VLLM_CMD} --enable-expert-parallel"
fi

# Prefix caching (helps with repeated prompt prefixes in agentic use)
if [ "${ENABLE_PREFIX_CACHING}" = "true" ]; then
    VLLM_CMD="${VLLM_CMD} --enable-prefix-caching"
fi

NODE_HOSTNAME=$(hostname)

echo ""
echo "============================================================"
echo " Launching vLLM Inference Server (background)"
echo "============================================================"
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
echo " >>> Server URL: http://${NODE_HOSTNAME}:${PORT}/v1 <<<"
echo "============================================================"
echo ""

# Launch vLLM server in background and capture PID
VLLM_LOAD_START=$(date +%s)
eval ${VLLM_CMD} &
VLLM_PID=$!
echo "vLLM server started with PID: ${VLLM_PID}"

# ==========================================================================
# Cleanup trap: ensure vLLM server is killed on exit (success or failure)
# ==========================================================================
cleanup() {
    echo ""
    echo "============================================================"
    echo " Cleaning up..."
    echo "============================================================"
    if kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill ${VLLM_PID}
        wait ${VLLM_PID} 2>/dev/null
        echo "vLLM server stopped."
    fi

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(( (ELAPSED % 3600) / 60 ))
    SECS=$((ELAPSED % 60))
    echo ""
    echo "END TIME: $(date)"
    echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECS}s (${ELAPSED} seconds)"
}
trap cleanup EXIT

# ==========================================================================
# Wait for vLLM server to become healthy
# ==========================================================================
echo "Waiting for vLLM server to become healthy..."
echo "  Health endpoint: http://localhost:${PORT}/health"
echo "  Timeout: ${HEALTH_TIMEOUT}s (polling every ${HEALTH_POLL_INTERVAL}s)"
echo ""

SECONDS_WAITED=0
while [ ${SECONDS_WAITED} -lt ${HEALTH_TIMEOUT} ]; do
    # Check if vLLM process is still alive
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM server process (PID: ${VLLM_PID}) died unexpectedly!"
        echo "Check the error log for details."
        exit 1
    fi

    # Poll the /health endpoint
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null)
    if [ "${HTTP_CODE}" = "200" ]; then
        VLLM_LOAD_END=$(date +%s)
        LOAD_DURATION=$((VLLM_LOAD_END - VLLM_LOAD_START))
        echo ""
        echo ">>> vLLM server is healthy! (took ${LOAD_DURATION}s to start) <<<"
        echo ""
        break
    fi

    echo "  [${SECONDS_WAITED}s/${HEALTH_TIMEOUT}s] Server not ready yet (HTTP: ${HTTP_CODE})..."
    sleep ${HEALTH_POLL_INTERVAL}
    SECONDS_WAITED=$((SECONDS_WAITED + HEALTH_POLL_INTERVAL))
done

if [ ${SECONDS_WAITED} -ge ${HEALTH_TIMEOUT} ]; then
    echo "ERROR: vLLM server did not become healthy within ${HEALTH_TIMEOUT} seconds!"
    exit 1
fi

# Quick validation: list available models
echo "Verifying server - listing available models:"
curl -s -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" | python -m json.tool 2>/dev/null || true
echo ""

# ==========================================================================
# Run Agentic RAG evaluation
# ==========================================================================
echo "============================================================"
echo " Starting Agentic RAG Evaluation"
echo "============================================================"
echo " Config: ${RAG_CONFIG}"
echo " vLLM API: http://localhost:${PORT}/v1"
echo "============================================================"
echo ""

RAG_START=$(date +%s)

python agentic_rag.py ${RAG_CONFIG}
RAG_EXIT_CODE=$?

RAG_END=$(date +%s)
RAG_DURATION=$((RAG_END - RAG_START))
RAG_HOURS=$((RAG_DURATION / 3600))
RAG_MINUTES=$(( (RAG_DURATION % 3600) / 60 ))
RAG_SECS=$((RAG_DURATION % 60))

echo ""
echo "============================================================"
echo " Agentic RAG Evaluation Complete"
echo "============================================================"
echo " Exit code:         ${RAG_EXIT_CODE}"
echo " Duration:          ${RAG_HOURS}h ${RAG_MINUTES}m ${RAG_SECS}s"
echo "============================================================"

exit ${RAG_EXIT_CODE}
