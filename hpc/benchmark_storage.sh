#!/bin/bash
#============================================================================
# Benchmark: Compare vLLM model loading time from Lustre vs NVMe SSD
#
# This script runs the vLLM server twice — once loading from the shared
# Lustre filesystem, once from local NVMe SSD — and logs the time to
# first token readiness for each. Useful for quantifying the benefit of
# NVMe staging for large MoE models.
#
# Usage:
#   sbatch benchmark_storage.sh
#
# Results are written to logs/benchmark_storage-<jobid>.out
#============================================================================

#SBATCH --job-name=vllm-bench-storage
#SBATCH --output=logs/benchmark_storage-%j.out
#SBATCH --error=logs/benchmark_storage-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --partition=capella

# ==========================================================================
# Configuration
# ==========================================================================
MODEL_NAME="${VLLM_MODEL:-zai-org/GLM-5}"
LUSTRE_MODEL_PATH="${VLLM_MODEL_PATH}"      # Path on shared Lustre filesystem
NVME_MOUNT="/tmp"
PORT=8000
TP_SIZE="${VLLM_TP_SIZE:-4}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}" # Keep short for benchmarking
GPU_MEM_UTIL=0.92
DTYPE="${VLLM_DTYPE:-auto}"
API_KEY="bench-key"

# ==========================================================================
# Environment setup — customize for your HPC
# ==========================================================================
# module purge
# module load CUDA/12.x
# source /path/to/your/vllm-env/bin/activate

mkdir -p logs

# ==========================================================================
# Helper function: start vLLM and measure time to ready
# ==========================================================================
measure_startup() {
    local LABEL=$1
    local MODEL_DIR=$2
    local LOG_FILE="logs/bench_${LABEL}_${SLURM_JOB_ID}.log"

    echo ""
    echo "============================================================"
    echo " Benchmarking: ${LABEL}"
    echo " Model path:   ${MODEL_DIR}"
    echo "============================================================"

    START_TIME=$(date +%s%N)  # nanoseconds

    # Start vLLM in background
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_DIR}" \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size ${TP_SIZE} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --max-model-len ${MAX_MODEL_LEN} \
        --dtype ${DTYPE} \
        --port ${PORT} \
        --api-key ${API_KEY} \
        --host 0.0.0.0 \
        --trust-remote-code \
        --enable-expert-parallel \
        2>&1 | tee "${LOG_FILE}" &

    VLLM_PID=$!

    # Wait for server to become ready (poll /v1/models endpoint)
    echo "Waiting for vLLM to become ready..."
    TIMEOUT=1800  # 30 minutes max
    ELAPSED=0

    while [ ${ELAPSED} -lt ${TIMEOUT} ]; do
        if curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${API_KEY}" \
            "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "200"; then
            break
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    END_TIME=$(date +%s%N)

    if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
        echo "ERROR: Server did not become ready within ${TIMEOUT}s"
        kill ${VLLM_PID} 2>/dev/null
        wait ${VLLM_PID} 2>/dev/null
        return 1
    fi

    # Calculate duration
    DURATION_NS=$((END_TIME - START_TIME))
    DURATION_S=$(echo "scale=2; ${DURATION_NS} / 1000000000" | bc)

    echo ""
    echo ">>> ${LABEL}: Server ready in ${DURATION_S} seconds <<<"
    echo ""

    # Run a quick inference test
    echo "Running inference smoke test..."
    INFER_START=$(date +%s%N)

    curl -s -X POST "http://localhost:${PORT}/v1/chat/completions" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL_NAME}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer in one word.\"}],
            \"max_tokens\": 10,
            \"temperature\": 0
        }" 2>/dev/null | python -m json.tool

    INFER_END=$(date +%s%N)
    INFER_DURATION=$(echo "scale=2; ($INFER_END - $INFER_START) / 1000000000" | bc)
    echo "First inference latency: ${INFER_DURATION} seconds"

    # Clean up
    echo "Stopping vLLM server..."
    kill ${VLLM_PID} 2>/dev/null
    wait ${VLLM_PID} 2>/dev/null
    sleep 10  # Give GPUs time to release memory

    # Write summary
    echo "${LABEL},${DURATION_S},${INFER_DURATION}" >> "logs/benchmark_results_${SLURM_JOB_ID}.csv"
}

# ==========================================================================
# Benchmark 1: Load from Lustre
# ==========================================================================
if [ -z "${LUSTRE_MODEL_PATH}" ]; then
    echo "ERROR: VLLM_MODEL_PATH not set. Cannot benchmark Lustre loading."
    echo "Usage: VLLM_MODEL_PATH=/path/to/model sbatch benchmark_storage.sh"
    exit 1
fi

echo "storage,startup_seconds,first_inference_seconds" > "logs/benchmark_results_${SLURM_JOB_ID}.csv"

measure_startup "lustre" "${LUSTRE_MODEL_PATH}"

# ==========================================================================
# Benchmark 2: Copy to NVMe SSD, then load
# ==========================================================================
LOCAL_MODEL_DIR="${NVME_MOUNT}/vllm_models/$(basename ${MODEL_NAME})"

echo ""
echo "============================================================"
echo " Copying model to NVMe SSD..."
echo " Source:      ${LUSTRE_MODEL_PATH}"
echo " Destination: ${LOCAL_MODEL_DIR}"
echo "============================================================"

COPY_START=$(date +%s)
mkdir -p "${LOCAL_MODEL_DIR}"
rsync -a --info=progress2 "${LUSTRE_MODEL_PATH}/" "${LOCAL_MODEL_DIR}/"
COPY_END=$(date +%s)
COPY_DURATION=$((COPY_END - COPY_START))

echo "Copy completed in ${COPY_DURATION} seconds"
echo "Model size on NVMe: $(du -sh ${LOCAL_MODEL_DIR} | cut -f1)"

measure_startup "nvme" "${LOCAL_MODEL_DIR}"

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "============================================================"
echo " BENCHMARK RESULTS"
echo "============================================================"
echo " Model:       ${MODEL_NAME}"
echo " Node:        $(hostname)"
echo " Copy time:   ${COPY_DURATION} seconds (Lustre → NVMe)"
echo ""
echo " Results:"
cat "logs/benchmark_results_${SLURM_JOB_ID}.csv" | column -t -s ','
echo ""
echo "============================================================"
echo " Full results: logs/benchmark_results_${SLURM_JOB_ID}.csv"
echo "============================================================"

# Cleanup NVMe
echo "Cleaning up NVMe staging area..."
rm -rf "${LOCAL_MODEL_DIR}"
