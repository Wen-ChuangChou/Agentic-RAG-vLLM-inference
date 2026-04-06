#!/bin/bash
#============================================================================
# SLURM job script: Launch a multi-node vLLM server using Ray
#
# For models that exceed single-node VRAM (e.g., GLM-5 744B requires ~860GB,
# needing at least 2 nodes × 4 H100 or more).
#
# Architecture:
#   - tensor_parallel_size = GPUs per node (intra-node sharding)
#   - pipeline_parallel_size = number of nodes (inter-node pipelining)
#
# Usage:
#   sbatch launch_vllm_multinode.sh
#============================================================================

#SBATCH --job-name=vllm-multinode
#SBATCH --output=logs/vllm-multinode-%j.out
#SBATCH --error=logs/vllm-multinode-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --partition=capella

# ==========================================================================
# Configuration
# ==========================================================================
MODEL_NAME="${VLLM_MODEL:-zai-org/GLM-5}"
MODEL_PATH="${VLLM_MODEL_PATH:-}"
PORT="${VLLM_PORT:-8000}"
TP_SIZE="${VLLM_TP_SIZE:-4}"                # GPUs per node
PP_SIZE="${VLLM_PP_SIZE:-2}"                # Number of nodes (pipeline parallel)
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
DTYPE="${VLLM_DTYPE:-auto}"
API_KEY="${VLLM_API_KEY:-vllm-secret-key}"
RAY_PORT=6379

# NVMe staging
USE_NVME="${VLLM_USE_NVME:-false}"
NVME_MOUNT="/tmp"

# ==========================================================================
# Environment setup — customize for your HPC
# ==========================================================================
# module purge
# module load CUDA/12.x
# source /path/to/your/vllm-env/bin/activate

mkdir -p logs

# ==========================================================================
# Identify nodes
# ==========================================================================
NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "${NODELIST}" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)

echo "============================================================"
echo " Multi-Node vLLM Deployment"
echo "============================================================"
echo " Head Node:    ${HEAD_NODE} (${HEAD_NODE_IP})"
echo " All Nodes:    $(echo ${NODELIST} | tr '\n' ' ')"
echo " TP Size:      ${TP_SIZE} (per node)"
echo " PP Size:      ${PP_SIZE} (across nodes)"
echo " Total GPUs:   $((TP_SIZE * PP_SIZE))"
echo " Model:        ${MODEL_NAME}"
echo "============================================================"

# ==========================================================================
# NVMe staging on ALL nodes (if enabled)
# ==========================================================================
SERVE_PATH=""

if [ "${USE_NVME}" = "true" ]; then
    if [ -n "${MODEL_PATH}" ] && [ -d "${MODEL_PATH}" ]; then
        LOCAL_MODEL_DIR="${NVME_MOUNT}/vllm_models/$(basename ${MODEL_NAME})"
        echo "Staging model to NVMe on all nodes..."

        # Copy in parallel on all nodes
        srun --ntasks-per-node=1 bash -c "
            mkdir -p ${LOCAL_MODEL_DIR}
            rsync -a ${MODEL_PATH}/ ${LOCAL_MODEL_DIR}/
            echo \"\$(hostname): Model staged at ${LOCAL_MODEL_DIR} — \$(du -sh ${LOCAL_MODEL_DIR} | cut -f1)\"
        "
        SERVE_PATH="${LOCAL_MODEL_DIR}"
    else
        echo "WARNING: MODEL_PATH not set or not found. Using HF download."
        SERVE_PATH="${MODEL_NAME}"
    fi
else
    SERVE_PATH="${MODEL_PATH:-${MODEL_NAME}}"
fi

# ==========================================================================
# Start Ray cluster
# ==========================================================================
echo "Starting Ray head on ${HEAD_NODE}..."
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    ray start --head \
    --node-ip-address="${HEAD_NODE_IP}" \
    --port=${RAY_PORT} \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    --num-gpus "${SLURM_GPUS_ON_NODE:-4}" \
    --block &
sleep 15

# Start Ray workers on remaining nodes
for NODE in $(echo "${NODELIST}" | tail -n +2); do
    echo "Starting Ray worker on ${NODE}..."
    srun --nodes=1 --ntasks=1 -w "${NODE}" \
        ray start \
        --address="${HEAD_NODE_IP}:${RAY_PORT}" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "${SLURM_GPUS_ON_NODE:-4}" \
        --block &
    sleep 5
done

# Wait for workers to register
echo "Waiting for Ray cluster to be ready..."
sleep 20

# Verify Ray cluster
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    ray status

# ==========================================================================
# Launch vLLM on head node
# ==========================================================================
echo ""
echo "Launching vLLM server on ${HEAD_NODE}..."
echo "Connect via: http://${HEAD_NODE}:${PORT}/v1"
echo ""

srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    python -m vllm.entrypoints.openai.api_server \
    --model "${SERVE_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --dtype ${DTYPE} \
    --port ${PORT} \
    --api-key ${API_KEY} \
    --host 0.0.0.0 \
    --trust-remote-code \
    --enable-expert-parallel \
    --enable-prefix-caching
