#!/bin/bash
#============================================================================
# SLURM job script: Launch a multi-node vLLM server using Ray
#
# Target Architecture:
#   2 nodes × 4 GPUs per node = 8 GPUs total
#   - Model: zai-org/GLM-5.3-Flash
#   - tensor_parallel_size: 1
#   - data_parallel_size: 8
#   - data_parallel_size_local: 4
#   - enable_expert_parallel: true
#   - all2all_backend: allgather_reducescatter
#   - data_parallel_backend: ray (Option A)
#
# Usage:
#   sbatch hpc/launch_vllm_multinode.sh
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
#SBATCH --licenses=cat

set -eo pipefail

# ==========================================================================
# Configuration
# ==========================================================================
MODEL_NAME="${VLLM_MODEL:-zai-org/GLM-5.3-Flash}"
MODEL_PATH="${VLLM_MODEL_PATH:-}"
PORT="${VLLM_PORT:-8000}"
TP_SIZE="${VLLM_TP_SIZE:-1}"                        # tensor_parallel_size
DP_SIZE="${VLLM_DP_SIZE:-8}"                        # data_parallel_size (total across all nodes)
DP_SIZE_LOCAL="${VLLM_DP_SIZE_LOCAL:-4}"            # data_parallel_size_local (per node)
DP_BACKEND="${VLLM_DP_BACKEND:-ray}"                # data_parallel_backend
ENABLE_EP="${VLLM_ENABLE_EP:-true}"                # enable_expert_parallel
ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-allgather_reducescatter}" # all2all_backend
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.94}"
DTYPE="${VLLM_DTYPE:-auto}"
API_KEY="${VLLM_API_KEY:-ai4all}"
ENABLE_PREFIX_CACHING="${VLLM_PREFIX_CACHE:-true}"
RAY_PORT=6379

# Required for DeepEP low-latency kernels under Ray
export VLLM_RAY_DP_PACK_STRATEGY=strict

# NVMe staging
USE_NVME="${VLLM_USE_NVME:-false}"
NVME_MOUNT="/tmp"

# ==========================================================================
# Environment setup
# ==========================================================================
module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
source .venv/bin/activate

mkdir -p logs

# ==========================================================================
# Identify nodes
# ==========================================================================
NODELIST=$(scontrol show hostname "$SLURM_JOB_NODELIST")
HEAD_NODE=$(echo "${NODELIST}" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)

echo "============================================================"
echo " Multi-Node vLLM Deployment (Ray Data Parallel)"
echo "============================================================"
echo " Head Node:       ${HEAD_NODE} (${HEAD_NODE_IP})"
echo " All Nodes:       $(echo ${NODELIST} | tr '\n' ' ')"
echo " Model:           ${MODEL_NAME}"
echo " TP Size:         ${TP_SIZE}"
echo " DP Size Total:   ${DP_SIZE}"
echo " DP Size Local:   ${DP_SIZE_LOCAL}"
echo " DP Backend:      ${DP_BACKEND}"
echo " Expert Parallel: ${ENABLE_EP}"
echo " All2All Backend: ${ALL2ALL_BACKEND}"
echo " Total GPUs:      $((SLURM_NNODES * 4))"
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
cleanup_ray() {
    echo "Stopping Ray cluster across all nodes..."
    srun --overlap --ntasks-per-node=1 ray stop --force 2>/dev/null || true
}
trap cleanup_ray EXIT

echo "Starting Ray head on ${HEAD_NODE} (${HEAD_NODE_IP}:${RAY_PORT})..."
ray start --head \
    --node-ip-address="${HEAD_NODE_IP}" \
    --port=${RAY_PORT} \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    --num-gpus "${SLURM_GPUS_ON_NODE:-4}" \
    --disable-usage-stats

# Start Ray workers on remaining nodes
for NODE in $(echo "${NODELIST}" | tail -n +2); do
    echo "Starting Ray worker on ${NODE}..."
    srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" \
        ray start \
        --address="${HEAD_NODE_IP}:${RAY_PORT}" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "${SLURM_GPUS_ON_NODE:-4}" \
        --disable-usage-stats
done

# Wait for workers to register
echo "Waiting for Ray cluster to be ready..."
sleep 5

# Verify Ray cluster
ray status

# ==========================================================================
# Launch vLLM server via Ray data-parallel on head node
# ==========================================================================
echo ""
echo "Launching vLLM server on ${HEAD_NODE}..."
echo "Connect via: http://${HEAD_NODE}:${PORT}/v1"
echo ""

VLLM_CMD="vllm serve ${SERVE_PATH} \
    --served-model-name ${MODEL_NAME} \
    --tensor-parallel-size ${TP_SIZE} \
    --data-parallel-size ${DP_SIZE} \
    --data-parallel-size-local ${DP_SIZE_LOCAL} \
    --data-parallel-backend ${DP_BACKEND} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --dtype ${DTYPE} \
    --port ${PORT} \
    --api-key ${API_KEY} \
    --host 0.0.0.0 \
    --trust-remote-code"

if [ "${ENABLE_EP}" = "true" ]; then
    VLLM_CMD="${VLLM_CMD} --enable-expert-parallel"
fi

if [ -n "${ALL2ALL_BACKEND}" ]; then
    VLLM_CMD="${VLLM_CMD} --all2all-backend ${ALL2ALL_BACKEND}"
fi

if [ "${ENABLE_PREFIX_CACHING}" = "true" ]; then
    VLLM_CMD="${VLLM_CMD} --enable-prefix-caching"
fi

srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" ${VLLM_CMD}
