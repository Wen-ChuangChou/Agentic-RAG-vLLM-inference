#!/usr/bin/env bash
# ==============================================================================
# Fast Pre-Submission GPU & Node Health Checker for Capella Cluster
# Usage:
#   ./hpc/check_gpus.sh                 # View cluster status & drained nodes
#   ./hpc/check_gpus.sh c36,c93         # Test specific nodes with live 5s srun probe
#   ./hpc/check_gpus.sh --probe-idle    # Probe all currently idle nodes
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${WORKSPACE_ROOT}/.venv/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON="python3"
fi

PARTITION="capella"

show_cluster_summary() {
    echo "============================================================"
    echo " Capella Cluster GPU & Node Status"
    echo "============================================================"
    
    echo "--- Drained / Unhealthy Nodes (Avoid in --exclude) ---"
    sinfo -p "${PARTITION}" -R 2>/dev/null || echo "No drain info available"
    echo ""

    echo "--- Node State Summary ---"
    sinfo -p "${PARTITION}" -o "%15P %15T %10D %N" 2>/dev/null || sinfo
    echo ""

    echo "--- Idle Nodes with 4 H100 GPUs ---"
    IDLE_NODES=$(sinfo -p "${PARTITION}" -N -o "%N %T %G" 2>/dev/null | grep "idle" | grep "gpu" || true)
    if [ -n "${IDLE_NODES}" ]; then
        echo "${IDLE_NODES}"
    else
        echo "No completely idle GPU nodes found at the moment."
    fi
    echo "============================================================"
}

probe_node() {
    local NODE="$1"
    echo -n "Probing node ${NODE}... "
    
    local OUT
    OUT=$(srun --partition="${PARTITION}" -w "${NODE}" --nodes=1 --ntasks=1 --gpus=4 --time=00:01:00 \
        "${VENV_PYTHON}" -c "
import os, sys
try:
    import torch
    available = torch.cuda.is_available()
    cnt = torch.cuda.device_count() if available else 0
    names = [torch.cuda.get_device_name(i) for i in range(cnt)] if cnt > 0 else []
    dev_str = names[0] if names else 'None'
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')
    print(f'SUCCESS|{cnt}|{cvd}|{dev_str}')
except Exception as e:
    print(f'ERROR|0|unset|{e}')
" 2>&1 || echo "ERROR|0|unset|srun_failed")

    local STATUS
    STATUS=$(echo "${OUT}" | grep -E "^(SUCCESS|ERROR)\|" | tail -n1)
    
    if [ -z "${STATUS}" ]; then
        echo "[FAIL] Could not run test on ${NODE}: ${OUT}"
        return 1
    fi

    local CODE=$(echo "${STATUS}" | cut -d'|' -f1)
    local COUNT=$(echo "${STATUS}" | cut -d'|' -f2)
    local CVD=$(echo "${STATUS}" | cut -d'|' -f3)
    local DEV_NAME=$(echo "${STATUS}" | cut -d'|' -f4)

    if [ "${CODE}" = "SUCCESS" ] && [ "${COUNT}" -ge 4 ]; then
        echo "[OK] ${COUNT}/4 GPUs accessible | ${DEV_NAME} | CVD=${CVD}"
    elif [ "${CODE}" = "SUCCESS" ]; then
        echo "[WARN] Only ${COUNT}/4 GPUs visible | ${DEV_NAME} | CVD=${CVD}"
    else
        echo "[FAIL] Error probing node: ${DEV_NAME}"
    fi
}

# Main routing
if [ $# -eq 0 ]; then
    show_cluster_summary
    echo ""
    echo "To test live CUDA device visibility on specific node(s), run:"
    echo "  $0 <nodename(s)>     e.g., $0 c61,c125"
    echo "  $0 --probe-idle      e.g., probes currently idle nodes"
    exit 0
fi

if [ "$1" = "--probe-idle" ]; then
    IDLE_LIST=$(sinfo -p "${PARTITION}" -N -h -o "%N %T" 2>/dev/null | awk '$2=="idle"{print $1}')
    if [ -z "${IDLE_LIST}" ]; then
        echo "No idle nodes currently available."
        exit 0
    fi
    echo "Probing all idle nodes: $(echo ${IDLE_LIST} | tr '\n' ' ')"
    for N in ${IDLE_LIST}; do
        probe_node "${N}"
    done
    exit 0
fi

# Comma-separated or space-separated node list
TARGETS=$(echo "$1" | tr ',' ' ')
for NODE in ${TARGETS}; do
    probe_node "${NODE}"
done
