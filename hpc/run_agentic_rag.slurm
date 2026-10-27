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
#   # Single node (default, 4 GPUs, e.g. Qwen3.8-Flash-Next-FP8, GLM-4.7-Flash):
#   sbatch hpc/run_agentic_rag.slurm
#   sbatch hpc/run_agentic_rag.slurm recipes/Qwen3.8-Flash-Next-FP8.yaml
#   sbatch hpc/run_agentic_rag.slurm --local-env recipes/Qwen3.6-35B-A3B-FP8.yaml
#   sbatch hpc/run_agentic_rag.slurm --test-ids 4 12 20 49 56
#
#   # Two nodes (8 GPUs, e.g. GLM-5.3-Flash with Ray Data Parallel):
#   sbatch --nodes=2 --ntasks=2 --cpus-per-task=56 --mem=0 --gpus-per-node=4 --gres-flags=disable-binding --exclusive --time=01:30:00 hpc/run_agentic_rag.slurm --local-env --nvme recipes/GLM-5.3-Flash.yaml
#   sbatch --nodes=3 --ntasks=3 --cpus-per-task=56 --mem=0 --gpus-per-node=4 --gres-flags=disable-binding --exclusive --time=01:30:00 hpc/run_agentic_rag.slurm --local-env --nvme recipes/DeepSeek-V4.1-Flash-1017.yaml
#
#   VLLM_MODEL_PATH=/path/to/model sbatch hpc/run_agentic_rag.slurm
#
# A pre-flight ABI check runs before model staging and fails fast (seconds)
# if a version-locked package pair is inconsistent. Bypass with:
#   RAG_SKIP_PREFLIGHT=true sbatch hpc/run_agentic_rag.slurm
#============================================================================

#SBATCH --job-name=agentic-rag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --gres-flags=disable-binding
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --partition=capella
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --licenses=cat

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
START_TIME=$(date +%s)

# Clear any inherited GPU restrictions from login/submission environment
unset CUDA_VISIBLE_DEVICES


# ==========================================================================
# Configuration (override via environment variables)
# ==========================================================================
USE_LOCAL_ENV="${VLLM_LOCAL_ENV:-false}"
USE_NVME="${VLLM_USE_NVME:-false}"
RAG_CONFIG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-env)
            USE_LOCAL_ENV=true
            shift
            ;;
        --use-nvme|--nvme)
            USE_NVME=true
            shift
            ;;
        *.yaml|*.yml)
            RAG_CONFIG="$1"
            shift
            ;;
        *)
            if [[ -z "${RAG_CONFIG}" && ! "$1" =~ ^- ]]; then
                RAG_CONFIG="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "${RAG_CONFIG}" ]]; then
    RAG_CONFIG="recipes/Qwen3.8-Flash-Next-FP8.yaml"
fi
set -- "${EXTRA_ARGS[@]}"
NUM_NODES=${SLURM_NNODES:-1}
RAG_JOB_TMP="/tmp/agentic-rag-${SLURM_JOB_ID:?Submit with sbatch}"
NVME_MOUNT="${RAG_JOB_TMP}"
cleanup_job_tmp() {
    if [ -n "${RAY_WORKER_PID:-}" ]; then
        kill "${RAY_WORKER_PID}" 2>/dev/null || true
    fi
    if [ "${RAY_STARTED:-false}" = "true" ]; then
        echo "Stopping Ray cluster across all nodes..."
        if [ "${NUM_NODES}" -gt 1 ]; then
            for WORKER in $(echo "${NODELIST}" | tail -n +2); do
                ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${WORKER}" \
                    "'${RAY_BIN:-ray}' stop --force" 2>/dev/null || true
            done
        fi
        "${RAY_BIN:-ray}" stop --force 2>/dev/null || true
    fi
    echo "Removing job-local files across all nodes: ${RAG_JOB_TMP}"
    if [ "${NUM_NODES}" -gt 1 ]; then
        srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" rm -rf -- "${RAG_JOB_TMP}" 2>/dev/null || true
    else
        rm -rf -- "${RAG_JOB_TMP}" 2>/dev/null || true
    fi
}
trap cleanup_job_tmp EXIT

# ==========================================================================
# Environment setup
# ==========================================================================
module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
RAG_PYTHON="${PWD}/.venv/bin/python"
# MoE backend configuration is dynamically configured below after loading RAG_CONFIG

# FlashInfer compilation workspace directory
if [[ "${USE_LOCAL_ENV}" == "true" ]]; then
    # Cached build files can contain an earlier job's /tmp source paths.
    # Isolate all compilation caches before importing vLLM or FlashInfer.
    export VLLM_CACHE_ROOT="${RAG_JOB_TMP}/cache/vllm"
    export TORCHINDUCTOR_CACHE_DIR="${RAG_JOB_TMP}/cache/torchinductor"
    export TRITON_CACHE_DIR="${RAG_JOB_TMP}/cache/triton"
    export FLASHINFER_WORKSPACE_DIR="${RAG_JOB_TMP}/cache/flashinfer"
    export FLASHINFER_WORKSPACE_BASE="${RAG_JOB_TMP}"
    mkdir -p "${FLASHINFER_WORKSPACE_DIR}"
else
    mkdir -p "${PWD}/.cache/flashinfer"
    export FLASHINFER_WORKSPACE_DIR="${PWD}/.cache/flashinfer"
    export FLASHINFER_WORKSPACE_BASE="${PWD}"
fi
export FLASHINFER_DISABLE_VERSION_CHECK=1

if [[ "${USE_LOCAL_ENV}" == "true" ]]; then
    echo "Staging Python environment to local SSD across ${NUM_NODES} node(s)..."
    if [ "${NUM_NODES}" -gt 1 ]; then
        srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" bash -c "
            mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
            python -u hpc/stage_python_env.py .venv "${RAG_JOB_TMP}/venv"
        "
    else
        mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
        python -u hpc/stage_python_env.py .venv "${RAG_JOB_TMP}/venv"
    fi
    RAG_PYTHON="${RAG_JOB_TMP}/venv/bin/python"
fi
# Do not source the copied activate script: it contains the old absolute path.
export VIRTUAL_ENV="${RAG_PYTHON%/bin/python}"
export PATH="${VIRTUAL_ENV}/bin:${PWD}/.venv/bin:${PATH}"
RAY_BIN="${VIRTUAL_ENV}/bin/ray"
echo "Pipeline Python: ${RAG_PYTHON}"
echo "Job-local files: ${RAG_JOB_TMP}"

# Now that Python + PyYAML is available, extract models from config
MODEL_NAME="${VLLM_MODEL:-$(${RAG_PYTHON} -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['model']['model_id'])")}"
MODEL_PATH="${VLLM_MODEL_PATH:-$(${RAG_PYTHON} -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['model'].get('model_path', ''))")}"

JUDGE_MODEL="${VLLM_JUDGE_MODEL:-$(${RAG_PYTHON} -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['evaluation']['model_id'])")}"
JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-$(${RAG_PYTHON} -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}'))['evaluation'].get('model_path', ''))")}"

# Extract distributed configuration
DP_BACKEND="$(${RAG_PYTHON} -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}')).get('model', {}).get('data_parallel_backend', ''))")"
DP_SIZE="$(${RAG_PYTHON} -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}')).get('model', {}).get('data_parallel_size', 1))")"
DP_SIZE_LOCAL="$(${RAG_PYTHON} -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}')).get('model', {}).get('data_parallel_size_local', 4))")"

# MoE backend configuration: detect recipe preference
MOE_BACKEND="$(${RAG_PYTHON} -c "import yaml; print(yaml.safe_load(open('${RAG_CONFIG}')).get('model', {}).get('moe_backend', ''))")"
if [ "${MOE_BACKEND}" = "triton" ] || [ "${MOE_BACKEND}" = "marlin" ]; then
    echo "MoE backend: ${MOE_BACKEND} requested by recipe. Disabling DeepGEMM."
    export VLLM_MOE_USE_DEEP_GEMM=0
    export VLLM_USE_DEEP_GEMM=0
else
    # Default to DeepGEMM to bypass 56-minute CUTLASS JIT compilation where supported
    export VLLM_MOE_USE_DEEP_GEMM=1
    export VLLM_USE_DEEP_GEMM=1
fi

NUM_NODES=${SLURM_NNODES:-1}
REQUIRED_NODES=$(( (DP_SIZE + DP_SIZE_LOCAL - 1) / DP_SIZE_LOCAL ))

# Validate allocation satisfies recipe requirements
if [ "${NUM_NODES}" -lt "${REQUIRED_NODES}" ]; then
    echo "============================================================"
    echo " CONFIG ERROR: Insufficient nodes allocated"
    echo "============================================================"
    echo " ${RAG_CONFIG} requires ${REQUIRED_NODES} node(s) (DP_SIZE=${DP_SIZE}, DP_SIZE_LOCAL=${DP_SIZE_LOCAL})"
    echo " but only ${NUM_NODES} node(s) were allocated."
    echo ""
    echo " Re-submit with:"
    echo "   sbatch --nodes=${REQUIRED_NODES} --cpus-per-task=32 --time=02:00:00 hpc/run_agentic_rag.slurm ${RAG_CONFIG}"
    echo "============================================================"
    exit 1
fi

if [ "${SLURM_CPUS_PER_TASK:-56}" -lt 56 ] && [ "${NUM_NODES}" -gt 1 ]; then
    echo "============================================================"
    echo " CONFIG ERROR: Insufficient CPUs per task (${SLURM_CPUS_PER_TASK:-56} < 56)"
    echo "============================================================"
    echo " Each Capella compute node has 4 H100 GPUs and 56 usable CPU cores (14 cores/GPU)."
    echo " Allocating fewer CPUs causes Slurm cgroups to restrict NUMA socket coverage,"
    echo " which hides GPU 3 (CUDA_VISIBLE_DEVICES=0,1,2) on compute nodes."
    echo ""
    echo " Please submit with --cpus-per-task=56 --exclusive:"
    echo "   sbatch --nodes=${NUM_NODES} --ntasks=${NUM_NODES} --cpus-per-task=56 --mem=0 --gpus-per-node=4 --gres-flags=disable-binding --exclusive --time=01:30:00 hpc/run_agentic_rag.slurm ${RAG_CONFIG}"
    echo "============================================================"
    exit 1
fi

mkdir -p logs

NODELIST=$(scontrol show hostname "${SLURM_JOB_NODELIST:-$(hostname)}")
HEAD_NODE=$(echo "${NODELIST}" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address 2>/dev/null || hostname -I | awk '{print $1}')
RAY_PORT=6379
DP_MASTER_PORT=13345

export HEAD_NODE_IP
export MASTER_ADDR="${HEAD_NODE_IP}"
export DP_MASTER_PORT

echo "============================================================"
echo " Hybrid Agentic RAG Pipeline"
echo "============================================================"
echo " Start time:        $(date)"
echo " Job ID:            ${SLURM_JOB_ID:-manual}"
echo " Head Node:         ${HEAD_NODE} (${HEAD_NODE_IP})"
echo " All Nodes:         $(echo ${NODELIST} | tr '\n' ' ')"
echo " Nodes Allocated:   ${NUM_NODES}"
echo " GPUs per Node:     ${SLURM_GPUS_ON_NODE:-4}"
echo " Total GPUs:        $((NUM_NODES * ${SLURM_GPUS_ON_NODE:-4}))"
echo " Model:             ${MODEL_NAME}"
echo " Judge:             ${JUDGE_MODEL}"
echo " Config:            ${RAG_CONFIG}"
echo " DP Size:           ${DP_SIZE}"
echo " DP Backend:        ${DP_BACKEND:-none}"
echo " MoE Backend:       ${MOE_BACKEND:-default} (VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM})"
echo " NVMe staging:      ${USE_NVME}"
echo " Local Python env:  ${USE_LOCAL_ENV}"
if [ "${DP_BACKEND}" = "ray" ] || [ "${NUM_NODES}" -gt 1 ]; then
    echo " Ray port:          ${RAY_PORT}"
    echo " DP master:         ${HEAD_NODE_IP}:${DP_MASTER_PORT}"
fi
echo "============================================================"
echo ""

# ==========================================================================
# Pre-flight Node Health & Filesystem Accessibility check
# ==========================================================================
if [ "${NUM_NODES}" -gt 1 ]; then
    echo "--- Pre-flight node health & filesystem check ---"
    UNHEALTHY_NODES=""
    CHECK_OUTPUT=$(srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" \
        "${RAG_PYTHON}" -c '
import os, sys, torch
node = os.uname().nodename
pwd_ok = os.path.isdir(os.environ.get("PWD", ""))
py_ok = os.access(sys.executable, os.X_OK)
gpu_cnt = torch.cuda.device_count() if torch.cuda.is_available() else 0
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
print(f"NODE_CHECK|{node}|{pwd_ok and py_ok}|{gpu_cnt}|{cvd}")
' 2>/dev/null || true)

    while IFS='|' read -r TAG NODE FS_OK GPU_COUNT CVD_VAL; do
        if [ "${TAG}" != "NODE_CHECK" ]; then
            continue
        fi
        if [ "${FS_OK}" != "True" ]; then
            echo "  [FAIL] Node ${NODE}: Workspace '${PWD}' or '${RAG_PYTHON}' not accessible"
            UNHEALTHY_NODES="${UNHEALTHY_NODES} ${NODE}"
        else
            echo "  [OK]   Node ${NODE}: Filesystem and Python environment accessible"
        fi

        if [ "${GPU_COUNT}" -lt "${SLURM_GPUS_ON_NODE:-4}" ]; then
            echo "  [FAIL] Node ${NODE}: Only ${GPU_COUNT}/${SLURM_GPUS_ON_NODE:-4} GPUs visible to CUDA (CUDA_VISIBLE_DEVICES=${CVD_VAL})"
            UNHEALTHY_NODES="${UNHEALTHY_NODES} ${NODE}"
        else
            echo "  [OK]   Node ${NODE}: ${GPU_COUNT}/${SLURM_GPUS_ON_NODE:-4} GPUs verified (CUDA_VISIBLE_DEVICES=${CVD_VAL})"
        fi
    done <<< "${CHECK_OUTPUT}" 

    if [ -n "${UNHEALTHY_NODES}" ]; then
        echo ""
        echo "============================================================"
        echo " PRE-FLIGHT NODE HEALTH FAILURE:"
        echo " Unhealthy node(s) detected:${UNHEALTHY_NODES}"
        echo " Either storage is unmounted or fewer than ${SLURM_GPUS_ON_NODE:-4} GPUs are accessible."
        echo ""
        echo " Please re-submit your job excluding the broken nodes:"
        BAD_LIST=$(echo "${UNHEALTHY_NODES}" | xargs | tr ' ' ',')
        echo "   sbatch --exclude=${BAD_LIST} ... hpc/run_agentic_rag.slurm ${RAG_CONFIG}"
        echo "============================================================"
        exit 1
    fi
    echo "--- All allocated nodes passed filesystem check ---"
    echo ""
fi

# ==========================================================================
# Pre-flight ABI / smoke check  (fail in seconds, not after 20 minutes)
# ==========================================================================
# Three version-locked families share one failure mode: upgrading ONE member
# of a pair breaks the whole engine. Each has cost a full job death before.
#   A) torch / torchvision / torchaudio
#   B) flashinfer-python / flashinfer-cubin
#   C) apache-tvm-ffi / tilelang
# Step 1 validates the version pairs. Step 2 imports the real stack, which
# catches ABI breaks that metadata alone cannot see (e.g. Family C aborts via
# C++ terminate() with no Python traceback, so it must be imported to detect).
# Set RAG_SKIP_PREFLIGHT=true to bypass.
# ==========================================================================
if [[ "${RAG_SKIP_PREFLIGHT:-false}" != "true" ]]; then
    PREFLIGHT_LOG="logs/preflight-${SLURM_JOB_ID:-manual}.log"
    echo "--- Pre-flight ABI check ---"

    # --- Step 1: version pairs (metadata only; no heavy imports) ---
    if ! "${RAG_PYTHON}" -u - 2>&1 <<'PY' | tee "${PREFLIGHT_LOG}"
import importlib.metadata as md
import sys

failures = []
notes = []


def get(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def base(v):
    """Strip local version segment: 2.13.0+cu130 -> 2.13.0"""
    return v.split("+")[0] if v else None


def parse(v):
    out = []
    for part in base(v).split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


print("Version pairs:")
for label, dists in (
    ("A torch/torchvision/torchaudio",
     ["torch", "torchvision", "torchaudio"]),
    ("B flashinfer-python/cubin",
     ["flashinfer-python", "flashinfer-cubin"]),
    ("C apache-tvm-ffi/tilelang",
     ["apache-tvm-ffi", "tilelang"]),
):
    shown = "  ".join(f"{d}={get(d) or 'MISSING'}" for d in dists)
    print(f"  [{label.split()[0]}] {shown}")

# --- Family A: torch family must be co-installed and ABI-compatible ---
torch_v = get("torch")
tv_v = get("torchvision")
ta_v = get("torchaudio")
for name, v in (("torch", torch_v), ("torchvision", tv_v), ("torchaudio", ta_v)):
    if v is None:
        failures.append(f"{name} is not installed")
if tv_v and not torch_v:
    failures.append("torchvision installed but torch is missing")


def torch_pin(requirer):
    """Return the exact torch version a package declares in Requires-Dist.

    torchvision/torchaudio record their ABI-coupled torch build here
    (e.g. "torch (==2.13.0)"), which is authoritative for Family A.
    """
    try:
        reqs = md.requires(requirer) or []
    except md.PackageNotFoundError:
        return None
    for req in reqs:
        req = req.strip()
        low = req.lower()
        if not low.startswith("torch"):
            continue
        # Ignore extras/environment markers on other torch subpackages.
        if low.startswith(("torchvision", "torchaudio", "torch-")):
            continue
        if "==" not in req:
            continue
        pinned = req.split("==", 1)[1]
        # Drop any trailing environment marker: "2.13.0 ; extra == 'x'"
        pinned = pinned.split(";", 1)[0]
        # Metadata formats this as "torch (==2.13.0)"; strip parens/space.
        pinned = pinned.strip().strip("()").strip()
        if pinned:
            return pinned
    return None


for requirer in ("torchvision", "torchaudio"):
    if get(requirer) is None or torch_v is None:
        continue
    pinned = torch_pin(requirer)
    if pinned and base(torch_v) != base(pinned):
        failures.append(
            f"{requirer} requires torch=={pinned} but torch {torch_v} is "
            f"installed -- the compiled extension will fail to load symbols"
        )

# --- Family B: flashinfer versions check ---
fp_v = get("flashinfer-python")
fc_v = get("flashinfer-cubin")
if fp_v and fc_v and fp_v != fc_v:
    if os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK") == "1":
        notes.append(
            f"flashinfer-python ({fp_v}) != flashinfer-cubin ({fc_v}), "
            "bypassed via FLASHINFER_DISABLE_VERSION_CHECK=1"
        )
    else:
        failures.append(
            f"flashinfer version mismatch: flashinfer-python={fp_v} vs "
            f"flashinfer-cubin={fc_v} (must be identical, or set FLASHINFER_DISABLE_VERSION_CHECK=1)"
        )
elif fp_v and not fc_v:
    notes.append(f"flashinfer-python ({fp_v}) installed without flashinfer-cubin (JIT mode enabled)")

# --- Family C: tilelang caps apache-tvm-ffi ---
tf_v = get("apache-tvm-ffi")
tl_v = get("tilelang")
if tf_v and tl_v and parse(tf_v) > (0, 1, 11):
    failures.append(
        f"apache-tvm-ffi {tf_v} exceeds the <=0.1.11 that tilelang {tl_v} "
        f"supports -- triggers a C++ abort in tilelang type registration"
    )

for n in notes:
    print(f"  WARNING: {n}")
if failures:
    print("")
    for f in failures:
        print(f"  FAIL: {f}")
    print("  Version-pair check FAILED")
    sys.exit(1)
print("  Version-pair check passed")
PY
    then
        echo ""
        echo "============================================================"
        echo " PRE-FLIGHT FAILED: environment version pairs are inconsistent."
        echo " See ${PREFLIGHT_LOG}"
        echo " This would have died mid-warmup after ~20 minutes."
        echo " Bypass with RAG_SKIP_PREFLIGHT=true (not recommended)."
        echo "============================================================"
        exit 1
    fi

    # --- Step 2: import the real stack (catches ABI breaks metadata misses) ---
    echo "--- Pre-flight import smoke test ---"
    if ! "${RAG_PYTHON}" -u - 2>&1 <<'PY' | tee -a "${PREFLIGHT_LOG}"
import sys

failures = []


def check(label, fn):
    try:
        fn()
        print(f"  ok    {label}")
    except BaseException as e:  # noqa: BLE001 - report everything
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")
        failures.append(label)


def _torch():
    import torch  # noqa: F401


def _torchvision():
    import torch
    import torchvision  # noqa: F401
    # The exact op that broke in Family A.
    assert hasattr(torch.ops.torchvision, "nms"), "torchvision::nms not registered"


def _transformers():
    import transformers.image_utils  # noqa: F401


def _flashinfer():
    import flashinfer  # noqa: F401


def _tilelang():
    # Family C canary. A too-new apache-tvm-ffi makes this import abort the
    # process via C++ terminate() (SIGABRT) with no Python traceback, so it
    # runs in a subprocess where we can detect the signal explicitly.
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-c", "import tilelang"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {proc.returncode}"
        raise RuntimeError(tail)


def _vllm():
    from vllm import LLM, SamplingParams  # noqa: F401


def _vllm_ext():
    import vllm._C_stable_libtorch
    import vllm._moe_C_stable_libtorch


for label, fn in (
    ("torch", _torch),
    ("torchvision + nms op", _torchvision),
    ("transformers.image_utils", _transformers),
    ("flashinfer", _flashinfer),
    ("tilelang", _tilelang),
    ("vllm", _vllm),
    ("vllm CUDA extensions", _vllm_ext),
):
    check(label, fn)

if failures:
    print("")
    print(f"  Import smoke test FAILED ({len(failures)} module(s))")
    sys.exit(1)
print("  Import smoke test passed")
PY
    then
        echo ""
        echo "============================================================"
        echo " PRE-FLIGHT FAILED: the vLLM kernel stack does not import."
        echo " See ${PREFLIGHT_LOG}"
        echo " Likely an ABI mismatch inside a version-locked pair"
        echo " (torch family, flashinfer, or apache-tvm-ffi/tilelang)."
        echo " Bypass with RAG_SKIP_PREFLIGHT=true (not recommended)."
        echo "============================================================"
        exit 1
    fi
    echo "--- Pre-flight passed (log: ${PREFLIGHT_LOG}) ---"
    echo ""
else
    echo "--- Pre-flight ABI check SKIPPED (RAG_SKIP_PREFLIGHT=true) ---"
    echo ""
fi

# ==========================================================================
# Performance & Communication environment variables
# ==========================================================================
export OMP_NUM_THREADS=1                                 # Avoid CPU thread contention
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1        # Better KV cache sizing
export NCCL_ASYNC_ERROR_HANDLING=1                       # Crash on NCCL hangs
export TOKENIZERS_PARALLELISM=false                      # Suppress tokenizer warnings
export TORCH_FLOAT32_MATMUL_PRECISION=high               # TF32 on H100s
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce GPU memory fragmentation
export VLLM_WORKER_MULTIPROC_METHOD=spawn                # Explicitly request spawn to quiet CUDA initialization warnings
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200           # 20m execution timeout for long thinking traces & multi-node MoE all2all
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200             # Watchdog timeout matching model execution timeout
export NCCL_BUFFSIZE=16777216                            # 16MB buffer for high-throughput all2all EP over InfiniBand
export NCCL_SOCKET_IFNAME=ibp3s0.0001,ibp,eno4           # Prioritize InfiniBand IPoIB subnet (172.24.192.0/20); avoid container interfaces
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5           # Bind only to active 200Gb/s HCAs (mlx5_0 and mlx5_1 are PORT_DOWN on Capella)
export NCCL_CROSS_NIC=1                                  # Enable multi-HCA routing across Hopper SXM5 rails
export NCCL_DEBUG=WARN                                   # Log actionable NCCL warnings without spamming
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
export VLLM_RAY_DP_PACK_STRATEGY=strict

# ==========================================================================
# Start Ray cluster across nodes (only when Ray backend is configured)
# ==========================================================================
RAY_STARTED=false
if [ "${DP_BACKEND}" = "ray" ] || { [ "${NUM_NODES}" -gt 1 ] && [ "${DP_SIZE}" -gt 1 ]; }; then
    RAY_STARTED=true

    echo "Starting Ray head on ${HEAD_NODE} (${HEAD_NODE_IP}:${RAY_PORT})..."
    "${RAY_BIN}" start --head \
        --node-ip-address="${HEAD_NODE_IP}" \
        --port=${RAY_PORT} \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "${SLURM_GPUS_ON_NODE:-4}" \
        --disable-usage-stats

    if [ "${NUM_NODES}" -gt 1 ]; then
        WORKER_NODES=$(echo "${NODELIST}" | tail -n +2)
        echo "Starting Ray worker(s) on ${WORKER_NODES} via SSH daemonization..."

        # Forward critical environment variables (local NVMe caches, InfiniBand NICs, timeouts) to remote Ray workers
        RAY_WORKER_ENV="export PATH='${PATH}'; \
export VIRTUAL_ENV='${VIRTUAL_ENV}'; \
export VLLM_CACHE_ROOT='${VLLM_CACHE_ROOT}'; \
export TORCHINDUCTOR_CACHE_DIR='${TORCHINDUCTOR_CACHE_DIR}'; \
export TRITON_CACHE_DIR='${TRITON_CACHE_DIR}'; \
export FLASHINFER_WORKSPACE_DIR='${FLASHINFER_WORKSPACE_DIR}'; \
export FLASHINFER_WORKSPACE_BASE='${FLASHINFER_WORKSPACE_BASE}'; \
export FLASHINFER_DISABLE_VERSION_CHECK=1; \
export OMP_NUM_THREADS=1; \
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1; \
export NCCL_ASYNC_ERROR_HANDLING=1; \
export TOKENIZERS_PARALLELISM=false; \
export TORCH_FLOAT32_MATMUL_PRECISION=high; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
export VLLM_WORKER_MULTIPROC_METHOD=spawn; \
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200; \
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200; \
export NCCL_BUFFSIZE=16777216; \
export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'; \
export NCCL_IB_HCA='${NCCL_IB_HCA}'; \
export NCCL_CROSS_NIC=1; \
export NCCL_DEBUG=WARN; \
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV; \
export VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-0}; \
export VLLM_RAY_DP_PACK_STRATEGY=strict;"

        for WORKER in ${WORKER_NODES}; do
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${WORKER}" \
                "${RAY_WORKER_ENV} CUDA_VISIBLE_DEVICES=0,1,2,3 '${RAY_BIN}' start \
                    --address='${HEAD_NODE_IP}:${RAY_PORT}' \
                    --num-cpus='${SLURM_CPUS_PER_TASK:-56}' \
                    --num-gpus='${SLURM_GPUS_ON_NODE:-4}' \
                    --disable-usage-stats"
        done
    fi

    echo "Waiting for Ray cluster to settle (${NUM_NODES} nodes)..."
    for _ in {1..15}; do
        ACTIVE_NODES=$("${RAY_BIN}" status 2>/dev/null | grep -E "Active:" -A 10 | grep -c "node_" || true)
        if [ "${ACTIVE_NODES}" -ge "${NUM_NODES}" ]; then
            break
        fi
        sleep 1
    done
    "${RAY_BIN}" status
else
    echo "Ray cluster not required (single-node non-Ray setup)."
fi

# ==========================================================================
# NVMe Model Staging
# ==========================================================================

resolve_model_source() {
    # Find the actual model directory given a model name and optional path
    local MODEL_ID="$1"
    local EXPLICIT_PATH="$2"

    if [ -n "${EXPLICIT_PATH}" ] && [ -d "${EXPLICIT_PATH}" ]; then
        if compgen -G "${EXPLICIT_PATH}/*.safetensors" >/dev/null || compgen -G "${EXPLICIT_PATH}/*.bin" >/dev/null || compgen -G "${EXPLICIT_PATH}/*.pt" >/dev/null; then
            echo "${EXPLICIT_PATH}"
            return
        fi
        echo "  [WARNING] Explicit path '${EXPLICIT_PATH}' does not contain weight files (*.safetensors, *.bin, *.pt)." >&2
    fi

    # Lookup snapshot directory in HuggingFace cache directly
    local HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
    local REPO_DIR="${HF_CACHE}/models--${MODEL_ID//\//--}"
    local CAND=""
    if [ -f "${REPO_DIR}/refs/main" ]; then
        local COMMIT
        COMMIT=$(head -n 1 "${REPO_DIR}/refs/main" 2>/dev/null)
        if [ -n "${COMMIT}" ] && [ -d "${REPO_DIR}/snapshots/${COMMIT}" ]; then
            CAND="${REPO_DIR}/snapshots/${COMMIT}"
        fi
    fi

    if [ -z "${CAND}" ] && [ -d "${REPO_DIR}/snapshots" ]; then
        for d in "${REPO_DIR}/snapshots"/*/; do
            if [ -d "${d}" ]; then
                CAND="${d%/}"
                break
            fi
        done
    fi

    # Crucial validation: ensure the snapshot actually contains model weight files
    # and has no active or incomplete downloads in blobs/
    if [ -n "${CAND}" ] && [ -d "${CAND}" ]; then
        if compgen -G "${REPO_DIR}/blobs/*.incomplete" >/dev/null; then
            echo "  [WARNING] Model '${MODEL_ID}' has active or incomplete downloads (*.incomplete) in blobs/. Skipping incomplete model." >&2
            echo ""
            return
        fi
        if compgen -G "${CAND}/*.safetensors" >/dev/null || compgen -G "${CAND}/*.bin" >/dev/null || compgen -G "${CAND}/*.pt" >/dev/null; then
            echo "${CAND}"
            return
        else
            echo "  [WARNING] Snapshot '${CAND}' contains no model weight files (*.safetensors, *.bin, *.pt)." >&2
        fi
    fi

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
        return 1
    fi

    if ! compgen -G "${SRC}/*.safetensors" >/dev/null && ! compgen -G "${SRC}/*.bin" >/dev/null && ! compgen -G "${SRC}/*.pt" >/dev/null; then
        echo "  [ERROR] Source directory '${SRC}' for ${BASENAME} contains no weight files (*.safetensors, *.bin, *.pt). Skipping NVMe staging."
        return 1
    fi

    echo "Staging ${BASENAME} to NVMe SSD ..."
    echo "  Source:  ${SRC}"
    echo "  Dest:    ${DEST}"
    local T0=$(date +%s)
    if [ "${NUM_NODES}" -gt 1 ]; then
        srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" bash -c "
            mkdir -p '${DEST}'
            rsync -aL '${SRC}/' '${DEST}/'
            echo \"\$(hostname): Copied in \$(du -sh '${DEST}' | cut -f1)\"
        "
    else
        mkdir -p "${DEST}"
        rsync -aL --info=progress2 "${SRC}/" "${DEST}/"
        echo "$(hostname): Copied in $(du -sh "${DEST}" | cut -f1)"
    fi
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
    else
        echo "  [ERROR] No valid local model weights found for main model '${MODEL_NAME}'."
        echo "  Ensure the model is fully downloaded via: sbatch hpc/download_model.slurm ${MODEL_NAME}"
        exit 1
    fi

    # Stage judge model (if different from main)
    if [ "${JUDGE_MODEL}" != "${MODEL_NAME}" ]; then
        JUDGE_SRC=$(resolve_model_source "${JUDGE_MODEL}" "${JUDGE_MODEL_PATH}")
        if [ -n "${JUDGE_SRC}" ]; then
            stage_to_nvme "${JUDGE_SRC}" "$(basename ${JUDGE_MODEL})" "VLLM_JUDGE_MODEL_PATH"
        else
            echo "  [WARNING] No complete local snapshot found for judge model '${JUDGE_MODEL}'. Skipping NVMe staging for judge."
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
if [ "${NUM_NODES}" -gt 1 ] && [ "${DP_SIZE}" -gt 1 ]; then
    echo "Launching multi-node pipeline across ${NUM_NODES} nodes (${NUM_NODES} tasks total, 1 task/node)..."
    srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" \
        "${RAG_PYTHON}" -u agentic_rag.py --config "${RAG_CONFIG}" "$@" || EXIT_CODE=$?
else
    echo "Launching single-node pipeline on ${HEAD_NODE}..."
    "${RAG_PYTHON}" -u agentic_rag.py --config "${RAG_CONFIG}" "$@" || EXIT_CODE=$?
fi

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
