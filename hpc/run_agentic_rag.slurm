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
#
# A pre-flight ABI check runs before model staging and fails fast (seconds)
# if a version-locked package pair is inconsistent. Bypass with:
#   RAG_SKIP_PREFLIGHT=true sbatch hpc/run_agentic_rag.slurm
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

# --- Family B: flashinfer versions must match EXACTLY ---
fp_v = get("flashinfer-python")
fc_v = get("flashinfer-cubin")
if fp_v and fc_v and fp_v != fc_v:
    failures.append(
        f"flashinfer version mismatch: flashinfer-python={fp_v} vs "
        f"flashinfer-cubin={fc_v} (must be identical)"
    )

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
