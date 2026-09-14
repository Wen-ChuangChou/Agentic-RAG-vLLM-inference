#!/bin/bash
# Run this on the HPC cluster to set up the vLLM environment
# Usage: bash hpc/download_model.sh <model_id>

if [ -z "$1" ]; then
    echo "Error: Please provide a model ID."
    echo "Usage: $0 <model_id>"
    exit 1
fi

MODEL_ID=$1

module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
if [ ! -d ".venv" ]; then
    python -m venv --system-site-packages .venv
fi
source .venv/bin/activate

echo "Downloading model: ${MODEL_ID}"
hf download "${MODEL_ID}" --exclude "*.bin"