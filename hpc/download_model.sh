#!/bin/bash
# Run this on the HPC cluster to set up the vLLM environment
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
python -m venv --system-site-packages .venv
source .venv/bin/activate
hf download Qwen/Qwen3.5-122B-A10B-FP8 --exclude "*.bin"
#huggingface-cli download zai-org/GLM-4.7-FP8 --exclude "*.bin"