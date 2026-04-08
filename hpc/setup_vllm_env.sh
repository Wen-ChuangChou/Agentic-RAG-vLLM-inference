#!/bin/bash
# Run this on the HPC cluster to set up the vLLM environment
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install uv
# Install vLLM and project dependencies needed on the client side
uv pip install -r requirement.txt --index-strategy unsafe-best-match
echo "vLLM environment ready."
echo "Test with: python -c 'import vllm; print(vllm.__version__)'"