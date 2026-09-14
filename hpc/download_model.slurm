#!/bin/bash
#SBATCH --job-name=download-model
#SBATCH --output=logs/download-model/%j.out
#SBATCH --error=logs/download-model/%j.err
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=01:00:00

# Usage: sbatch download_model.slurm <model_id>

if [ -z "$1" ]; then
    echo "Error: Please provide a model ID."
    echo "Usage: $0 <model_id>"
    exit 1
fi

MODEL_ID=$1

module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3
if [ ! -d ".venv" ]; then
    python -m venv --system-site-packages .venv
fi
source .venv/bin/activate

echo "Downloading model: ${MODEL_ID}"
hf download "${MODEL_ID}" --exclude "*.bin"