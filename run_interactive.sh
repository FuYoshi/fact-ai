#!/bin/bash
# =============================================================================
# Interactive Session Script for FCGrad/IPPO Testing
# =============================================================================
# Usage:
#   1. Start interactive session first:
#      srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=9 --time=01:00:00 --pty bash -i
#   
#   2. Then run this script:
#      source ~/simon/fact_fcgrad/run_interactive.sh
#
#   Or run a specific config:
#      source ~/simon/fact_fcgrad/run_interactive.sh ippo_unfair_coin
# =============================================================================

# Get config name from argument (default: test config)
CONFIG_NAME="${1:-ippo_unfair_coin_test}"

echo "=============================================="
echo "  FCGrad Interactive Session Setup"
echo "=============================================="

# Load modules
echo "📦 Loading modules..."
module purge
module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source ~/.venvs/fcgrad/bin/activate

# Set XLA flags for JAX GPU
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

# Verify setup
echo ""
echo "🔍 Verifying JAX GPU setup..."
python -c "import jax; print(f'   JAX {jax.__version__}, Backend: {jax.default_backend()}, Devices: {jax.devices()}')"

# Navigate to project
cd ~/simon/fact_fcgrad

echo ""
echo "=============================================="
echo "  Running: ${CONFIG_NAME}"
echo "=============================================="
echo ""

# Run training
python main.py --config "${CONFIG_NAME}"

echo ""
echo "✅ Training complete!"
