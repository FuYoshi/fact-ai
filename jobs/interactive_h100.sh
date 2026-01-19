#!/bin/bash
# =============================================================================
# Interactive H100 GPU Session for IPPO Training
# =============================================================================
# Usage: bash jobs/interactive_h100.sh
# Or: srun --partition=gpu --gres=gpu:h100:1 --cpus-per-task=9 --time=02:00:00 --pty bash
#
# Note: Partition name may vary (gpu, gpu_h100, h100, etc.)
# Adjust --partition and --gres flags as needed for your cluster
# =============================================================================

echo "=============================================="
echo "  Requesting Interactive H100 GPU Session"
echo "=============================================="
echo ""

# Use gpu_h100 partition (found via sinfo)
PARTITION="gpu_h100"
GPU_TYPE="h100:1"  # Request 1 H100 GPU

echo "Using partition: $PARTITION"
echo "GPU type: $GPU_TYPE"
echo ""

# Request interactive session
srun \
  --partition="${PARTITION}" \
  --gres="gpu:${GPU_TYPE}" \
  --ntasks=1 \
  --cpus-per-task=9 \
  --time=02:00:00 \
  --pty \
  bash -c "
    echo '============================================'
    echo '  Interactive H100 Session Started'
    echo '============================================'
    echo 'Job ID: \$SLURM_JOB_ID'
    echo 'Node: \$SLURMD_NODENAME'
    echo ''
    
    # Check GPU
    echo '🔍 Checking GPU...'
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    
    # Load modules
    echo ''
    echo '📦 Loading modules...'
    module purge
    module load 2024
    module load CUDA/12.6.0
    module load cuDNN/9.5.0.50-CUDA-12.6.0
    module load Python/3.12.3-GCCcore-13.3.0
    
    # Activate virtual environment
    echo '🐍 Activating virtual environment...'
    source ~/.venvs/fcgrad/bin/activate
    
    # Set XLA flags for JAX GPU
    export XLA_FLAGS=\"--xla_gpu_cuda_data_dir=\$CUDA_HOME\"
    
    # Verify setup
    echo ''
    echo '🔍 Verifying JAX GPU setup...'
    python -c \"import jax; print(f'   JAX {jax.__version__}, Backend: {jax.default_backend()}, Devices: {jax.devices()}')\"
    
    # Navigate to IPPO directory
    cd ~/simon/fact_fcgrad/external/SocialJax/algorithms/IPPO
    
    echo ''
    echo '============================================'
    echo '  Ready for Interactive Commands'
    echo '============================================'
    echo ''
    echo 'Quick start commands:'
    echo '  python ippo_cnn_coins.py --config-name=fcgrad_unfair_coin_fast'
    echo '  python ippo_cnn_coins.py --config-name=ippo_unfair_coin_col_fast'
    echo '  python ippo_cnn_coins.py --config-name=ippo_unfair_coin_ind_quick'
    echo ''
    echo 'Or use main.py from project root:'
    echo '  cd ~/simon/fact_fcgrad'
    echo '  python main.py --config fcgrad_unfair_coin_fast'
    echo ''
    
    # Start interactive bash
    exec bash
  "
