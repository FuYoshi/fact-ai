#!/bin/bash
# =============================================================================
# Interactive GPU MIG Session for IPPO Training
# =============================================================================
# Usage: bash jobs/interactive_gpu_mig.sh
# Or: srun --partition=gpu_mig --gpus=1 --cpus-per-task=9 --time=02:00:00 --pty bash

echo "=============================================="
echo "  Requesting Interactive GPU MIG Session"
echo "=============================================="
echo ""

# Request interactive session
srun \
  --partition=gpu_mig \
  --gpus=1 \
  --ntasks=1 \
  --cpus-per-task=9 \
  --time=02:00:00 \
  --pty \
  bash -c "
    echo '============================================'
    echo '  Interactive Session Started'
    echo '============================================'
    echo 'Job ID: \$SLURM_JOB_ID'
    echo 'Node: \$SLURMD_NODENAME'
    echo ''
    
    # Load modules
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
    echo '  python ippo_cnn_coins.py --config-name=ippo_unfair_coin_ind_quick'
    echo '  python ippo_cnn_coins.py --config-name=ippo_unfair_coin_ind'
    echo ''
    echo 'Or use main.py from project root:'
    echo '  cd ~/simon/fact_fcgrad'
    echo '  python main.py --config ippo_unfair_coin_ind_quick'
    echo ''
    
    # Start interactive bash
    exec bash
  "
