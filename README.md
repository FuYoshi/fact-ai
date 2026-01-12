# FCGrad

Factored Gradient methods for multi-agent reinforcement learning with JAX.

## Quick Start

The environment is **already installed** at `~/.venvs/fcgrad`. Just run your code!

### Running a Batch Job

Create a job script (e.g., `run_experiment.job`):

```bash
#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=FCGrad
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%A.out

module purge
module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0
source ~/.venvs/fcgrad/bin/activate

# Required for JAX GPU compilation
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

cd $HOME/simon/fact_fcgrad
python main.py
```

Submit with: `sbatch run_experiment.job`

### Interactive Session (for testing)

```bash
# Request a GPU node
srun --partition=gpu_mig --gpus=1 --time=01:00:00 --pty bash

# Load modules and activate
module purge
module load 2024 CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 Python/3.12.3-GCCcore-13.3.0
source ~/.venvs/fcgrad/bin/activate
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

# Run your code
python main.py
```

### Verify GPU Works

```python
import jax
print(jax.devices())  # Should show [cuda(id=0)]
```

## Environment Details

| Package | Version |
|---------|---------|
| JAX | 0.4.30 |
| Flax | 0.8.5 |
| JaxMarl | 0.0.7 |
| Hydra | 1.3.2 |

## Reinstallation (if needed)

Only run this if the environment is broken:

```bash
sbatch install_working.job
```

## Troubleshooting

### JAX not detecting GPU

Make sure you:
1. Are on a GPU node (not login node)
2. Loaded all modules **before** activating the venv

### Pip upgraded JAX and broke things

```bash
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
pip install "jax[cuda12_local]==0.4.30"
```
