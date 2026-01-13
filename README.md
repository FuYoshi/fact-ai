# FCGrad

Implementation of "Fair Cooperation in Mixed-Motive Games via Conflict-Aware Gradient Adjustment" (Kim & Sycara, CMU).

## Quick Start

The environment is **already installed** at `~/.venvs/fcgrad`. Just run your code!

## Running Experiments

### Interactive Session (Recommended for Testing)

Use the helper script for easy setup:

```bash
# Step 1: Request GPU node (choose time based on config)
srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=9 --time=01:00:00 --pty bash -i  # 1hr for test
srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=9 --time=04:00:00 --pty bash -i  # 4hr for full

# Step 2: Run with helper script
source ~/simon/fact_fcgrad/run_interactive.sh                    # Test config (fast)
source ~/simon/fact_fcgrad/run_interactive.sh ippo_unfair_coin   # Full IPPO baseline
source ~/simon/fact_fcgrad/run_interactive.sh fcgrad_unfair_coin # FCGrad
```

**Available Configs:**

| Config | Minibatches | Timesteps | JIT Time | Total Time |
|--------|-------------|-----------|----------|------------|
| `ippo_unfair_coin_test` | 4 | 1M | ~2-5 min | ~10 min |
| `ippo_unfair_coin` | 500 | 1B | ~1-4 hr | ~24 hr |
| `fcgrad_unfair_coin` | 500 | 1B | ~1-4 hr | ~24 hr |

⚠️ **Note:** With 500 minibatches, JAX JIT compilation takes 1-4 hours before training starts. The progress bar will stay at 0% during compilation.

### Batch Job (For Full Training)

For full training runs, use a batch job with sufficient time:

```bash
#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=IPPO_Unfair
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=24:00:00          # 24 hours for full training
#SBATCH --output=slurm_%A_%a.out
#SBATCH --array=0-3              # 4 seeds

module purge
module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0
source ~/.venvs/fcgrad/bin/activate
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

cd $HOME/simon/fact_fcgrad

# Map array task to seed
SEEDS=(42 52 62 72)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python main.py --config ippo_unfair_coin --seed $SEED
```

Submit with: `sbatch run_experiment.job`

## Paper Hyperparameters (Unfair Coin Game)

From Appendix B of the FCGrad paper:

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (linear annealing) |
| Parallel envs | 256 |
| Steps per rollout | 1000 |
| Total timesteps | 1e9 |
| Update epochs | 2 |
| Minibatches | 500 |
| Discount (γ) | 0.99 |
| GAE (λ) | 0.95 |
| Clip ε | 0.2 |
| Entropy coef | 0.1 |
| Value coef | 0.1 |
| Grad clip | 0.5 |
| Green coin prob | 0.9375 (15/16) |

## Environment Details

| Package | Version |
|---------|---------|
| JAX | 0.4.30 |
| Flax | 0.8.5 |
| SocialJax | local |
| Hydra | 1.3.2 |

## Cost Estimation (Snellius)

| Config | Time | SBUs per seed | 4 seeds |
|--------|------|---------------|---------|
| Test | ~10 min | ~11 | ~44 |
| Full | ~24 hr | ~1,536 | ~6,144 |

## Learning Curves

Training automatically saves CSV logs to `./logs/`. Plot them with:

```bash
source ~/.venvs/fcgrad/bin/activate
pip install pandas matplotlib  # if not already installed
python plot_learning_curve.py --log-dir ./logs
```

This creates `./logs/learning_curves.png` showing all runs.

## Troubleshooting

### JAX not detecting GPU
1. Make sure you're on a GPU node (not login node)
2. Load modules **before** activating the venv

### JIT compilation takes forever
With 500 minibatches, JIT takes 1-4 hours. This is expected per the paper's hyperparameters.

### Reinstall environment (if broken)
```bash
sbatch install_working.job
```
