# FCGrad: A Reproducibility Study

Implementation and reproducibility study of **"Fair Cooperation in Mixed-Motive Games via Conflict-Aware Gradient Adjustment"** by Woojun Kim and Katia Sycara (Carnegie Mellon University).

This repository contains a JAX-based implementation of the FCGrad algorithm applied to the **Unfair Coin Game** environment, as described in the paper.

## Repository Structure

```
fact-ai/
├── main.py                          # Main entry point
├── README.md                        # This file
├── pyproject.toml                   # Project configuration
├── requirements.txt                 # Python dependencies
├── jobs/                            # SLURM job templates
├── algorithms/IPPO/                 # Training algorithm
│   ├── ippo_cnn_coins.py            # Main training script
│   ├── fcgrad_utils.py              # FCGrad implementation
│   ├── fairness_metrics.py          # Metrics computation
│   └── config/                      # Hydra configs (6 configs)
└── socialjax/                       # Environment implementation
    ├── environments/
    │   └── coin_game/               # Unfair Coin Game environment
    ├── registration.py
    └── wrappers/
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA 12.x (for GPU acceleration)
- JAX with CUDA support

### Setup

```bash
# Clone the repository
git clone <repo-url> fact-ai
cd fact-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify JAX GPU support
python -c "import jax; print(jax.devices())"
```

### Dependencies

| Package | Version |
|---------|---------|
| JAX | 0.4.30+ |
| Flax | 0.8.5+ |
| Optax | 0.2.2+ |
| Hydra | 1.3.2+ |
| WandB | 0.16+ |

## Running Experiments

### Quick Start

```bash
# Run IPPO with individual rewards (baseline)
python main.py --config ippo_unfair_coin_ind

# Run IPPO with collective rewards (baseline)
python main.py --config ippo_unfair_coin_col

# Run FCGrad
python main.py --config fcgrad_unfair_coin
```

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --config NAME       Config file name (without .yaml extension)
  --total-timesteps N Override total timesteps (e.g., 1e6 for quick test)
  --seed N            Random seed
  --fcgrad            Enable FCGrad (overrides config)
  --tune              Run hyperparameter tuning
```

### Examples

```bash
# Quick test run (1M timesteps)
python main.py --config ippo_unfair_coin_ind --total-timesteps 1e6

# Full training with specific seed
python main.py --config fcgrad_unfair_coin --seed 42

# 3-agent experiment
python main.py --config fcgrad_unfair_coin_3agents_711
```

## Configuration Files

| Config | Method | Agents | Rewards | Description |
|--------|--------|--------|---------|-------------|
| `ippo_unfair_coin_ind` | IPPO | 2 | Individual | Baseline with individual rewards |
| `ippo_unfair_coin_col` | IPPO | 2 | Collective | Baseline with collective rewards |
| `fcgrad_unfair_coin` | FCGrad | 2 | Individual | Main FCGrad method |
| `ippo_unfair_coin_3agents_711_ind` | IPPO | 3 | Individual | 7:1:1 distribution, individual |
| `ippo_unfair_coin_3agents_711_col` | IPPO | 3 | Collective | 7:1:1 distribution, collective |
| `fcgrad_unfair_coin_3agents_711` | FCGrad | 3 | Individual | 7:1:1 distribution, FCGrad |

## Hyperparameters

From Appendix B of the FCGrad paper:

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (linear annealing) |
| Parallel environments | 256 |
| Steps per rollout | 1000 |
| Total timesteps | 1e9 |
| Update epochs | 2 |
| Minibatches | 500 |
| Discount (gamma) | 0.99 |
| GAE (lambda) | 0.95 |
| Clip epsilon | 0.2 |
| Entropy coefficient | 0.1 |
| Value coefficient | 0.1 |
| Max gradient norm | 0.5 |
| FCGrad beta | 0.5 |

## Environment: Unfair Coin Game

The Unfair Coin Game is a two-player mixed-motive game where:

- **Grid**: 16x11 cells
- **Agents**: 2 (or 3 for extended experiments)
- **Coins**: Spawn with biased probabilities
  - 2-agent: 15/16 green (agent 0's color), 1/16 red (agent 1's color)
  - 3-agent (7:1:1): 77.78% agent 0, 11.11% each for agents 1 and 2

**Reward Structure**:
- Picking up any coin: +1 reward
- Your coin picked by another agent: -2 penalty

This creates a dilemma where the optimal individual strategy (collect all coins) conflicts with fair cooperation (only collect your own coins).

## FCGrad Algorithm

FCGrad (Fair Conflict-aware Gradient Adjustment) addresses unfairness in multi-agent learning by:

1. **Dual Value Heads**: Maintains two value functions:
   - V_ind: Estimates individual returns
   - V_col: Estimates collective (average) returns

2. **Conflict Detection**: Identifies when individual and collective gradients conflict

3. **Gradient Adjustment**: When conflicts occur, adjusts the policy gradient to balance individual and collective objectives using the beta parameter

## Metrics

Key metrics tracked during training:

| Metric | Description |
|--------|-------------|
| `episode_return` | Total reward per episode |
| `eat_own_coins` | Number of own-colored coins collected |
| `fairness_gini` | Gini coefficient of rewards (0 = perfect equality) |
| `fairness_jain` | Jain's fairness index (1 = perfect fairness) |
| `efficiency` | Total coins collected / maximum possible |

## SLURM Job Submission

For HPC clusters with SLURM:

```bash
# Submit a single job
sbatch jobs/run_ippo_unfair_coin_ind.job

# Submit with seed override
sbatch --export=SEED=52 jobs/run_ippo_unfair_coin_ind.job
```

## Logging

Training logs to Weights & Biases by default. To disable:

```bash
export WANDB_MODE=disabled
python main.py --config ippo_unfair_coin_ind
```

## Troubleshooting

### JAX not detecting GPU
1. Ensure you're on a GPU node (not login node)
2. Load CUDA modules before activating the virtual environment

### Memory issues
Reduce `NUM_ENVS` or `NUM_MINIBATCHES` in the config file.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{kim2024fcgrad,
  title={Fair Cooperation in Mixed-Motive Games via Conflict-Aware Gradient Adjustment},
  author={Kim, Woojun and Sycara, Katia},
  journal={arXiv preprint arXiv:2402.xxxxx},
  year={2024}
}
```

## License

This project is for academic research purposes.
