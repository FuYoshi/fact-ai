"""
FCGrad - Main entry point for running IPPO/FCGrad experiments

FCGrad: Fair Conflict-aware Gradient Adjustment
Reproducibility study of Kim & Sycara (CMU) paper on the Unfair Coin Game.
"""
import sys
import os
import argparse

# Add project root to Python path for socialjax imports
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# Add IPPO directory for fcgrad_utils import
ippo_dir = os.path.join(project_root, 'algorithms', 'IPPO')
sys.path.insert(0, ippo_dir)


def main():
    parser = argparse.ArgumentParser(description='Run IPPO/FCGrad on Unfair Coin Game')
    parser.add_argument(
        '--config',
        type=str,
        default='ippo_unfair_coin_ind',
        help='Config name (e.g., ippo_unfair_coin_ind, ippo_unfair_coin_col, fcgrad_unfair_coin)'
    )
    parser.add_argument(
        '--total-timesteps',
        type=float,
        default=None,
        help='Total timesteps (e.g., 1e6 for quick test, 1e9 for full run)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--fcgrad',
        action='store_true',
        help='Enable FCGrad gradient adjustment (overrides config)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run hyperparameter tuning instead of single run'
    )
    parser.add_argument(
        '--num-minibatches',
        type=int,
        default=None,
        help='Override NUM_MINIBATCHES (default: from config)'
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=None,
        help='Override NUM_ENVS (default: from config)'
    )

    args = parser.parse_args()

    # Use custom config if provided, otherwise default
    config_name = args.config
    # Remove .yaml if user included it
    if config_name.endswith('.yaml'):
        config_name = config_name[:-5]

    algo_name = "FCGrad" if args.fcgrad else "IPPO"
    print(f"Running {algo_name} on Unfair Coin Game")
    print(f"Config: {config_name}")
    print(f"Working directory: {os.getcwd()}")
    print()

    try:
        # Import Hydra and required modules
        from hydra import compose
        from omegaconf import OmegaConf
        import importlib.util

        # Load the IPPO script module from file
        script_path = os.path.join(ippo_dir, 'ippo_cnn_coins.py')
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        spec = importlib.util.spec_from_file_location('ippo_cnn_coins', script_path)
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)

        # Initialize Hydra using initialize_config_dir with absolute path
        config_dir = os.path.abspath(os.path.join(ippo_dir, 'config'))
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        # Clear any existing Hydra instance to avoid conflicts
        GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Compose the config
            cfg = compose(config_name=config_name)

            # Override config values from command line
            if args.total_timesteps is not None:
                cfg['TOTAL_TIMESTEPS'] = args.total_timesteps
            if args.seed is not None:
                cfg['SEED'] = args.seed
            if args.tune:
                cfg['TUNE'] = True
            if args.fcgrad:
                cfg['FCGRAD'] = True
            if args.num_minibatches is not None:
                cfg['NUM_MINIBATCHES'] = args.num_minibatches
            if args.num_envs is not None:
                cfg['NUM_ENVS'] = args.num_envs
            # Override WANDB_MODE from environment variable if set.
            if 'WANDB_MODE' in os.environ:
                cfg['WANDB_MODE'] = os.environ['WANDB_MODE']

            # Run the training
            if cfg.get('TUNE', False):
                script_module.tune(cfg)
            else:
                script_module.single_run(cfg)

    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("\nMake sure dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running IPPO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
