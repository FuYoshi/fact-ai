"""
FCGrad - Main entry point for running IPPO baseline from SocialJax
"""
import sys
import os
import argparse

# Add SocialJax to Python path
socialjax_path = os.path.join(os.path.dirname(__file__), 'external', 'SocialJax')
if os.path.exists(socialjax_path):
    sys.path.insert(0, socialjax_path)

# Get IPPO directory path
ippo_dir = os.path.join(socialjax_path, 'algorithms', 'IPPO')


def main():
    parser = argparse.ArgumentParser(description='Run IPPO baseline from SocialJax')
    parser.add_argument(
        '--env',
        type=str,
        default='coins',
        choices=['coins', 'cleanup', 'harvest_common', 'coop_mining', 'gift', 
                 'mushrooms', 'pd_arena', 'territory_open', 'harvest_common_closed',
                 'harvest_common_partnership'],
        help='Environment to run (default: coins)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Custom config name (e.g., ippo_cnn_coins_sanity for sanity check)'
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
        '--tune',
        action='store_true',
        help='Run hyperparameter tuning instead of single run'
    )
    
    args = parser.parse_args()
    
    # Map environment name to script name
    env_to_script = {
        'coins': 'ippo_cnn_coins',
        'cleanup': 'ippo_cnn_cleanup',
        'harvest_common': 'ippo_cnn_harvest_common',
        'coop_mining': 'ippo_cnn_coop_mining',
        'gift': 'ippo_cnn_gift',
        'mushrooms': 'ippo_cnn_mushrooms',
        'pd_arena': 'ippo_cnn_pd_arena',
        'territory_open': 'ippo_cnn_territory_open',
        'harvest_common_closed': 'ippo_cnn_harvest_common_closed',
        'harvest_common_partnership': 'ippo_cnn_harvest_common_partnership',
    }
    
    script_name = env_to_script[args.env]
    # Use custom config if provided, otherwise default
    if args.config:
        config_name = args.config
        # Remove .yaml if user included it
        if config_name.endswith('.yaml'):
            config_name = config_name[:-5]
    else:
        config_name = script_name
    
    print(f"Running IPPO baseline: {script_name}")
    print(f"Config: {config_name}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    try:
        # Import Hydra and required modules
        from hydra import compose
        from omegaconf import OmegaConf
        import importlib.util
        
        # Load the IPPO script module from file
        script_path = os.path.join(ippo_dir, f'{script_name}.py')
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)
        
        # Initialize Hydra using initialize_config_dir with absolute path
        # This is the recommended way when config dir is not relative to CWD
        config_dir = os.path.abspath(os.path.join(ippo_dir, 'config'))
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        # Use initialize_config_dir with absolute path (Hydra 1.1+)
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing Hydra instance to avoid conflicts
        GlobalHydra.instance().clear()
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Compose the config (use config_name which may be custom)
            cfg = compose(config_name=config_name)
            
            # Override config values from command line
            if args.total_timesteps is not None:
                cfg['TOTAL_TIMESTEPS'] = args.total_timesteps
            if args.seed is not None:
                cfg['SEED'] = args.seed
            if args.tune:
                cfg['TUNE'] = True
            
            # Run the IPPO training
            if cfg.get('TUNE', False):
                script_module.tune(cfg)
            else:
                script_module.single_run(cfg)
                
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("\nMake sure:")
        print("  1. SocialJax submodule is initialized:")
        print("     git submodule update --init --recursive")
        print("  2. Dependencies are installed:")
        print("     pip install -r external/SocialJax/requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running IPPO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
