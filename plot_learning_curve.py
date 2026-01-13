#!/usr/bin/env python3
"""Plot learning curves and fairness metrics from CSV log files."""

import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def compute_fairness(df):
    """Compute Gini and Jain fairness indices."""
    r_red = df['return_red'].values
    r_green = df['return_green'].values
    
    # Gini coefficient (0 = fair, 1 = unfair) - use abs to handle negatives
    total = np.abs(r_red) + np.abs(r_green)
    gini = np.abs(r_red - r_green) / (total + 1e-8)
    
    # Jain's fairness index (1 = fair, 0.5 = maximally unfair for 2 agents)
    # Only meaningful when both returns are positive
    jain = np.where(
        (r_red > 0) & (r_green > 0),
        (r_red + r_green)**2 / (2 * (r_red**2 + r_green**2 + 1e-8)),
        0.0
    )
    
    return gini, jain

def plot_learning_curves(log_dir="./logs", output_file=None):
    """Plot all learning curves from CSV files in log_dir."""
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {log_dir}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for csv_file in csv_files:
        name = os.path.basename(csv_file).replace('.csv', '')
        try:
            df = pd.read_csv(csv_file)
            window = max(1, len(df) // 50)
            
            # Check if we have per-agent returns
            has_per_agent = 'return_red' in df.columns and 'return_green' in df.columns
            
            # Plot 1: Mean return
            ax = axes[0, 0]
            smoothed = df['returned_episode_returns'].rolling(window=window, min_periods=1).mean()
            ax.plot(df['env_step'], smoothed, label=name, alpha=0.8)
            ax.set_xlabel('Environment Steps')
            ax.set_ylabel('Mean Episode Return')
            ax.set_title('Learning Curve (Mean Return)')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if has_per_agent:
                # Plot 2: Per-agent returns
                ax = axes[0, 1]
                red_smooth = df['return_red'].rolling(window=window, min_periods=1).mean()
                green_smooth = df['return_green'].rolling(window=window, min_periods=1).mean()
                ax.plot(df['env_step'], red_smooth, label=f'{name} (Red)', color='red', alpha=0.7)
                ax.plot(df['env_step'], green_smooth, label=f'{name} (Green)', color='green', alpha=0.7)
                ax.set_xlabel('Environment Steps')
                ax.set_ylabel('Agent Return')
                ax.set_title('Per-Agent Returns (Red vs Green)')
                ax.legend(loc='lower right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Plot 3: Fairness metrics
                ax = axes[1, 0]
                gini, jain = compute_fairness(df)
                gini_smooth = pd.Series(gini).rolling(window=window, min_periods=1).mean()
                jain_smooth = pd.Series(jain).rolling(window=window, min_periods=1).mean()
                ax.plot(df['env_step'], gini_smooth, label=f'{name} Gini', linestyle='-', alpha=0.8)
                ax.plot(df['env_step'], jain_smooth, label=f'{name} Jain', linestyle='--', alpha=0.8)
                ax.axhline(y=0.509, color='gray', linestyle=':', label='Paper Gini (Ind)')
                ax.axhline(y=0.498, color='gray', linestyle='-.', label='Paper Jain (Ind)')
                ax.set_xlabel('Environment Steps')
                ax.set_ylabel('Fairness Index')
                ax.set_title('Fairness Metrics (Gini↓ better, Jain↑ better)')
                ax.legend(loc='right', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Plot 4: Eat own coins
            ax = axes[1, 1]
            own_smooth = df['eat_own_coins'].rolling(window=window, min_periods=1).mean()
            ax.plot(df['env_step'], own_smooth, label=name, alpha=0.8)
            ax.set_xlabel('Environment Steps')
            ax.set_ylabel('Eat Own Coins')
            ax.set_title('Cooperation Metric (lower = more stealing)')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Print summary
            print(f"\n📊 {name}:")
            print(f"   Final Mean Return: {df['returned_episode_returns'].iloc[-1]:.2f}")
            if has_per_agent:
                final_red = df['return_red'].iloc[-1]
                final_green = df['return_green'].iloc[-1]
                final_gini = gini[-1]
                final_jain = jain[-1]
                print(f"   Final Red Return:  {final_red:.2f}")
                print(f"   Final Green Return: {final_green:.2f}")
                print(f"   Final Gini: {final_gini:.3f} (paper Ind: 0.509)")
                print(f"   Final Jain: {final_jain:.3f} (paper Ind: 0.498)")
                print(f"   Red > Green: {final_red > final_green} (paper expects True for Ind)")
            print(f"   Final Eat Own Coins: {df['eat_own_coins'].iloc[-1]:.2f}")
                
        except Exception as e:
            print(f"✗ Error reading {csv_file}: {e}")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n📈 Plot saved to: {output_file}")
    else:
        out_path = os.path.join(log_dir, "learning_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 Plot saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from CSV logs")
    parser.add_argument("--log-dir", default="./logs", help="Directory with CSV files")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()
    
    plot_learning_curves(args.log_dir, args.output)
