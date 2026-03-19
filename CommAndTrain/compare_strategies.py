#!/usr/bin/env python
"""
Compare the performance of four allocation strategies.
All parameters are set via the DFLConfig class in config.py, no command line arguments needed.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
from dataclasses import asdict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_default_config
from dfl_trainer import DFLTrainer


def run_single_strategy(strategy_name, base_config):
    """Run a single strategy, base_config is the base configuration."""
    print(f"\n{'='*70}")
    print(f"Running strategy: {strategy_name.upper()}")
    print(f"{'='*70}")

    # Create strategy-specific configuration copy to avoid modifying base config
    config = deepcopy(base_config)

    # Set strategy-related parameters
    config.allocation_strategy = strategy_name
    config.experiment_name = f"{base_config.experiment_name}_{strategy_name}"

    # Set output directories: under base output dir, create subdirectory per strategy with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.result_dir = os.path.join(base_config.base_output_dir, 'results', f"{strategy_name}_{timestamp}")
    config.model_dir = os.path.join(config.result_dir, 'models')
    config.log_dir = os.path.join(config.result_dir, 'logs')

    # Create directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Print configuration summary
    print(f"Experiment name: {config.experiment_name}")
    print(f"Allocation strategy: {config.allocation_strategy}")
    print(f"Number of agents: {config.num_agents}")
    print(f"Agent weights: {config.weights}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Data distribution: {config.data_distribution}")
    print(f"Non-IID degree: {config.non_iid_degree}")
    print(f"Communication rounds: {config.communication_rounds}")
    print(f"Local epochs: {config.local_epochs}")
    print(f"SINR threshold: {config.sinr_threshold_db} dB")

    try:
        trainer = DFLTrainer(config)
        history, results_path, plot_path = trainer.train()

        # Extract timing statistics from trainer
        strategy_result = {
            'strategy': strategy_name,
            'training_time': trainer.training_times.get('total_training_time', 0),
            'total_allocation_time': trainer.communication_stats.get('total_allocation_time', 0),
            'avg_allocation_time': trainer.communication_stats.get('avg_allocation_time', 0),
            'max_allocation_time': trainer.communication_stats.get('max_allocation_time', 0),
            'results_path': results_path,
            'plot_path': plot_path,
            'history': history,
            'final_weighted_accuracy': history['weighted_accuracy'][-1] if history['weighted_accuracy'] else 0,
            'final_simple_accuracy': history['simple_accuracy'][-1] if history['simple_accuracy'] else 0,
        }
        return strategy_result

    except Exception as e:
        print(f"Training for strategy {strategy_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_strategies(strategies, base_config):
    """Compare multiple strategies."""
    print(f"\n{'='*70}")
    print(f"Starting multi-strategy comparison")
    print(f"Strategies to compare: {strategies}")
    print(f"{'='*70}")

    # Run each strategy
    results = {}
    for strategy in strategies:
        strategy_result = run_single_strategy(strategy, base_config)
        if strategy_result:
            results[strategy] = strategy_result

    # Plot comparison
    if len(results) >= 2:
        plot_comparison_results(results, base_config)

    # Save comparison results
    comparison_dir = os.path.join(base_config.base_output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)

    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'strategies': strategies,
        'config': asdict(base_config),  # save base configuration
        'results': {
            strategy: {
                'final_weighted_accuracy': results[strategy]['final_weighted_accuracy'],
                'final_simple_accuracy': results[strategy]['final_simple_accuracy'],
                'training_time': results[strategy]['training_time'],
                'results_path': results[strategy]['results_path'],
                'plot_path': results[strategy]['plot_path']
            } for strategy in results
        }
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(comparison_dir, f"comparison_{timestamp}.json")

    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\nComparison results saved to: {comparison_file}")

    # Print summary (including allocation times)
    print(f"\n{'='*70}")
    print("Strategy Comparison Summary")
    print(f"{'='*70}")
    for strategy in results:
        r = results[strategy]
        print(f"{strategy.upper():15} | Weighted Acc: {r['final_weighted_accuracy']:.2f}% | "
              f"Simple Acc: {r['final_simple_accuracy']:.2f}% | "
              f"Training Time: {r['training_time']:.2f}s | "
              f"Total Alloc Time: {r['total_allocation_time']:.2f}s | "
              f"Avg Alloc Time: {r['avg_allocation_time']:.3f}s")

    # Also store allocation time statistics in comparison_data
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'strategies': strategies,
        'config': asdict(base_config),
        'results': {
            strategy: {
                'final_weighted_accuracy': r['final_weighted_accuracy'],
                'final_simple_accuracy': r['final_simple_accuracy'],
                'training_time': r['training_time'],
                'total_allocation_time': r['total_allocation_time'],
                'avg_allocation_time': r['avg_allocation_time'],
                'max_allocation_time': r['max_allocation_time'],
                'results_path': r['results_path'],
                'plot_path': r['plot_path']
            } for strategy, r in results.items()
        }
    }
    return results


def plot_comparison_results(results, base_config):
    """Plot comparison results: weighted accuracy comparison + cumulative allocation time comparison."""
    if not results or len(results) < 2:
        print("Not enough data for comparison")
        return

    plt.figure(figsize=(14, 6))

    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange']
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    # Mark every 10 points
    marker_interval = 10

    # Subplot 1: Weighted accuracy
    ax1 = plt.subplot(1, 2, 1)
    for i, (strategy, result) in enumerate(results.items()):
        history = result['history']
        if 'weighted_accuracy' not in history or not history['weighted_accuracy']:
            continue

        rounds = np.array(history.get('communication_rounds', list(range(len(history['weighted_accuracy'])))))
        acc = np.array(history['weighted_accuracy'])

        # Plot original curve (thin line)
        ax1.plot(rounds, acc, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)],
                 linewidth=1.5, alpha=0.6)

        # Plot markers at every 10th point
        marker_idx = np.arange(0, len(rounds), marker_interval)
        ax1.plot(rounds[marker_idx], acc[marker_idx],
                 color=colors[i % len(colors)], marker=markers[i % len(markers)],
                 linestyle='', markersize=6, label=f'{strategy.upper()}')

        # Plot smoothed curve using moving average (window size 5)
        if len(acc) >= 5:
            window = 5
            smoothed = np.convolve(acc, np.ones(window)/window, mode='valid')
            smooth_rounds = rounds[window-1:]
            ax1.plot(smooth_rounds, smoothed, color=colors[i % len(colors)],
                     linestyle='-', linewidth=2, alpha=0.8)

    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Weighted Accuracy (%)')
    ax1.set_title('Weighted Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Cumulative allocation time
    ax2 = plt.subplot(1, 2, 2)
    for i, (strategy, result) in enumerate(results.items()):
        history = result['history']
        if 'allocation_times' not in history or not history['allocation_times']:
            continue

        rounds = np.array(history.get('communication_rounds', list(range(len(history['allocation_times'])))))
        alloc_times = np.array(history['allocation_times'])
        cum_alloc = np.cumsum(alloc_times)   # cumulative allocation time

        # Original cumulative curve
        ax2.plot(rounds, cum_alloc, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)],
                 linewidth=1.5, alpha=0.6)

        # Mark every 10 points
        marker_idx = np.arange(0, len(rounds), marker_interval)
        ax2.plot(rounds[marker_idx], cum_alloc[marker_idx],
                 color=colors[i % len(colors)], marker=markers[i % len(markers)],
                 linestyle='', markersize=6, label=f'{strategy.upper()}')

        # Moving average smoothing (window size 5)
        if len(cum_alloc) >= 5:
            window = 5
            smoothed = np.convolve(cum_alloc, np.ones(window)/window, mode='valid')
            smooth_rounds = rounds[window-1:]
            ax2.plot(smooth_rounds, smoothed, color=colors[i % len(colors)],
                     linestyle='-', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Cumulative Allocation Time (seconds)')
    ax2.set_title('Cumulative Allocation Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save comparison plot
    comparison_dir = os.path.join(base_config.base_output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(comparison_dir, f"strategy_comparison_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(comparison_dir, f"strategy_comparison_{timestamp}.pdf"))

    print(f"\nComparison plot saved to: {plot_file}")
    plt.show()


def main():
    """Main function."""
    # Get default configuration from config
    base_config = get_default_config()

    # Get list of strategies to compare from configuration
    strategies = base_config.strategies_to_compare

    print(f"\nDFL Strategy Comparison Tool")
    print(f"Base output directory: {base_config.base_output_dir}")
    print(f"Random seed: {base_config.seed}")
    print(f"Strategies to compare: {strategies}")

    # Run comparison
    results = compare_strategies(strategies, base_config)

    if results:
        print(f"\nStrategy comparison completed!")
        return True
    else:
        print(f"\nStrategy comparison failed!")
        return False


if __name__ == "__main__":
    main()