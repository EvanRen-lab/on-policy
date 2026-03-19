#!/usr/bin/env python
"""
DFL Training Results Plotting Utilities
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import time


def plot_results(history, config, result_dir, allocation_strategy):
    """Plot training results."""
    if not history['communication_rounds']:
        print("No training history data to plot")
        return None

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # 1. Accuracy curves
    ax = axes[0, 0]
    ax.plot(history['communication_rounds'], history['weighted_accuracy'],
            label='weighted_accuracy', linewidth=2, color='blue')
    ax.plot(history['communication_rounds'], history['simple_accuracy'],
            label='simple_accuracy', linewidth=2, linestyle='--', color='orange')
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy on Global Testset')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Terminal accuracies
    ax = axes[0, 1]
    terminal_accuracies = np.array(history['terminal_accuracies']).T
    for i in range(config.num_agents):
        ax.plot(history['communication_rounds'], terminal_accuracies[i],
                label=f'terminal{i}', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Terminal Accuracies')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Loss curves
    ax = axes[0, 2]
    terminal_losses = np.array(history['terminal_losses']).T
    avg_losses = np.mean(terminal_losses, axis=0)
    ax.plot(history['communication_rounds'], avg_losses,
            label='Average Loss', linewidth=2, color='red')
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Loss')
    ax.set_title('Loss on Global Testset')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Neighbor count variation
    ax = axes[1, 0]
    ax.plot(history['communication_rounds'], history['neighbor_counts'],
            label='Average Neighbors', linewidth=2, color='green')
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Number of Neighbors')
    ax.set_title('Average Number of Neighbors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Resource allocation solving time
    ax = axes[1, 1]
    if config.record_allocation_time and history['allocation_times']:
        rounds_to_plot = list(range(min(len(history['allocation_times']), len(history['communication_rounds']))))
        allocation_times = history['allocation_times'][:len(rounds_to_plot)]

        ax.plot(rounds_to_plot, allocation_times,
                label='Allocation Time', linewidth=2, color='purple')
        ax.set_xlabel('Communication Rounds')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Resource Allocation Solving Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        avg_time = np.mean(allocation_times)
        ax.axhline(y=avg_time, color='r', linestyle='--', alpha=0.7,
                   label=f'Average: {avg_time:.3f}s')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Allocation time not recorded',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Resource Allocation Solving Time')

    # 6. Terminal weights distribution
    ax = axes[1, 2]
    terminal_ids = list(range(config.num_agents))
    weights = config.weights
    colors = plt.cm.Set3(np.linspace(0, 1, config.num_agents))
    bars = ax.bar(terminal_ids, weights, color=colors, edgecolor='black')
    ax.set_xlabel('Terminal ID')
    ax.set_ylabel('Weight')
    ax.set_title('Terminal Weights Distribution')

    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}', ha='center', va='bottom')

    ax.grid(True, alpha=0.3, axis='y')

    # 7. Allocation time distribution histogram
    ax = axes[2, 0]
    if config.record_allocation_time and history['allocation_times']:
        allocation_times = history['allocation_times']
        ax.hist(allocation_times, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('Allocation Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Allocation Times')
        ax.grid(True, alpha=0.3, axis='y')

        stats_text = f'Mean: {np.mean(allocation_times):.3f}s\n'
        stats_text += f'Std: {np.std(allocation_times):.3f}s\n'
        stats_text += f'Max: {np.max(allocation_times):.3f}s\n'
        stats_text += f'Min: {np.min(allocation_times):.3f}s'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'Allocation time not recorded',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution of Allocation Times')

    # 8. Accuracy improvement comparison (final - initial)
    ax = axes[2, 1]
    if len(history['terminal_accuracies']) > 1:
        initial_accs = history['terminal_accuracies'][0]
        final_accs = history['terminal_accuracies'][-1]
        improvements = [final - initial for final, initial in zip(final_accs, initial_accs)]

        colors = plt.cm.coolwarm(np.linspace(0, 1, config.num_agents))
        bars = ax.bar(terminal_ids, improvements, color=colors, edgecolor='black')
        ax.set_xlabel('Terminal ID')
        ax.set_ylabel('Accuracy Improvement (%)')
        ax.set_title('Accuracy Improvement (Final - Initial)')
        ax.grid(True, alpha=0.3, axis='y')

        for i, (bar, improvement) in enumerate(zip(bars, improvements)):
            ax.text(bar.get_x() + bar.get_width()/2.,
                    improvement + (0.5 if improvement >= 0 else -1.0),
                    f'{improvement:+.1f}%\nw={weights[i]:.2f}',
                    ha='center', va='bottom' if improvement >= 0 else 'top',
                    fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No improvement data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Accuracy Improvement')

    # 9. Accuracy vs neighbor count scatter plot
    ax = axes[2, 2]
    if len(history['weighted_accuracy']) > 0:
        accuracies = history['weighted_accuracy']
        neighbor_counts = history['neighbor_counts']

        scatter = ax.scatter(neighbor_counts, accuracies,
                             c=history['communication_rounds'],
                             cmap='viridis', alpha=0.7, s=50)
        ax.set_xlabel('Average Neighbors')
        ax.set_ylabel('Weighted Accuracy (%)')
        ax.set_title('Accuracy vs Neighbor Count')
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Communication Round')
    else:
        ax.text(0.5, 0.5, 'No data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Accuracy vs Neighbor Count')

    strategy_name = allocation_strategy.upper()
    plt.suptitle(f'DFL Training Results - Strategy: {strategy_name} - Rounds: {config.communication_rounds}', fontsize=14)
    plt.tight_layout()

    # Save image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(result_dir, f"training_plots_{allocation_strategy}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Training plots saved to: {plot_path}")
    return plot_path


def plot_time_breakdown(history, config, result_dir, allocation_strategy):
    """Plot time breakdown."""
    if not config.record_allocation_time or not history.get('allocation_times'):
        print("Allocation time not recorded or no data, cannot plot time breakdown")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Allocation time trend
    ax = axes[0, 0]
    rounds = list(range(len(history['allocation_times'])))
    ax.plot(rounds, history['allocation_times'],
            label='Allocation Time', linewidth=2, color='purple')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Resource Allocation Solving Time Trend')
    ax.grid(True, alpha=0.3)

    window_size = min(10, len(history['allocation_times']) // 5)
    if window_size > 1:
        moving_avg = np.convolve(history['allocation_times'],
                                  np.ones(window_size)/window_size,
                                  mode='valid')
        ax.plot(rounds[window_size-1:], moving_avg,
                label=f'{window_size}-round Moving Avg',
                linewidth=2, linestyle='--', color='orange')
    ax.legend()

    # 2. Allocation time distribution box plot
    ax = axes[0, 1]
    allocation_times = history['allocation_times']
    bp = ax.boxplot(allocation_times, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')

    stats_text = f'Count: {len(allocation_times)}\n'
    stats_text += f'Mean: {np.mean(allocation_times):.3f}s\n'
    stats_text += f'Median: {np.median(allocation_times):.3f}s\n'
    stats_text += f'Std: {np.std(allocation_times):.3f}s\n'
    stats_text += f'Q1: {np.percentile(allocation_times, 25):.3f}s\n'
    stats_text += f'Q3: {np.percentile(allocation_times, 75):.3f}s'

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Allocation Time (seconds)')
    ax.set_title('Allocation Time Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Cumulative allocation time
    ax = axes[1, 0]
    cumulative_time = np.cumsum(history['allocation_times'])
    rounds = list(range(len(cumulative_time)))

    ax.plot(rounds, cumulative_time,
            label='Cumulative Allocation Time', linewidth=2, color='green')
    ax.fill_between(rounds, 0, cumulative_time, alpha=0.3, color='green')

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Cumulative Time (seconds)')
    ax.set_title('Cumulative Resource Allocation Time')
    ax.grid(True, alpha=0.3)

    if cumulative_time.size > 0:
        total_time = cumulative_time[-1]
        ax.text(0.95, 0.05, f'Total: {total_time:.2f}s',
                transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. Strategy time comparison
    ax = axes[1, 1]
    strategy_times = history.get('allocation_strategy_times', {})
    strategies = list(strategy_times.keys())
    times = list(strategy_times.values())

    valid_indices = [i for i, t in enumerate(times) if t > 0]
    if valid_indices:
        valid_strategies = [strategies[i] for i in valid_indices]
        valid_times = [times[i] for i in valid_indices]

        bars = ax.bar(valid_strategies, valid_times,
                      color=plt.cm.tab20c(np.linspace(0, 1, len(valid_strategies))))

        ax.set_xlabel('Allocation Strategy')
        ax.set_ylabel('Total Time (seconds)')
        ax.set_title('Total Allocation Time by Strategy')
        ax.set_xticklabels(valid_strategies, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, time_val in zip(bars, valid_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.2f}s', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No strategy time data',
                ha='center', va='center', transform=ax.transAxes)

    plt.suptitle(f'Time Analysis - Allocation Strategy: {allocation_strategy}', fontsize=14)
    plt.tight_layout()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    time_plot_path = os.path.join(result_dir, f"time_analysis_{allocation_strategy}_{timestamp}.png")
    plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Time analysis plots saved to: {time_plot_path}")
    return time_plot_path