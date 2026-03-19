#!/usr/bin/env python
"""
Independent test of resource allocation algorithms (multiple rounds):
Load complete configuration from dfl_config.py
Each test seed (random scenario) simulates multiple communication rounds,
with terminal positions moving each round and resources reallocated.
"""
import numpy as np
import time
import sys
import os
from copy import deepcopy

try:
    from config import DFLConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Error: Cannot find dfl_config.py. Please ensure the configuration file exists.")
    sys.exit(1)

# Import trajectory generator (for mobility simulation)
try:
    from mobility import TrajectoryGenerator
    MOBILITY_AVAILABLE = True
except ImportError:
    MOBILITY_AVAILABLE = False
    print("Warning: mobility module not available, will use static positions (positions unchanged each round).")

# Import allocator modules
try:
    from allocator_mappo import mappo_allocator
    MAPPO_AVAILABLE = True
except ImportError:
    MAPPO_AVAILABLE = False
    print("Warning: mappo_allocator not available")

try:
    from allocator_ao import alternating_allocator
    ALTERNATING_AVAILABLE = True
except ImportError:
    ALTERNATING_AVAILABLE = False
    print("Warning: alternating_allocator not available")

try:
    from allocator_avg import avg_allocator
    AVG_AVAILABLE = True
except ImportError:
    AVG_AVAILABLE = False
    print("Warning: avg_allocator not available")

try:
    from allocator_random import random_allocator
    RANDOM_AVAILABLE = True
except ImportError:
    RANDOM_AVAILABLE = False
    print("Warning: random_allocator not available")


def get_wireless_config(config: DFLConfig, seed=None):
    """Construct a configuration dictionary required by allocators from a DFLConfig instance."""
    cfg = {
        'num_agents': config.num_agents,
        'num_channels': config.num_channels,
        'max_power': config.max_power,
        'max_power_dbm': config.max_power_dbm,
        'bandwidth': config.bandwidth,
        'noise_dbm': config.noise_dbm,
        'sinr_threshold_db': config.sinr_threshold_db,
        'distance_scale': config.distance_scale,
        'pathloss_a': config.pathloss_a,
        'pathloss_b': config.pathloss_b,
        'alternating_iterations': config.alternating_iterations,
        'random_iterations': config.random_iterations,
        'power_optimization_method': config.power_optimization_method,
        'weights': config.weights,
        'coefficients': config.coefficients,
        'seed': seed,  # Pass current test seed for internal use by allocators
        'mappo_config_path': config.mappo_config_path,
        'mappo_model_path': config.mappo_model_path,
    }
    return cfg


def generate_positions(num_agents, radius, seed=None):
    """Randomly generate terminal positions within a circular area (using given seed for reproducibility)."""
    if seed is not None:
        np.random.seed(seed)
    positions = []
    for _ in range(num_agents):
        r = radius * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        positions.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(positions)


def compute_neighbor_stats(neighbor_lists):
    """Compute statistics from neighbor lists."""
    num_agents = len(neighbor_lists)
    neighbor_counts = [len(n) for n in neighbor_lists]
    avg_neighbors = np.mean(neighbor_counts)
    std_neighbors = np.std(neighbor_counts)
    min_neighbors = np.min(neighbor_counts)
    max_neighbors = np.max(neighbor_counts)
    total_links = sum(neighbor_counts)
    max_possible = num_agents * (num_agents - 1)
    connectivity_ratio = total_links / max_possible if max_possible > 0 else 0
    return {
        'avg_neighbors': avg_neighbors,
        'std_neighbors': std_neighbors,
        'min_neighbors': min_neighbors,
        'max_neighbors': max_neighbors,
        'total_links': total_links,
        'connectivity_ratio': connectivity_ratio,
        'neighbor_counts': neighbor_counts,
    }


def test_allocation(strategy, positions, config_dict):
    """Test a single allocation strategy, return neighbor lists and execution time."""
    start = time.perf_counter()
    if strategy == 'mappo' and MAPPO_AVAILABLE:
        neighbor_lists = mappo_allocator(positions, config_dict)
    elif strategy == 'alternating' and ALTERNATING_AVAILABLE:
        neighbor_lists = alternating_allocator(positions, config_dict)
    elif strategy == 'avg' and AVG_AVAILABLE:
        neighbor_lists = avg_allocator(positions, config_dict)
    elif strategy == 'random' and RANDOM_AVAILABLE:
        neighbor_lists = random_allocator(positions, config_dict)
    else:
        raise ValueError(f"Strategy {strategy} unavailable or unknown")
    elapsed = time.perf_counter() - start
    return neighbor_lists, elapsed


def main():
    # Create configuration instance (using default values, or can be overridden via command line)
    config = DFLConfig()
    print("Loaded configuration:")
    print(f"  Number of agents: {config.num_agents}")
    print(f"  Number of channels: {config.num_channels}")
    print(f"  Deployment radius: {config.deployment_radius} m")
    print(f"  Mobility: {'Enabled' if config.mobility_enabled else 'Disabled'}")
    print(f"  Communication rounds: {config.communication_rounds}")
    print(f"  Strategies to compare: {config.strategies_to_compare}")
    print("-" * 60)

    num_tests = 5  # Number of tests (different random seeds), can be adjusted as needed
    num_rounds = config.communication_rounds  # One allocation per round

    # Determine strategies to test (only those specified in config and available)
    strategies = [s for s in config.strategies_to_compare if (
        (s == 'mappo' and MAPPO_AVAILABLE) or
        (s == 'alternating' and ALTERNATING_AVAILABLE) or
        (s == 'avg' and AVG_AVAILABLE) or
        (s == 'random' and RANDOM_AVAILABLE)
    )]
    if not strategies:
        print("No allocators available. Please check imports or strategies_to_compare in configuration.")
        return

    print(f"Running {num_tests} random scenario tests, each simulating {num_rounds} communication rounds...")
    # Collect results: results[strategy][test_idx][round_idx] = {'avg_neighbors': val, 'time': val}
    results = {s: [] for s in strategies}

    for test_idx in range(num_tests):
        print(f"\n{'='*60}")
        base_seed = config.seed + test_idx if config.seed else test_idx
        print(f"Test scenario {test_idx+1}/{num_tests} (seed={base_seed})")
        print(f"{'='*60}")

        # Initialize trajectory generator if available and mobility enabled
        if MOBILITY_AVAILABLE and config.mobility_enabled:
            # Deep copy config and set seed to ensure trajectory for this test scenario is independent and reproducible
            test_config = deepcopy(config)
            test_config.seed = base_seed
            traj_gen = TrajectoryGenerator(test_config)
        else:
            traj_gen = None
            # If no mobility, generate a set of initial positions, then keep them unchanged each round
            static_positions = generate_positions(config.num_agents, config.deployment_radius, seed=base_seed)

        # For each strategy, initialize an empty list to store data for this test scenario
        for s in strategies:
            results[s].append([])

        # Simulate multiple communication rounds
        for round_idx in range(num_rounds):
            # Get positions for current round
            if traj_gen is not None:
                positions = traj_gen.get_positions(round_idx)
            else:
                positions = static_positions

            # Build configuration dictionary (pass base_seed as allocator seed to ensure consistent internal randomness)
            config_dict = get_wireless_config(config, seed=base_seed)

            for strategy in strategies:
                try:
                    neighbor_lists, exec_time = test_allocation(strategy, positions, config_dict)
                    stats = compute_neighbor_stats(neighbor_lists)

                    # Record results for this round
                    results[strategy][test_idx].append({
                        'round': round_idx,
                        'avg_neighbors': stats['avg_neighbors'],
                        'std_neighbors': stats['std_neighbors'],
                        'min_neighbors': stats['min_neighbors'],
                        'max_neighbors': stats['max_neighbors'],
                        'connectivity_ratio': stats['connectivity_ratio'],
                        'time': exec_time
                    })

                    # Print brief info (every 5 rounds and the last round)
                    if round_idx % 5 == 0 or round_idx == num_rounds-1:
                        print(f"  Round {round_idx:3d} | {strategy:12s}: avg neighbors = {stats['avg_neighbors']:.2f}, "
                              f"connectivity ratio = {stats['connectivity_ratio']:.1%}, time = {exec_time:.4f}s")
                except Exception as e:
                    print(f"  Round {round_idx:3d} | {strategy:12s}: failed - {e}")
                    # Fill with a placeholder to keep list length consistent
                    results[strategy][test_idx].append({
                        'round': round_idx,
                        'avg_neighbors': np.nan,
                        'std_neighbors': np.nan,
                        'min_neighbors': np.nan,
                        'max_neighbors': np.nan,
                        'connectivity_ratio': np.nan,
                        'time': np.nan
                    })

    # Summary statistics: compute average performance for each strategy across all test scenarios and rounds
    print("\n" + "="*70)
    print("Summary results (average over all test scenarios and rounds)")
    print("="*70)

    for strategy in strategies:
        # Extract all valid records (ignore NaN)
        all_avg_neighbors = []
        all_times = []
        for test_data in results[strategy]:
            for round_data in test_data:
                if not np.isnan(round_data['avg_neighbors']):
                    all_avg_neighbors.append(round_data['avg_neighbors'])
                    all_times.append(round_data['time'])

        if all_avg_neighbors:
            avg_nei = np.mean(all_avg_neighbors)
            std_nei = np.std(all_avg_neighbors)
            avg_time = np.mean(all_times)
            std_time = np.std(all_times)
            print(f"{strategy:12s}: avg neighbors = {avg_nei:.2f} ± {std_nei:.2f}, avg time = {avg_time:.4f}s ± {std_time:.4f}s")
        else:
            print(f"{strategy:12s}: no valid data")

    # Optional: Print trend of average neighbors per round for each strategy
    print("\n" + "="*70)
    print("Average neighbors per round (averaged across test scenarios)")
    print("="*70)
    print("Round", end="")
    for strategy in strategies:
        print(f"  {strategy:>10}", end="")
    print()

    for round_idx in range(num_rounds):
        print(f"{round_idx:4d}", end="")
        for strategy in strategies:
            # Collect avg_neighbors for this round across all test scenarios
            round_values = []
            for test_data in results[strategy]:
                if round_idx < len(test_data) and not np.isnan(test_data[round_idx]['avg_neighbors']):
                    round_values.append(test_data[round_idx]['avg_neighbors'])
            if round_values:
                avg_val = np.mean(round_values)
                print(f"  {avg_val:10.2f}", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()


if __name__ == "__main__":
    main()