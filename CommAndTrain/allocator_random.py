#!/usr/bin/env python
"""
Random search allocator - revised version (using actual loss J_hat as evaluation criterion)
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from wireless_utils import (
    calculate_sinr_matrix,
    get_neighbors_from_adjacency,
    calculate_connectivity_stats
)

def random_allocator(positions, config):
    """
    Random search allocator aiming to minimize the actual observed loss J_hat.

    Args:
        positions: array of terminal positions [N, 2]
        config: configuration dictionary, must contain the following keys:
                num_agents, num_channels, max_power, bandwidth, noise_dbm,
                sinr_threshold_db, distance_scale, pathloss_a, pathloss_b,
                random_iterations, weights, coefficients, seed

    Returns:
        neighbor_lists: neighbor list for each agent, each element is a list of neighbor IDs
    """
    # Extract parameters from config
    num_agents = config['num_agents']
    weights = config['weights']
    coefficients = config['coefficients']
    max_power = config['max_power']
    num_channels = config['num_channels']
    sinr_threshold_db = config['sinr_threshold_db']
    bandwidth = config['bandwidth']
    noise_dbm = config['noise_dbm']
    distance_scale = config['distance_scale']
    a = config['pathloss_a']
    b = config['pathloss_b']
    random_iterations = config['random_iterations']
    seed = config['seed']

    try:
        # Initialize best solution
        best_J = float('inf')                   # best loss (lower is better)
        best_neighbor_lists = None
        best_sinr_matrix = None
        best_power_allocations = None
        best_adjacency_matrix = None
        best_stats = None

        print(f"Running random search algorithm ({random_iterations} iterations, objective: minimize J_hat)...")

        # Precompute penalty coefficients C_m
        rho = np.array(weights)
        K = num_agents
        C = rho**2 + (K - 1) * rho               # shape (num_agents,)

        for iteration in range(random_iterations):
            # Randomly generate power allocations and channel selections
            power_allocations = []
            channel_selections = []

            for i in range(num_agents):
                power_alloc = {}
                total_possible_power = 0
                for j in range(num_agents):
                    if i != j:
                        random_weight = np.random.rand()
                        power_alloc[j] = random_weight
                        total_possible_power += random_weight

                if total_possible_power > 0:
                    scaling_factor = max_power / total_possible_power
                    for j in power_alloc:
                        power_alloc[j] *= scaling_factor
                else:
                    for j in power_alloc:
                        power_alloc[j] = max_power / (num_agents - 1)

                power_allocations.append(power_alloc)

                channel_select = {}
                for j in range(num_agents):
                    if i != j:
                        channel_select[j] = np.random.randint(0, num_channels)
                channel_selections.append(channel_select)

            # Compute SINR matrix and adjacency matrix (single small-scale fading realization)
            sinr_matrix, adjacency_matrix, _, _ = calculate_sinr_matrix(
                positions=positions,
                power_allocations=power_allocations,
                channel_selections=channel_selections,
                coefficients=coefficients,
                max_power=max_power,
                num_channels=num_channels,
                bandwidth=bandwidth,
                noise_dbm=noise_dbm,
                sinr_threshold_db=sinr_threshold_db,
                distance_scale=distance_scale,
                a=a,
                b=b,
                seed=seed + iteration if seed is not None else None
            )

            neighbor_lists = get_neighbors_from_adjacency(adjacency_matrix)
            stats = calculate_connectivity_stats(adjacency_matrix)

            # Compute actual loss J_hat
            J_hat = 0.0
            for m in range(num_agents):
                for n in range(num_agents):
                    if m != n and adjacency_matrix[m, n] == 0:
                        J_hat += C[m]

            # Update best solution (lower J_hat is better)
            if J_hat < best_J:
                best_J = J_hat
                best_neighbor_lists = neighbor_lists
                best_sinr_matrix = sinr_matrix
                best_power_allocations = power_allocations
                best_adjacency_matrix = adjacency_matrix
                best_stats = stats

        # If no valid solution found, return empty lists
        if best_neighbor_lists is None:
            best_neighbor_lists = [[] for _ in range(num_agents)]

        # Print best result statistics (using saved best information)
        if best_sinr_matrix is not None and best_stats is not None:
            # Calculate total power used
            total_power_used = 0
            for i in range(num_agents):
                device_power = sum(best_power_allocations[i].values())
                total_power_used += device_power

            print(f"\nRandom search completed (iterations={random_iterations}):")
            print(f"  Best loss J_hat: {best_J:.6f}")
            print(f"  Total links: {best_stats['total_connected_links']}")
            print(f"  Total power used: {total_power_used:.6f} W")
            print(f"  Neighbor lists: {best_neighbor_lists}")

        return best_neighbor_lists

    except Exception as e:
        print(f"Random search allocation failed: {e}")
        import traceback
        traceback.print_exc()
        return [[] for _ in range(num_agents)]