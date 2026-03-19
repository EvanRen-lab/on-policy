#!/usr/bin/env python
"""
Average allocation allocator - revised version (aligned with random/ao interface)
Each device's total transmit power does not exceed the maximum power, and the total power is evenly distributed among all links.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from wireless_utils import (
    calculate_sinr_matrix,
    get_neighbors_from_adjacency,
    calculate_connectivity_stats
)

def avg_allocator(positions, config):
    """
    Average allocation allocator (aligned with random/ao interface)
    Each device's total transmit power does not exceed the maximum power, and the total power is evenly distributed among all links.

    Args:
        positions: array of terminal positions [N, 2]
        config: configuration dictionary, must contain the following keys:
                num_agents, num_channels, max_power, bandwidth, noise_dbm,
                sinr_threshold_db, distance_scale, pathloss_a, pathloss_b,
                weights, coefficients, seed

    Returns:
        neighbor_lists: neighbor list for each agent, each element is a list of neighbor IDs
    """
    # Extract parameters from config
    num_agents = config['num_agents']
    max_power = config['max_power']
    num_channels = config['num_channels']
    bandwidth = config['bandwidth']
    noise_dbm = config['noise_dbm']
    sinr_threshold_db = config['sinr_threshold_db']
    distance_scale = config['distance_scale']
    a = config['pathloss_a']
    b = config['pathloss_b']
    weights = config['weights']
    coefficients = config['coefficients']
    seed = config['seed']

    if seed is not None:
        np.random.seed(seed)

    try:
        # Compute penalty coefficients C[m] = rho[m]^2 + (K-1)*rho[m]
        rho = np.array(weights)
        K = num_agents
        C = rho**2 + (K - 1) * rho

        # Average allocation: each device's total power is evenly distributed among all other devices
        # Each device has at most (N-1) possible links, each link gets average power
        avg_power_per_link = max_power / (num_agents - 1) if num_agents > 1 else max_power

        # Create power allocation dictionary
        power_allocations = []
        for i in range(num_agents):
            power_alloc = {}
            for j in range(num_agents):
                if i != j:
                    power_alloc[j] = avg_power_per_link
            power_allocations.append(power_alloc)

        # Random channel allocation
        channel_selections = []
        for i in range(num_agents):
            channel_select = {}
            for j in range(num_agents):
                if i != j:
                    channel_select[j] = np.random.randint(0, num_channels)
            channel_selections.append(channel_select)

        # Compute SINR matrix and adjacency matrix
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
            seed=seed
        )

        # Compute J_hat: sum penalty C[m] for each disconnected link (m,n)
        J_hat = 0.0
        for m in range(num_agents):
            for n in range(num_agents):
                if m != n and adjacency_matrix[m, n] == 0:
                    J_hat += C[m]

        # Get neighbor lists
        neighbor_lists = get_neighbors_from_adjacency(adjacency_matrix)

        # Compute statistics and print
        stats = calculate_connectivity_stats(adjacency_matrix)
        total_power_used = sum(sum(p.values()) for p in power_allocations)

        print(f"\nAverage allocation completed:")
        print(f"  Best loss J_hat: {J_hat:.6f}")
        print(f"  Total links: {stats['total_connected_links']}")
        print(f"  Total power used: {total_power_used:.6f} W")
        print(f"  Neighbor lists: {neighbor_lists}")

        return neighbor_lists

    except Exception as e:
        print(f"Average allocation failed: {e}")
        import traceback
        traceback.print_exc()
        return [[] for _ in range(num_agents)]