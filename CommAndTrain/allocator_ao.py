#!/usr/bin/env python
"""
Alternating optimization allocator
Non-convex optimization based on link success probability, using alternating optimization of power and channels.
"""
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add path to import wireless utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wireless_utils import (
    calculate_sinr_matrix,
    get_neighbors_from_adjacency,
    calculate_connectivity_stats
)

def compute_J_hat(power_allocations, channel_selections, positions, coefficients,
                  max_power, num_channels, bandwidth, noise_dbm, sinr_threshold_db,
                  distance_scale, a, b, seed, C):
    """
    Helper function: compute the actual loss J_hat given power and channel allocations.
    """
    # Calculate SINR matrix and adjacency matrix (single small-scale fading realization)
    _, adjacency_matrix, _, _ = calculate_sinr_matrix(
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
    num_agents = len(positions)
    for m in range(num_agents):
        for n in range(num_agents):
            if m != n and adjacency_matrix[m, n] == 0:
                J_hat += C[m]
    return J_hat

def optimize_power_allocation_fixed_channels(
    positions, channel_matrix, coefficients, C, weights,
    max_power, num_channels, bandwidth, noise_dbm, sinr_threshold_db,
    distance_scale, a, b, seed, method='SLSQP'
):
    """
    Power optimization under fixed channel allocation (aiming to minimize J_hat).
    """
    num_agents = positions.shape[0]

    # Build list of all possible links (i,j) i≠j
    active_links = []
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                active_links.append((i, j))

    num_active_links = len(active_links)
    if num_active_links == 0:
        return np.zeros((num_agents, num_agents)), {'success': False, 'message': 'no links'}

    # Mapping from transmitter to link indices
    tx_to_link_indices = {}
    for idx, (i, j) in enumerate(active_links):
        tx_to_link_indices.setdefault(i, []).append(idx)

    # Objective function: input power vector, return J_hat
    def objective(power_vector):
        # Reconstruct power allocation dictionary and channel selection dictionary
        power_allocations = [{} for _ in range(num_agents)]
        channel_selections = [{} for _ in range(num_agents)]

        for idx, (i, j) in enumerate(active_links):
            power_allocations[i][j] = max(power_vector[idx], 0.0)      # ensure non-negative
            channel_selections[i][j] = channel_matrix[i, j]            # fixed channel

        # Compute J_hat
        J = compute_J_hat(
            power_allocations, channel_selections, positions, coefficients,
            max_power, num_channels, bandwidth, noise_dbm, sinr_threshold_db,
            distance_scale, a, b, seed, C
        )
        return J

    # Linear constraint: total transmit power per transmitter ≤ max_power
    A = np.zeros((num_agents, num_active_links))
    for idx, (i, j) in enumerate(active_links):
        A[i, idx] = 1
    linear_constraint = LinearConstraint(A, 0, max_power)

    # Initial guess: uniformly allocate power to all links of each transmitter
    x0 = np.zeros(num_active_links)
    for i, indices in tx_to_link_indices.items():
        per_link_power = max_power / len(indices) if indices else 0
        for idx in indices:
            x0[idx] = per_link_power

    # Bounds: each link power in [0, max_power]
    bounds = Bounds(lb=0, ub=max_power)

    # Optimization options
    options = {'maxiter': 500, 'ftol': 1e-6, 'disp': False}

    try:
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=[linear_constraint],
            options=options
        )

        # Reconstruct optimal power matrix
        optimal_powers = np.zeros((num_agents, num_agents))
        for idx, (i, j) in enumerate(active_links):
            optimal_powers[i, j] = max(result.x[idx], 0.0)

        # Check constraints and correct (numerical tolerance)
        for i in range(num_agents):
            total = np.sum(optimal_powers[i, :])
            if total > max_power + 1e-6:
                scale = max_power / total
                optimal_powers[i, :] *= scale

        result_info = {
            'success': result.success,
            'message': result.message,
            'fun': result.fun,
            'nit': result.nit,
            'nfev': result.nfev
        }
        return optimal_powers, result_info

    except Exception as e:
        print(f"Power optimization exception: {e}")
        # Fallback: uniform allocation
        fallback_powers = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    fallback_powers[i, j] = max_power / (num_agents - 1)
        return fallback_powers, {'success': False, 'message': str(e), 'fun': np.inf, 'nit': 0, 'nfev': 0}

def optimize_channel_allocation_fixed_powers(
    positions, power_matrix, coefficients, C, weights,
    num_channels, bandwidth, noise_dbm, sinr_threshold_db,
    distance_scale, a, b, seed, max_iter=5
):
    """
    Channel allocation optimization under fixed power allocation (aiming to minimize J_hat).
    """
    num_agents = positions.shape[0]

    # Build all possible links (i,j)
    active_links = []
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                active_links.append((i, j))

    # Initialize channel assignment matrix
    channel_assignments = np.zeros((num_agents, num_agents), dtype=int)

    # Construct mapping from transmitter to links
    tx_to_links = {}
    for i, j in active_links:
        tx_to_links.setdefault(i, []).append((i, j))

    # Initial channel assignment: assign different channels to links from the same transmitter (cyclic allocation)
    for i, links in tx_to_links.items():
        for idx, (_, j) in enumerate(links):
            channel_assignments[i, j] = idx % num_channels

    # Convert power matrix to dictionary format
    power_allocations = [{} for _ in range(num_agents)]
    for i, j in active_links:
        power_allocations[i][j] = power_matrix[i, j]

    # Current channel selection dictionary
    channel_selections = [{} for _ in range(num_agents)]
    for i, j in active_links:
        channel_selections[i][j] = channel_assignments[i, j]

    # Compute initial J_hat
    current_J = compute_J_hat(
        power_allocations, channel_selections, positions, coefficients,
        max_power=np.max(power_matrix),  # actual max_power is already ensured externally
        num_channels=num_channels,
        bandwidth=bandwidth,
        noise_dbm=noise_dbm,
        sinr_threshold_db=sinr_threshold_db,
        distance_scale=distance_scale,
        a=a,
        b=b,
        seed=seed,
        C=C
    )

    print(f"  Initial channel allocation J_hat = {current_J:.6f}")

    # Greedy iterative optimization
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        # Process links in descending order of transmit power (links with higher power prioritized for adjustment)
        links_with_power = []
        for i, j in active_links:
            if power_matrix[i, j] > 1e-12:
                links_with_power.append((power_matrix[i, j], (i, j)))
        links_with_power.sort(reverse=True, key=lambda x: x[0])

        for _, (i, j) in links_with_power:
            current_channel = channel_assignments[i, j]
            best_channel = current_channel
            best_J = current_J

            # Get set of channels already occupied by the same transmitter
            used_channels = set()
            for (tx_i, tx_j) in tx_to_links.get(i, []):
                if tx_j != j:
                    used_channels.add(channel_assignments[tx_i, tx_j])

            # Try all possible channels
            for new_ch in range(num_channels):
                if new_ch in used_channels:
                    continue

                # Temporarily change channel
                old_ch = channel_assignments[i, j]
                channel_assignments[i, j] = new_ch
                channel_selections[i][j] = new_ch

                # Compute new J_hat
                new_J = compute_J_hat(
                    power_allocations, channel_selections, positions, coefficients,
                    max_power=np.max(power_matrix),
                    num_channels=num_channels,
                    bandwidth=bandwidth,
                    noise_dbm=noise_dbm,
                    sinr_threshold_db=sinr_threshold_db,
                    distance_scale=distance_scale,
                    a=a,
                    b=b,
                    seed=seed,
                    C=C
                )

                if new_J < best_J - 1e-12:   # strict improvement
                    best_J = new_J
                    best_channel = new_ch
                    improved = True

                # Restore original channel (wait to apply best channel later)
                channel_assignments[i, j] = old_ch
                channel_selections[i][j] = old_ch

            # Apply best channel if improvement
            if best_channel != current_channel:
                channel_assignments[i, j] = best_channel
                channel_selections[i][j] = best_channel
                current_J = best_J

        if improved:
            print(f"    Channel optimization iteration {iteration}: J_hat = {current_J:.6f}")

    return channel_assignments, current_J

def alternating_allocator(positions, config):
    """
    Alternating optimization allocator aiming to minimize the actual observed loss J_hat.

    Args:
        positions: array of terminal positions [N, 2]
        config: configuration dictionary, must contain the following keys:
                num_agents, num_channels, max_power, bandwidth, noise_dbm,
                sinr_threshold_db, distance_scale, pathloss_a, pathloss_b,
                weights, coefficients, alternating_iterations, seed
                optional: power_optimization_method (default 'SLSQP')

    Returns:
        neighbor_lists: neighbor list for each agent, each element is a list of neighbor IDs
    """
    # Extract parameters from config
    num_agents = config['num_agents']
    weights = np.array(config['weights'])                 # ensure array
    coefficients = np.array(config['coefficients'])       # ensure array
    max_power = config['max_power']
    num_channels = config['num_channels']
    sinr_threshold_db = config['sinr_threshold_db']
    bandwidth = config['bandwidth']
    noise_dbm = config['noise_dbm']
    distance_scale = config['distance_scale']
    a = config['pathloss_a']
    b = config['pathloss_b']
    alternating_iterations = config.get('alternating_iterations', 5)
    power_method = config.get('power_optimization_method', 'SLSQP')
    seed = config.get('seed', None)

    # Compute penalty coefficients C[m] = rho[m]^2 + (K-1)*rho[m]
    rho = weights
    K = num_agents
    C = rho**2 + (K - 1) * rho

    print(f"Running alternating optimization algorithm (max {alternating_iterations} inner-outer iterations, objective: minimize J_hat)...")
    print(f"Number of agents: {num_agents}, Number of channels: {num_channels}")

    try:
        # Initialize: random channel assignment (ensure different channels for links from same transmitter)
        channel_assignments = np.zeros((num_agents, num_agents), dtype=int)
        for i in range(num_agents):
            # Cyclic allocation for outgoing links of agent i
            for idx, j in enumerate(range(num_agents)):
                if i != j:
                    channel_assignments[i, j] = idx % num_channels

        # Initial power allocation: uniform (each transmitter total power = max_power, uniformly distributed to all outgoing links)
        power_matrix = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            per_link_power = max_power / (num_agents - 1)
            for j in range(num_agents):
                if i != j:
                    power_matrix[i, j] = per_link_power

        # Convert to dictionary format
        power_allocations = [{} for _ in range(num_agents)]
        channel_selections = [{} for _ in range(num_agents)]
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    power_allocations[i][j] = power_matrix[i, j]
                    channel_selections[i][j] = channel_assignments[i, j]

        # Compute initial J_hat
        best_J = compute_J_hat(
            power_allocations, channel_selections, positions, coefficients,
            max_power, num_channels, bandwidth, noise_dbm, sinr_threshold_db,
            distance_scale, a, b, seed, C
        )
        best_power = power_matrix.copy()
        best_channel = channel_assignments.copy()
        print(f"Initial J_hat = {best_J:.6f}")

        # Alternating optimization main loop
        for ao_iter in range(alternating_iterations):
            print(f"\n--- Alternating optimization iteration {ao_iter+1}/{alternating_iterations} ---")

            # Step A: fix channels, optimize power
            print("A. Power optimization...")
            opt_power, power_info = optimize_power_allocation_fixed_channels(
                positions=positions,
                channel_matrix=best_channel,
                coefficients=coefficients,
                C=C,
                weights=weights,
                max_power=max_power,
                num_channels=num_channels,
                bandwidth=bandwidth,
                noise_dbm=noise_dbm,
                sinr_threshold_db=sinr_threshold_db,
                distance_scale=distance_scale,
                a=a,
                b=b,
                seed=seed,                 # use fixed seed to ensure consistent fading realization
                method=power_method
            )

            if power_info['success']:
                print(f"  Power optimization completed, J_hat = {power_info['fun']:.6f}")
            else:
                print(f"  Power optimization warning: {power_info['message']}")

            # Update power matrix
            power_matrix = opt_power

            # Reconstruct power allocation dictionary for subsequent computation
            power_allocations = [{} for _ in range(num_agents)]
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:
                        power_allocations[i][j] = power_matrix[i, j]

            # Step B: fix powers, optimize channels
            print("B. Channel optimization...")
            opt_channel, channel_J = optimize_channel_allocation_fixed_powers(
                positions=positions,
                power_matrix=power_matrix,
                coefficients=coefficients,
                C=C,
                weights=weights,
                num_channels=num_channels,
                bandwidth=bandwidth,
                noise_dbm=noise_dbm,
                sinr_threshold_db=sinr_threshold_db,
                distance_scale=distance_scale,
                a=a,
                b=b,
                seed=seed,                 # also fixed seed
                max_iter=5
            )
            channel_assignments = opt_channel
            print(f"  Channel optimization completed, J_hat = {channel_J:.6f}")

            # Update channel selection dictionary
            channel_selections = [{} for _ in range(num_agents)]
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:
                        channel_selections[i][j] = channel_assignments[i, j]

            # Compute current total J_hat
            current_J = compute_J_hat(
                power_allocations, channel_selections, positions, coefficients,
                max_power, num_channels, bandwidth, noise_dbm, sinr_threshold_db,
                distance_scale, a, b, seed, C
            )
            print(f"  Current total J_hat = {current_J:.6f}")

            # Update best solution
            if current_J < best_J - 1e-12:
                best_J = current_J
                best_power = power_matrix.copy()
                best_channel = channel_assignments.copy()
                print(f"  → Updated best solution, J_hat = {best_J:.6f}")

        # Compute final adjacency matrix and neighbor lists using best solution
        print(f"\n--- Optimization completed, best J_hat = {best_J:.6f} ---")

        # Reconstruct best power and channel allocations
        best_power_alloc = [{} for _ in range(num_agents)]
        best_channel_sel = [{} for _ in range(num_agents)]
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    best_power_alloc[i][j] = best_power[i, j]
                    best_channel_sel[i][j] = best_channel[i, j]

        # Compute final SINR matrix and adjacency matrix (still using the same seed)
        sinr_matrix, final_adjacency, _, _ = calculate_sinr_matrix(
            positions=positions,
            power_allocations=best_power_alloc,
            channel_selections=best_channel_sel,
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

        neighbor_lists = get_neighbors_from_adjacency(final_adjacency)
        stats = calculate_connectivity_stats(final_adjacency)

        # Print final statistics
        total_power_used = np.sum(best_power)
        print(f"Connectivity statistics:")
        print(f"  Total possible links: {stats['total_possible_links']}")
        print(f"  Total connected links: {stats['total_connected_links']}")
        print(f"  Connectivity ratio: {stats['connectivity_ratio']:.2%}")
        print(f"  Average connections per device: {stats['avg_connections_per_device']:.2f}")
        print(f"  Total power used: {total_power_used:.6f} W")
        print(f"  Neighbor lists: {neighbor_lists}")

        return neighbor_lists

    except Exception as e:
        print(f"Alternating optimization allocation failed: {e}")
        import traceback
        traceback.print_exc()
        return [[] for _ in range(num_agents)]