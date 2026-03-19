#!/usr/bin/env python
"""
Wireless computation utility module - Provides path loss and SINR calculation consistent with MAPPO.
Extended version: Adds link success probability calculation and optimization functions.
"""
import numpy as np
import math
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from typing import Dict, Tuple, List, Optional
import warnings


def calculate_pathloss(distance, distance_scale=1000.0, a=128.1, b=37.6):
    """
    Calculate path loss (dB) - exactly consistent with the MAPPO training environment.

    Args:
        distance: Distance (meters)
        distance_scale: Distance scaling factor (default 1000.0, converts to km)
        a: Path loss constant (128.1 dB)
        b: Path loss coefficient (37.6)

    Returns:
        pl_db: Path loss (dB)
    """
    if distance <= 0:
        return 0.0

    # Convert to km
    d_km = distance / distance_scale

    # Path loss model: PL(d) = a + b*log10(d_km) dB
    pl_db = a + b * np.log10(d_km)

    return pl_db


def calculate_channel_gains(positions, distance_scale=1000.0, a=128.1, b=37.6):
    """
    Compute large-scale channel gain matrix - exactly consistent with the MAPPO training environment.

    Args:
        positions: Array of terminal positions [N, 2]
        distance_scale: Distance scaling factor
        a: Path loss constant
        b: Path loss coefficient

    Returns:
        G: Large-scale channel gain matrix [N, N]
    """
    num_agents = positions.shape[0]
    G = np.zeros((num_agents, num_agents))

    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                G[i, j] = 0.0
                continue

            # Calculate distance (meters)
            dist = np.linalg.norm(positions[i] - positions[j])

            # Calculate path loss (dB)
            pl_db = calculate_pathloss(dist, distance_scale, a, b)

            # Convert to linear gain
            G[i, j] = 10 ** (-pl_db / 10.0)

    return G


def add_rayleigh_fading(G, seed=None):
    """
    Add Rayleigh small-scale fading - exactly consistent with the MAPPO training environment.

    Args:
        G: Large-scale channel gain matrix [N, N]
        seed: Random seed

    Returns:
        G_total: Total channel gain matrix [N, N]
        H_small_scale: Small-scale fading matrix [N, N]
    """
    if seed is not None:
        np.random.seed(seed)

    num_agents = G.shape[0]

    # Rayleigh fading: |h|^2 follows exponential distribution with mean 1
    H_small_scale = np.random.exponential(scale=1.0, size=(num_agents, num_agents))
    np.fill_diagonal(H_small_scale, 0)  # Set diagonal to 0

    # Total channel gain = large-scale gain * small-scale gain
    G_total = G * H_small_scale

    return G_total, H_small_scale


def calculate_sinr_matrix(positions, power_allocations, channel_selections,
                         coefficients, max_power=0.2, num_channels=3,
                         bandwidth=1e6, noise_dbm=-174, sinr_threshold_db=10.0,
                         distance_scale=1000.0, a=128.1, b=37.6, seed=None):
    """
    Compute SINR matrix - exactly consistent with the MAPPO training environment.

    Args:
        positions: Terminal positions [N, 2]
        power_allocations: List of power allocations, each element is a dictionary {target terminal: power (W)}
        channel_selections: List of channel selections, each element is a dictionary {target terminal: channel index}
        coefficients: Self-interference coefficients [N]
        max_power: Maximum power (W)
        num_channels: Number of channels
        bandwidth: Bandwidth (Hz)
        noise_dbm: Noise power spectral density (dBm/Hz)
        sinr_threshold_db: SINR threshold (dB)
        distance_scale: Distance scaling factor
        a: Path loss constant
        b: Path loss coefficient
        seed: Random seed

    Returns:
        sinr_matrix: SINR matrix [N, N] (dB)
        adjacency_matrix: Adjacency matrix [N, N] (boolean)
        G_total: Total channel gain matrix [N, N]
        H_small_scale: Small-scale fading matrix [N, N]
    """
    num_agents = positions.shape[0]

    # 1. Compute large-scale channel gains (path loss)
    G = calculate_channel_gains(positions, distance_scale, a, b)

    # 2. Add small-scale fading (Rayleigh fading)
    G_total, H_small_scale = add_rayleigh_fading(G, seed)

    # 3. Noise power calculation
    noise_psd_linear = 10 ** ((noise_dbm - 30) / 10.0)  # dBm/Hz -> W/Hz
    noise_power = noise_psd_linear * bandwidth  # Total noise power (W)

    # 4. SINR threshold (linear value)
    sinr_threshold_linear = 10 ** (sinr_threshold_db / 10.0)

    # 5. Initialize SINR matrix
    sinr_matrix = np.full((num_agents, num_agents), -np.inf)  # in dB
    adjacency_matrix = np.zeros((num_agents, num_agents), dtype=bool)

    # 6. Compute SINR for each link
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue

            # Check if power is allocated
            if j not in power_allocations[i]:
                continue

            # Transmit power and channel
            p_ij = power_allocations[i][j]
            channel_ij = channel_selections[i][j]

            # Desired signal power
            signal_power = p_ij * G_total[i, j]

            # Calculate total interference
            total_interference = noise_power  # start with noise

            # Self-interference: receiver j transmits on the same channel to other terminals
            for k in range(num_agents):
                if k == j:  # skip itself
                    continue

                # Check if j transmits to k and uses the same channel
                if k in power_allocations[j]:
                    channel_jk = channel_selections[j][k]
                    if channel_jk == channel_ij:  # same channel
                        p_jk = power_allocations[j][k]
                        # Self-interference term: multiplied by self-interference coefficient
                        total_interference += p_jk * coefficients[j]

            # Co-channel interference: from other transmitters using the same channel
            for k in range(num_agents):
                if k == i or k == j:  # exclude current transmitter and receiver
                    continue

                # Check if k transmits to j and uses the same channel
                if j in power_allocations[k]:
                    channel_kj = channel_selections[k][j]
                    if channel_kj == channel_ij:  # same channel
                        p_kj = power_allocations[k][j]
                        total_interference += p_kj * G_total[k, j]

            # Compute SINR (linear value)
            sinr_linear = signal_power / total_interference if total_interference > 0 else 0

            # Convert to dB
            sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -np.inf
            sinr_matrix[i, j] = sinr_db

            # Determine connectivity (hard decision)
            adjacency_matrix[i, j] = sinr_linear >= sinr_threshold_linear

    return sinr_matrix, adjacency_matrix, G_total, H_small_scale


def get_neighbors_from_adjacency(adjacency_matrix):
    """
    Obtain neighbor lists from adjacency matrix.

    Args:
        adjacency_matrix: Adjacency matrix [N, N] (boolean)

    Returns:
        neighbor_lists: List of neighbor lists, each element is a list of neighbor IDs for terminal i
    """
    num_agents = adjacency_matrix.shape[0]
    neighbor_lists = []

    for i in range(num_agents):
        neighbors = np.where(adjacency_matrix[i, :])[0].tolist()
        neighbor_lists.append(neighbors)

    return neighbor_lists


def calculate_connectivity_stats(adjacency_matrix):
    """
    Compute connectivity statistics.

    Args:
        adjacency_matrix: Adjacency matrix [N, N]

    Returns:
        stats: Dictionary of statistics
    """
    num_agents = adjacency_matrix.shape[0]

    # Calculate total number of links
    total_possible_links = num_agents * (num_agents - 1)
    total_connected_links = np.sum(adjacency_matrix)

    # Calculate connections per device
    connections_per_device = np.sum(adjacency_matrix, axis=1)

    # Statistics
    stats = {
        'num_agents': num_agents,
        'total_possible_links': total_possible_links,
        'total_connected_links': total_connected_links,
        'connectivity_ratio': total_connected_links / total_possible_links if total_possible_links > 0 else 0,
        'avg_connections_per_device': np.mean(connections_per_device),
        'min_connections_per_device': np.min(connections_per_device),
        'max_connections_per_device': np.max(connections_per_device),
    }

    return stats


def calculate_link_success_probability(
    positions, power_allocations, channel_selections, coefficients,
    bandwidth=1e6, noise_dbm=-174, sinr_threshold_db=10.0,
    distance_scale=1000.0, a=128.1, b=37.6, seed=None
):
    """
    Compute the link transmission success probability matrix φ_{k,k'}.

    Based on formula: φ_{k,k'} = P[SINR_{k,k'} > γ_th]
                     = A_{k,k'}^n ∏_{m≠k,k'} ∏_{m'≠m}^n 1/(1 + B_{k,k'}^{m,m'})

    Args:
        positions: Terminal positions [N, 2]
        power_allocations: List of power allocations, each element is a dictionary {target terminal: power (W)}
        channel_selections: List of channel selections, each element is a dictionary {target terminal: channel index}
        coefficients: Self-interference coefficients [N]
        bandwidth: Bandwidth (Hz)
        noise_dbm: Noise power spectral density (dBm/Hz)
        sinr_threshold_db: SINR threshold (dB)
        distance_scale: Distance scaling factor
        a: Path loss constant
        b: Path loss coefficient
        seed: Random seed

    Returns:
        phi_matrix: Link success probability matrix [N, N]
    """
    num_agents = positions.shape[0]

    # 1. Compute large-scale channel gains
    G = calculate_channel_gains(positions, distance_scale, a, b)

    # 2. Noise power calculation
    noise_psd_linear = 10 ** ((noise_dbm - 30) / 10.0)  # dBm/Hz -> W/Hz
    noise_power = noise_psd_linear * bandwidth  # Total noise power (W)

    # 3. SINR threshold (linear value)
    gamma_th = 10 ** (sinr_threshold_db / 10.0)

    # 4. Initialize matrix
    phi_matrix = np.zeros((num_agents, num_agents))

    # 5. Compute success probability for each link
    for k in range(num_agents):
        for kp in range(num_agents):
            if k == kp:
                continue

            # Check if the link exists
            if kp not in power_allocations[k]:
                phi_matrix[k, kp] = 0
                continue

            # Get link parameters
            p_kkp = power_allocations[k][kp]
            c_kkp = channel_selections[k][kp]

            if p_kkp < 1e-12:
                phi_matrix[k, kp] = 0
                continue

            # Compute channel gain
            g_kkp = G[k, kp]

            if g_kkp < 1e-12:
                phi_matrix[k, kp] = 0
                continue

            # 1. Calculate self-interference part
            self_interference_sum = 0
            for n in range(num_agents):
                if n == kp:
                    continue

                # Check if receiver kp transmits to n on the same channel
                if (kp in channel_selections and
                    n in channel_selections[kp] and
                    n in power_allocations[kp] and
                    channel_selections[kp][n] == c_kkp):

                    p_kp_n = power_allocations[kp][n]
                    self_interference_sum += p_kp_n * coefficients[kp]

            # 2. Compute term A
            denominator = p_kkp * g_kkp
            exponent_A = -(gamma_th * (noise_power + self_interference_sum)) / denominator
            term_A = np.exp(exponent_A)

            # 3. Compute product of B terms
            prod_B = 1.0

            for m in range(num_agents):
                if m == k or m == kp:
                    continue

                for mp in range(num_agents):
                    if mp == m:
                        continue

                    # Check if link (m, mp) exists and uses the same channel
                    if (mp in power_allocations[m] and
                        mp in channel_selections[m] and
                        channel_selections[m][mp] == c_kkp):

                        p_m_mp = power_allocations[m][mp]
                        g_m_kp = G[m, kp]

                        # Compute B value
                        B_val = (gamma_th * p_m_mp * g_m_kp) / denominator

                        # Update product term
                        prod_B *= (1.0 / (1.0 + B_val))

            # 4. Compute success probability
            phi_matrix[k, kp] = term_A * prod_B

            # Ensure probability is within [0, 1]
            phi_matrix[k, kp] = max(0.0, min(1.0, phi_matrix[k, kp]))

    return phi_matrix


def calculate_system_objective(
    positions, power_allocations, channel_selections, coefficients,
    weights=None, bandwidth=1e6, noise_dbm=-174, sinr_threshold_db=10.0,
    distance_scale=1000.0, a=128.1, b=37.6, seed=None
):
    """
    Compute the system objective function value.

    Objective function: J(P, C) = ∑_{k} ∑_{k'} (1 - φ_{k,k'}) [weights_{k}^2 + (n-1)weights_{k}]

    Args:
        positions: Terminal positions
        power_allocations: Power allocation
        channel_selections: Channel allocation
        coefficients: Self-interference coefficients
        weights: Terminal weights
        bandwidth: Bandwidth
        noise_dbm: Noise power
        sinr_threshold_db: SINR threshold
        distance_scale: Distance scaling factor
        a: Path loss constant
        b: Path loss coefficient
        seed: Random seed

    Returns:
        total_cost: Objective function value
        phi_matrix: Success probability matrix
    """
    num_agents = positions.shape[0]

    # Compute success probability matrix
    phi_matrix = calculate_link_success_probability(
        positions, power_allocations, channel_selections, coefficients,
        bandwidth, noise_dbm, sinr_threshold_db,
        distance_scale, a, b, seed
    )

    # Compute objective function
    total_cost = 0.0

    for k in range(num_agents):
        for kp in range(num_agents):
            if k == kp:
                continue

            # Check if the link exists
            if kp in power_allocations[k]:
                p_kkp = power_allocations[k][kp]
                if p_kkp > 1e-12:
                    phi = phi_matrix[k, kp]
                    cost_term = (1 - phi) * (weights[k]*2 + (num_agents-1) * weights[k])
                    total_cost += cost_term

    return total_cost, phi_matrix