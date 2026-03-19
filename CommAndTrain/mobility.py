#!/usr/bin/env python
"""
Terminal mobility trajectory generator
"""
import numpy as np


class TrajectoryGenerator:
    """Simple trajectory generator for simulating terminal mobility."""

    def __init__(self, config):
        self.num_agents = config.num_agents
        self.config = config
        self.radius = config.deployment_radius
        np.random.seed(config.seed)

        # Generate initial positions
        self.positions = self._generate_random_positions()
        self.step_count = 0

    def _generate_random_positions(self):
        positions = []
        for _ in range(self.num_agents):
            r = self.radius * np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            positions.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(positions)

    def get_positions(self, round_idx=None):
        """Get current positions, optionally add random movement."""
        if not self.config.mobility_enabled:
            return self.positions.copy()

        # Simple movement: add small random perturbation each step
        max_speed = self.config.max_speed
        noise = np.random.normal(0, max_speed, self.positions.shape)
        self.positions += noise

        # Ensure positions stay within radius
        distances = np.linalg.norm(self.positions, axis=1)
        mask = distances > self.radius
        if mask.any():
            self.positions[mask] = self.positions[mask] * (self.radius / distances[mask].reshape(-1, 1))

        return self.positions.copy()