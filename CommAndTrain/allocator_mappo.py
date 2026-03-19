#!/usr/bin/env python
"""
DFL MAPPO Allocator - revised version (aligned with random/ao/avg interface)
Allocate power and channels using a trained MAPPO model based on terminal positions, weights, and self-interference coefficients.
"""
import sys
import os
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import wireless computation tools (including small-scale fading)
from wireless_utils import (
    calculate_channel_gains,
    calculate_sinr_matrix,
    get_neighbors_from_adjacency,
    calculate_connectivity_stats
)

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = str(Path(current_dir).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

mappo_path = "/data/8T/zhouh/on-policy"
if mappo_path not in sys.path:
    sys.path.append(mappo_path)

try:
    from gym import spaces
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    print("✓ Successfully imported MAPPO module")
except ImportError as e:
    print(f"✗ Failed to import MAPPO module: {e}")
    raise


class Namespace:
    """Simple namespace class to convert dictionary to object."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MAPPOAllocator:
    """DFL MAPPO allocator - perform communication allocation based on trained MAPPO model."""

    def __init__(self, config_path: str, actor_path: str):
        """
        Initialize allocator (model parameters are loaded from training configuration file).
        """
        print("=" * 60)
        print("DFL MAPPO Allocator Initialization")
        print("=" * 60)

        # Load training configuration file
        self.config = self._load_config(config_path)
        print(f"✓ Configuration file loaded successfully: {config_path}")

        # Extract model-related parameters (consistent with training environment)
        self.num_agents = int(self.config.get('num_agents', 5))
        self.num_channels = int(self.config.get('num_channels', 8))
        self.max_power = float(self.config.get('max_power', 0.2))
        self.bandwidth = float(self.config.get('bandwidth', 1e6))
        self.noise_dbm = float(self.config.get('noise_dbm', -174))
        self.gamma_th = float(self.config.get('gamma_th', 10.0))
        self.deployment_radius = float(self.config.get('deployment_radius', 200.0))
        self.normalization_method = str(self.config.get('normalization_method', 'log_standard'))

        # Wireless parameters (consistent with training environment)
        self.distance_scale = float(self.config.get('distance_scale', 1000.0))
        self.pathloss_a = float(self.config.get('pathloss_a', 128.1))
        self.pathloss_b = float(self.config.get('pathloss_b', 37.6))

        # Network parameters
        self.hidden_size = int(self.config.get('hidden_size', 64))
        self.recurrent_N = int(self.config.get('recurrent_N', 1))
        self.layer_N = int(self.config.get('layer_N', 1))
        self.use_ReLU = bool(self.config.get('use_ReLU', True))
        self.use_orthogonal = bool(self.config.get('use_orthogonal', True))
        self.use_popart = bool(self.config.get('use_popart', False))
        self.use_feature_normalization = bool(self.config.get('use_feature_normalization', False))
        self.use_naive_recurrent_policy = bool(self.config.get('use_naive_recurrent_policy', False))
        self.use_recurrent_policy = bool(self.config.get('use_recurrent_policy', True))

        # Compute observation and action dimensions
        self.obs_dim = 4 * self.num_agents + 2
        self.num_links_per_agent = self.num_agents - 1
        # Action dimension = power part (1 per link) + channel part (num_channels per link)
        self.action_dim_per_agent = self.num_links_per_agent + self.num_links_per_agent * self.num_channels

        print(f"\nModel configuration:")
        print(f"  Number of agents: {self.num_agents}")
        print(f"  Number of channels: {self.num_channels}")
        print(f"  Observation dimension: {self.obs_dim}")
        print(f"  Action dimension: {self.action_dim_per_agent}")
        print(f"  Normalization method: {self.normalization_method}")

        # Initialize policy network
        try:
            self.policy = self._init_policy()
            self._load_weights(actor_path)
            print("✓ Policy network initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Policy network initialization failed: {e}")

        print("=" * 60)

    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        processed_config = {}
        for key, value in config.items():
            if key == '_wandb':
                continue
            if isinstance(value, dict) and 'value' in value:
                processed_config[key] = value['value']
            else:
                processed_config[key] = value
        return processed_config

    def _init_policy(self):
        """Initialize policy network (consistent with training)."""
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        act_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_dim_per_agent,), dtype=np.float32)

        policy_config = {
            'algorithm_name': 'mappo',
            'recurrent_N': self.recurrent_N,
            'hidden_size': self.hidden_size,
            'layer_N': self.layer_N,
            'use_ReLU': self.use_ReLU,
            'use_popart': self.use_popart,
            'use_feature_normalization': self.use_feature_normalization,
            'use_orthogonal': self.use_orthogonal,
            'gain': 0.01,
            'use_naive_recurrent_policy': self.use_naive_recurrent_policy,
            'use_recurrent_policy': self.use_recurrent_policy,
            'use_cuda': False,
            'cuda': False,
            'share_policy': True,
            'use_centralized_V': True,
            'use_obs_instead_of_state': False,
            # Training parameters (ignored during inference, but required for initialization)
            'lr': 3e-4,
            'critic_lr': 5e-4,
            'opti_eps': 1e-5,
            'weight_decay': 0,
            'clip_param': 0.2,
            'ppo_epoch': 15,
            'num_mini_batch': 1,
            'data_chunk_length': 10,
            'value_loss_coef': 1,
            'entropy_coef': 0.01,
            'max_grad_norm': 10,
            'use_huber_loss': True,
            'use_valuenorm': True,
            'huber_delta': 10,
            'use_linear_lr_decay': True,
            'use_clipped_value_loss': True,
            'use_gae': True,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'use_proper_time_limits': True,
            'use_policy_active_masks': True,
            'use_value_active_masks': True,
            'use_max_grad_norm': True,
            'use_stacked_frames': False,
            'stacked_frames': 1,
            'n_rollout_threads': 1,
            'n_eval_rollout_threads': 1,
            'n_render_rollout_threads': 1,
            'use_render': False,
            'use_eval': False,
            'save_gifs': False,
            'model_dir': None,
            'run_dir': './',
        }

        policy_args = Namespace(**policy_config)
        return R_MAPPOPolicy(
            policy_args,
            obs_space,
            share_obs_space,
            act_space,
            device=torch.device('cpu')
        )

    def _load_weights(self, actor_path: str):
        """Load model weights."""
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Model file not found: {actor_path}")
        actor_dict = torch.load(actor_path, map_location='cpu')
        if isinstance(actor_dict, dict):
            if 'actor' in actor_dict:
                actor_dict = actor_dict['actor']
            elif 'model' in actor_dict:
                actor_dict = actor_dict['model']
            elif 'state_dict' in actor_dict:
                actor_dict = actor_dict['state_dict']
            elif 'module' in actor_dict:
                actor_dict = actor_dict['module']
        self.policy.actor.load_state_dict(actor_dict)
        self.policy.actor.eval()
        print(f"✓ Model weights loaded successfully: {actor_path}")

    def _calculate_channel_gains(self, positions: np.ndarray) -> np.ndarray:
        """Compute large-scale channel gains (path loss)."""
        return calculate_channel_gains(
            positions,
            distance_scale=self.distance_scale,
            a=self.pathloss_a,
            b=self.pathloss_b
        )

    def _compute_normalization_stats(self, G: np.ndarray):
        """Compute normalization statistics for channel gains (consistent with training environment)."""
        G_log = np.log10(G + 1e-20)
        G_log_offdiag = G_log[~np.eye(self.num_agents, dtype=bool)].flatten()
        if self.normalization_method == "log_standard":
            self.G_log_mean = np.mean(G_log_offdiag)
            self.G_log_std = max(np.std(G_log_offdiag), 1e-6)
        elif self.normalization_method == "log_minmax":
            self.g_log_min = np.min(G_log_offdiag)
            self.g_log_max = np.max(G_log_offdiag)
            if self.g_log_max - self.g_log_min < 1e-6:
                self.g_log_max = self.g_log_min + 1.0
        elif self.normalization_method == "relative":
            G_nonzero = G[G > 0]
            self.g_min_nonzero = np.min(G_nonzero) if len(G_nonzero) > 0 else 1e-20

    def _normalize_channel_gains(self, gains: np.ndarray, agent_id: int) -> np.ndarray:
        """Normalize channel gain vector (consistent with training environment)."""
        if self.normalization_method == "none":
            return gains.copy()
        elif self.normalization_method == "log_standard":
            gains_log = np.log10(gains + 1e-20)
            return (gains_log - self.G_log_mean) / self.G_log_std
        elif self.normalization_method == "log_minmax":
            gains_log = np.log10(gains + 1e-20)
            return 2 * (gains_log - self.g_log_min) / (self.g_log_max - self.g_log_min) - 1
        elif self.normalization_method == "relative":
            gains_normalized = np.zeros_like(gains)
            for j, g in enumerate(gains):
                if j == agent_id:
                    gains_normalized[j] = 0.0
                elif g <= 0:
                    gains_normalized[j] = -1.0
                else:
                    gains_normalized[j] = np.log10(g / self.g_min_nonzero) / 5.0
            return gains_normalized
        elif self.normalization_method == "simple":
            gains_log = np.log10(gains + 1e-20)
            return gains_log / 10.0
        else:  # Default log standardization
            gains_log = np.log10(gains + 1e-20)
            return (gains_log - self.G_log_mean) / self.G_log_std

    def _generate_observations(self, positions: np.ndarray,
                               weights: np.ndarray,
                               coefficients: np.ndarray) -> np.ndarray:
        """Generate observation vectors (exactly as in training environment)."""
        num_agents = positions.shape[0]
        G = self._calculate_channel_gains(positions)
        self._compute_normalization_stats(G)

        global_static_table = np.concatenate([weights, coefficients])
        noise_psd_linear = 10**((self.noise_dbm - 30) / 10.0)
        gamma_th_lin = 10**(self.gamma_th / 10.0)
        constants = np.array([noise_psd_linear, gamma_th_lin], dtype=np.float32)

        obs_list = []
        for i in range(num_agents):
            my_id = np.zeros(num_agents, dtype=np.float32)
            my_id[i] = 1.0
            my_gains = G[i, :].copy()
            my_gains_normalized = self._normalize_channel_gains(my_gains, i)
            obs = np.concatenate([
                my_id,
                global_static_table,
                my_gains_normalized,
                constants
            ], dtype=np.float32)
            obs_list.append(obs)

        return np.array(obs_list)

    def _get_model_actions(self, obs: np.ndarray) -> np.ndarray:
        """Get model actions (deterministic inference)."""
        obs_tensor = torch.from_numpy(obs).float()
        batch_size = obs.shape[0]
        recurrent_n = self.recurrent_N
        hidden_size = self.hidden_size

        rnn_states = torch.zeros(batch_size, recurrent_n, hidden_size)
        masks = torch.ones(batch_size, 1)

        with torch.no_grad():
            actions, _, _ = self.policy.actor(
                obs_tensor,
                rnn_states,
                masks,
                deterministic=True  # Use deterministic actions during inference
            )
        return actions.detach().cpu().numpy()

    def _parse_actions(self, actions: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse action vectors into power allocations and channel selections (hard selection, consistent with training step).
        Returns:
            power_allocations: List[Dict], each element is {target agent: power value}
            channel_selections: List[Dict], each element is {target agent: channel index}
        """
        num_agents = actions.shape[0]
        power_allocations = []
        channel_selections = []

        for i in range(num_agents):
            act = actions[i].flatten()

            # 1. Power part: first num_links_per_agent dimensions, softmax normalized then multiplied by max_power
            power_logits = act[:self.num_links_per_agent]
            # Numerically stable softmax
            exp_power = np.exp(power_logits - np.max(power_logits))
            power_weights = exp_power / (np.sum(exp_power) + 1e-12)
            actual_powers = power_weights * self.max_power  # Total power per link

            # 2. Channel part: remaining num_links_per_agent * num_channels dimensions, reshape then softmax per row
            channel_logits = act[self.num_links_per_agent:].reshape(self.num_links_per_agent, self.num_channels)
            # Softmax for each link
            exp_channel = np.exp(channel_logits - np.max(channel_logits, axis=1, keepdims=True))
            channel_probs = exp_channel / (np.sum(exp_channel, axis=1, keepdims=True) + 1e-12)
            # Hard channel selection: pick channel with highest probability
            selected_channels = np.argmax(channel_probs, axis=1)  # shape (num_links_per_agent,)

            # 3. Build allocation dictionaries
            power_alloc = {}
            channel_select = {}
            target_neighbors = [j for j in range(num_agents) if j != i]
            for idx, neighbor in enumerate(target_neighbors):
                power_alloc[neighbor] = actual_powers[idx]
                channel_select[neighbor] = int(selected_channels[idx])  # ensure Python int

            power_allocations.append(power_alloc)
            channel_selections.append(channel_select)

        return power_allocations, channel_selections

    def _calculate_sinr_matrix(self, positions: np.ndarray,
                               power_allocations: List[Dict],
                               channel_selections: List[Dict],
                               coefficients: np.ndarray,
                               seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SINR matrix and adjacency matrix (wireless_utils.calculate_sinr_matrix already includes small-scale fading).
        """
        sinr_matrix, adjacency_matrix, _, _ = calculate_sinr_matrix(
            positions=positions,
            power_allocations=power_allocations,
            channel_selections=channel_selections,
            coefficients=coefficients,
            max_power=self.max_power,
            num_channels=self.num_channels,
            bandwidth=self.bandwidth,
            noise_dbm=self.noise_dbm,
            sinr_threshold_db=self.gamma_th,
            distance_scale=self.distance_scale,
            a=self.pathloss_a,
            b=self.pathloss_b,
            seed=seed  # If None, random each time; if provided, reproducible
        )
        return sinr_matrix, adjacency_matrix

    def allocate(self, positions: np.ndarray,
                 weights: np.ndarray,
                 coefficients: np.ndarray,
                 seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform allocation and return a result dictionary containing adjacency matrix etc.
        """
        num_agents = positions.shape[0]
        if num_agents != self.num_agents:
            raise ValueError(f"Number of agents mismatch: model expects {self.num_agents}, input has {num_agents}")

        try:
            # Generate observations
            obs = self._generate_observations(positions, weights, coefficients)
            # Get actions
            actions = self._get_model_actions(obs)
            # Parse allocations
            power_allocations, channel_selections = self._parse_actions(actions)
            # Compute SINR and adjacency matrix (includes small-scale fading)
            sinr_matrix, adjacency_matrix = self._calculate_sinr_matrix(
                positions, power_allocations, channel_selections, coefficients, seed
            )
            # Neighbor lists
            neighbor_lists = get_neighbors_from_adjacency(adjacency_matrix)
            # Statistics
            stats = calculate_connectivity_stats(adjacency_matrix)
            total_power_used = sum(sum(alloc.values()) for alloc in power_allocations)

            return {
                'success': True,
                'positions': positions.copy(),
                'weights': weights.copy(),
                'coefficients': coefficients.copy(),
                'power_allocations': power_allocations,
                'channel_selections': channel_selections,
                'neighbor_lists': neighbor_lists,
                'sinr_matrix': sinr_matrix,
                'adjacency_matrix': adjacency_matrix,
                'total_power_used': total_power_used,
                'stats': stats,
            }

        except Exception as e:
            print(f"Allocation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'neighbor_lists': [[] for _ in range(num_agents)],
                'adjacency_matrix': np.zeros((num_agents, num_agents), dtype=bool),
                'total_power_used': 0.0,
            }


def mappo_allocator(positions, config):
    """
    MAPPO allocator aiming to minimize the actual observed loss J_hat (interface aligned with random/ao/avg).

    Args:
        positions: array of terminal positions [N, 2]
        config: configuration dictionary, must contain the following keys:
                num_agents, mappo_config_path, mappo_model_path
                optional: weights, coefficients, seed

    Returns:
        neighbor_lists: neighbor list for each agent, each element is a list of neighbor IDs
    """
    # Extract parameters from config
    num_agents = config['num_agents']
    weights = config.get('weights', np.ones(num_agents) / num_agents)
    coefficients = config.get('coefficients', np.ones(num_agents) * 1e-9)
    seed = config.get('seed', None)

    # Compute penalty coefficients C
    rho = np.array(weights)
    C = rho**2 + (num_agents - 1) * rho

    # Model paths (must be provided)
    mappo_config_path = config['mappo_config_path']
    mappo_model_path = config['mappo_model_path']
    if mappo_config_path is None or mappo_model_path is None:
        raise ValueError("config must contain 'mappo_config_path' and 'mappo_model_path'")

    # Cache allocator instance (function attribute)
    if not hasattr(mappo_allocator, '_allocator'):
        mappo_allocator._allocator = MAPPOAllocator(mappo_config_path, mappo_model_path)
        print("Initialized MAPPO allocator (cached)")

    try:
        # Call allocator
        result = mappo_allocator._allocator.allocate(positions, weights, coefficients, seed)

        if not result['success']:
            raise RuntimeError(result.get('error', 'Unknown error'))

        neighbor_lists = result['neighbor_lists']
        adjacency_matrix = result['adjacency_matrix']
        total_power_used = result['total_power_used']

        # Compute J_hat
        J_hat = 0.0
        for m in range(num_agents):
            for n in range(num_agents):
                if m != n and not adjacency_matrix[m, n]:
                    J_hat += C[m]

        # Print statistics (consistent with random/ao)
        print(f"\nMAPPO allocation completed:")
        print(f"  Best loss J_hat: {J_hat:.6f}")
        print(f"  Total power used: {total_power_used:.6f} W")
        print(f"  Neighbor lists: {neighbor_lists}")

        return neighbor_lists

    except Exception as e:
        print(f"MAPPO allocation failed: {e}")
        import traceback
        traceback.print_exc()
        return [[] for _ in range(num_agents)]