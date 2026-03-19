import gym
from gym import spaces
import numpy as np

class DFLEnv(gym.Env):
    def __init__(self, args):
        """
        Completely static decentralized federated learning communication environment.
        Hard channel selection + small-scale Rayleigh fading
        """
        self.n = args.num_agents
        self.num_channels = args.num_channels
        
        # Constants loaded from configuration file
        self.bandwidth = args.bandwidth 
        self.noise_psd_linear = 10**((args.noise_dbm - 30) / 10.0)  # convert to linear
        self.gamma_th_lin = 10**(args.gamma_th / 10.0)  # SINR threshold, linear
        self.p_max_uniform = args.max_power  # uniform maximum power limit
        self.num_links_per_agent = self.n - 1  # number of outgoing links per agent
        
        # Normalization method
        self.normalization_method = args.normalization_method
        
        # Penalty coefficients (currently unused)
        self.channel_repeat_penalty_weight = getattr(args, 'channel_penalty_weight', 0.0)
        self.bidirectional_same_channel_penalty_weight = getattr(args, 'bidirectional_same_channel_penalty_weight', 0.0)
        
        # Reward scaling factor
        self.reward_scaling_factor = getattr(args, 'reward_scaling_factor', 1.0)
        
        # Auxiliary reward weight (actual number of successful links)
        self.aux_reward_weight = getattr(args, 'aux_reward_weight', 1)  # suggested to increase
        
        # Whether to enable small-scale fading
        self.include_fading = getattr(args, 'include_fading', True)
        
        # Deployment radius
        self.deployment_radius = args.deployment_radius
        
        # Training mode flag
        self.training = True
        
        # --- Dimension calculation ---
        # Actor observation: ID(n) + static table(2n: rho and beta_si) + channel gains(n) + common constants(2) = 4n+2
        self.obs_dim = 4 * self.n + 2
        
        # Action space: power weights (K-1) + channel logits (K-1)*C
        self.action_dim_per_agent = self.num_links_per_agent + self.num_links_per_agent * self.num_channels
        
        # Critic shared observation: all actor observations + all actor action features
        self.action_feat_per_link = 1 + self.num_channels  # normalized power + one-hot channel
        self.action_feat_dim = self.n * self.num_links_per_agent * self.action_feat_per_link
        self.share_obs_dim = (self.obs_dim * self.n) + self.action_feat_dim

        # --- Define spaces ---
        self.action_space = [spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.action_dim_per_agent,), 
            dtype=np.float32
        ) for _ in range(self.n)]
        
        self.observation_space = [spaces.Box(
            low=-np.inf, 
            high=+np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        ) for _ in range(self.n)]
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, 
            high=+np.inf, 
            shape=(self.share_obs_dim,), 
            dtype=np.float32
        ) for _ in range(self.n)]
        
        # Normalization statistics (initialized in reset)
        self.G_log_mean = 0
        self.G_log_std = 1
        self.g_log_min = -1
        self.g_log_max = 1

    def reset(self):
        """Reset the environment and generate a new static scenario."""
        self.current_step = 0
        
        # 1. Randomly generate terminal positions
        self.positions = self._init_positions()
        
        # 2. Compute large-scale channel gains (based on distance)
        self.G = self._init_channel_gains()
        
        # 3. Compute normalization statistics for channel gains
        self._compute_normalization_stats()
        
        # 4. Generate heterogeneous parameters for each terminal
        # 4.1 Weights (sum to 1)
        self.rho_n = np.random.dirichlet(np.ones(self.n) * 2.0)
        # 4.2 Self-interference coefficients (small value)
        self.beta_si_n = np.random.uniform(1e-9, 1e-8, self.n)
        
        # 5. Build global static table (2n dimension)
        self.global_static_table = np.concatenate([self.rho_n, self.beta_si_n])

        # 6. Generate observations for each agent
        obs_n = self._get_obs_n()
        
        # 7. Initialize critic shared observation (using zero action features)
        empty_action_features = np.zeros(self.action_feat_dim, dtype=np.float32)
        share_obs_n = self._get_share_obs(obs_n, empty_action_features)
        
        return obs_n, share_obs_n, [None]*self.n
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics for channel gains."""
        G_log = np.log10(self.G + 1e-20)
        G_log_offdiag = G_log[~np.eye(self.n, dtype=bool)].flatten()
        
        if self.normalization_method == "log_standard":
            self.G_log_mean = np.mean(G_log_offdiag)
            self.G_log_std = max(np.std(G_log_offdiag), 1e-6)
        elif self.normalization_method == "log_minmax":
            self.g_log_min = np.min(G_log_offdiag)
            self.g_log_max = np.max(G_log_offdiag)
            if self.g_log_max - self.g_log_min < 1e-6:
                self.g_log_max = self.g_log_min + 1.0
        elif self.normalization_method == "relative":
            G_nonzero = self.G[self.G > 0]
            self.g_min_nonzero = np.min(G_nonzero) if len(G_nonzero) > 0 else 1e-20

    def step(self, action_n):
        """
        Execute one step of environment interaction (hard channel selection + small-scale fading).
        """
        self.current_step += 1

        # --- 1. Action parsing and assignment (hard channel selection) ---
        tx_power = np.zeros((self.n, self.n, self.num_channels), dtype=np.float32)
        link_selected_channel = {}  # (i, j) -> selected channel index
        action_feat_list = []

        for i in range(self.n):
            act = np.clip(action_n[i], 0.0, 1.0).flatten()

            # Power part (first num_links_per_agent dimensions)
            power_logits = act[:self.num_links_per_agent]
            power_weights = np.exp(power_logits) / (np.sum(np.exp(power_logits)) + 1e-12)
            link_powers = power_weights * self.p_max_uniform  # total power per link

            # Channel part (remaining num_links_per_agent * num_channels dimensions)
            channel_logits = act[self.num_links_per_agent:].reshape(self.num_links_per_agent, self.num_channels)
            channel_probs = np.exp(channel_logits) / (np.sum(np.exp(channel_logits), axis=1, keepdims=True) + 1e-12)

            # Hard channel selection: pick the channel with highest probability for each link
            selected_channels = np.argmax(channel_probs, axis=1)  # shape (num_links_per_agent,)

            target_neighbors = [j for j in range(self.n) if j != i]
            for idx, j in enumerate(target_neighbors):
                ch = selected_channels[idx]
                tx_power[i, j, ch] = link_powers[idx]          # allocate total link power to selected channel
                link_selected_channel[(i, j)] = ch

            # Build action features (for critic)
            power_feat = link_powers / self.p_max_uniform      # normalized power
            channel_onehot = np.zeros((self.num_links_per_agent, self.num_channels), dtype=np.float32)
            channel_onehot[np.arange(self.num_links_per_agent), selected_channels] = 1.0
            channel_feat = channel_onehot.flatten()
            action_feat_list.append(np.concatenate([power_feat, channel_feat]))

        action_features = np.concatenate(action_feat_list)

        # Penalty terms (can be ignored)
        channel_repeat_penalty = 0.0
        bidirectional_penalty = 0.0

        # --- 2. Generate small-scale fading coefficients (Rayleigh fading, |h|^2 ~ Exp(1)) ---
        if self.include_fading:
            fading = np.random.exponential(scale=1.0, size=(self.n, self.n, self.num_channels))
        else:
            fading = np.ones((self.n, self.n, self.num_channels))

        # --- 3. Physical layer simulation: compute SINR and adjacency matrix ---
        noise_power = self.noise_psd_linear * self.bandwidth
        sinr_threshold_linear = self.gamma_th_lin

        adjacency_matrix = np.zeros((self.n, self.n), dtype=bool)
        sinr_matrix = np.full((self.n, self.n), -np.inf)
        phi_matrix = np.zeros((self.n, self.n))  # theoretical success probability (without fading) for monitoring

        # Compute SINR for each link
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                # Check if any power is allocated to this link
                if np.sum(tx_power[i, j, :]) < 1e-12:
                    continue

                ch = link_selected_channel[(i, j)]          # channel used by this link
                p_ij = tx_power[i, j, ch]                  # power on that channel (total link power)

                # Desired signal power (including large-scale and small-scale fading)
                signal_power = p_ij * self.G[i, j] * fading[i, j, ch]

                # Compute interference (only on the same channel)
                interference = noise_power

                # Self-interference: transmitter j's own transmissions on the same channel (no fading)
                for target in range(self.n):
                    if target == j:
                        continue
                    p_j_target = tx_power[j, target, ch]
                    if p_j_target > 0:
                        interference += p_j_target * self.beta_si_n[j]

                # Interference from other transmitters: transmitters other than i transmitting to j on same channel (with fading)
                for m in range(self.n):
                    if m == i or m == j:
                        continue
                    p_m_j = tx_power[m, j, ch]
                    if p_m_j > 0:
                        interference += p_m_j * self.G[m, j] * fading[m, j, ch]

                # Compute SINR
                sinr_linear = signal_power / interference if interference > 0 else 0
                sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -np.inf
                sinr_matrix[i, j] = sinr_db

                # Hard decision
                if sinr_linear >= sinr_threshold_linear:
                    adjacency_matrix[i, j] = True

                # Monitoring success probability (using original exponential model, for information only, not used in reward)
                phi_matrix[i, j] = np.exp(-sinr_threshold_linear * noise_power / (p_ij * self.G[i, j])) if p_ij > 0 else 0.0

        successful_links = np.sum(adjacency_matrix)

        # --- 4. Basic reward (based on theoretical convergence upper bound, still using fading-free phi_matrix) ---
        total_j = 0.0
        for k in range(self.n):
            weight_k = (self.rho_n[k]**2 + (self.n - 1) * self.rho_n[k])
            for kp in range(self.n):
                if k == kp:
                    continue
                total_j += (1.0 - phi_matrix[k, kp]) * weight_k
        basic_reward = -total_j

        # --- 5. Auxiliary reward: actual number of successful links ---
        max_possible_links = self.n * (self.n - 1)
        aux_reward = successful_links / max_possible_links  # normalized [0,1]

        # --- 6. Total reward (adjustable combination) ---
        # Combine basic and auxiliary rewards to enrich training signal
        total_reward = self.aux_reward_weight * aux_reward
        scaled_reward = basic_reward + total_reward * self.reward_scaling_factor

        # --- 7. Build return values ---
        obs_n = self._get_obs_n()
        share_obs_n = self._get_share_obs(obs_n, action_features)

        info_n = {
            'basic_reward': basic_reward,
            'aux_reward': aux_reward,
            'successful_links': successful_links,
            'mean_success_prob': np.mean(phi_matrix[phi_matrix > 0]),
            'channel_repeat_penalty': channel_repeat_penalty,
            'bidirectional_penalty': bidirectional_penalty,
            'total_reward': total_reward,
            'j_cost': total_j
        }

        done_n = [True] * self.n

        # Debug print (can be commented out)
        # print(f"[DEBUG] scaled_reward = {scaled_reward:.4f} (aux={aux_reward:.3f}, basic={basic_reward:.3f})")

        return (
            obs_n,
            share_obs_n,
            [[scaled_reward]] * self.n,
            done_n,
            [info_n] * self.n,
            [None] * self.n
        )

    # ========== Helper methods (unchanged) ==========
    def _get_obs_n(self):
        """Construct observation vector for each agent."""
        obs_n = []
        for i in range(self.n):
            my_id = np.zeros(self.n, dtype=np.float32)
            my_id[i] = 1.0
            static_features = self.global_static_table.copy()
            my_gains = self.G[i, :].copy()
            my_gains_normalized = self._normalize_channel_gains(my_gains, i)
            constants = np.array([self.noise_psd_linear, self.gamma_th_lin], dtype=np.float32)
            obs = np.concatenate([my_id, static_features, my_gains_normalized, constants], dtype=np.float32)
            obs_n.append(obs)
        return np.array(obs_n)

    def _normalize_channel_gains(self, gains, agent_id):
        """Normalize channel gain vector."""
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
        else:
            gains_log = np.log10(gains + 1e-20)
            return (gains_log - self.G_log_mean) / self.G_log_std

    def _get_share_obs(self, obs_n, action_features):
        """Build shared observation for critic."""
        combined = np.concatenate([obs_n.flatten(), action_features], dtype=np.float32)
        return np.array([combined for _ in range(self.n)])

    def _init_positions(self):
        """Randomly generate terminal positions (circular area)."""
        positions = []
        radius = self.deployment_radius if hasattr(self, 'deployment_radius') else 200.0
        for _ in range(self.n):
            r = radius * np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            positions.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(positions)

    def _init_channel_gains(self):
        """Compute large-scale path loss based on distance."""
        G = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    G[i, j] = 0.0
                    continue
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                dist_km = max(dist, 1.0) / 1000.0
                pl_db = 128.1 + 37.6 * np.log10(dist_km)
                G[i, j] = 10**(-pl_db / 10.0)
        return G

    def set_training_mode(self, mode=True):
        self.training = mode

    def get_env_info(self):
        return {
            "n_agents": self.n,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim_per_agent,
            "share_obs_dim": self.share_obs_dim,
            "num_channels": self.num_channels,
            "num_links_per_agent": self.num_links_per_agent,
            "normalization_method": self.normalization_method,
            "aux_reward_weight": self.aux_reward_weight,
        }