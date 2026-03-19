import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb



def _t2n(x):
    return x.detach().cpu().numpy()


class DFLRunner(Runner):
    def __init__(self, config):
        super(DFLRunner, self).__init__(config)
        
        self.wandb_initialized = False  # flag to record whether WandB initialization succeeded

    def warmup(self):
        # Reset environment and get initial observations
        results = self.envs.reset()
        # Unpack according to environment return format (obs, share_obs, available_actions)
        obs, share_obs, available_actions = results

        # If not using centralized critic, share_obs equals local obs
        if not self.use_centralized_V:
            share_obs = obs

        # This share_obs will be directly stored in buffer at step 0
        # Expected dimension should be (8, 5, 216)
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

        # Initialize WandB if enabled
        if self.all_args.use_wandb and not self.wandb_initialized:
            # Ensure necessary parameters exist
            if not hasattr(self.all_args, 'user_name'):
                self.all_args.user_name = 'default_user'
            if not hasattr(self.all_args, 'experiment_name'):
                # If experiment name not set, use timestamp
                import datetime
                self.all_args.experiment_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            try:
                # Check if there is already an active wandb run
                if wandb.run is not None:
                    wandb.finish()  # terminate previous run
                
                import os
                
                # Create wandb directory
                wandb_dir = os.path.join(self.run_dir, "wandb")
                os.makedirs(wandb_dir, exist_ok=True)
                
                # Initialize wandb
                wandb.init(
                    project=self.all_args.env_name,
                    entity=self.all_args.user_name,
                    config=vars(self.all_args),
                    name=self.all_args.experiment_name,
                    dir=wandb_dir,
                    reinit=True,
                    force=True,
                    settings=wandb.Settings(start_method="fork")  # add settings to avoid multiprocessing issues
                )
                
                self.wandb_initialized = True
                print(f">>> WandB initialized successfully!")
                print(f">>> Project: {self.all_args.env_name}")
                print(f">>> Run name: {self.all_args.experiment_name}")
                print(f">>> Entity: {self.all_args.user_name}")
                
                # Save configuration files
                import glob
                config_files = glob.glob(os.path.join(self.run_dir, "*.py"))
                for config_file in config_files:
                    wandb.save(config_file)
                
                # Log initial parameters
                if wandb.run is not None:  # check again if wandb initialized successfully
                    wandb.log({
                        "phys_params/bandwidth_hz": self.all_args.bandwidth,
                        "phys_params/noise_psd_dbm": self.all_args.noise_dbm,
                        "phys_params/max_power_w": self.all_args.max_power,
                        "phys_params/gamma_th_db": self.all_args.gamma_th,
                        "phys_params/deployment_radius_m": self.all_args.deployment_radius,
                        "phys_params/num_agents": self.all_args.num_agents,
                        "phys_params/num_channels": self.all_args.num_channels
                    }, step=0)
                    
            except Exception as e:
                print(f">>> WandB initialization failed: {e}")
                print(">>> Continuing without WandB...")
                self.all_args.use_wandb = False
                self.wandb_initialized = False
        elif self.all_args.use_wandb and self.wandb_initialized:
            print(">>> WandB already initialized")
        else:
            print(">>> WandB is disabled")
        
    def safe_wandb_log(self, data_dict, step=None):
        """Safe WandB logging function."""
        if not self.all_args.use_wandb or not self.wandb_initialized:
            return
        
        try:
            if wandb.run is not None:
                if step is not None:
                    wandb.log(data_dict, step=step)
                else:
                    wandb.log(data_dict)
            else:
                print(f"[WARNING] WandB run is None, skipping log")
        except Exception as e:
            print(f"[WARNING] WandB logging error: {e}")
    
    def log_train(self, train_infos, total_num_steps):
        """Training logging function."""
        self.safe_wandb_log(train_infos, total_num_steps)

    def run(self):
        self.warmup() 

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # --- Sampling phase ---
            sample_start = time.perf_counter()
            
            for step in range(self.episode_length):
                # collect will get current step's obs/share_obs from buffer internally
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                # Environment step: obtain next observations, note share_obs here is 216-dimensional
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                # Modification: pass share_obs along with other data to insert
                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)
            
            sample_end = time.perf_counter()
            sample_time = sample_end - sample_start

            # --- Training phase ---
            train_start = time.perf_counter()
            
            self.compute()
            train_infos = self.train()
            
            train_end = time.perf_counter()
            train_time = train_end - train_start

            # --- Logging ---
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / (end - start))
                # Note: rewards in buffer are of shape (8, 5, 1)
                avg_rew = np.mean(self.buffer.rewards) * self.episode_length
                
                total_cycle_time = sample_time + train_time
                sample_ratio = (sample_time / total_cycle_time) * 100 if total_cycle_time > 0 else 0

                print(f"\n" + "="*50)
                print(f"DFL Exp {self.experiment_name} | Episode {episode}/{episodes}")
                print(f"Steps {total_num_steps} | FPS {fps} | Avg Reward: {avg_rew:.4f}")
                print(f"--- Time Profiling ---")
                print(f"Physics Simulation (CPU): {sample_time:.2f}s ({sample_ratio:.1f}%)")
                print(f"Neural Net Training (GPU): {train_time:.2f}s")
                print(f"="*50)

                if self.all_args.use_wandb and self.wandb_initialized:  # conditional check
                    # Log timing information
                    train_infos["time/sample_time_s"] = sample_time
                    train_infos["time/train_time_s"] = train_time
                    train_infos["time/total_cycle_time_s"] = total_cycle_time
                    train_infos["time/sample_ratio_percent"] = sample_ratio
                    
                    # Log training metrics
                    train_infos["performance/average_episode_rewards"] = avg_rew
                    train_infos["performance/fps"] = fps
                    train_infos["training/total_num_steps"] = total_num_steps
                    train_infos["training/episode"] = episode
                    
                    # Log physical layer metrics if available
                    if 'infos' in locals() and len(infos) > 0:
                        # Assume infos is a list of dictionaries
                        # Take information from the first environment as representative
                        if isinstance(infos, list) and len(infos) > 0:
                            info = infos[0]
                            if isinstance(info, dict):
                                for key, value in info.items():
                                    if isinstance(value, (int, float)):
                                        train_infos[f"physics/{key}"] = value
                    
                    self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        
    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # Obtain concatenated observations from buffer and feed to model
        value, action, action_log_prob, rnn_states, rnn_states_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step])
            )

        # Reshape to match multi-thread environment dimensions
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions


    def insert(self, data):
        # Key modification: explicitly receive share_obs
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # Reset RNN states for episodes that are done
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        # Construct masks
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # Core fix: directly use the oracle share_obs from environment instead of manually reshaping obs
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)


    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        results = self.eval_envs.reset()
        eval_obs, eval_share_obs, eval_avail = results

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Environment step
            eval_obs, eval_s_obs, eval_rews, eval_dones, eval_infos, eval_avails = self.eval_envs.step(eval_actions)
            eval_episode_rewards.append(eval_rews)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_avg_rew = np.mean(np.sum(np.array(eval_episode_rewards), axis=0))
        print(f">>> Evaluation Average Reward: {eval_avg_rew:.4f}")
        
        # Use safe logging function
        self.safe_wandb_log({"eval_average_episode_rewards": eval_avg_rew}, step=total_num_steps)