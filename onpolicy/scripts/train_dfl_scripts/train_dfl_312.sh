#!/bin/bash

# --- 1. Path configuration ---
cd /data/8T/zhouh/on-policy/onpolicy/scripts/train/


# --- 2. Experiment name and environment settings ---
env="DFL"
algo="mappo"
user="1wuwwow1-shanghai-university" 

# Generate unique experiment name
timestamp=$(date +%m%d_%H%M)
exp="dfl_ep1_64t_${timestamp}"

# --- 3. Physical scenario parameters ---
num_agents=5
num_channels=8
max_power=0.2
gamma_th=5
bandwidth=1000000   # 1MHz bandwidth
noise_dbm=-174      # noise power spectral density (dBm/Hz)
deployment_radius=200.0

# --- 4. Normalization method configuration ---
normalization_method="simple"

# --- 5. Penalty coefficient configuration ---
channel_penalty_weight=0.2
bidirectional_same_channel_penalty_weight=0.2
bidirectional_same_channel_penalty_weight=0.2
aux_reward_weight=5.0
reward_scaling_factor=1.0

# --- 6. Training hyperparameters ---
threads=128
steps=500000
ep_len=1  # In DFL each step is an independent scenario, so ep_len=1
gamma=0

echo "================================================"
echo ">>> Starting DFL Training"
echo ">>> Experiment Name: ${exp}"
echo ">>> Number of Agents: ${num_agents}"
echo ">>> Number of Channels: ${num_channels}"
echo "================================================"

# --- 7. Execute training ---
python train_dfl_312.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --user_name ${user} \
    --wandb_name "dfl_trial" \
    --num_agents ${num_agents} \
    --num_channels ${num_channels} \
    --max_power ${max_power} \
    --bandwidth ${bandwidth} \
    --noise_dbm ${noise_dbm} \
    --gamma_th ${gamma_th} \
    --deployment_radius ${deployment_radius} \
    --normalization_method ${normalization_method} \
    --channel_penalty_weight ${channel_penalty_weight} \
    --bidirectional_same_channel_penalty_weight ${bidirectional_same_channel_penalty_weight} \
    --aux_reward_weight ${aux_reward_weight}  \
    --reward_scaling_factor ${reward_scaling_factor} \
    --n_rollout_threads ${threads} \
    --num_env_steps ${steps} \
    --episode_length ${ep_len} \
    --clip_param 0.2 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.1 \
    --use_linear_lr_decay \
    --use_proper_time_limits \
    --use_eval \
    --n_eval_rollout_threads 1 \
    --gamma ${gamma}\
    --max_grad_norm 0.5\
    --eval_interval 100 \
    --save_interval 100 \
    --use_render False \
    --use_cuda \
    --cuda_deterministic \
    --use_recurrent_policy \