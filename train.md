

# 350kj 最高72
python train.py --M 3 --N 1 --K 40 --env_name "MAPPO-3UAV-version-18-350kj-dim384-mlp-local" --decoupled_clip --update_every_episodes 10 --K_epochs 10 --map_size 2000 --T 2400 --speed_levels 6-20 --history_horizon 20 --init_uav_energies "350000, 300000, 250000" --transformer_dim 384 --transformer_heads 2 --transformer_layers 2 --max_ep_len 1000 --max_training_timesteps 15000000 --entropy_ratio 0.03  --reward_divisor 400 --no_normalize_advantage



# 全200 结果63
python train.py --M 3 --N 1 --K 40 --env_name "MAPPO-3UAV-version-18-all200kj-dim384-mlp-local" --decoupled_clip --update_every_episodes 10 --K_epochs 10 --map_size 2000 --T 2400 --speed_levels 6-20 --history_horizon 20 --init_uav_energies "200000, 200000, 200000" --transformer_dim 384 --transformer_heads 2 --transformer_layers 2 --max_ep_len 1000 --max_training_timesteps 15000000 --entropy_ratio 0.05  --reward_divisor 400 --no_normalize_advantage

python train.py --M 3 --N 1 --K 40 --env_name "MAPPO-3UAV-version-18-all200kj-dim384-mlp-local-noDe-clip008" --eps_clip 0.08 --update_every_episodes 10 --K_epochs 10 --map_size 2000 --T 2400 --speed_levels 6-20 --history_horizon 20 --init_uav_energies "200000, 200000, 200000" --transformer_dim 384 --transformer_heads 2 --transformer_layers 2 --max_ep_len 1000 --max_training_timesteps 15000000 --entropy_ratio 0.05  --reward_divisor 400 --no_normalize_advantage




