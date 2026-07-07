"""
smoke_test.py — 冒烟测试: 验证全部代码可跑通，无维度/逻辑/边界报错

测试项:
  1. 数据生成
  2. 环境创建 + 维度检查
  3. P2P 通信功能检查 (UAV 初始同位置 → 应在通信范围内)
  4. 随机策略跑 1 个 episode
  5. PPO agent 创建 + action 选择
  6. CTDE 更新 (1 轮)
  7. 测试评估
"""
import sys
import os
import numpy as np
import torch
from argparse import Namespace

# ================================================================
# 1. 数据生成
# ================================================================
print("[1/7] Generating data...")
sys.path.insert(0, os.path.dirname(__file__))
from data_generate import generate_data

DATA_PATH = "./data/poi_8_map_300x300.npy"
generate_data(K=8, N=1, map_size=300, output_path=DATA_PATH, seed=42)

# ================================================================
# 2. 环境创建 + 维度检查
# ================================================================
print("\n[2/7] Creating environment...")
from Env import MultiDroneAoIEnv

args = Namespace(
    speed_levels="6-20",
    init_uav_energy=150000.0,
    init_uav_energies=None,
    reward_divisor=10.0,
    R_comm=100.0,
    history_horizon=5,
    BS_back_times=4,
    buffer_punishment=False,
    punishment_value=4.0,
    pre_reward_ratio=3.0,
    reward_scale_size=10000.0,
    print_info=False,
    position_file=DATA_PATH,
)

env = MultiDroneAoIEnv(
    M=3, N=1, K=8, T=300.0, map_size=300.0,
    args=args, position_file=DATA_PATH,
)

obs = env.reset()
obs_dim = env.observation_space.shape[0]
token_dim = env.token_dim
comm_dim = env.comm_dim
expected_obs = 5 * env.K + env.K * env.M + env.M + 4 + (env.M - 1) * env.comm_obs_per_neighbor

print(f"  obs_dim={obs_dim}  expected={expected_obs}  match={obs_dim == expected_obs}")
print(f"  token_dim={token_dim}  comm_dim={comm_dim}")
print(f"  encoder_len={env.encoder_token_len}  decoder_len={env.decoder_token_len}")
print(f"  critic_state_dim={env.get_global_critic_state().shape[0]}")
assert obs_dim == expected_obs, f"obs_dim mismatch: {obs_dim} != {expected_obs}"
assert obs.shape[0] == obs_dim, f"obs shape mismatch: {obs.shape[0]} != {obs_dim}"
assert not np.any(np.isnan(obs)), "obs contains NaN"
print("  ✓ Dimensions OK")

# ================================================================
# 3. P2P 通信检查
# ================================================================
print("\n[3/7] Checking P2P communication...")
# 所有 UAV 初始在 BS 位置, 距离=0 < R_comm=100
n_pairs = int(np.sum(env.comm_adjacency) / 2)
print(f"  Initial adjacency:\n{env.comm_adjacency}")
print(f"  Active pairs: {n_pairs}")
assert n_pairs > 0, "Expected active communication at start (all UAVs at BS)"
print("  ✓ P2P communication active at start")

# ================================================================
# 4. 随机策略跑 1 episode
# ================================================================
print("\n[4/7] Running random policy episode...")
env.reset()
done_count = 0
done_bool = np.zeros(env.M, dtype=np.int32)
action_queue = [None] * env.M
step_count = 0

for step in range(100):
    cand = np.full(env.M, np.inf, dtype=np.float32)
    for i in range(env.M):
        if done_bool[i]:
            continue
        if action_queue[i] is None:
            masks = env.get_action_masks(i)
            valid = np.where(masks["target"] > 0)[0]
            if len(valid) == 0:
                done_bool[i] = 1
                continue
            t_act = int(np.random.choice(valid))
            s_act = int(np.random.randint(env.speed_action_dim))
            action_queue[i] = (t_act, s_act)
        cand[i] = env.drone_timing_now[i] + env.time_cost(i, action_queue[i])

    if np.isinf(cand).all():
        break

    actor = int(np.argmin(cand))
    obs, reward, done, info = env.step(actor, action_queue[actor])
    action_queue[actor] = None
    step_count += 1

    assert obs.shape[0] == obs_dim, f"obs dim changed: {obs.shape[0]}"
    assert not np.any(np.isnan(obs)), f"NaN in obs at step {step}"
    assert not np.isnan(reward), f"NaN reward at step {step}"

    if done:
        done_bool[actor] = 1
        done_count += 1
    if done_count == env.M:
        break

stats = env.get_comm_stats()
print(f"  Steps: {step_count}  Done: {done_count}/{env.M}")
print(f"  Comm active ratio: {stats['comm_active_ratio']:.3f}")
print(f"  Repeat visits: {stats['repeat_visits']}/{stats['total_poi_visits']}")
print(f"  Final AoI mean: {np.mean(env.aoi):.2f}")
print("  ✓ Random episode OK")

# ================================================================
# 5. PPO Agent 创建 + Action 选择
# ================================================================
print("\n[5/7] Creating PPO agents and selecting actions...")
from PPO import PPO, CentralizedCritic

env.reset()
critic_state_dim = int(env.get_global_critic_state().shape[0])

ppo_agents = []
for i in range(env.M):
    ppo_agents.append(PPO(
        agent_id=i, token_dim=token_dim,
        target_action_dim=env.target_action_dim,
        speed_action_dim=env.speed_action_dim,
        max_encoder_len=env.encoder_token_len,
        max_decoder_len=env.decoder_token_len,
        max_critic_len=env.critic_token_len,
        max_other_agents=env.max_other_agents,
        lr_actor=3e-4, lr_critic=3e-4,
        gamma=0.99, K_epochs=3, eps_clip=0.2,
        summary_dir="logs/smoke_test",
        entropy_ratio=0.05, gae_lambda=0.97, gae_flag=True,
        d_model=64, nhead=2, num_layers=1, dropout=0.1,
    ))

shared_critic = CentralizedCritic(
    critic_state_dim=critic_state_dim, lr_critic=3e-4,
    gamma=0.99, K_epochs=3, gae_lambda=0.97, gae_flag=True,
    summary_writer=ppo_agents[0].writer, hidden_dim=64,
)

# 选择动作
obs_pack = env.get_transformer_inputs(0)
masks = env.get_action_masks(0)
action, trans = ppo_agents[0].select_action(obs_pack, masks)
print(f"  Agent 0 action: target={action[0]} speed={action[1]}")
assert 0 <= action[0] < env.target_action_dim
assert 0 <= action[1] < env.speed_action_dim
print("  ✓ PPO agents created and acting OK")

# ================================================================
# 6. CTDE 更新 (2 个 episode 后)
# ================================================================
print("\n[6/7] Running CTDE update (2 mini episodes)...")

for ep in range(2):
    env.reset()
    done_bool = np.zeros(env.M, dtype=np.int32)
    action_queue = [None] * env.M
    target_masks = np.ones((env.M, env.target_action_dim), dtype=np.float32)
    speed_masks = np.ones((env.M, env.speed_action_dim), dtype=np.float32)

    for step in range(50):
        cand = np.full(env.M, np.inf, dtype=np.float32)
        for i in range(env.M):
            if done_bool[i]:
                continue
            if action_queue[i] is None:
                obs_pack = env.get_transformer_inputs(i)
                mp = {"target": target_masks[i], "speed": speed_masks[i]}
                action, trans = ppo_agents[i].select_action(obs_pack, mp)
                c_snap = shared_critic.snapshot(obs_pack)
                action_queue[i] = action
                # 暂存
                if not hasattr(env, '_meta'):
                    env._meta = {}
                env._meta[i] = {"trans": trans, "snap": c_snap}
            cand[i] = env.drone_timing_now[i] + env.time_cost(i, action_queue[i])

        if np.isinf(cand).all():
            break

        actor = int(np.argmin(cand))
        _, reward, done, info = env.step(actor, action_queue[actor])
        meta = env._meta[actor]
        ppo_agents[actor].store_transition(meta["trans"])
        li = ppo_agents[actor].last_transition_index()

        # 暂存 critic 数据
        if not hasattr(env, '_critic_data'):
            env._critic_data = []
        env._critic_data.append({
            "snap": meta["snap"], "reward": float(reward),
            "done": 1.0 if done else 0.0, "ref": (actor, li),
        })

        target_masks[actor] = info["target"]
        speed_masks[actor] = info["speed"]
        action_queue[actor] = None

        if done:
            done_bool[actor] = 1
        if done_bool.all():
            break

    # Episode 结束, 标记最后一个 done
    if hasattr(env, '_critic_data') and len(env._critic_data) > 0:
        env._critic_data[-1]["done"] = 1.0

# CTDE 更新
if hasattr(env, '_critic_data') and len(env._critic_data) > 0:
    snaps = [d["snap"] for d in env._critic_data]
    rewards = [d["reward"] for d in env._critic_data]
    dones = [d["done"] for d in env._critic_data]
    refs = [d["ref"] for d in env._critic_data]

    old_values = [s["value"] for s in snaps]
    advantages, returns = shared_critic.compute_advantages_and_returns(rewards, dones, old_values)

    if advantages.numel() > 0:
        actor_adv = [
            torch.zeros(len(a.buffer.target_actions), dtype=torch.float32)
            for a in ppo_agents
        ]
        for si, (aid, li) in enumerate(refs):
            if 0 <= li < actor_adv[aid].numel():
                actor_adv[aid][li] = advantages[si]

        shared_critic.update(snaps, returns)
        for aid in range(env.M):
            ppo_agents[aid].update(actor_adv[aid])

    print("  ✓ CTDE update completed")

# ================================================================
# 7. 测试评估
# ================================================================
print("\n[7/7] Running test evaluation...")
env.reset()
done_bool = np.zeros(env.M, dtype=np.int32)
action_queue = [None] * env.M
target_masks = np.ones((env.M, env.target_action_dim), dtype=np.float32)
speed_masks = np.ones((env.M, env.speed_action_dim), dtype=np.float32)
test_reward = [0.0] * env.M

for step in range(50):
    cand = np.full(env.M, np.inf, dtype=np.float32)
    for i in range(env.M):
        if done_bool[i]:
            continue
        if action_queue[i] is None:
            obs_pack = env.get_transformer_inputs(i)
            action_queue[i] = ppo_agents[i].action_test(
                obs_pack, {"target": target_masks[i], "speed": speed_masks[i]}
            )
        cand[i] = env.drone_timing_now[i] + env.time_cost(i, action_queue[i])
    if np.isinf(cand).all():
        break
    actor = int(np.argmin(cand))
    _, reward, done, info = env.step(actor, action_queue[actor])
    target_masks[actor] = info["target"]
    speed_masks[actor] = info["speed"]
    test_reward[actor] += reward
    action_queue[actor] = None
    if done:
        done_bool[actor] = 1
    if done_bool.all():
        break

print(f"  Test rewards: {[f'{r:.4f}' for r in test_reward]}")
print(f"  Test team reward: {sum(test_reward):.4f}")
print(f"  Test mean AoI: {np.mean(env.aoi):.2f}")
print("  ✓ Test evaluation OK")

# ================================================================
print("\n" + "=" * 60)
print("  ALL SMOKE TESTS PASSED ✓")
print("=" * 60)
