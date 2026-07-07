"""
verify.py — P2P 通信消融验证

对比同一训练好的模型 (或随机策略) 在以下两种条件下的表现:
  1. R_comm=0   (P2P 通信关闭)
  2. R_comm>0   (P2P 通信开启)

指标: 总 reward, 平均 AoI, 加权 AoI, 重复访问率, 通信统计
"""
import argparse
import numpy as np
import os
import json
from argparse import Namespace

from Env import MultiDroneAoIEnv
from PPO import PPO, CentralizedCritic


def evaluate(env, ppo_agents, shared_critic, n_episodes=10, max_steps=300):
    """运行 n_episodes 轮评估，返回统计均值"""
    M = env.M
    all_rewards = []
    all_mean_aoi = []
    all_weighted_aoi = []
    all_repeat_rates = []
    all_comm_active = []

    for ep in range(n_episodes):
        env.reset()
        action_queue = [None] * M
        target_masks = np.ones((M, env.target_action_dim), dtype=np.float32)
        speed_masks = np.ones((M, env.speed_action_dim), dtype=np.float32)
        done_bool = np.zeros(M, dtype=np.int32)
        ep_reward = [0.0] * M

        for _ in range(max_steps):
            cand = np.full(M, np.inf, dtype=np.float32)
            for i in range(M):
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
            ep_reward[actor] += reward
            action_queue[actor] = None
            if done:
                done_bool[actor] = 1
            if done_bool.all():
                break

        team_r = sum(ep_reward)
        mean_aoi = float(np.mean(env.aoi))
        w_aoi = float(np.sum(env.aoi * env.poi_weights) / max(np.sum(env.poi_weights), 1e-6))
        stats = env.get_comm_stats()

        all_rewards.append(team_r)
        all_mean_aoi.append(mean_aoi)
        all_weighted_aoi.append(w_aoi)
        all_repeat_rates.append(stats["repeat_visit_rate"])
        all_comm_active.append(stats["comm_active_ratio"])

        print(f"  [Ep {ep}] reward={team_r:.4f}  mean_aoi={mean_aoi:.2f}  "
              f"repeat={stats['repeat_visit_rate']:.3f}  comm={stats['comm_active_ratio']:.3f}")

    return {
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "mean_aoi": float(np.mean(all_mean_aoi)),
        "weighted_aoi": float(np.mean(all_weighted_aoi)),
        "repeat_visit_rate": float(np.mean(all_repeat_rates)),
        "comm_active_ratio": float(np.mean(all_comm_active)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--N", type=int, default=1)
    p.add_argument("--T", type=float, default=600.0)
    p.add_argument("--map_size", type=float, default=500.0)
    p.add_argument("--position_file", type=str, default=None)
    p.add_argument("--R_comm_on", type=float, default=150.0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--checkpoint_dir", type=str, default="PPO_preTrained")
    p.add_argument("--env_name", type=str, default="P2P-MARL")
    p.add_argument("--speed_levels", type=str, default="6-20")
    p.add_argument("--init_uav_energy", type=float, default=2e5)
    p.add_argument("--init_uav_energies", type=str, default=None)
    p.add_argument("--reward_scale_size", type=float, default=10000.0)
    p.add_argument("--reward_divisor", type=float, default=10.0)
    p.add_argument("--pre_reward_ratio", type=float, default=3.0)
    p.add_argument("--BS_back_times", type=int, default=5)
    p.add_argument("--buffer_punishment", action="store_true")
    p.add_argument("--punishment_value", type=float, default=4.0)
    p.add_argument("--history_horizon", type=int, default=10)
    p.add_argument("--transformer_dim", type=int, default=128)
    p.add_argument("--transformer_heads", type=int, default=2)
    p.add_argument("--transformer_layers", type=int, default=2)
    p.add_argument("--transformer_dropout", type=float, default=0.1)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--print_info", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print("  P2P Communication Ablation Verification")
    print("=" * 70)

    results = {}

    for label, r_comm in [("COMM_OFF", 0.0), ("COMM_ON", args.R_comm_on)]:
        print(f"\n--- {label} (R_comm={r_comm}) ---")
        args.R_comm = r_comm
        env = MultiDroneAoIEnv(
            args.M, args.N, args.K, args.T, args.map_size,
            args=args, position_file=args.position_file,
        )

        # 创建 agents (随机策略, 无需加载模型)
        from PPO import PPO, CentralizedCritic
        import torch

        token_dim = env.token_dim
        ppo_agents = []
        for i in range(args.M):
            ppo_agents.append(PPO(
                agent_id=i, token_dim=token_dim,
                target_action_dim=env.target_action_dim,
                speed_action_dim=env.speed_action_dim,
                max_encoder_len=env.encoder_token_len,
                max_decoder_len=env.decoder_token_len,
                max_critic_len=env.critic_token_len,
                max_other_agents=env.max_other_agents,
                lr_actor=1e-4, lr_critic=1e-4,
                gamma=0.99, K_epochs=5, eps_clip=0.2,
                summary_dir=f"logs/verify_{label}",
                entropy_ratio=0.03, gae_lambda=0.97, gae_flag=True,
                d_model=args.transformer_dim, nhead=args.transformer_heads,
                num_layers=args.transformer_layers, dropout=args.transformer_dropout,
            ))

        shared_critic = CentralizedCritic(
            critic_state_dim=int(env.get_global_critic_state().shape[0]),
            lr_critic=1e-4, gamma=0.99, K_epochs=5,
            gae_lambda=0.97, gae_flag=True,
            summary_writer=ppo_agents[0].writer,
            hidden_dim=args.transformer_dim,
        )

        # 尝试加载模型
        ckpt_prefix = os.path.join(
            args.checkpoint_dir, args.env_name,
            f"PPO_{args.env_name}_{args.random_seed}"
        )
        loaded = False
        if os.path.exists(f"{ckpt_prefix}_agent0.pth"):
            for i in range(args.M):
                ppo_agents[i].load(f"{ckpt_prefix}_agent{i}.pth")
            shared_critic.load(f"{ckpt_prefix}_critic.pth")
            loaded = True
            print(f"  Loaded checkpoints from {ckpt_prefix}")
        else:
            print(f"  No checkpoint found, using random policy")

        result = evaluate(env, ppo_agents, shared_critic,
                          n_episodes=args.episodes, max_steps=args.max_steps)
        results[label] = result

    # 对比表
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'COMM_OFF':<20} {'COMM_ON':<20} {'Delta':<15}")
    print("-" * 70)
    for key in ["reward_mean", "mean_aoi", "weighted_aoi", "repeat_visit_rate", "comm_active_ratio"]:
        off = results["COMM_OFF"][key]
        on = results["COMM_ON"][key]
        delta = on - off
        print(f"{key:<25} {off:<20.4f} {on:<20.4f} {delta:<+15.4f}")
    print("=" * 70)

    # 保存结果
    out_path = "verify_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
