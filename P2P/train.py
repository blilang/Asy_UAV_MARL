"""
train.py — 异步多无人机 PPO 训练 (CTDE) + P2P 通信

功能:
  1. 断点续训: 自动检测 checkpoint, 恢复 episode/timestep/best_reward
  2. 定期绘图: 每 plot_freq 个 episode 保存 reward/AoI 曲线 + 路径可视化
  3. 定期保存: 每 save_freq 个 episode 保存 checkpoint
  4. 后台友好: nohup python train.py & 即可

用法:
  # 首次训练
  python train.py

  # 断开后继续训练 (自动加载 checkpoint)
  python train.py

  # 自定义参数
  python train.py --K 30 --map_size 1500 --T 2400 --R_comm 450

  # 后台运行
  nohup python train.py > train_log.txt 2>&1 &
"""
import os
import json
from datetime import datetime
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PPO import PPO, CentralizedCritic
from Env import MultiDroneAoIEnv


# ================================================================
#  参数解析
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(description="P2P Comm MARL Training")

    # 环境
    p.add_argument("--env_name", type=str, default="P2P-MARL")
    p.add_argument("--M", type=int, default=3, help="UAV 数量")
    p.add_argument("--N", type=int, default=1, help="基站数量")
    p.add_argument("--K", type=int, default=30, help="POI 数量")
    p.add_argument("--T", type=float, default=2400.0, help="时间上限 (秒)")
    p.add_argument("--map_size", type=float, default=1500.0, help="地图边长 (m)")
    p.add_argument("--position_file", type=str, default=None)
    p.add_argument("--R_comm", type=float, default=450.0, help="P2P 通信半径 (m), 0=关闭")

    # 速度 & 能量
    p.add_argument("--speed_levels", type=str, default="6-20")
    p.add_argument("--init_uav_energy", type=float, default=8e5)
    p.add_argument("--init_uav_energies", type=str, default=None)

    # 奖励
    p.add_argument("--reward_scale_size", type=float, default=10000.0)
    p.add_argument("--reward_divisor", type=float, default=10.0)
    p.add_argument("--pre_reward_ratio", type=float, default=3.0)
    p.add_argument("--BS_back_times", type=int, default=8)
    p.add_argument("--buffer_punishment", action="store_true")
    p.add_argument("--punishment_value", type=float, default=4.0)

    # PPO
    p.add_argument("--max_ep_len", type=int, default=500)
    p.add_argument("--max_training_timesteps", type=int, default=int(5e6))
    p.add_argument("--K_epochs", type=int, default=10)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--decoupled_clip", action="store_true")
    p.add_argument("--eps_clip_pos", type=float, default=0.35)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--entropy_ratio", type=float, default=0.03)
    p.add_argument("--gae_lambda", type=float, default=0.97)
    p.add_argument("--gae_flag", action="store_true")
    p.add_argument("--no_normalize_advantage", dest="normalize_advantage", action="store_false")
    p.add_argument("--update_every_episodes", type=int, default=5)
    p.add_argument("--force_return_bs_threshold", type=int, default=10)
    p.add_argument("--random_seed", type=int, default=0)

    # Transformer
    p.add_argument("--history_horizon", type=int, default=10)
    p.add_argument("--transformer_dim", type=int, default=128)
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--transformer_layers", type=int, default=2)
    p.add_argument("--transformer_dropout", type=float, default=0.1)

    # 日志 & 保存
    p.add_argument("--log_dir", type=str, default="PPO_logs")
    p.add_argument("--checkpoint_dir", type=str, default="PPO_preTrained")
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument("--save_freq", type=int, default=50, help="每 N episode 保存 checkpoint")
    p.add_argument("--plot_freq", type=int, default=50, help="每 N episode 保存曲线图")
    p.add_argument("--test_freq", type=int, default=20, help="每 N episode 测试评估")
    p.add_argument("--print_freq", type=int, default=10, help="每 N episode 打印日志")
    p.add_argument("--print_info", action="store_true")

    p.set_defaults(normalize_advantage=True, gae_flag=True)
    return p.parse_args()


# ================================================================
#  绘图工具
# ================================================================
def plot_training_curves(history, output_dir, episode):
    """保存 reward, AoI, 通信统计 三张曲线图"""
    os.makedirs(output_dir, exist_ok=True)
    eps = history["episodes"]
    if len(eps) < 2:
        return

    # ---- 图 1: Reward ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(eps, history["team_rewards"], alpha=0.3, color="blue", linewidth=0.5)
    # 滑动平均
    window = min(50, len(eps) // 3 + 1)
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(history["team_rewards"], kernel, mode="valid")
        axes[0].plot(eps[window - 1:], smooth, color="blue", linewidth=2, label=f"MA-{window}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Team Reward")
    axes[0].set_title("Training Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if len(history["test_episodes"]) > 0:
        axes[1].plot(history["test_episodes"], history["test_rewards"],
                     "o-", color="red", linewidth=2, markersize=4, label="Test Reward")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Test Reward")
        axes[1].set_title("Test Reward")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No test data yet", ha="center", va="center",
                     transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"reward_ep{episode}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 图 2: AoI ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (key, label, color) in enumerate([
        ("mean_aois", "Mean AoI", "green"),
        ("weighted_aois", "Weighted AoI", "orange"),
        ("max_aois", "Max AoI", "red"),
    ]):
        axes[idx].plot(eps, history[key], alpha=0.3, color=color, linewidth=0.5)
        if window > 1:
            smooth = np.convolve(history[key], kernel, mode="valid")
            axes[idx].plot(eps[window - 1:], smooth, color=color, linewidth=2, label=f"MA-{window}")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel(label)
        axes[idx].set_title(label)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"aoi_ep{episode}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 图 3: 通信统计 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(eps, history["comm_active_ratios"], alpha=0.3, color="purple", linewidth=0.5)
    if window > 1:
        smooth = np.convolve(history["comm_active_ratios"], kernel, mode="valid")
        axes[0].plot(eps[window - 1:], smooth, color="purple", linewidth=2, label=f"MA-{window}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Comm Active Ratio")
    axes[0].set_title("P2P Communication Activity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, history["repeat_visit_rates"], alpha=0.3, color="brown", linewidth=0.5)
    if window > 1:
        smooth = np.convolve(history["repeat_visit_rates"], kernel, mode="valid")
        axes[1].plot(eps[window - 1:], smooth, color="brown", linewidth=2, label=f"MA-{window}")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Repeat Visit Rate")
    axes[1].set_title("Repeat POI Visits")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comm_ep{episode}.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ================================================================
#  训练状态 保存/加载
# ================================================================
def save_training_state(path, i_episode, time_step, best_test_reward, history):
    state = {
        "i_episode": i_episode,
        "time_step": time_step,
        "best_test_reward": best_test_reward,
        "history": history,
    }
    torch.save(state, path)


def load_training_state(path):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    return None


# ================================================================
#  测试评估
# ================================================================
def run_test_episode(env, ppo_agents, max_ep_len, speed_action_dim):
    """跑一个 greedy 测试 episode, 返回 (team_reward, mean_aoi, weighted_aoi, positions)"""
    env.reset()
    M = env.M
    action_queue = [None] * M
    target_masks = np.ones((M, env.target_action_dim), dtype=np.float32)
    done_bool = np.zeros(M, dtype=np.int32)
    rewards = [0.0] * M
    positions = [[] for _ in range(M)]

    for _ in range(max_ep_len):
        cand = np.full(M, np.inf, dtype=np.float32)
        for i in range(M):
            if done_bool[i]:
                continue
            if action_queue[i] is None:
                obs_pack = env.get_transformer_inputs(i)
                action_queue[i] = ppo_agents[i].action_test(
                    obs_pack, {"target": target_masks[i], "speed": np.ones(speed_action_dim)}
                )
            cand[i] = env.drone_timing_now[i] + env.time_cost(i, action_queue[i])
        if np.isinf(cand).all():
            break
        actor = int(np.argmin(cand))
        _, reward, done, info = env.step(actor, action_queue[actor])
        target_masks[actor] = info["target"]
        rewards[actor] += reward
        positions[actor].append(env.drone_position_now[actor].copy())
        action_queue[actor] = None
        if done:
            done_bool[actor] = 1
        if done_bool.all():
            break

    team_r = float(sum(rewards))
    mean_aoi = float(np.mean(env.aoi))
    w_aoi = float(np.sum(env.aoi * env.poi_weights) / max(np.sum(env.poi_weights), 1e-6))
    return team_r, mean_aoi, w_aoi, positions


# ================================================================
#  主训练函数
# ================================================================
def train():
    args = parse_args()

    # 目录
    ckpt_dir = os.path.join(args.checkpoint_dir, args.env_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_prefix = os.path.join(ckpt_dir, f"PPO_{args.env_name}_{args.random_seed}")
    state_path = f"{ckpt_prefix}_state.pt"

    output_dir = os.path.join(args.output_dir, args.env_name)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, args.env_name)
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_dir = os.path.join("logs", args.env_name)

    # 环境
    env = MultiDroneAoIEnv(
        args.M, args.N, args.K, args.T, args.map_size,
        args=args, position_file=args.position_file,
    )

    token_dim = env.token_dim
    target_action_dim = env.target_action_dim
    speed_action_dim = env.speed_action_dim
    critic_state_dim = int(env.get_global_critic_state().shape[0])

    # 创建 agents
    ppo_agents = []
    for i in range(args.M):
        ppo_agents.append(PPO(
            agent_id=i, token_dim=token_dim,
            target_action_dim=target_action_dim, speed_action_dim=speed_action_dim,
            max_encoder_len=env.encoder_token_len, max_decoder_len=env.decoder_token_len,
            max_critic_len=env.critic_token_len, max_other_agents=env.max_other_agents,
            lr_actor=args.lr_actor, lr_critic=args.lr_critic,
            gamma=args.gamma, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
            summary_dir=tensorboard_dir, entropy_ratio=args.entropy_ratio,
            gae_lambda=args.gae_lambda, gae_flag=args.gae_flag,
            decoupled_clip=args.decoupled_clip, eps_clip_pos=args.eps_clip_pos,
            d_model=args.transformer_dim, nhead=args.transformer_heads,
            num_layers=args.transformer_layers, dropout=args.transformer_dropout,
        ))

    shared_critic = CentralizedCritic(
        critic_state_dim=critic_state_dim, lr_critic=args.lr_critic,
        gamma=args.gamma, K_epochs=args.K_epochs,
        gae_lambda=args.gae_lambda, gae_flag=args.gae_flag,
        summary_writer=ppo_agents[0].writer, hidden_dim=args.transformer_dim,
    )

    # ================================================================
    #  断点续训: 尝试加载 checkpoint
    # ================================================================
    time_step = 0
    i_episode = 0
    best_test_reward = -np.inf
    history = {
        "episodes": [], "team_rewards": [], "mean_aois": [], "weighted_aois": [],
        "max_aois": [], "comm_active_ratios": [], "repeat_visit_rates": [],
        "test_episodes": [], "test_rewards": [], "test_mean_aois": [],
    }

    resumed = False
    if os.path.exists(f"{ckpt_prefix}_agent0.pth"):
        # 加载模型
        for i in range(args.M):
            ppo_agents[i].load(f"{ckpt_prefix}_agent{i}.pth")
        shared_critic.load(f"{ckpt_prefix}_critic.pth")

        # 加载训练状态
        saved_state = load_training_state(state_path)
        if saved_state is not None:
            i_episode = saved_state["i_episode"]
            time_step = saved_state["time_step"]
            best_test_reward = saved_state["best_test_reward"]
            history = saved_state["history"]
            resumed = True

    # 打印信息
    print("=" * 80)
    if resumed:
        print(f"  *** RESUMED from Episode {i_episode}, Timestep {time_step} ***")
    else:
        print(f"  *** NEW TRAINING ***")
    print(f"  P2P Comm MARL Training")
    print(f"  M={args.M}  K={args.K}  N={args.N}  T={args.T}  Map={args.map_size}")
    print(f"  R_comm={args.R_comm}  Speed={args.speed_levels}  Energy={args.init_uav_energy}")
    print(f"  Transformer dim={args.transformer_dim} heads={args.transformer_heads} layers={args.transformer_layers}")
    print(f"  obs_dim={env.observation_space.shape[0]}  token_dim={token_dim}  critic_dim={critic_state_dim}")
    print(f"  save_freq={args.save_freq}  plot_freq={args.plot_freq}  test_freq={args.test_freq}")
    print(f"  Total target: {args.max_training_timesteps} timesteps")
    print("=" * 80)

    # CSV 日志 (追加模式)
    log_f_name = os.path.join(log_dir, f"PPO_{args.env_name}_log.csv")
    if not os.path.exists(log_f_name) or not resumed:
        log_f = open(log_f_name, "w")
        log_f.write("episode,timestep,team_reward,mean_aoi,weighted_aoi,max_aoi,"
                     "comm_active_ratio,avg_neighbors,repeat_visit_rate\n")
    else:
        log_f = open(log_f_name, "a")

    start_time = datetime.now().replace(microsecond=0)
    print(f"Training started at: {start_time}")

    episodes_since_update = 0
    critic_rollout = []
    critic_rewards = []
    critic_dones = []
    critic_refs = []

    def run_ctde_update():
        nonlocal episodes_since_update
        if len(critic_rewards) == 0:
            return
        old_values = [s["value"] for s in critic_rollout]
        advantages, returns = shared_critic.compute_advantages_and_returns(
            rewards=critic_rewards, dones=critic_dones, old_values=old_values,
        )
        if advantages.numel() == 0:
            return
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_adv = [
            torch.zeros(len(a.buffer.target_actions), dtype=torch.float32)
            for a in ppo_agents
        ]
        for si, (aid, li) in enumerate(critic_refs):
            if 0 <= li < actor_adv[aid].numel():
                actor_adv[aid][li] = advantages[si]

        shared_critic.update(critic_rollout, returns)
        for aid in range(args.M):
            ppo_agents[aid].update(actor_adv[aid])

        critic_rollout.clear()
        critic_rewards.clear()
        critic_dones.clear()
        critic_refs.clear()

    def save_checkpoint(tag=""):
        for i in range(args.M):
            ppo_agents[i].save(f"{ckpt_prefix}_agent{i}.pth")
        shared_critic.save(f"{ckpt_prefix}_critic.pth")
        save_training_state(state_path, i_episode, time_step, best_test_reward, history)
        if tag:
            print(f"  [Checkpoint] saved at Ep {i_episode} t={time_step} ({tag})")

    # ================================================================
    #  主训练循环
    # ================================================================
    while time_step <= args.max_training_timesteps:
        env.reset()
        ep_reward = [0.0] * args.M
        done_tag = 0
        ep_start = len(critic_rewards)

        action_queue = [None] * args.M
        meta_queue = [None] * args.M
        target_masks = np.ones((args.M, target_action_dim), dtype=np.float32)
        speed_masks = np.ones((args.M, speed_action_dim), dtype=np.float32)
        done_bool = np.zeros(args.M, dtype=np.int32)
        non_base_count = np.zeros(args.M, dtype=np.int32)

        for _ in range(1, args.max_ep_len + 1):
            candidate_times = np.full(args.M, np.inf, dtype=np.float32)
            for i in range(args.M):
                if done_bool[i]:
                    continue
                if action_queue[i] is None:
                    obs_pack = env.get_transformer_inputs(i)
                    mask_pack = {"target": target_masks[i], "speed": speed_masks[i]}

                    threshold = int(args.force_return_bs_threshold)
                    if threshold > 0 and non_base_count[i] >= threshold:
                        forced = (env.K + env.N - 1, env.default_speed_idx)
                        action, trans = ppo_agents[i].select_action(
                            obs_pack, mask_pack, deter_action=forced
                        )
                        non_base_count[i] = 0
                    else:
                        action, trans = ppo_agents[i].select_action(obs_pack, mask_pack)
                        if action[0] < env.K:
                            non_base_count[i] += 1
                        else:
                            non_base_count[i] = 0

                    critic_snap = shared_critic.snapshot(obs_pack)
                    action_queue[i] = action
                    meta_queue[i] = {"actor_trans": trans, "critic_snap": critic_snap}

                candidate_times[i] = env.drone_timing_now[i] + env.time_cost(i, action_queue[i])

            if np.isinf(candidate_times).all():
                break

            actor = int(np.argmin(candidate_times))
            _, reward, done, info = env.step(actor, action_queue[actor])

            meta = meta_queue[actor]
            ppo_agents[actor].store_transition(meta["actor_trans"])
            li = ppo_agents[actor].last_transition_index()
            critic_rollout.append(meta["critic_snap"])
            critic_rewards.append(float(reward))
            critic_dones.append(0.0)
            critic_refs.append((actor, li))

            target_masks[actor] = info["target"]
            speed_masks[actor] = info["speed"]
            action_queue[actor] = None
            meta_queue[actor] = None
            ep_reward[actor] += reward
            time_step += 1

            if done:
                done_tag += 1
                done_bool[actor] = 1
            if done_tag == args.M:
                break
            if time_step > args.max_training_timesteps:
                break

        # ---- Episode 结束 ----
        if len(critic_dones) > ep_start:
            critic_dones[-1] = 1.0

        team_reward = sum(ep_reward)
        mean_aoi = float(np.mean(env.aoi))
        weighted_aoi = float(np.sum(env.aoi * env.poi_weights) / max(np.sum(env.poi_weights), 1e-6))
        max_aoi = float(np.max(env.aoi))
        comm_stats = env.get_comm_stats()

        i_episode += 1
        episodes_since_update += 1

        # 记录历史
        history["episodes"].append(i_episode)
        history["team_rewards"].append(team_reward)
        history["mean_aois"].append(mean_aoi)
        history["weighted_aois"].append(weighted_aoi)
        history["max_aois"].append(max_aoi)
        history["comm_active_ratios"].append(comm_stats["comm_active_ratio"])
        history["repeat_visit_rates"].append(comm_stats["repeat_visit_rate"])

        # CSV
        log_f.write(f"{i_episode},{time_step},{team_reward:.6f},{mean_aoi:.2f},"
                     f"{weighted_aoi:.2f},{max_aoi:.2f},"
                     f"{comm_stats['comm_active_ratio']:.4f},"
                     f"{comm_stats['avg_neighbors']:.4f},"
                     f"{comm_stats['repeat_visit_rate']:.4f}\n")
        log_f.flush()

        # TensorBoard
        ppo_agents[0].writer.add_scalar("reward/train_team", team_reward, i_episode)
        ppo_agents[0].writer.add_scalar("aoi/mean", mean_aoi, i_episode)
        ppo_agents[0].writer.add_scalar("aoi/weighted", weighted_aoi, i_episode)
        ppo_agents[0].writer.add_scalar("aoi/max", max_aoi, i_episode)
        ppo_agents[0].writer.add_scalar("comm/active_ratio", comm_stats["comm_active_ratio"], i_episode)
        ppo_agents[0].writer.add_scalar("comm/avg_neighbors", comm_stats["avg_neighbors"], i_episode)
        ppo_agents[0].writer.add_scalar("comm/repeat_visit_rate", comm_stats["repeat_visit_rate"], i_episode)

        # 打印
        if i_episode % args.print_freq == 0:
            elapsed = datetime.now().replace(microsecond=0) - start_time
            print(f"[Ep {i_episode:>5d}] t={time_step:<8d} team_r={team_reward:>8.4f}  "
                  f"aoi={mean_aoi:>7.1f}/{weighted_aoi:>7.1f}  "
                  f"comm={comm_stats['comm_active_ratio']:.3f}  "
                  f"repeat={comm_stats['repeat_visit_rate']:.3f}  "
                  f"elapsed={elapsed}")

        # CTDE 更新
        if episodes_since_update >= args.update_every_episodes:
            run_ctde_update()
            episodes_since_update = 0

        # ---- 测试评估 ----
        if i_episode % args.test_freq == 0:
            test_r, test_aoi, test_waoi, test_pos = run_test_episode(
                env, ppo_agents, args.max_ep_len, speed_action_dim
            )
            history["test_episodes"].append(i_episode)
            history["test_rewards"].append(test_r)
            history["test_mean_aois"].append(test_aoi)
            ppo_agents[0].call_2_record(i_episode, test_r)

            print(f"  [Test] reward={test_r:.4f}  aoi={test_aoi:.1f}  w_aoi={test_waoi:.1f}")

            # 保存路径可视化
            env.visualize_routes(
                test_pos, output_dir=output_dir,
                file=f"routes_ep{i_episode}_", nums=50,
            )

            if test_r > best_test_reward:
                best_test_reward = test_r
                env.visualize_routes(
                    test_pos, output_dir=output_dir,
                    file="best_test_", nums=50,
                )
                print(f"  [Best] new best reward: {best_test_reward:.4f}")

        # ---- 定期绘图 ----
        if i_episode % args.plot_freq == 0:
            plot_training_curves(history, output_dir, i_episode)

        # ---- 定期保存 checkpoint ----
        if i_episode % args.save_freq == 0:
            save_checkpoint(tag=f"periodic ep{i_episode}")

    # ================================================================
    #  训练结束
    # ================================================================
    if episodes_since_update > 0:
        run_ctde_update()

    save_checkpoint(tag="final")
    plot_training_curves(history, output_dir, i_episode)

    log_f.close()
    end_time = datetime.now().replace(microsecond=0)
    print("=" * 80)
    print(f"Training complete: {start_time} → {end_time} ({end_time - start_time})")
    print(f"Best test reward: {best_test_reward:.4f}")
    print(f"Total episodes: {i_episode}  Total timesteps: {time_step}")
    print(f"Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train()
