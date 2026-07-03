"""
run_baselines.py  —— 统一评估入口
=============================================
用法:
    python run_baselines.py --algo voronoi        # 仅跑 Voronoi 分区贪心
    python run_baselines.py --algo ga             # 仅跑遗传算法
    python run_baselines.py --algo all            # 两个都跑（默认）
    python run_baselines.py --algo all --episodes 10 --position_file ./data/poi_20_map_1000x1000.npy

所有参数都与原始 env 保持一致，可直接复用训练脚本的参数。
"""

import argparse
import time
import json
import os
import sys
import numpy as np


# ============================================================
# 1. 参数解析 —— 与 env 所需 args 对齐
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Multi-UAV AoI Baseline Evaluation")

    # --- 选择算法 ---
    p.add_argument("--algo", type=str, default="all",
                   choices=["voronoi", "ga", "all"],
                   help="要评估的算法: voronoi / ga / all")

    # --- 评估设置 ---
    p.add_argument("--episodes", type=int, default=5,
                   help="评估轮数")
    p.add_argument("--max_steps", type=int, default=5000,
                   help="每轮最大执行步数")
    p.add_argument("--verbose", action="store_true",
                   help="打印每步详情")
    p.add_argument("--save_results", type=str, default=None,
                   help="保存结果 JSON 的路径")

    # --- 环境参数 ---
    p.add_argument("--M", type=int, default=3, help="无人机数量")
    p.add_argument("--N", type=int, default=2, help="基站数量")
    p.add_argument("--K", type=int, default=5, help="POI 数量")
    p.add_argument("--T", type=float, default=100.0, help="时间上限(秒)")
    p.add_argument("--map_size", type=float, default=100.0, help="地图边长")
    p.add_argument("--position_file", type=str, default=None,
                   help="POI/BS 位置 .npy 文件路径")

    # --- Env args 兼容字段 ---
    p.add_argument("--speed_levels", type=str, default="6-20")
    p.add_argument("--init_uav_energy", type=float, default=2e5)
    p.add_argument("--init_uav_energies", type=str, default=None)
    p.add_argument("--reward_scale_size", type=float, default=10000.0)
    p.add_argument("--reward_divisor", type=float, default=10.0)
    p.add_argument("--pre_reward_ratio", type=float, default=3.0)
    p.add_argument("--BS_back_times", type=int, default=5,
                   help="Buffer 上限 / 返回基站阈值")
    p.add_argument("--buffer_punishment", action="store_true")
    p.add_argument("--punishment_value", type=float, default=4.0)
    p.add_argument("--print_info", action="store_true")
    p.add_argument("--history_horizon", type=int, default=10)

    # --- GA 专属参数 ---
    p.add_argument("--ga_pop_size", type=int, default=200,
                   help="遗传算法种群大小")
    p.add_argument("--ga_generations", type=int, default=2000,
                   help="遗传算法迭代代数")
    p.add_argument("--ga_mutation_rate", type=float, default=0.3,
                   help="遗传算法变异率")

    # --- 随机种子 ---
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ============================================================
# 2. 环境创建
# ============================================================
def make_env(args):
    """
    尝试导入真实环境；如果导入失败则提示用户调整 sys.path。
    """
    try:
        # 尝试多种可能的导入方式
        try:
            from env import MultiDroneAoIEnv
        except ImportError:
            from Env import MultiDroneAoIEnv
    except ImportError:
        print("=" * 60)
        print("[ERROR] 无法导入 MultiDroneAoIEnv。")
        print("请确保环境文件 (env.py / Env.py) 在当前目录或 PYTHONPATH 中。")
        print("例如:  cp /path/to/your/Env.py ./env.py")
        print("=" * 60)
        sys.exit(1)

    env = MultiDroneAoIEnv(
        M=args.M,
        N=args.N,
        K=args.K,
        T=args.T,
        map_size=args.map_size,
        args=args,
        position_file=args.position_file,
    )
    return env


# ============================================================
# 3. 运行单个算法的评估
# ============================================================
def evaluate_algorithm(name, env, policy_cls, policy_kwargs,
                       n_episodes, max_steps, verbose=False):
    """
    对指定算法进行多轮评估，返回统计结果。
    """
    all_metrics = []

    for ep in range(n_episodes):
        env.reset()

        # 创建 & 初始化策略
        policy = policy_cls(env, **policy_kwargs)
        policy.setup(env)

        # 异步执行循环
        M = env.M
        action_queue = [None] * M
        done_flag = [False] * M
        target_masks = [
            env.get_action_masks(i)["target"] for i in range(M)
        ]

        total_reward = 0.0
        per_uav_reward = np.zeros(M)
        per_uav_steps = np.zeros(M, dtype=int)

        for step_idx in range(max_steps):
            # Phase 1: 填充空闲无人机的动作
            candidate_times = np.full(M, np.inf)
            for uav in range(M):
                if done_flag[uav]:
                    continue
                if action_queue[uav] is None:
                    action_queue[uav] = policy.choose_action(
                        env, uav, target_masks[uav]
                    )
                if action_queue[uav] is not None:
                    candidate_times[uav] = (
                        env.drone_timing_now[uav]
                        + env.time_cost(uav, action_queue[uav])
                    )

            # Phase 2: 选择最早完成的无人机
            if np.all(np.isinf(candidate_times)):
                break
            actor = int(np.argmin(candidate_times))

            # Phase 3: 执行
            obs, reward, done, info = env.step(actor, [action_queue[actor], 10])

            total_reward += reward
            per_uav_reward[actor] += reward
            per_uav_steps[actor] += 1

            if verbose:
                action_val = action_queue[actor]
                target = int(action_val) if isinstance(action_val, (int, np.integer)) else int(action_val[0])
                kind = "POI" if target < env.K else "BS"
                print(f"  [Ep{ep} Step{step_idx}] UAV {actor} -> {kind} {target}, "
                      f"reward={reward:.6f}, time={env.drone_timing_now[actor]:.2f}")

            # Phase 4: 更新
            target_masks[actor] = info["target"]
            action_queue[actor] = None
            if done:
                done_flag[actor] = True
            if all(done_flag):
                break

        # 收集指标
        mean_aoi = float(np.mean(env.aoi))
        weighted_aoi = float(np.sum(env.aoi * env.poi_weights) / max(np.sum(env.poi_weights), 1e-6))
        max_aoi = float(np.max(env.aoi))

        metrics = {
            "episode": ep,
            "total_reward": float(total_reward),
            "mean_aoi": mean_aoi,
            "weighted_aoi": weighted_aoi,
            "max_aoi": max_aoi,
            "per_uav_steps": per_uav_steps.tolist(),
            "per_uav_reward": per_uav_reward.tolist(),
            "total_steps": int(np.sum(per_uav_steps)),
        }
        all_metrics.append(metrics)

        print(f"  [{name}] Episode {ep}: reward={total_reward:.6f}, "
              f"mean_aoi={mean_aoi:.2f}, weighted_aoi={weighted_aoi:.2f}, "
              f"max_aoi={max_aoi:.2f}, steps={metrics['total_steps']}")

    # 汇总统计
    rewards = [m["total_reward"] for m in all_metrics]
    mean_aois = [m["mean_aoi"] for m in all_metrics]
    weighted_aois = [m["weighted_aoi"] for m in all_metrics]
    max_aois = [m["max_aoi"] for m in all_metrics]

    summary = {
        "algorithm": name,
        "n_episodes": n_episodes,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_aoi_mean": float(np.mean(mean_aois)),
        "mean_aoi_std": float(np.std(mean_aois)),
        "weighted_aoi_mean": float(np.mean(weighted_aois)),
        "weighted_aoi_std": float(np.std(weighted_aois)),
        "max_aoi_mean": float(np.mean(max_aois)),
        "max_aoi_std": float(np.std(max_aois)),
        "per_episode": all_metrics,
    }
    return summary


# ============================================================
# 4. 打印对比表
# ============================================================
def print_comparison_table(results):
    """
    以表格形式打印所有算法的对比结果。
    """
    print("\n" + "=" * 90)
    print(f"{'算法':<25} {'总奖励(mean±std)':<25} {'平均AoI':<18} {'加权AoI':<18} {'最大AoI':<18}")
    print("-" * 90)
    for r in results:
        name = r["algorithm"]
        rew = f"{r['reward_mean']:.4f}±{r['reward_std']:.4f}"
        maoi = f"{r['mean_aoi_mean']:.2f}±{r['mean_aoi_std']:.2f}"
        waoi = f"{r['weighted_aoi_mean']:.2f}±{r['weighted_aoi_std']:.2f}"
        xaoi = f"{r['max_aoi_mean']:.2f}±{r['max_aoi_std']:.2f}"
        print(f"{name:<25} {rew:<25} {maoi:<18} {waoi:<18} {xaoi:<18}")
    print("=" * 90)


# ============================================================
# 5. 主入口
# ============================================================
def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("Multi-UAV AoI Baseline Evaluation")
    print(f"  算法:    {args.algo}")
    print(f"  评估轮数: {args.episodes}")
    print(f"  环境:    M={args.M}, K={args.K}, N={args.N}, T={args.T}")
    print("=" * 60)

    # 创建环境
    env = make_env(args)
    print(f"[INFO] 环境创建成功: M={env.M}, K={env.K}, N={env.N}, T={env.T}")
    print(f"       POI 权重: {env.poi_weights}")
    print(f"       速度级别: {env.speed_levels}")
    print()

    results = []
    algos_to_run = []
    if args.algo in ("voronoi", "all"):
        algos_to_run.append("voronoi")
    if args.algo in ("ga", "all"):
        algos_to_run.append("ga")

    # ---- Voronoi 分区贪心 ----
    if "voronoi" in algos_to_run:
        from voronoi_greedy import VoronoiGreedyPolicy
        print("[*] 评估 Voronoi 分区贪心算法 ...")
        t0 = time.time()
        summary = evaluate_algorithm(
            name="Voronoi-Greedy",
            env=env,
            policy_cls=VoronoiGreedyPolicy,
            policy_kwargs={"buffer_threshold": args.BS_back_times, "seed": args.seed},
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        summary["wall_time_sec"] = time.time() - t0
        results.append(summary)
        print(f"  完成，耗时 {summary['wall_time_sec']:.2f}s\n")

    # ---- 遗传算法 ----
    if "ga" in algos_to_run:
        from genetic_algorithm import GAPolicy
        print("[*] 评估遗传算法 ...")
        t0 = time.time()
        summary = evaluate_algorithm(
            name="Genetic-Algorithm",
            env=env,
            policy_cls=GAPolicy,
            policy_kwargs={
                "pop_size": args.ga_pop_size,
                "generations": args.ga_generations,
                "mutation_rate": args.ga_mutation_rate,
                "buffer_threshold": args.BS_back_times,
                "seed": args.seed,
            },
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        summary["wall_time_sec"] = time.time() - t0
        results.append(summary)
        print(f"  完成，耗时 {summary['wall_time_sec']:.2f}s\n")

    # ---- 输出对比 ----
    if len(results) > 1:
        print_comparison_table(results)

    # ---- 保存结果 ----
    if args.save_results:
        # 移除 per_episode 中的 numpy 类型
        for r in results:
            for ep_data in r.get("per_episode", []):
                for k, v in ep_data.items():
                    if isinstance(v, np.ndarray):
                        ep_data[k] = v.tolist()
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存到 {args.save_results}")


if __name__ == "__main__":
    main()

    """

# 同时评估两个算法（默认）
python run_baselines.py --algo all --episodes 10

# 仅跑 Voronoi 贪心
python run_baselines.py --algo voronoi --episodes 5

# 仅跑遗传算法，自定义参数
python run_baselines.py --algo ga --ga_pop_size 80 --ga_generations 200

# 指定环境参数 + 保存结果
python run_baselines.py --algo all --M 3 --K 40 --N 1 --T 2400 --map_size 2000 --position_file ./data/poi_40_map_2000x2000.npy --reward_divisor 400 --episodes 1 --save_results results.json --init_uav_energies "250000, 200000, 150000"
    
    """