import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from Env import MultiDroneAoIEnv


@dataclass
class EpisodeResult:
    team_reward: float
    uav_rewards: List[float]
    avg_aoi: float
    routes: List[List[int]]
    positions: List[List[np.ndarray]]
    uav_exec_times: List[float]
    uav_task_counts: List[int]
    uav_energy_used: List[float]
    global_events: List[Tuple[int, np.ndarray]]


def parse_args():
    parser = argparse.ArgumentParser(description="Round-robin baseline for asynchronous multi-UAV collection.")

    parser.add_argument("--env_name", type=str, default="A-MAPPO", help="Environment/run name used for log folder.")
    parser.add_argument("--M", type=int, default=2, help="Number of UAVs.")
    parser.add_argument("--N", type=int, default=1, help="Number of BS.")
    parser.add_argument("--K", type=int, default=20, help="Number of POIs (used if map file is not provided).")
    parser.add_argument("--T", type=int, default=1800, help="Episode time horizon.")
    parser.add_argument("--map_size", type=float, default=1000.0, help="Map size.")
    parser.add_argument("--position_file", type=str, default=None, help="Path to map .npy file.")

    parser.add_argument("--episodes", type=int, default=5, help="How many episodes to run.")
    parser.add_argument("--max_ep_len", type=int, default=2000, help="Max async interaction steps per episode.")

    parser.add_argument("--speed_levels", type=str, default="6-20", help="Speed levels, e.g. 6-20.")
    parser.add_argument("--fixed_speed", type=float, default=12.0, help="Fixed speed value for all UAV actions.")
    parser.add_argument(
        "--single_speed_only",
        action="store_true",
        help="Run baseline on one fixed speed only (default runs all speed levels).",
    )
    parser.add_argument("--return_buffer_size", type=int, default=5, help="Return to BS if local buffer length reaches this.")

    # Keep environment arguments compatible.
    parser.add_argument("--history_horizon", type=int, default=20)
    parser.add_argument("--init_uav_energy", type=float, default=300000.0)
    parser.add_argument(
        "--init_uav_energies",
        type=str,
        default=None,
        help="Optional per-UAV initial energies, e.g. \"250000,200000,150000\" or \"[250000 200000 150000]\"",
    )
    parser.add_argument("--pre_reward_ratio", type=float, default=3.0)
    parser.add_argument("--buffer_punishment", action="store_true")
    parser.add_argument("--punishment_value", type=float, default=4.0)
    parser.add_argument("--reward_scale_size", type=float, default=10000.0)
    parser.add_argument("--reward_divisor", type=float, default=10.0)
    parser.add_argument("--BS_back_times", type=int, default=5)
    parser.add_argument("--print_info", action="store_true")

    parser.add_argument("--visualize_last_episode", action="store_true", help="Additionally save route figure of last episode.")
    parser.add_argument(
        "--visualize_best_episode",
        dest="visualize_best_episode",
        action="store_true",
        help="Save route figure of best team-reward episode (default: enabled).",
    )
    parser.add_argument(
        "--no_visualize_best_episode",
        dest="visualize_best_episode",
        action="store_false",
        help="Disable saving route figure of best team-reward episode.",
    )
    parser.add_argument("--visualize_dir", type=str, default=None, help="Route figure output directory (default: logs/<env_name>).")
    parser.add_argument("--visualize_num_tasks", type=int, default=20, help="Number of tasks to visualize per UAV")

    parser.set_defaults(visualize_best_episode=True)
    return parser.parse_args()


def partition_center_from_bs(base_pos: np.ndarray) -> np.ndarray:
    # If there are multiple BS points, use their centroid as angular center.
    if base_pos.shape[0] == 1:
        return base_pos[0].astype(np.float32)
    return np.mean(base_pos, axis=0).astype(np.float32)


def split_pois_by_angle(sensor_pos: np.ndarray, center: np.ndarray, num_uavs: int) -> List[List[int]]:
    if num_uavs <= 0:
        raise ValueError(f"num_uavs must be positive, got {num_uavs}")

    rel = sensor_pos - center[None, :]
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    angles = np.mod(angles + 2.0 * np.pi, 2.0 * np.pi)  # [0, 2pi)
    radii = np.linalg.norm(rel, axis=1)

    sector_width = 2.0 * np.pi / float(num_uavs)
    sector_ids = np.floor(angles / sector_width).astype(np.int64)
    sector_ids = np.minimum(sector_ids, num_uavs - 1)

    regions: List[List[int]] = [[] for _ in range(num_uavs)]
    for poi_idx, sector in enumerate(sector_ids.tolist()):
        regions[sector].append(int(poi_idx))

    # Deterministic visiting order inside each angular sector.
    for uav_id in range(num_uavs):
        region = regions[uav_id]
        if len(region) == 0:
            continue
        idx = np.asarray(region, dtype=np.int64)
        order = np.lexsort((radii[idx], angles[idx]))
        regions[uav_id] = [region[i] for i in order]

    return regions


def nearest_bs_action(env: MultiDroneAoIEnv, uav_id: int) -> int:
    dist = np.linalg.norm(env.base_pos - env.drone_position_now[uav_id], axis=1)
    return env.K + int(np.argmin(dist))


def map_fixed_speed_to_idx(env: MultiDroneAoIEnv, fixed_speed: float) -> int:
    speed_levels = np.asarray(env.speed_levels, dtype=np.float32)
    return int(np.argmin(np.abs(speed_levels - fixed_speed)))


def choose_next_action(
    env: MultiDroneAoIEnv,
    uav_id: int,
    assigned_cycle: List[int],
    cursor: int,
    speed_idx: int,
    return_buffer_size: int,
) -> Tuple[Tuple[int, int], int]:
    # Return to BS if local buffer is large.
    if len(env.drone_buffer[uav_id]) >= return_buffer_size:
        return (nearest_bs_action(env, uav_id), speed_idx), cursor

    # Round-robin on assigned POIs, skip POIs already in local buffer.
    if len(assigned_cycle) > 0:
        for _ in range(len(assigned_cycle)):
            poi = assigned_cycle[cursor]
            cursor = (cursor + 1) % len(assigned_cycle)
            if poi not in env.drone_buffer[uav_id]:
                return (poi, speed_idx), cursor

    # If all assigned POIs are already buffered, go to BS.
    return (nearest_bs_action(env, uav_id), speed_idx), cursor


def build_global_tail_positions(global_events: List[Tuple[int, np.ndarray]], num_uavs: int, tail_n: int = 15):
    tail = global_events[-tail_n:] if tail_n > 0 else global_events
    positions = [[] for _ in range(num_uavs)]
    for uav_id, pos in tail:
        positions[int(uav_id)].append(np.asarray(pos, dtype=np.float32).copy())
    return positions


def run_one_episode(
    env: MultiDroneAoIEnv,
    speed_idx: int,
    return_buffer_size: int,
    max_ep_len: int,
    assigned: List[List[int]],
) -> EpisodeResult:
    env.reset()
    if len(assigned) != env.M:
        raise ValueError(f"assigned region count {len(assigned)} does not match M={env.M}")
    cursor = [0 for _ in range(env.M)]

    done_bool = np.zeros(env.M, dtype=np.int32)
    done_tag = 0

    queued_actions = [None for _ in range(env.M)]
    routes = [[] for _ in range(env.M)]
    positions = [[] for _ in range(env.M)]
    rewards = [0.0 for _ in range(env.M)]
    global_events = []

    for _ in range(max_ep_len):
        candidate_times = np.full(env.M, np.inf, dtype=np.float32)

        for uav_id in range(env.M):
            if done_bool[uav_id]:
                continue
            if queued_actions[uav_id] is None:
                queued_actions[uav_id], cursor[uav_id] = choose_next_action(
                    env=env,
                    uav_id=uav_id,
                    assigned_cycle=assigned[uav_id],
                    cursor=cursor[uav_id],
                    speed_idx=speed_idx,
                    return_buffer_size=return_buffer_size,
                )
            candidate_times[uav_id] = env.drone_timing_now[uav_id] + env.time_cost(uav_id, queued_actions[uav_id])

        if np.isinf(candidate_times).all():
            break

        action_uav = int(np.argmin(candidate_times))
        action = queued_actions[action_uav]
        _, reward, done, _ = env.step(action_uav, action)

        rewards[action_uav] += float(reward)
        routes[action_uav].append(int(action[0]))
        positions[action_uav].append(env.drone_position_now[action_uav].copy())
        global_events.append((action_uav, env.drone_position_now[action_uav].copy()))
        queued_actions[action_uav] = None

        if done:
            done_bool[action_uav] = 1
            done_tag += 1
            if done_tag == env.M:
                break

    return EpisodeResult(
        team_reward=float(sum(rewards)),
        uav_rewards=[float(r) for r in rewards],
        avg_aoi=float(np.mean(env.aoi)),
        routes=routes,
        positions=positions,
        uav_exec_times=[float(t) for t in env.drone_timing_now],
        uav_task_counts=[int(len(task_log)) for task_log in env.drone_task_time_log],
        uav_energy_used=[float(sum(costs)) for costs in env.drone_task_energy_log],
        global_events=global_events,
    )


def main():
    args = parse_args()
    visualize_dir = args.visualize_dir if args.visualize_dir is not None else f"./logs/{args.env_name}"

    env = MultiDroneAoIEnv(
        M=args.M,
        N=args.N,
        K=args.K,
        T=args.T,
        map_size=args.map_size,
        args=args,
        position_file=args.position_file,
    )

    partition_center = partition_center_from_bs(env.base_pos)
    assigned_regions = split_pois_by_angle(env.sensor_pos, partition_center, env.M)
    region_counts = [len(r) for r in assigned_regions]
    print(
        f"Angular partition center={partition_center.tolist()}, "
        f"uav_region_poi_counts={region_counts}, total_pois={int(sum(region_counts))}"
    )

    if args.single_speed_only:
        speed_indices = [map_fixed_speed_to_idx(env, args.fixed_speed)]
    else:
        speed_indices = list(range(len(env.speed_levels)))

    overall_summary = []
    last_result = None
    last_speed = None
    best_result = None
    best_speed = None
    best_episode = None
    best_team_reward = -np.inf

    for speed_idx in speed_indices:
        chosen_speed = float(env.speed_levels[speed_idx])
        print("=" * 80)
        print(f"Evaluating speed {chosen_speed:.2f} (speed_idx={speed_idx})")

        all_rewards = []
        all_aoi = []
        uav_cum_rewards = np.zeros(env.M, dtype=np.float64)
        uav_cum_tasks = np.zeros(env.M, dtype=np.int64)

        for ep in range(1, args.episodes + 1):
            result = run_one_episode(
                env=env,
                speed_idx=speed_idx,
                return_buffer_size=args.return_buffer_size,
                max_ep_len=args.max_ep_len,
                assigned=assigned_regions,
            )
            last_result = result
            last_speed = chosen_speed
            if result.team_reward > best_team_reward:
                best_team_reward = float(result.team_reward)
                best_result = result
                best_speed = chosen_speed
                best_episode = ep
            all_rewards.append(result.team_reward)
            all_aoi.append(result.avg_aoi)
            uav_cum_rewards += np.asarray(result.uav_rewards, dtype=np.float64)
            uav_cum_tasks += np.asarray(result.uav_task_counts, dtype=np.int64)

            print(
                f"[Speed {chosen_speed:.2f} | Episode {ep}] team_reward={result.team_reward:.6f}, "
                f"uav_rewards={result.uav_rewards}, avg_aoi={result.avg_aoi:.6f}, "
                f"uav_exec_times={result.uav_exec_times}, "
                f"uav_task_counts={result.uav_task_counts}, total_tasks={int(sum(result.uav_task_counts))}"
            )

        print(
            f"[Speed {chosen_speed:.2f}] avg_team_reward={np.mean(all_rewards):.6f}, "
            f"avg_aoi={np.mean(all_aoi):.6f}"
        )
        print(
            f"[Speed {chosen_speed:.2f}] uav_cumulative_rewards="
            f"{[round(float(v), 6) for v in uav_cum_rewards.tolist()]}, "
            f"uav_cumulative_task_counts={uav_cum_tasks.tolist()}, "
            f"total_tasks={int(np.sum(uav_cum_tasks))}"
        )
        if last_result is not None:
            print(f"[Speed {chosen_speed:.2f}] last_episode_uav_exec_times={last_result.uav_exec_times}")
            print(f"[Speed {chosen_speed:.2f}] last_episode_uav_energy_used={last_result.uav_energy_used}")

        overall_summary.append(
            {
                "speed": chosen_speed,
                "avg_team_reward": float(np.mean(all_rewards)),
                "avg_aoi": float(np.mean(all_aoi)),
                "uav_cum_rewards": [float(v) for v in uav_cum_rewards.tolist()],
                "uav_cum_tasks": [int(v) for v in uav_cum_tasks.tolist()],
            }
        )

    print("=" * 80)
    print("Summary across speeds:")
    for item in overall_summary:
        print(
            f"speed={item['speed']:.2f}, "
            f"uav_cumulative_rewards={[round(float(v), 6) for v in item['uav_cum_rewards']]}, "
            f"uav_cumulative_task_counts={item['uav_cum_tasks']}, "
            f"total_tasks={int(sum(item['uav_cum_tasks']))}, "
            f"avg_team_reward={item['avg_team_reward']:.6f}, "
            f"avg_aoi={item['avg_aoi']:.6f}"
        )

    if args.visualize_best_episode and best_result is not None:
        env.visualize_routes(
            best_result.positions,
            output_dir=visualize_dir,
            file="polling_best_routes_",
            nums=args.visualize_num_tasks,
        )
        tail15_positions = build_global_tail_positions(best_result.global_events, env.M, tail_n=15)
        env.visualize_routes(
            tail15_positions,
            output_dir=visualize_dir,
            file="polling_best_routes_last15_global_",
            nums=15,
        )
        print(
            f"Saved BEST route figure to {visualize_dir} "
            f"(speed={best_speed:.2f}, episode={best_episode}, reward={best_team_reward:.6f})"
        )

    if args.visualize_last_episode and last_result is not None:
        env.visualize_routes(
            last_result.positions,
            output_dir=visualize_dir,
            file="polling_last_",
            nums=args.visualize_num_tasks,
        )
        print(f"Saved LAST route figure to {visualize_dir} (speed={last_speed:.2f})")


if __name__ == "__main__":
    main()
