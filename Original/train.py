import os
from datetime import datetime
import argparse
import torch
import numpy as np
from PPO_double import PPO_Hierarchical
from Env import MultiDroneAoIEnv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Joint PPO algorithm")

    parser.add_argument('--env_name', type=str, default="MAPPO", help="gym环境的名称")
    parser.add_argument('--mode', type=str, default="train", help="模式（训练/测试）")
    parser.add_argument('--has_continuous_action_space', action='store_true', help="动作空间是否为连续型（默认：离散型）")
    parser.add_argument('--max_ep_len', type=int, default=1000, help="每个回合的最大步数")
    parser.add_argument('--max_episodes', type=int, default=150000, help="最大训练episode数")

    parser.add_argument('--print_freq', type=int, default=50, help="打印平均奖励的间隔（episode）")
    parser.add_argument('--log_freq', type=int, default=50, help="记录日志的间隔（episode）")
    parser.add_argument('--save_model_freq', type=int, default=500, help="模型保存的频率（episode）")
    parser.add_argument('--test_freq', type=int, default=100, help="测试评估的频率（episode）")
    parser.add_argument('--plot_freq', type=int, default=100, help="绘制图表的频率（episode）")

    parser.add_argument('--action_std', type=float, default=0.3, help="动作分布的初始标准差（连续动作空间）")
    parser.add_argument('--action_std_decay_rate', type=float, default=0.03, help="动作标准差的衰减率")
    parser.add_argument('--min_action_std', type=float, default=0.01, help="动作标准差的最小值（达到后停止衰减）")
    parser.add_argument('--action_std_decay_freq', type=int, default=200, help="动作标准差的衰减频率（episode）")

    parser.add_argument('--K_epochs', type=int, default=5, help="每次PPO更新中策略更新的轮数")
    parser.add_argument('--eps_clip', type=float, default=0.2, help="PPO的裁剪参数")
    parser.add_argument('--gamma', type=float, default=0.99, help="折扣因子")
    parser.add_argument('--lr_actor', type=float, default=0.0002, help="上层actor网络的学习率")
    parser.add_argument('--lr_critic', type=float, default=0.001, help="上层critic网络的学习率")

    parser.add_argument('--entropy_ratio_upper', type=float, default=0.01, help="高层奖励熵的系数")
    parser.add_argument('--entropy_ratio_lower', type=float, default=0.1, help="下层奖励熵的系数")

    parser.add_argument('--gae-lambda', type=float, default=0.97, help='GAE的lambda参数')
    parser.add_argument('--gae_flag', action='store_true', default=True, help="是否使用GAE")

    parser.add_argument('--n_step_td_upper', type=int, default=3, help="上层模型n步TD（目标选择）")
    parser.add_argument('--n_step_td_lower', type=int, default=5, help="下层模型n步TD（速度选择）")

    parser.add_argument('--log_dir', type=str, default="PPO_logs", help="日志文件的目录")
    parser.add_argument('--checkpoint_dir', type=str, default="PPO_model", help="模型检查点的保存目录")
    parser.add_argument('--run_num_pretrained', type=int, default=0, help="预训练模型的编号，用于防止覆盖")

    parser.add_argument('--M', type=int, default=3, help="无人机数量")
    parser.add_argument('--N', type=int, default=1, help="基站数量")
    parser.add_argument('--K', type=int, default=60, help="兴趣点（PoI）数量")
    parser.add_argument('--map_size', type=int, default=3000, help="地图大小")
    parser.add_argument('--T', type=int, default=2400, help="时间限制")
    parser.add_argument('--BS_back_times', type=int, default=7, help="次数")
    parser.add_argument('--pre_reward_ratio', type=float, default=float(4), help="前置奖励系数")
    parser.add_argument('--buffer_punishment', default=True, help="是否对缓存区进行惩罚")
    parser.add_argument('--punishment_value', type=float, default=float(4), help="惩罚值")
    parser.add_argument('--punishment_value_2', type=float, default=float(2), help="惩罚值2")
    parser.add_argument('--reward_scale_size', type=float, default=float(100000), help="奖励缩放系数")
    parser.add_argument('--print_info', action='store_true', help="是否打印详细信息")
    parser.add_argument('--max_speed', type=int, default=25, help="默认最大速度")
    parser.add_argument('--min_speed', type=int, default=5, help="最小速度")
    parser.add_argument('--max_hover_radius', type=int, default=50, help="最大悬停半径")
    parser.add_argument('--Energy', type=int, default=300, help="默认能量")

    # 新增参数：每个无人机的能量和最大速度
    parser.add_argument('--uav_energies', nargs='+', type=int, default=[400, 400, 400], help="每个无人机的初始能量")
    parser.add_argument('--uav_max_speeds', nargs='+', type=int, default=[25, 25, 25], help="每个无人机的最大速度")

    return parser.parse_args()


def calculate_energy_consumption(initial_energy, remaining_energy):
    """计算能耗"""
    return initial_energy - remaining_energy


def calculate_average_speed(speeds_list):
    """计算平均速度"""
    if len(speeds_list) == 0:
        return 0.0
    return np.mean(speeds_list)


def plot_training_progress(all_reward, AOI_GAIN, episode, output_dir="training_plots"):
    """绘制训练进度图"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 绘制奖励曲线
    plt.figure(figsize=(15, 10))

    # 总奖励
    plt.subplot(2, 2, 1)
    total_rewards = [sum(rewards) for rewards in zip(*all_reward)]
    episodes = list(range(len(total_rewards)))
    plt.plot(episodes, total_rewards, 'b-', alpha=0.7, label='Total Reward')
    plt.title('Total Reward Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 各无人机奖励
    plt.subplot(2, 2, 2)
    for i in range(len(all_reward)):
        plt.plot(episodes, all_reward[i], label=f'UAV {i}', alpha=0.7)
    plt.title('Individual UAV Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # AoI增益
    plt.subplot(2, 2, 3)
    aoi_episodes = list(range(len(AOI_GAIN)))
    plt.plot(aoi_episodes, AOI_GAIN, 'g-', alpha=0.7)
    plt.title('AoI Gain Progress')
    plt.xlabel('Episode')
    plt.ylabel('AoI Gain')
    plt.grid(True, alpha=0.3)

    # 最近100个episode的滑动平均
    plt.subplot(2, 2, 4)
    if len(total_rewards) >= 100:
        moving_avg = []
        for i in range(99, len(total_rewards)):
            moving_avg.append(np.mean(total_rewards[i - 99:i + 1]))
        moving_episodes = list(range(99, len(total_rewards)))
        plt.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label='100-episode Moving Average')
        plt.title('Moving Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_progress.png', dpi=400, bbox_inches='tight')
    plt.close()


def plot_episode_speed_timeline(speeds_all_uavs, episode, output_dir="speed_analysis"):
    """Speed Timeline"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(15, 8))

    for uav_id, speeds in enumerate(speeds_all_uavs):
        if len(speeds) > 0:
            steps = np.arange(len(speeds))
            plt.plot(steps, speeds, marker='o', markersize=4, alpha=0.8,
                     linewidth=2, label=f'UAV {uav_id}')
    plt.xlabel('Action Step')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Speed Timeline for All UAVs (Episode {episode})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(5, 30)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/speed_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_violin_speed_analysis(speeds_all_uavs, episode, output_dir="speed_analysis"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_uavs = len(speeds_all_uavs)
    n_cols = 3
    n_rows = (n_uavs + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    # 为每个UAV绘制小提琴图样式的速度分布
    for uav_id, speeds in enumerate(speeds_all_uavs):
        if len(speeds) > 0:
            plt.subplot(n_rows, n_cols, uav_id + 1)
            try:
                create_violin_plot(speeds, f'UAV {uav_id}', episode)
            except Exception as e:
                print(f"Error creating violin plot for UAV {uav_id} in episode {episode}: {e}")
                # 创建一个简单的散点图作为备选
                plt.scatter(range(len(speeds)), speeds, alpha=0.6)
                plt.title(f'UAV {uav_id} Speed Data (Episode {episode})')
                plt.ylabel('Speed (m/s)')
                plt.xlabel('Action Step')

    plt.tight_layout()

    try:
        plt.savefig(f'{output_dir}/speed_distribution.png', dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving speed distribution plot: {e}")
    finally:
        plt.close()


def create_violin_plot(speeds, title, episode):
    speeds = np.array(speeds)

    # 检查数据有效性
    if len(speeds) == 0:
        print(f"Warning: No speed data for {title} in episode {episode}")
        return

    # 移除NaN和无穷大值
    speeds = speeds[np.isfinite(speeds)]

    if len(speeds) == 0:
        print(f"Warning: No valid speed data for {title} in episode {episode}")
        return

    min_speed = np.min(speeds)
    max_speed = np.max(speeds)
    mean_speed = np.mean(speeds)
    speed_range = np.linspace(5, 25, 100)

    # 检查是否所有速度值都相同或数据点太少
    if len(speeds) <= 1 or np.std(speeds) < 1e-6:
        # 如果只有一个点或所有点都相同，创建一个简单的分布
        print(f"Warning: Insufficient variation in speed data for {title} in episode {episode}")
        density = np.zeros_like(speed_range)
        # 在均值附近创建一个小的分布
        center_idx = np.argmin(np.abs(speed_range - mean_speed))
        if 0 <= center_idx < len(density):
            density[center_idx] = 0.4
    else:
        try:
            kde = stats.gaussian_kde(speeds)
            density = kde(speed_range)
            # 标准化密度，使最大宽度为0.4
            if np.max(density) > 0:
                density = density / np.max(density) * 0.4
            else:
                density = np.zeros_like(speed_range)
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: KDE failed for {title} in episode {episode}: {e}")
            # 创建一个简单的分布作为备选
            density = np.zeros_like(speed_range)
            center_idx = np.argmin(np.abs(speed_range - mean_speed))
            if 0 <= center_idx < len(density):
                density[center_idx] = 0.4

    plt.fill_betweenx(speed_range, -density, density, alpha=0.3, color='lightblue', label='Distribution')
    plt.plot([0, 0], [5, 25], 'b-', linewidth=2, alpha=0.8)

    # 为每个点添加随机的左右偏移
    np.random.seed(42)  # 保证可重复性
    jitter = np.random.uniform(-0.15, 0.15, len(speeds))

    plt.scatter(jitter, speeds, alpha=0.6, s=30, color='darkblue', label='Speed values')

    # 最小值线
    plt.plot([-0.3, 0.3], [min_speed, min_speed], 'r-', linewidth=2,
             label=f'Min: {min_speed:.1f} m/s')

    # 最大值线
    plt.plot([-0.3, 0.3], [max_speed, max_speed], 'r-', linewidth=2,
             label=f'Max: {max_speed:.1f} m/s')

    # 平均值线
    plt.plot([-0.3, 0.3], [mean_speed, mean_speed], 'orange', linewidth=3,
             label=f'Mean: {mean_speed:.1f} m/s')

    # 设置图形属性
    plt.ylim(5, 25)
    plt.xlim(-0.5, 0.5)
    plt.ylabel('Speed (m/s)')
    plt.title(f'{title} Speed Distribution (Episode {episode})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    # 移除x轴刻度标签，因为x轴没有实际意义
    plt.xticks([])


def run_test_episode(ppo_agents, env, args, max_steps=400, visualize=True, viz_steps=15):
    """运行测试episode并可视化轨迹"""
    env.reset()
    masks = np.ones((args.M, args.K + args.N))
    UAVs_actions_queue = [-1 for i in range(args.M)]
    done_tag = 0
    test_rewards = [0] * args.M
    test_actions = [[[0, 0]] for i in range(args.M)]
    test_speeds = [[] for i in range(args.M)]

    # 用于可视化的轨迹记录
    trajectory_positions = [[] for i in range(args.M)]
    trajectory_actions = [[] for i in range(args.M)]

    for i in range(args.M):
        trajectory_positions[i].append(env.drone_position_now[i].copy())

    UAVs_timing_next = env.drone_timing_now.copy()

    for t in range(1, max_steps + 1):
        for i in range(args.M):
            if UAVs_actions_queue[i] == -1:
                UAVs_actions_queue[i] = ppo_agents[i].action_test(env._get_obs(i), masks[i], i)
                UAVs_timing_next[i] += env.time_cost(i, UAVs_actions_queue[i])

        out_of_T = np.where(UAVs_timing_next >= args.T)
        UAVs_timing_next[out_of_T] = 100000
        action_UAV = np.argmin(UAVs_timing_next)

        state, reward, done, masked = env.step(action_UAV, UAVs_actions_queue[action_UAV])
        masks[action_UAV] = masked
        test_rewards[action_UAV] += reward
        test_actions[action_UAV].append(list(env.get_pos(int(UAVs_actions_queue[action_UAV][0]))))
        test_speeds[action_UAV].append(float(UAVs_actions_queue[action_UAV][1]))

        if len(trajectory_positions[action_UAV]) <= viz_steps:
            trajectory_positions[action_UAV].append(env.drone_position_now[action_UAV].copy())
            trajectory_actions[action_UAV].append(int(UAVs_actions_queue[action_UAV][0]))

        UAVs_actions_queue[action_UAV] = -1

        if done:
            done_tag += 1

        if done_tag == args.M:
            break

    if visualize:
        visualize_trajectory(env, trajectory_positions, trajectory_actions, viz_steps)

    return test_rewards, test_actions, test_speeds


def visualize_trajectory(env, trajectory_positions, trajectory_actions, viz_steps, output_dir="training_plots"):
    """在POI权重地图基础上可视化无人机轨迹"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    weights = env.weights
    poi_positions = env.sensor_pos
    bs_positions = env.base_pos

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 14))

    # 标准化权重用于显示（1-10的范围）
    display_weights = 1 + (weights - weights.min()) * 9 / (weights.max() - weights.min())

    # 1. 绘制POI点（根据权重着色）
    poi_scatter = ax.scatter(poi_positions[:, 0], poi_positions[:, 1],
                             c=display_weights, cmap='viridis', s=120,
                             alpha=0.8, edgecolors='black', linewidths=1.5,
                             vmin=1, vmax=10, zorder=3, label='POI')

    # 2. 绘制基站
    bs_scatter = ax.scatter(bs_positions[:, 0], bs_positions[:, 1],
                            c='red', s=150, marker='^',
                            alpha=1.0, edgecolors='darkred', linewidths=2,
                            zorder=5, label='Base Station')

    # 3. 绘制无人机轨迹
    uav_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive']

    for uav_id in range(len(trajectory_positions)):
        positions = trajectory_positions[uav_id]
        if len(positions) < 2:
            continue

        color = uav_colors[uav_id % len(uav_colors)]

        # 限制到viz_steps步
        limited_positions = positions[:min(viz_steps + 1, len(positions))]

        # 绘制轨迹路径
        if len(limited_positions) > 1:
            for i in range(len(limited_positions) - 1):
                start_pos = limited_positions[i]
                end_pos = limited_positions[i + 1]

                # 绘制轨迹线段
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                        color=color, linewidth=3, alpha=0.8, zorder=2,
                        label=f'UAV {uav_id} Path' if i == 0 else "")

                # 添加方向箭头
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                length = np.sqrt(dx ** 2 + dy ** 2)

                if length > 20:  # 只在距离足够大时添加箭头
                    # 计算箭头位置（线段中点偏向终点）
                    arrow_x = start_pos[0] + 0.7 * dx
                    arrow_y = start_pos[1] + 0.7 * dy

                    # 归一化方向向量
                    dx_norm = dx / length * 25  # 箭头长度
                    dy_norm = dy / length * 25

                    ax.arrow(arrow_x, arrow_y, dx_norm, dy_norm,
                             head_width=20, head_length=25, fc=color, ec=color,
                             alpha=0.8, zorder=4)

        # 标记起始和结束位置
        if limited_positions:
            # 起始位置（大星形）
            ax.scatter(limited_positions[0][0], limited_positions[0][1],
                       c=color, s=300, marker='*', alpha=1.0,
                       edgecolors='black', linewidths=2, zorder=6,
                       label=f'UAV {uav_id} Start')

            # 结束位置（如果不是起始位置）
            if len(limited_positions) > 1:
                ax.scatter(limited_positions[-1][0], limited_positions[-1][1],
                           c=color, s=250, marker='*', alpha=1.0,
                           edgecolors='white', linewidths=2, zorder=6)

            # 添加步数标签
            for i, pos in enumerate(limited_positions):
                if i > 0:  # 不标记起点
                    ax.annotate(f'{i}', (pos[0], pos[1]), xytext=(8, 8),
                                textcoords='offset points', fontsize=11,
                                color=color, fontweight='bold')

    # 4. 添加POI权重颜色条
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(poi_scatter, cax=cax)
    cbar.set_label('POI Weights', fontsize=12, fontweight='bold')
    cbar.set_ticks(range(1, 11))
    cbar.set_ticklabels([f'{i}' for i in range(1, 11)])

    # 5. 添加POI编号标签
    # for i, (pos, weight) in enumerate(zip(poi_positions, display_weights)):
    #     ax.annotate(f'P{i}', (pos[0], pos[1]), xytext=(0, -25),
    #                 textcoords='offset points', fontsize=9,
    #                 color='black', fontweight='bold', ha='center',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue',
    #                           edgecolor='blue', alpha=0.8))

    # 6. 添加基站编号标签
    # for i, pos in enumerate(bs_positions):
    #     ax.annotate(f'BS{i}', (pos[0], pos[1]), xytext=(0, 25),
    #                 textcoords='offset points', fontsize=10,
    #                 color='white', fontweight='bold', ha='center',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='red',
    #                           edgecolor='darkred', alpha=0.9))

    # 7. 设置图表属性
    ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'UAV Trajectory Visualization with POI Weights (First {viz_steps} Steps)\n'
                 f'Dashed lines: UAV to POI connections, Dotted lines: UAV to BS connections',
                 fontsize=14, fontweight='bold', pad=20)

    # 8. 网格和图例
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9,
              bbox_to_anchor=(0, 1), ncol=1)

    # 9. 设置坐标轴范围
    ax.set_xlim(-50, env.map_size + 50)
    ax.set_ylim(-50, env.map_size + 50)
    ax.set_aspect('equal', adjustable='box')

    # 10. 保存图片
    filename = f"trajectory_weights_first_{viz_steps}_steps.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"轨迹可视化图已保存: {filepath}")


def train():
    args = parse_args()

    # 构建实验名称，包含能量和最大速度信息
    experiment_name = f'{args.env_name}_joint'
    if args.uav_energies:
        energy_str = '_'.join(map(str, args.uav_energies[:args.M]))
        experiment_name += f'_energy_{energy_str}'
    if args.uav_max_speeds:
        speed_str = '_'.join(map(str, args.uav_max_speeds[:args.M]))
        experiment_name += f'_speed_{speed_str}'

    tensorboard_dir = os.path.join('tensorboard_logs', experiment_name)

    # 创建环境
    env = MultiDroneAoIEnv(args.M, args.N, args.K, args.T, args.map_size, args=args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_dim_lower = env.lower_observation_space.shape[0]

    # 设置日志目录
    log_dir = os.path.join(args.log_dir, args.env_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取运行编号
    current_num_files = len([f for f in os.listdir(log_dir) if f.endswith('.csv')])
    run_num = current_num_files
    log_f_name = os.path.join(log_dir, f'PPO_joint_{experiment_name}_log_{run_num}.csv')
    print(f"Current logging run number for {args.env_name}: {run_num}")
    print(f"Logging at: {log_f_name}")

    # 设置模型保存目录
    directory = os.path.join(args.checkpoint_dir, args.env_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    best_total_reward = float('-inf')
    best_episode = 0
    best_total_gain = float('-inf')
    best_episode_g = 0
    best_model_dir = os.path.join(directory, "best_models")
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    # 创建智能体
    ppo_agents = []
    for i in range(args.M):
        agent = PPO_Hierarchical(
            i, state_dim, action_dim, state_dim_lower, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs,
            args.eps_clip, args.has_continuous_action_space,
            tensorboard_dir, args.entropy_ratio_upper, args.entropy_ratio_lower,
            args.gae_lambda, args.gae_flag, args.min_speed, args.max_speed,
            args.n_step_td_upper, args.n_step_td_lower, args.action_std
        )
        ppo_agents.append(agent)

    # 开始训练
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT):", start_time)
    print("=" * 100)

    # 打印无人机配置信息
    print("无人机配置信息：")
    for i in range(args.M):
        print(f"UAV {i}: 初始能量={env.uav_energies[i]}, 最大速度={env.uav_max_speeds[i]}")
    print("=" * 100)

    # 创建日志文件
    log_f = open(log_f_name, "w+")

    # 动态创建CSV头部
    header = 'episode,total_reward,aoi_gain'
    for i in range(args.M):
        header += f',uav{i}_energy_remaining,uav{i}_energy_consumed,uav{i}_avg_speed,uav{i}_final_time'
    header += '\n'
    log_f.write(header)

    # 训练相关变量
    all_reward = [[] for _ in range(args.M)]
    AOI_GAIN = []

    # 速度监测相关变量
    avg_speed_history = [[] for _ in range(args.M)]
    energy_consumption_history = [[] for _ in range(args.M)]

    # 打印和记录相关变量
    print_reward_window = []
    window_size = args.print_freq

    print("=" * 50)
    print("开始联合训练：同时训练上层和下层PPO")
    print("=" * 50)

    # 主训练循环
    for i_episode in range(1, args.max_episodes + 1):

        # 重置环境开始新的episode
        env.reset()
        current_ep_reward = [0] * args.M
        done_tag = 0
        UAVs_actions_queue = [-1 for i in range(args.M)]

        masks = np.ones((args.M, args.K + args.N))
        actions_set = [[] for i in range(args.M)]
        positions_set = [[] for i in range(args.M)]
        speeds_set = [[] for i in range(args.M)]
        done_bool = [0 for i in range(args.M)]
        UAVs_timing_next = [0 for i in range(args.M)]

        # 记录初始能量
        initial_energies = env.uav_energies.copy()

        # 初始化episode访问计数器
        episode_poi_visits = [0 for _ in range(args.M)]  # 每个UAV在当前episode中访问POI的次数
        episode_bs_visits = [0 for _ in range(args.M)]  # 每个UAV在当前episode中访问BS的次数

        # 设置环境引用
        for agent in ppo_agents:
            agent.set_env(env)

        # Episode内的step循环
        for t in range(1, args.max_ep_len + 1):
            # 为每个无人机选择动作
            for i in range(args.M):
                if UAVs_actions_queue[i] == -1:
                    UAVs_actions_queue[i] = ppo_agents[i].select_action(env._get_obs(i), masks[i], i)
                    UAVs_timing_next[i] += env.time_cost(i, UAVs_actions_queue[i])

            # 处理已完成的无人机
            for i in range(args.M):
                UAVs_timing_next[i] += done_bool[i] * 100000

            # 选择下一个执行动作的无人机
            action_UAV = np.argmin(UAVs_timing_next)

            # 统计访问次数
            current_action = int(UAVs_actions_queue[action_UAV][0])
            if current_action < args.K:  # 访问POI
                episode_poi_visits[action_UAV] += 1
            elif args.K <= current_action < args.K + args.N:  # 访问BS
                episode_bs_visits[action_UAV] += 1

            # 执行动作
            state, reward, done, masked = env.step(action_UAV, UAVs_actions_queue[action_UAV])

            # 添加mask到buffer
            ppo_agents[action_UAV].add_mask_to_buffer(masks[action_UAV])

            # 记录动作和位置
            actions_set[action_UAV].append(UAVs_actions_queue[action_UAV])
            positions_set[action_UAV].append(env.drone_position_now[action_UAV].copy())
            current_speed = float(UAVs_actions_queue[action_UAV][1])
            speeds_set[action_UAV].append(current_speed)

            # 更新mask和重置动作队列
            masks[action_UAV] = masked
            UAVs_actions_queue[action_UAV] = -1

            # 添加reward和terminal到buffer
            ppo_agents[action_UAV].add_reward_to_buffer(reward, done)

            if done:
                done_tag += 1
                done_bool[action_UAV] = 1

            current_ep_reward[action_UAV] += reward

            # 如果所有无人机都完成了，结束episode
            if done_tag == args.M:
                break

        # Episode结束后的处理
        total_reward = sum(current_ep_reward)
        current_aoi_gain = env.aoi_gain_total

        # 计算能耗和平均速度
        remaining_energies = env.drone_energys.copy()
        energy_consumed = [initial_energies[i] - remaining_energies[i] for i in range(args.M)]
        avg_speeds = [calculate_average_speed(speeds_set[i]) for i in range(args.M)]
        final_times = env.drone_timing_now.copy()

        # 更新历史记录
        for i in range(args.M):
            all_reward[i].append(current_ep_reward[i])
            avg_speed_history[i].append(avg_speeds[i])
            energy_consumption_history[i].append(energy_consumed[i])

        AOI_GAIN.append(current_aoi_gain)
        print_reward_window.append(total_reward)
        if len(print_reward_window) > window_size:
            print_reward_window.pop(0)

        # 更新最佳奖励
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_episode = i_episode
        if current_aoi_gain > best_total_gain:
            best_total_gain = current_aoi_gain
            best_episode_g = i_episode
            # for idx, agent in enumerate(ppo_agents):
            #     best_path = os.path.join(best_model_dir, f"best_{experiment_name}_{idx}.pth")
            #     agent.save(best_path)
            if i_episode > 1000:
                # 保存最佳模型
                for idx, agent in enumerate(ppo_agents):
                    best_path = os.path.join(best_model_dir, f"best_{experiment_name}_{idx}.pth")
                    agent.save(best_path)

        # 打印训练信息
        if i_episode % args.print_freq == 0:
            avg_reward = np.mean(print_reward_window)
            print("-" * 100)
            print(f"Episode: {i_episode}  current_ep_reward: {total_reward:.2f} AoI Gain: {current_aoi_gain:.2f}")
            print(f"Average Reward (last {len(print_reward_window)} episodes): {avg_reward:.2f}  Best Total Reward: {best_total_reward:.2f} (Episode {best_episode}) Best Total Gain: {best_total_gain:.2f} (Episode {best_episode_g})")
            print(f"UAV Energy Remaining: {[f'{e:.1f}' for e in remaining_energies]}  Work time: {[f'{t:.1f}' for t in final_times]}  UAV Average Speeds: {[f'{speed:.2f}' for speed in avg_speeds]}")
            print(f"POI Visit Numbers: {episode_poi_visits}  BS Visit Numbers: {episode_bs_visits}")
            print("-" * 100)

        # 记录日志
        if i_episode % args.log_freq == 0:
            log_line = f'{i_episode},{total_reward:.4f},{current_aoi_gain:.4f}'
            for i in range(args.M):
                log_line += f',{remaining_energies[i]:.4f},{energy_consumed[i]:.4f},{avg_speeds[i]:.4f},{final_times[i]:.4f}'
            log_line += '\n'
            log_f.write(log_line)
            log_f.flush()

        # 更新模型
        for agent in ppo_agents:
            agent.update()
            torch.cuda.empty_cache()

        # 保存模型
        if i_episode % args.save_model_freq == 0:
            print(f"\nSaving models at episode {i_episode}...")
            for idx, agent in enumerate(ppo_agents):
                checkpoint_path = os.path.join(directory, f"{experiment_name}_uav{idx}.pth")
                agent.save(checkpoint_path)

        # 绘制训练进度图
        if i_episode % args.plot_freq == 0:
            plot_training_progress(all_reward, AOI_GAIN, i_episode)
            plot_episode_speed_timeline(speeds_set, i_episode)
            plot_violin_speed_analysis(speeds_set, i_episode)

        # 运行测试
        if i_episode % args.test_freq == 0:
            test_rewards, test_actions, test_speeds = run_test_episode(ppo_agents, env, args)
            test_total_reward = sum(test_rewards)

            # 记录测试结果到tensorboard
            if ppo_agents[0].writer:
                ppo_agents[0].call_2_record('test/total_reward', i_episode, test_total_reward)
                ppo_agents[0].call_2_record('test/aoi_gain', i_episode, env.aoi_gain_total)
                for idx in range(args.M):
                    ppo_agents[0].call_2_record(f'test/uav{idx}_reward', i_episode, test_rewards[idx])
                    if len(test_speeds[idx]) > 0:
                        ppo_agents[0].call_2_record(f'test/uav{idx}_avg_speed', i_episode, np.mean(test_speeds[idx]))

            print(f"Test Episode {i_episode}: Total Reward = {test_total_reward:.2f}")

    # 训练完成，保存最终模型
    print(f"\n{'=' * 50}")
    print("训练完成！保存最终模型...")
    for idx, agent in enumerate(ppo_agents):
        final_path = os.path.join(directory, f"{experiment_name}_final_uav{idx}.pth")
        agent.save(final_path)

    # 最终统计和绘图
    plot_training_progress(all_reward, AOI_GAIN, args.max_episodes)

    log_f.close()
    env.close()

    end_time = datetime.now().replace(microsecond=0)
    total_time = end_time - start_time

    print("=" * 100)
    print(f"训练开始时间: {start_time}")
    print(f"训练结束时间: {end_time}")
    print(f"总训练时间: {total_time}")
    print(f"最佳总奖励: {best_total_reward:.2f} (Episode {best_episode})")
    print(f"最佳总增益: {best_total_gain:.2f} (Episode {best_episode_g})")
    print(f"日志文件: {log_f_name}")
    print("=" * 100)


if __name__ == '__main__':
    train()
