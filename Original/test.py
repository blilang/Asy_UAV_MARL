import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from PPO_double import PPO_Hierarchical
from Env import MultiDroneAoIEnv
from scipy import stats
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for Joint PPO algorithm")

    parser.add_argument('--env_name', type=str, default="MAPPO", help="环境名称")
    parser.add_argument('--max_ep_len', type=int, default=1000, help="每个回合的最大步数")
    parser.add_argument('--test_episodes', type=int, default=1, help="测试episodes数量")
    parser.add_argument('--viz_steps', type=int, default=15, help="可视化轨迹步数")
    parser.add_argument('--enable_gif', action='store_true', default=False, help="是否生成GIF动画")
    parser.add_argument('--checkpoint_dir', type=str, default="PPO_model", help="模型检查点目录")
    parser.add_argument('--fps', type=int, default=10, help="GIF帧率")
    parser.add_argument('--speed_multiplier', type=float, default=20.0, help="动画速度倍率")

    # 新增轮次可视化参数
    parser.add_argument('--viz_specific_round', type=int, default=1, help="可视化指定的轮次（如第2轮）")
    parser.add_argument('--viz_round_start', type=int, default=2, help="可视化轮次范围的起始轮次")
    parser.add_argument('--viz_round_end', type=int, default=3, help="可视化轮次范围的结束轮次")
    parser.add_argument('--viz_mode', type=str, default='specific', choices=['specific', 'range'], 
                        help="可视化模式：specific（指定单轮）或range（轮次范围）")

    # 新增字体大小控制参数
    parser.add_argument('--font_scale', type=float, default=1.0, help="统一字体大小缩放因子 (0.5-3.0)")
    parser.add_argument('--poi_label_size', type=float, default=None, help="POI标签字体大小")
    parser.add_argument('--bs_label_size', type=float, default=None, help="基站标签字体大小")
    parser.add_argument('--legend_size', type=float, default=None, help="图例字体大小")
    parser.add_argument('--axis_label_size', type=float, default=None, help="坐标轴标签字体大小")
    parser.add_argument('--axis_tick_size', type=float, default=None, help="坐标轴刻度字体大小")
    parser.add_argument('--title_size', type=float, default=None, help="标题字体大小")
    parser.add_argument('--colorbar_label_size', type=float, default=None, help="颜色条标签字体大小")
    parser.add_argument('--stats_text_size', type=float, default=None, help="统计文本字体大小")
    parser.add_argument('--round_annotation_size', type=float, default=None, help="轮次标注字体大小")

    # 环境参数 - 必须与训练时保持一致
    parser.add_argument('--M', type=int, default=3, help="无人机数量")
    parser.add_argument('--N', type=int, default=1, help="基站数量")
    parser.add_argument('--K', type=int, default=60, help="兴趣点数量")
    parser.add_argument('--map_size', type=int, default=3000, help="地图大小")
    parser.add_argument('--T', type=int, default=2400, help="时间限制")
    parser.add_argument('--BS_back_times', type=int, default=7, help="次数")
    parser.add_argument('--pre_reward_ratio', type=float, default=4.0, help="前置奖励系数")
    parser.add_argument('--buffer_punishment', default=False, help="是否对缓存区进行惩罚")
    parser.add_argument('--punishment_value', type=float, default=4.0, help="惩罚值")
    parser.add_argument('--punishment_value_2', type=float, default=2.0, help="惩罚值2")
    parser.add_argument('--reward_scale_size', type=float, default=100000.0, help="奖励缩放系数")
    parser.add_argument('--max_speed', type=int, default=25, help="默认最大速度")
    parser.add_argument('--min_speed', type=int, default=5, help="最小速度")
    parser.add_argument('--max_hover_radius', type=int, default=50, help="最大悬停半径")
    parser.add_argument('--Energy', type=int, default=300, help="默认能量")
    parser.add_argument('--print_info', action='store_true', help="是否打印详细信息")

    # 无人机配置 - 与训练时保持一致
    parser.add_argument('--uav_energies', nargs='+', type=int, default=[600, 400, 200], help="每个无人机的初始能量")
    parser.add_argument('--uav_max_speeds', nargs='+', type=int, default=[25, 25, 25], help="每个无人机的最大速度")

    return parser.parse_args()


class FontSizeConfig:
    """字体大小配置类"""
    def __init__(self, args):
        self.font_scale = max(0.5, min(3.0, args.font_scale))  # 限制在合理范围内
        
        # 基础字体大小（这些是matplotlib的默认大小）
        base_sizes = {
            'poi_label': 14,
            'bs_label': 18,
            'legend': 17,
            'axis_label': 26,
            'axis_tick': 24,
            'title': 15,
            'colorbar_label': 24,
            'stats_text': 14,
            'round_annotation': 15
        }
        
        # 应用用户自定义大小或使用缩放后的基础大小
        self.poi_label_size = args.poi_label_size if args.poi_label_size is not None else base_sizes['poi_label'] * self.font_scale
        self.bs_label_size = args.bs_label_size if args.bs_label_size is not None else base_sizes['bs_label'] * self.font_scale
        self.legend_size = args.legend_size if args.legend_size is not None else base_sizes['legend'] * self.font_scale
        self.axis_label_size = args.axis_label_size if args.axis_label_size is not None else base_sizes['axis_label'] * self.font_scale
        self.axis_tick_size = args.axis_tick_size if args.axis_tick_size is not None else base_sizes['axis_tick'] * self.font_scale
        self.title_size = args.title_size if args.title_size is not None else base_sizes['title'] * self.font_scale
        self.colorbar_label_size = args.colorbar_label_size if args.colorbar_label_size is not None else base_sizes['colorbar_label'] * self.font_scale
        self.stats_text_size = args.stats_text_size if args.stats_text_size is not None else base_sizes['stats_text'] * self.font_scale
        self.round_annotation_size = args.round_annotation_size if args.round_annotation_size is not None else base_sizes['round_annotation'] * self.font_scale
        
        # 其他相关的缩放元素
        self.poi_bbox_pad = 0.2 * self.font_scale
        self.bs_bbox_pad = 0.3 * self.font_scale
        self.stats_bbox_pad = 0.5 * self.font_scale
        self.round_bbox_pad = 0.3 * self.font_scale
        
        # POI和BS标签的偏移量也要相应缩放
        self.poi_label_offset = -25 * self.font_scale
        self.bs_label_offset = 20 * self.font_scale
        
    def print_config(self):
        """打印当前字体配置"""
        print(f"\n字体大小配置 (缩放因子: {self.font_scale:.1f}):")
        print(f"  POI标签: {self.poi_label_size:.1f}")
        print(f"  基站标签: {self.bs_label_size:.1f}")
        print(f"  图例: {self.legend_size:.1f}")
        print(f"  坐标轴标签: {self.axis_label_size:.1f}")
        print(f"  坐标轴刻度: {self.axis_tick_size:.1f}")
        print(f"  标题: {self.title_size:.1f}")
        print(f"  颜色条标签: {self.colorbar_label_size:.1f}")
        print(f"  统计文本: {self.stats_text_size:.1f}")
        print(f"  轮次标注: {self.round_annotation_size:.1f}")


def load_models(args, state_dim, action_dim, state_dim_lower):
    """加载训练好的模型"""
    experiment_name = f'{args.env_name}_joint'
    if args.uav_energies:
        energy_str = '_'.join(map(str, args.uav_energies[:args.M]))
        experiment_name += f'_energy_{energy_str}'
    if args.uav_max_speeds:
        speed_str = '_'.join(map(str, args.uav_max_speeds[:args.M]))
        experiment_name += f'_speed_{speed_str}'

    # 模型目录
    model_dir = os.path.join(args.checkpoint_dir, args.env_name, "best_models")

    ppo_agents = []
    for i in range(args.M):
        agent = PPO_Hierarchical(
            i, state_dim, action_dim, state_dim_lower, 0.0002, 0.001, 0.99, 5,
            0.2, False, None, 0.01, 0.1, 0.97, True,
            args.min_speed, args.max_speed, 3, 5, 0.3
        )

        # 加载最佳模型
        model_path = os.path.join(model_dir, f"best_{experiment_name}_{i}.pth")
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"成功加载UAV {i}的模型: {model_path}")
        else:
            print(f"警告: 找不到UAV {i}的模型文件: {model_path}")

        ppo_agents.append(agent)

    return ppo_agents


def UAV_Energy(v):
    """无人机能耗模型"""
    P_b = 79.86
    P_i = 88.63
    V_tip = 120
    u_0 = 4.03
    f_0 = 0.6
    a = 1.225
    n = 0.05
    R = 0.503
    energy = P_b * (1 + (3 * v * v) / (V_tip * V_tip)) + P_i * np.sqrt(
        np.sqrt(1 + (v * v * v * v) / (4 * (u_0 ** 4))) - v * v / (2 * u_0 * u_0)) + f_0 * a * n * R * v * v * v / 2
    return energy / 1000


def analyze_collection_rounds(trajectory_data, args):
    """
    分析每个无人机的收集轮次
    一轮定义：从开始收集POI到访问基站结束
    返回每个无人机的轮次信息
    """
    uav_rounds = []
    
    for uav_id in range(args.M):
        positions = trajectory_data['positions'][uav_id]
        actions = trajectory_data['actions'][uav_id]
        timestamps = trajectory_data['timestamps'][uav_id]
        
        rounds = []
        current_round_positions = []
        current_round_actions = []
        current_round_timestamps = []
        
        # 添加起始位置
        if len(positions) > 0:
            current_round_positions.append(positions[0])
            current_round_timestamps.append(timestamps[0] if len(timestamps) > 0 else 0)
        
        for i, action in enumerate(actions):
            # 添加当前动作对应的位置
            if i + 1 < len(positions):  # actions比positions少1个
                current_round_positions.append(positions[i + 1])
                current_round_timestamps.append(timestamps[i + 1] if i + 1 < len(timestamps) else 0)
                current_round_actions.append(action)
            
            # 检查是否访问基站（一轮结束）
            if action >= args.K:  # 访问基站
                round_info = {
                    'round_id': len(rounds) + 1,
                    'positions': current_round_positions.copy(),
                    'actions': current_round_actions.copy(),
                    'timestamps': current_round_timestamps.copy(),
                    'start_time': current_round_timestamps[0] if current_round_timestamps else 0,
                    'end_time': current_round_timestamps[-1] if current_round_timestamps else 0,
                    'poi_collected': [a for a in current_round_actions if a < args.K]
                }
                rounds.append(round_info)
                
                # 重置当前轮次，但保留基站位置作为下一轮的起点
                current_round_positions = [current_round_positions[-1]] if current_round_positions else []
                current_round_timestamps = [current_round_timestamps[-1]] if current_round_timestamps else []
                current_round_actions = []
        
        # 如果还有未完成的轮次（没有访问基站）
        if len(current_round_actions) > 0:
            round_info = {
                'round_id': len(rounds) + 1,
                'positions': current_round_positions.copy(),
                'actions': current_round_actions.copy(),
                'timestamps': current_round_timestamps.copy(),
                'start_time': current_round_timestamps[0] if current_round_timestamps else 0,
                'end_time': current_round_timestamps[-1] if current_round_timestamps else 0,
                'poi_collected': [a for a in current_round_actions if a < args.K]
            }
            rounds.append(round_info)
        
        uav_rounds.append(rounds)
    
    return uav_rounds


def get_available_rounds_info(uav_rounds, args):
    """
    获取所有无人机的轮次信息，确定可视化的有效范围
    修改版本：返回更详细的信息，不再限制最小轮数
    """
    min_rounds = float('inf')
    max_rounds = 0
    uav_round_counts = []
    
    for uav_id in range(args.M):
        round_count = len(uav_rounds[uav_id])
        uav_round_counts.append(round_count)
        if round_count > 0:
            min_rounds = min(min_rounds, round_count)
        max_rounds = max(max_rounds, round_count)
    
    if min_rounds == float('inf'):
        min_rounds = 0
    
    return {
        'min_rounds': min_rounds,
        'max_rounds': max_rounds,
        'uav_round_counts': uav_round_counts
    }


def validate_round_parameters(args, rounds_info):
    """
    验证轮次参数的有效性
    修改版本：基于最大轮数而不是最小轮数进行验证
    """
    min_rounds = rounds_info['min_rounds']
    max_rounds = rounds_info['max_rounds']
    
    if args.viz_mode == 'specific':
        if args.viz_specific_round is None:
            print(f"警告: 指定轮次模式但未设置具体轮次，使用默认值2")
            args.viz_specific_round = 2
        
        # 修改：基于最大轮数而不是最小轮数进行验证
        if args.viz_specific_round > max_rounds:
            print(f"警告: 指定轮次 {args.viz_specific_round} 超过最大轮次数 {max_rounds}，调整为 {max_rounds}")
            args.viz_specific_round = max_rounds
        
        if args.viz_specific_round < 1:
            print(f"警告: 指定轮次 {args.viz_specific_round} 小于1，调整为1")
            args.viz_specific_round = 1
    
    elif args.viz_mode == 'range':
        # 验证轮次范围 - 修改：基于最大轮数
        if args.viz_round_start < 1:
            print(f"警告: 起始轮次 {args.viz_round_start} 小于1，调整为1")
            args.viz_round_start = 1
        
        if args.viz_round_end > max_rounds:
            print(f"警告: 结束轮次 {args.viz_round_end} 超过最大轮次数 {max_rounds}，调整为 {max_rounds}")
            args.viz_round_end = max_rounds
        
        if args.viz_round_start > args.viz_round_end:
            print(f"警告: 起始轮次 {args.viz_round_start} 大于结束轮次 {args.viz_round_end}，交换顺序")
            args.viz_round_start, args.viz_round_end = args.viz_round_end, args.viz_round_start
    
    return args


def plot_flexible_rounds_trajectory(results, env, args, output_dir="test_results"):
    """
    灵活的轮次轨迹可视化函数 - 增强字体大小控制
    支持指定具体轮次或轮次范围
    修改版本：支持显示任意无人机的轨迹，即使其他无人机已经退出
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建字体大小配置
    font_config = FontSizeConfig(args)
    font_config.print_config()
    
    episode_id = results['episode_id']
    trajectory_data = results['trajectory_data']
    
    # 分析收集轮次
    uav_rounds = analyze_collection_rounds(trajectory_data, args)
    
    # 获取轮次信息并验证参数
    rounds_info = get_available_rounds_info(uav_rounds, args)
    args = validate_round_parameters(args, rounds_info)
    
    # 确定要可视化的轮次
    if args.viz_mode == 'specific':
        rounds_to_show = [args.viz_specific_round]
        title_suffix = f"Round {args.viz_specific_round}"
        filename_suffix = f"round_{args.viz_specific_round}"
    else:  # range mode
        rounds_to_show = list(range(args.viz_round_start, args.viz_round_end + 1))
        if len(rounds_to_show) == 1:
            title_suffix = f"Round {rounds_to_show[0]}"
            filename_suffix = f"round_{rounds_to_show[0]}"
        else:
            title_suffix = f"Rounds {args.viz_round_start}-{args.viz_round_end}"
            filename_suffix = f"rounds_{args.viz_round_start}_to_{args.viz_round_end}"
    
    # 定义颜色方案：每个UAV一种基础颜色，每轮用不同深浅
    base_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']
    
    # 为每轮生成颜色（由浅到深）
    def get_round_color(uav_id, round_id, total_rounds):
        base = base_colors[uav_id % len(base_colors)]
        if base == 'blue':
            colors = ['lightblue', 'blue', 'darkblue', 'navy', 'midnightblue']
        elif base == 'orange':
            colors = ['moccasin', 'orange', 'darkorange', 'orangered', 'darkred']
        elif base == 'green':
            colors = ['lightgreen', 'green', 'darkgreen', 'forestgreen', 'darkslategray']
        elif base == 'purple':
            colors = ['plum', 'purple', 'darkviolet', 'indigo', 'darkmagenta']
        elif base == 'brown':
            colors = ['burlywood', 'brown', 'saddlebrown', 'maroon', 'black']
        elif base == 'pink':
            colors = ['lightpink', 'hotpink', 'deeppink', 'crimson', 'darkred']
        else:
            colors = ['lightgray', 'gray', 'dimgray', 'darkgray', 'black']
        
        # 选择颜色，确保有足够的区分度
        color_idx = min(round_id - 1, len(colors) - 1)
        return colors[color_idx]
    
    # 创建图形，根据字体缩放调整图形大小
    base_figsize = (16, 14)
    figsize = (base_figsize[0] * max(1.0, font_config.font_scale * 0.8), 
               base_figsize[1] * max(1.0, font_config.font_scale * 0.8))
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制环境
    weights = env.weights
    poi_positions = env.sensor_pos
    bs_positions = env.base_pos
    display_weights = 1 + (weights - weights.min()) * 9 / (weights.max() - weights.min())
    
    # 计算散点大小，根据字体缩放调整
    base_poi_size = 120
    base_bs_size = 150
    poi_size = base_poi_size * font_config.font_scale
    bs_size = base_bs_size * font_config.font_scale
    
    # 绘制POI点（根据权重着色）
    poi_scatter = ax.scatter(poi_positions[:, 0], poi_positions[:, 1],
                             c=display_weights, cmap='viridis', s=poi_size,
                             alpha=0.8, edgecolors='black', 
                             linewidths=1.5 * font_config.font_scale,
                             vmin=1, vmax=10, zorder=3)
    
    # 绘制基站
    ax.scatter(bs_positions[:, 0], bs_positions[:, 1],
               c='red', s=bs_size, marker='^',
               alpha=1.0, edgecolors='darkred', 
               linewidths=2 * font_config.font_scale,
               zorder=5, label='Base Station')
    
    # 统计信息
    legend_elements = []
    total_trajectories = 0
    uav_stats = []
    
    # 绘制每个无人机的指定轮次轨迹
    # 修改：不再受最小轮数限制，每个无人机独立检查
    for uav_id in range(args.M):
        rounds = uav_rounds[uav_id]
        uav_total_rounds = len(rounds)
        uav_shown_rounds = 0
        uav_poi_collected = 0
        
        for round_id in rounds_to_show:
            # 检查该无人机是否有这一轮（独立检查，不受其他无人机影响）
            if round_id <= len(rounds):
                round_info = rounds[round_id - 1]  # round_id从1开始，索引从0开始
                positions = round_info['positions']
                actions = round_info['actions']
                poi_collected = round_info['poi_collected']
                
                if len(positions) < 2:
                    continue
                
                uav_shown_rounds += 1
                uav_poi_collected += len(poi_collected)
                total_trajectories += 1
                
                color = get_round_color(uav_id, round_id, uav_total_rounds)
                
                # 绘制轨迹路径，线宽根据字体缩放调整
                for i in range(len(positions) - 1):
                    start_pos = positions[i]
                    end_pos = positions[i + 1]
                    
                    # 调整线宽，不同轮次使用不同线宽
                    base_line_width = 6 - min(round_id - 1, 3)  # 第一轮最粗，后面逐渐变细
                    line_width = base_line_width * font_config.font_scale
                    alpha = 0.9
                    
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                            color=color, linewidth=line_width, alpha=alpha, zorder=2)
                    
                    # 添加方向箭头，大小根据字体缩放调整
                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]
                    length = np.sqrt(dx ** 2 + dy ** 2)
                    
                    if length > 20:
                        arrow_x = start_pos[0] + 0.6 * dx
                        arrow_y = start_pos[1] + 0.6 * dy
                        dx_norm = dx / length * 25 * font_config.font_scale
                        dy_norm = dy / length * 25 * font_config.font_scale
                        
                        ax.arrow(arrow_x, arrow_y, dx_norm, dy_norm,
                                 head_width=25 * font_config.font_scale, 
                                 head_length=30 * font_config.font_scale, 
                                 fc=color, ec=color,
                                 alpha=alpha, zorder=4)
                
                # 标记起始位置，大小根据字体缩放调整
                ax.scatter(positions[0][0], positions[0][1],
                           c=color, s=300 * font_config.font_scale, marker='*', alpha=1.0,
                           edgecolors='black', linewidths=2 * font_config.font_scale, zorder=6)
                
                # 标记轮次结束位置（基站位置）
                if len(positions) > 1:
                    end_marker = 's' if round_id == 1 else 'o'
                    base_marker_size = 200 if round_id == 1 else 150
                    marker_size = base_marker_size * font_config.font_scale
                    ax.scatter(positions[-1][0], positions[-1][1],
                               c=color, s=marker_size, marker=end_marker, alpha=1.0,
                               edgecolors='white', linewidths=2 * font_config.font_scale, zorder=6)
                
                # 添加轮次标签，使用配置的字体大小
                if len(positions) > 1:
                    mid_pos = positions[len(positions)//2]
                    ax.annotate(f'U{uav_id}R{round_id}', (mid_pos[0], mid_pos[1]), 
                               xytext=(10 * font_config.font_scale, 10 * font_config.font_scale), 
                               textcoords='offset points',
                               fontsize=font_config.round_annotation_size, 
                               fontweight='bold', color=color,
                               bbox=dict(boxstyle='round,pad=' + str(font_config.round_bbox_pad), 
                                       facecolor='white', edgecolor=color, alpha=0.8))
                
                # 添加图例元素
                from matplotlib.lines import Line2D
                legend_elements.append(Line2D([0], [0], color=color, lw=line_width, 
                                            label=f'UAV {uav_id} R{round_id} ({len(poi_collected)} POIs)'))
            else:
                # 记录无人机没有此轮次的信息
                print(f"UAV {uav_id} 没有第 {round_id} 轮")
        
        # 记录UAV统计信息
        uav_stats.append({
            'uav_id': uav_id,
            'total_rounds': uav_total_rounds,
            'shown_rounds': uav_shown_rounds,
            'poi_collected': uav_poi_collected
        })
    
    # 添加POI编号标签，使用配置的字体大小
    for i, pos in enumerate(poi_positions):
        ax.annotate(f'P{i}', (pos[0], pos[1]), 
                    xytext=(0, font_config.poi_label_offset),
                    textcoords='offset points', 
                    fontsize=font_config.poi_label_size,
                    color='black', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=' + str(font_config.poi_bbox_pad), 
                            facecolor='lightblue', edgecolor='blue', alpha=0.7))
    
    # 添加基站编号标签，使用配置的字体大小
    for i, pos in enumerate(bs_positions):
        ax.annotate(f'BS{i}', (pos[0], pos[1]), 
                    xytext=(0, font_config.bs_label_offset),
                    textcoords='offset points', 
                    fontsize=font_config.bs_label_size,
                    color='white', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=' + str(font_config.bs_bbox_pad), 
                            facecolor='red', alpha=0.9))
    
    # 添加颜色条，使用配置的字体大小
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(poi_scatter, cax=cax)
    cbar.set_label('POI Weights', fontsize=font_config.colorbar_label_size, fontweight='bold')
    
    # 设置颜色条刻度字体大小
    cbar.ax.tick_params(labelsize=font_config.axis_tick_size)
    
    # 统计信息文本，使用配置的字体大小
    stats_text = f"Visualization: {title_suffix}\n"
    stats_text += f"Available Rounds: Min={rounds_info['min_rounds']}, Max={rounds_info['max_rounds']}\n"
    stats_text += "UAV Summary:\n"
    
    for stat in uav_stats:
        stats_text += f"UAV {stat['uav_id']}: {stat['shown_rounds']}/{stat['total_rounds']} rounds, {stat['poi_collected']} POIs\n"
    
    stats_text += f"Total Trajectories Shown: {total_trajectories}"
    
    # 显示统计信息，使用配置的字体大小
    # ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes,
    #         fontsize=font_config.stats_text_size, verticalalignment='top',
    #         bbox=dict(boxstyle='round,pad=' + str(font_config.stats_bbox_pad), 
    #                 facecolor='wheat', alpha=0.9))
    
    # 设置图表属性，使用配置的字体大小
    ax.set_xlabel('X Coordinate (m)', fontsize=font_config.axis_label_size, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=font_config.axis_label_size, fontweight='bold')
    # ax.set_title(f'PPO UAV {title_suffix} Trajectory Analysis - Episode {episode_id + 1}\n'
    #              f'Collection Rounds with POI Weights', 
    #              fontsize=font_config.title_size, fontweight='bold', 
    #              pad=20 * font_config.font_scale)
    
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=font_config.axis_tick_size)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 调整图例位置和大小，使用配置的字体大小 - 修改为右上角
    if len(legend_elements) <= 6:
        ax.legend(handles=legend_elements, loc='upper right', 
              fontsize=font_config.legend_size, framealpha=0.9,
              bbox_to_anchor=(0.95, 0.95))
    else:
        # 对于较多图例项，使用单列布局
        ax.legend(handles=legend_elements, loc='upper right', 
              fontsize=font_config.legend_size * 0.9, framealpha=0.9,
              bbox_to_anchor=(0.95, 0.95), ncol=1)
    
    ax.set_xlim(-50, env.map_size + 50)
    ax.set_ylim(-50, env.map_size + 50)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # 保存图片
    filename = f'trajectory_{filename_suffix}_episode_{episode_id + 1}.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"轨迹图已保存: {output_dir}/{filename}")
    
    # 打印详细的轮次分析结果
    print(f"\nEpisode {episode_id + 1} {title_suffix} 轮次分析:")
    print(f"可视化模式: {args.viz_mode}")
    if args.viz_mode == 'specific':
        print(f"指定轮次: {args.viz_specific_round}")
    else:
        print(f"轮次范围: {args.viz_round_start} - {args.viz_round_end}")
    
    print(f"所有无人机轮次分布: {rounds_info['uav_round_counts']}")
    print(f"最少轮次: {rounds_info['min_rounds']}, 最多轮次: {rounds_info['max_rounds']}")
    
    for uav_id in range(args.M):
        rounds = uav_rounds[uav_id]
        print(f"\nUAV {uav_id}: 共 {len(rounds)} 轮")
        
        for round_id in rounds_to_show:
            if round_id <= len(rounds):
                round_info = rounds[round_id - 1]
                poi_collected = round_info['poi_collected']
                duration = round_info['end_time'] - round_info['start_time']
                print(f"  第 {round_id} 轮: 收集了 {len(poi_collected)} 个POI "
                      f"(POI IDs: {poi_collected}), 耗时: {duration:.1f}s")
            else:
                print(f"  第 {round_id} 轮: 不存在")


def run_single_test_episode_with_detailed_tracking(ppo_agents, env, args, episode_id=0):
    """运行单个测试episode并记录详细的时间线信息用于GIF生成"""
    # 重置环境
    env.reset()

    # 记录初始状态
    initial_energies = env.uav_energies.copy()
    initial_positions = env.drone_position_now.copy()

    print(f"\nEpisode {episode_id + 1} 开始:")
    print(f"  初始能量: {initial_energies}")
    print(f"  初始位置: {initial_positions}")
    print(f"  时间限制: {args.T}s")

    masks = np.ones((args.M, args.K + args.N))
    UAVs_actions_queue = [-1 for i in range(args.M)]
    done_tag = 0
    episode_rewards = [0.0] * args.M

    # 记录详细轨迹信息
    trajectory_data = {
        'positions': [[] for _ in range(args.M)],
        'speeds': [[] for _ in range(args.M)],
        'actions': [[] for _ in range(args.M)],
        'timestamps': [[] for _ in range(args.M)]
    }

    # 详细的时间线事件记录（用于GIF）
    detailed_events = []

    # 访问计数器
    poi_visit_counts = [[] for _ in range(args.M)]
    bs_visit_counts = [0 for _ in range(args.M)]

    # 记录初始位置和时间，以及初始事件
    for i in range(args.M):
        trajectory_data['positions'][i].append(env.drone_position_now[i].copy())
        trajectory_data['timestamps'][i].append(env.drone_timing_now[i])

        # 记录初始事件
        detailed_events.append({
            'time': 0.0,
            'uav_id': i,
            'event_type': 'start',
            'position': env.drone_position_now[i].copy(),
            'target_id': -1,
            'status': 'ready',
            'energy_before': initial_energies[i],
            'energy_after': initial_energies[i],
            'speed': 0.0,
            'action_type': 'init'
        })

    # 设置环境引用
    for agent in ppo_agents:
        agent.set_env(env)

    step_count = 0
    max_steps = args.max_ep_len

    # 主循环
    while step_count < max_steps:
        # 检查是否所有无人机都已完成任务
        if done_tag >= args.M:
            print(f"  所有无人机已完成任务，step={step_count}")
            break

        # 检查是否还有活跃的无人机
        active_uavs = [i for i in range(args.M) if env.drone_alive[i]]
        if not active_uavs:
            print(f"  没有活跃的无人机，step={step_count}")
            break

        # 为每个活跃的无人机选择动作
        UAVs_timing_next = env.drone_timing_now.copy()

        for i in range(args.M):
            if UAVs_actions_queue[i] == -1 and env.drone_alive[i]:
                # 检查时间约束
                if env.drone_timing_now[i] >= args.T:
                    env.drone_alive[i] = 0
                    UAVs_timing_next[i] = float('inf')
                    continue

                UAVs_actions_queue[i] = ppo_agents[i].action_test(env._get_obs(i), masks[i], i)
                time_cost = env.time_cost(i, UAVs_actions_queue[i])
                UAVs_timing_next[i] += time_cost

                # 预检查：如果执行这个动作会超时，则不执行
                if UAVs_timing_next[i] > args.T:
                    print(f"  UAV {i} 动作会导致超时 ({UAVs_timing_next[i]:.1f} > {args.T}), 提前结束")
                    env.drone_alive[i] = 0
                    UAVs_actions_queue[i] = -1
                    UAVs_timing_next[i] = float('inf')
                    done_tag += 1

        # 处理已完成的无人机
        for i in range(args.M):
            if not env.drone_alive[i]:
                UAVs_timing_next[i] = float('inf')

        # 如果所有无人机都完成，退出
        if all(UAVs_timing_next[i] == float('inf') for i in range(args.M)):
            print(f"  所有无人机时间已满或已完成，step={step_count}")
            break

        # 选择下一个执行动作的无人机
        action_UAV = np.argmin(UAVs_timing_next)

        if not env.drone_alive[action_UAV] or UAVs_actions_queue[action_UAV] == -1:
            print(f"  选中的UAV {action_UAV} 不可用，step={step_count}")
            break

        # 记录执行前的状态
        pre_time = env.drone_timing_now[action_UAV]
        pre_energy = env.drone_energys[action_UAV]
        pre_position = env.drone_position_now[action_UAV].copy()

        current_action = UAVs_actions_queue[action_UAV]
        target_id = int(current_action[0])
        speed = float(current_action[1])

        print(f"  Step {step_count}: UAV {action_UAV} 执行动作 [{target_id}, {speed:.1f}]")

        # 统计访问
        if target_id < args.K:  # 访问POI
            poi_visit_counts[action_UAV].append(target_id)
        elif target_id >= args.K:  # 访问BS
            bs_visit_counts[action_UAV] += 1

        # 执行动作
        state, reward, done, masked = env.step(action_UAV, current_action)

        # 记录执行后的状态
        post_time = env.drone_timing_now[action_UAV]
        post_energy = env.drone_energys[action_UAV]
        post_position = env.drone_position_now[action_UAV].copy()

        time_cost = post_time - pre_time
        energy_cost = pre_energy - post_energy

        # 检查时间合理性
        if post_time > args.T:
            env.drone_timing_now[action_UAV] = min(post_time, args.T)
            post_time = env.drone_timing_now[action_UAV]

        # 记录详细事件用于GIF
        # 移动事件
        move_distance = np.linalg.norm(post_position - pre_position)
        if move_distance > 1e-6:
            move_time = move_distance / speed if speed > 0 else time_cost * 0.8
            energy_move = move_time * UAV_Energy(speed)

            detailed_events.append({
                'time': pre_time,
                'uav_id': action_UAV,
                'event_type': 'move_start',
                'position': pre_position.copy(),
                'target_id': target_id,
                'status': f'moving_to_{target_id}',
                'energy_before': pre_energy,
                'energy_after': pre_energy - energy_move,
                'speed': speed,
                'action_type': 'move',
                'duration': move_time,
                'end_time': pre_time + move_time
            })

            detailed_events.append({
                'time': pre_time + move_time,
                'uav_id': action_UAV,
                'event_type': 'move_end',
                'position': post_position.copy(),
                'target_id': target_id,
                'status': f'arrived_at_{target_id}',
                'energy_before': pre_energy - energy_move,
                'energy_after': post_energy,
                'speed': 0.0,
                'action_type': 'arrive',
                'duration': time_cost - move_time,
                'end_time': post_time
            })

        # 通信事件
        if target_id < args.K:
            detailed_events.append({
                'time': post_time,
                'uav_id': action_UAV,
                'event_type': 'collect',
                'position': post_position.copy(),
                'target_id': target_id,
                'status': f'collecting_POI_{target_id}',
                'energy_before': post_energy,
                'energy_after': post_energy,
                'speed': 0.0,
                'action_type': 'collect',
                'duration': 0.1,
                'end_time': post_time + 0.1
            })
        else:
            detailed_events.append({
                'time': post_time,
                'uav_id': action_UAV,
                'event_type': 'offload',
                'position': post_position.copy(),
                'target_id': target_id,
                'status': f'offloading_BS_{target_id - args.K}',
                'energy_before': post_energy,
                'energy_after': post_energy,
                'speed': 0.0,
                'action_type': 'offload',
                'duration': 0.1,
                'end_time': post_time + 0.1
            })

        # 记录轨迹数据
        trajectory_data['positions'][action_UAV].append(post_position)
        trajectory_data['speeds'][action_UAV].append(speed)
        trajectory_data['actions'][action_UAV].append(target_id)
        trajectory_data['timestamps'][action_UAV].append(post_time)

        # 更新状态
        masks[action_UAV] = masked
        episode_rewards[action_UAV] += reward
        UAVs_actions_queue[action_UAV] = -1

        if done:
            print(f"    UAV {action_UAV} 任务完成 (done=True)")
            detailed_events.append({
                'time': post_time,
                'uav_id': action_UAV,
                'event_type': 'mission_complete',
                'position': post_position.copy(),
                'target_id': -1,
                'status': 'mission_complete',
                'energy_before': post_energy,
                'energy_after': post_energy,
                'speed': 0.0,
                'action_type': 'exit',
                'duration': 0.0,
                'end_time': post_time
            })
            done_tag += 1

        step_count += 1

        if step_count > max_steps:
            break

    # 计算最终统计
    final_energies = env.drone_energys.copy()
    energy_consumed = [max(0, initial_energies[i] - final_energies[i]) for i in range(args.M)]
    final_times = env.drone_timing_now.copy()
    final_times = [min(t, args.T) for t in final_times]

    # 计算平均速度
    avg_speeds = []
    for i in range(args.M):
        if len(trajectory_data['speeds'][i]) > 0:
            avg_speeds.append(np.mean(trajectory_data['speeds'][i]))
        else:
            avg_speeds.append(0.0)

    # 打印当前episode信息
    total_reward = sum(episode_rewards)
    aoi_gain = env.aoi_gain_total

    print(f"\nEpisode {episode_id + 1} 结果:")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  总增益: {aoi_gain:.2f}")
    print(f"  初始能量: {initial_energies}")
    print(f"  最终能量: {final_energies}")
    print(f"  能量消耗: {energy_consumed}")
    print(f"  结束时间: {final_times}")
    print(f"  平均速度: {[f'{s:.2f}' for s in avg_speeds]}")
    print(f"  POI访问次数: {[len(visits) for visits in poi_visit_counts]}")
    print(f"  BS访问次数: {bs_visit_counts}")
    print(f"  总执行步数: {step_count}")
    print("-" * 60)

    # 返回结果
    results = {
        'episode_id': episode_id,
        'total_reward': total_reward,
        'individual_rewards': episode_rewards,
        'aoi_gain': aoi_gain,
        'initial_energies': initial_energies,
        'final_energies': final_energies,
        'energy_consumed': energy_consumed,
        'final_times': final_times,
        'avg_speeds': avg_speeds,
        'poi_visits': poi_visit_counts,
        'bs_visits': bs_visit_counts,
        'trajectory_data': trajectory_data,
        'detailed_events': detailed_events,  # 新增详细事件记录
        'total_steps': step_count
    }

    return results


def run_single_test_episode(ppo_agents, env, args, episode_id=0):
    """简化版的测试episode运行函数"""
    return run_single_test_episode_with_detailed_tracking(ppo_agents, env, args, episode_id)


def create_synchronized_gif_with_energy_monitoring(results, env, args, output_dir="test_results"):
    """创建同步的轨迹和能量监控双面板GIF动画 - 修改版本，轨迹不会消失"""
    if not args.enable_gif:
        print("GIF生成已禁用")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    episode_id = results['episode_id']
    detailed_events = results['detailed_events']

    print(f"开始生成Episode {episode_id + 1}的双面板GIF动画...")

    # 计算动画时间范围
    max_time = max([event.get('end_time', event['time']) for event in detailed_events])
    if max_time <= 0:
        print("警告: 没有有效的时间数据，跳过GIF生成")
        return

    time_step = args.speed_multiplier / args.fps
    time_points = np.arange(0, max_time + time_step, time_step)

    print(f"动画时长: {max_time:.1f}s, 帧数: {len(time_points)}, 时间步长: {time_step:.1f}s")

    # 为每个时间点计算无人机状态
    def get_uav_state_at_time(uav_id, current_time):
        # 找到当前时间对应的最新状态
        relevant_events = [e for e in detailed_events if e['uav_id'] == uav_id and e['time'] <= current_time]

        if not relevant_events:
            # 返回初始状态
            initial_pos = env.base_pos[0] + np.array([[10, 0], [-10, 0], [0, 10], [0, -10]])[uav_id % 4]
            return initial_pos, "Ready", True, 0.0, results['initial_energies'][uav_id]

        # 找到最新的事件
        latest_event = max(relevant_events, key=lambda x: x['time'])

        # 检查是否在移动中
        move_events = [e for e in relevant_events if e['event_type'] == 'move_start' and e['time'] <= current_time]
        end_events = [e for e in relevant_events if e['event_type'] == 'move_end' and e['time'] <= current_time]

        # 如果有未完成的移动
        if len(move_events) > len(end_events):
            # 找到最后一个移动开始事件
            last_move = max(move_events, key=lambda x: x['time'])
            # 找到对应的移动结束事件
            corresponding_end = None
            for e in detailed_events:
                if (e['uav_id'] == uav_id and e['event_type'] == 'move_end' and
                        e['target_id'] == last_move['target_id'] and e['time'] > last_move['time']):
                    corresponding_end = e
                    break

            if corresponding_end and current_time < corresponding_end['time']:
                # 正在移动中，进行插值
                progress = (current_time - last_move['time']) / (corresponding_end['time'] - last_move['time'])
                progress = np.clip(progress, 0, 1)

                pos = last_move['position'] + (corresponding_end['position'] - last_move['position']) * progress
                energy = last_move['energy_before'] + (
                            corresponding_end['energy_after'] - last_move['energy_before']) * progress
                status = f"Moving to {last_move['target_id']} ({progress * 100:.1f}%)"
                return pos, status, True, last_move['speed'], energy

        # 检查任务完成状态
        if latest_event['event_type'] == 'mission_complete':
            status = f"Mission Complete ({latest_event['time']:.1f}s)"
            is_active = False
        elif latest_event['event_type'] == 'collect':
            status = f"Collecting POI {latest_event['target_id']}"
            is_active = True
        elif latest_event['event_type'] == 'offload':
            status = f"Offloading BS {latest_event['target_id'] - args.K}"
            is_active = True
        else:
            status = latest_event['status']
            is_active = True

        return latest_event['position'], status, is_active, latest_event['speed'], latest_event.get('energy_after', 0)

    # 准备环境数据
    weights = env.weights
    poi_positions = env.sensor_pos
    bs_positions = env.base_pos
    display_weights = 1 + (weights - weights.min()) * 9 / (weights.max() - weights.min())

    # 设置颜色
    uav_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']

    # 预计算所有时间点的轨迹数据（关键修改：存储整个轨迹历史）
    uav_trajectory_histories = [[] for _ in range(args.M)]
    
    # 为每个时间点预计算所有无人机的位置
    for t_idx, current_time in enumerate(time_points):
        for uav_id in range(args.M):
            pos, status, is_active, speed, energy = get_uav_state_at_time(uav_id, current_time)
            uav_trajectory_histories[uav_id].append(pos.copy())

    # 创建双面板图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    def animate(frame):
        ax1.clear()
        ax2.clear()
        current_time = time_points[frame]

        # 左侧面板: 轨迹图
        # 绘制POI点
        poi_scatter = ax1.scatter(poi_positions[:, 0], poi_positions[:, 1],
                                  c=display_weights, cmap='viridis', s=100,
                                  alpha=0.8, edgecolors='black', linewidths=1,
                                  vmin=1, vmax=10, zorder=3)

        # 绘制基站
        ax1.scatter(bs_positions[:, 0], bs_positions[:, 1],
                    c='red', s=120, marker='^',
                    alpha=1.0, edgecolors='darkred', linewidths=2, zorder=5)

        # 绘制无人机轨迹和当前位置
        active_uavs = 0
        uav_energies = []
        uav_statuses = []

        for uav_id in range(args.M):
            pos, status, is_active, speed, energy = get_uav_state_at_time(uav_id, current_time)
            color = uav_colors[uav_id % len(uav_colors)]

            uav_energies.append(energy)
            uav_statuses.append(status)

            # 绘制完整的历史轨迹（关键修改：显示到当前帧为止的所有轨迹）
            if frame > 0:  # 如果不是第一帧
                trajectory_to_now = uav_trajectory_histories[uav_id][:frame + 1]
                if len(trajectory_to_now) > 1:
                    traj = np.array(trajectory_to_now)
                    # 绘制轨迹线，使用渐变透明度效果
                    for i in range(len(traj) - 1):
                        alpha = 0.3 + 0.4 * (i / max(1, len(traj) - 1))  # 从0.3渐变到0.7
                        ax1.plot([traj[i][0], traj[i+1][0]], [traj[i][1], traj[i+1][1]], 
                                color=color, linewidth=2, alpha=alpha, zorder=2)

            # 绘制无人机当前位置
            if is_active:
                marker = 'o'
                size = 150
                edge_color = 'black'
                active_uavs += 1
            else:
                marker = 'X'
                size = 200
                edge_color = 'red'

            ax1.scatter(pos[0], pos[1], c=color, marker=marker, s=size,
                        edgecolors=edge_color, linewidth=2, zorder=10,
                        label=f'UAV {uav_id}')

            # 添加速度标签
            if speed > 0:
                ax1.annotate(f'{speed:.1f}m/s', (pos[0], pos[1]), xytext=(15, 15),
                             textcoords='offset points', fontsize=8, color=color,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # 状态信息文本框
        status_text = ""
        for uav_id in range(args.M):
            _, status, is_active, _, energy = get_uav_state_at_time(uav_id, current_time)
            status_symbol = "●" if is_active else "✗"
            status_text += f"UAV {uav_id} {status_symbol}: {status}\n"

        ax1.text(0.02, 0.98, status_text.strip(), transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9))

        # 设置左侧面板标题和信息
        ax1.set_title(f'PPO UAV Mission Progress - Episode {results["episode_id"] + 1}\n'
                      f'Time: {current_time:.1f}s | Active UAVs: {active_uavs}/{args.M} | Total Reward: {results["total_reward"]:.1f}',
                      fontsize=14, fontweight='bold')

        # 设置左侧面板属性
        ax1.set_xlim(-100, env.map_size + 100)
        ax1.set_ylim(-100, env.map_size + 100)
        ax1.set_xlabel('X Coordinate (m)', fontsize=12)
        ax1.set_ylabel('Y Coordinate (m)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_aspect('equal', adjustable='box')

        # 右侧面板: 能量监控
        # 计算能量百分比
        uav_ids = list(range(args.M))
        initial_energies = results['initial_energies']

        energy_percentages = []
        for i in range(args.M):
            if initial_energies[i] > 0:
                percentage = uav_energies[i] / initial_energies[i] * 100
            else:
                percentage = 0
            energy_percentages.append(percentage)

        # 绘制能量条形图
        bars = ax2.barh(uav_ids, energy_percentages,
                        color=[uav_colors[i] for i in range(args.M)], alpha=0.7)

        # 添加能量值标签和配置信息
        for i, (bar, energy, percentage) in enumerate(zip(bars, uav_energies, energy_percentages)):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{energy:.1f}J ({percentage:.1f}%)',
                     va='center', ha='left', fontsize=10)

            # 添加UAV配置信息
            config_text = f'Init:{initial_energies[i]}J, Max:{env.uav_max_speeds[i]}m/s'
            ax2.text(-5, bar.get_y() + bar.get_height() / 2, config_text,
                     va='center', ha='right', fontsize=8, alpha=0.8)

        ax2.set_xlabel('Energy Remaining (%)', fontsize=12)
        ax2.set_ylabel('UAV ID', fontsize=12)
        ax2.set_title(f'UAV Energy Status & Configuration\nTime: {current_time:.1f}s', fontsize=14)
        ax2.set_xlim(-30, 110)
        ax2.set_yticks(uav_ids)
        ax2.set_yticklabels([f'UAV {i}' for i in uav_ids])
        ax2.grid(True, alpha=0.3, axis='x')

        # 添加能量警告线
        ax2.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='Low Energy (20%)')
        ax2.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='Medium Energy (50%)')
        ax2.legend(loc='upper right', fontsize=8)

        plt.tight_layout()

    # 创建动画
    print(f"正在生成{len(time_points)}帧双面板动画...")
    anim = FuncAnimation(fig, animate, frames=len(time_points), interval=1000 / args.fps, repeat=True, blit=False)

    # 保存GIF
    energy_str = '_'.join(map(str, args.uav_energies[:args.M]))
    speed_str = '_'.join(map(str, args.uav_max_speeds[:args.M]))
    gif_path = f'{output_dir}/ppo_dual_panel_E{energy_str}_S{speed_str}_episode_{results["episode_id"] + 1}.gif'
    try:
        writer = PillowWriter(fps=args.fps)
        anim.save(gif_path, writer=writer)
        print(f"双面板GIF动画已保存: {gif_path}")
    except Exception as e:
        print(f"保存GIF失败: {e}")
        # 尝试备用方法
        try:
            anim.save(gif_path, writer='pillow', fps=args.fps)
            print(f"使用备用方法保存GIF成功: {gif_path}")
        except Exception as e2:
            print(f"备用方法也失败: {e2}")

    plt.close()


def plot_speed_analysis(all_results, args, output_dir="test_results"):
    """绘制速度分析图"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 收集所有episodes的速度数据
    all_speeds = [[] for _ in range(args.M)]
    for result in all_results:
        trajectory_data = result['trajectory_data']
        for uav_id in range(args.M):
            all_speeds[uav_id].extend(trajectory_data['speeds'][uav_id])

    # 绘制速度分布图
    fig, axes = plt.subplots(1, args.M, figsize=(6 * args.M, 6))
    if args.M == 1:
        axes = [axes]

    for uav_id in range(args.M):
        speeds = all_speeds[uav_id]
        if len(speeds) > 0:
            axes[uav_id].hist(speeds, bins=20, alpha=0.7, color=f'C{uav_id}', edgecolor='black')
            axes[uav_id].axvline(np.mean(speeds), color='red', linestyle='--',
                                 label=f'Mean: {np.mean(speeds):.1f} m/s')
            axes[uav_id].set_xlabel('Speed (m/s)')
            axes[uav_id].set_ylabel('Frequency')
            axes[uav_id].set_title(f'UAV {uav_id} Speed Distribution')
            axes[uav_id].legend()
            axes[uav_id].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/speed_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_test_results(all_results, args, output_dir="test_results"):
    """保存测试结果到CSV文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 汇总数据
    summary_data = []
    for result in all_results:
        episode_data = {
            'episode': result['episode_id'] + 1,
            'total_reward': result['total_reward'],
            'aoi_gain': result['aoi_gain'],
            'total_steps': result['total_steps']
        }

        # 添加每个无人机的数据
        for i in range(args.M):
            poi_visits_str = ','.join(map(str, result['poi_visits'][i]))
            episode_data.update({
                f'uav{i}_reward': result['individual_rewards'][i],
                f'uav{i}_initial_energy': result['initial_energies'][i],
                f'uav{i}_final_energy': result['final_energies'][i],
                f'uav{i}_energy_consumed': result['energy_consumed'][i],
                f'uav{i}_work_time': result['final_times'][i],
                f'uav{i}_avg_speed': result['avg_speeds'][i],
                f'uav{i}_poi_visits': poi_visits_str,
                f'uav{i}_bs_visits': result['bs_visits'][i],
                f'uav{i}_poi_visit_count': len(result['poi_visits'][i])
            })

        summary_data.append(episode_data)

    # 保存汇总数据
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{output_dir}/test_summary.csv', index=False)

    print(f"测试结果已保存到: {output_dir}/test_summary.csv")


def calculate_statistics(all_results, args):
    """计算并打印统计信息"""
    num_episodes = len(all_results)

    if num_episodes == 0:
        print("没有测试结果可以统计")
        return

    # 提取所有数据
    total_rewards = [r['total_reward'] for r in all_results]
    aoi_gains = [r['aoi_gain'] for r in all_results]

    # 按UAV分组的数据
    uav_rewards = [[] for _ in range(args.M)]
    uav_energy_consumed = [[] for _ in range(args.M)]
    uav_work_times = [[] for _ in range(args.M)]
    uav_avg_speeds = [[] for _ in range(args.M)]
    uav_poi_visit_counts = [[] for _ in range(args.M)]
    uav_bs_visit_counts = [[] for _ in range(args.M)]

    for result in all_results:
        for i in range(args.M):
            uav_rewards[i].append(result['individual_rewards'][i])
            uav_energy_consumed[i].append(result['energy_consumed'][i])
            uav_work_times[i].append(result['final_times'][i])
            uav_avg_speeds[i].append(result['avg_speeds'][i])
            uav_poi_visit_counts[i].append(len(result['poi_visits'][i]))
            uav_bs_visit_counts[i].append(result['bs_visits'][i])

    # 计算统计量
    def calc_stats(data):
        if len(data) == 0:
            return 0.0, 0.0
        return np.mean(data), np.std(data)

    print("\n" + "=" * 80)
    print(f"测试结果统计 (基于 {num_episodes} 个episodes):")
    print("=" * 80)

    # 整体统计
    total_reward_mean, total_reward_std = calc_stats(total_rewards)
    aoi_gain_mean, aoi_gain_std = calc_stats(aoi_gains)

    print(f"总奖励: 平均={total_reward_mean:.2f}, 标准差={total_reward_std:.2f}")
    print(f"AoI增益: 平均={aoi_gain_mean:.2f}, 标准差={aoi_gain_std:.2f}")

    # 按UAV统计
    for i in range(args.M):
        print(f"\nUAV {i} 统计:")

        reward_mean, reward_std = calc_stats(uav_rewards[i])
        energy_mean, energy_std = calc_stats(uav_energy_consumed[i])
        time_mean, time_std = calc_stats(uav_work_times[i])
        speed_mean, speed_std = calc_stats(uav_avg_speeds[i])
        poi_visit_mean, poi_visit_std = calc_stats(uav_poi_visit_counts[i])
        bs_visit_mean, bs_visit_std = calc_stats(uav_bs_visit_counts[i])

        print(f"  奖励: 平均={reward_mean:.2f}, 标准差={reward_std:.2f}")
        print(f"  能耗: 平均={energy_mean:.1f}, 标准差={energy_std:.1f}")
        print(f"  工作时间: 平均={time_mean:.1f}s, 标准差={time_std:.1f}s")
        print(f"  平均速度: 平均={speed_mean:.2f}m/s, 标准差={speed_std:.2f}m/s")
        print(f"  POI访问次数: 平均={poi_visit_mean:.1f}, 标准差={poi_visit_std:.1f}")
        print(f"  BS访问次数: 平均={bs_visit_mean:.1f}, 标准差={bs_visit_std:.1f}")

    print("=" * 80)


def main():
    args = parse_args()

    print("=" * 80)
    print("无人机多层次PPO测试 - 字体大小可调节版")
    print("=" * 80)

    # 创建环境
    env = MultiDroneAoIEnv(args.M, args.N, args.K, args.T, args.map_size, args=args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_dim_lower = env.lower_observation_space.shape[0]

    print(f"环境配置: M={args.M}, N={args.N}, K={args.K}")
    print(f"时间限制: {args.T}s")
    print(f"无人机能量配置: {args.uav_energies[:args.M]}")
    print(f"无人机最大速度配置: {args.uav_max_speeds[:args.M]}")
    print(f"GIF动画: {'启用' if args.enable_gif else '禁用'}")
    
    # 显示可视化参数
    print(f"\n可视化配置:")
    print(f"  模式: {args.viz_mode}")
    if args.viz_mode == 'specific':
        print(f"  指定轮次: {args.viz_specific_round}")
    else:
        print(f"  轮次范围: {args.viz_round_start} - {args.viz_round_end}")

    # 加载模型
    print("\n加载训练好的模型...")
    ppo_agents = load_models(args, state_dim, action_dim, state_dim_lower)

    # 运行测试
    print(f"\n开始运行 {args.test_episodes} 个测试episodes...")
    print("-" * 60)

    all_results = []

    for episode in range(args.test_episodes):
        result = run_single_test_episode(ppo_agents, env, args, episode)
        all_results.append(result)

        # 为第一个episode生成轨迹可视化
        if episode == 0:
            # 生成指定轮次的轨迹图
            print(f"\n生成轨迹可视化图...")
            plot_flexible_rounds_trajectory(result, env, args)
            
            # 生成GIF动画
            if args.enable_gif:
                create_synchronized_gif_with_energy_monitoring(result, env, args)

    # 分析和可视化结果
    print("\n生成分析图表...")
    plot_speed_analysis(all_results, args)
    save_test_results(all_results, args)

    # 计算并打印最终统计
    calculate_statistics(all_results, args)

    print(f"\n所有结果已保存到 test_results/ 目录")


if __name__ == '__main__':
    main()