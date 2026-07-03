"""
遗传算法 (Genetic Algorithm, GA) 基线 -- 多无人机 AoI 最小化路径规划
=======================================================================

核心思路
--------
使用遗传算法在 **离线阶段** 为每架无人机规划一条访问序列（POI 索引 + 基站返回点），
然后在环境中按照规划序列异步执行。

染色体编码
----------
- 一条染色体 = M 条子路径，每条子路径是一个动作索引列表。
  动作 0..K-1 表示访问对应 POI，K..K+N-1 表示返回对应基站。
- 约束：每个 POI 在整条染色体中至多出现一次；
  基站动作按 buffer_threshold 间隔插入。
- 示例：route_uav_0 = [3, 7, 1, K, 5, 9, K]
  含义：访问 POI3 -> POI7 -> POI1 -> 返回 BS0 -> 访问 POI5 -> POI9 -> 返回 BS0

适应度评估
----------
为了避免对真实环境进行 pop_size * generations 次模拟（太慢），
本模块实现了一个 **轻量级适应度模拟器** (LightweightSimulator)，
它用事件驱动方式近似计算总加权 AoI 奖励，比调用真实 env.step() 快数个数量级。

遗传算子
--------
- 选择：锦标赛选择（tournament size = 3）
- 交叉：单 UAV 路径级别的顺序交叉 (OX) + 整条路径交换
- 变异：段内交换 / 跨 UAV 迁移 / 2-opt 段反转 / 段内打乱
- 精英保留策略：每代保留 top-2 个体

执行阶段
--------
GAPolicy 按照最佳染色体的规划序列依次输出动作；
若规划用尽或目标被 mask，则回退到贪心策略（加权 AoI/距离 最大的 POI 或最近 BS）。
"""

import numpy as np
import copy
import time
from typing import List, Tuple, Optional, Dict, Any

# ============================================================================
#  能量模型常量（与环境一致的 UAV 功耗公式）
# ============================================================================
P_B = 79.86       # 叶片剖面功率 (W)
P_I = 88.63       # 感应功率 (W)
V_TIP = 120.0     # 叶尖速度 (m/s)
U_0 = 4.03        # 悬停时平均旋翼感应速度 (m/s)
F_0 = 0.6         # 机身阻力比
A_RHO = 1.225     # 空气密度 (kg/m^3)
N_ROTOR = 0.05    # 旋翼实度
R_ROTOR = 0.503   # 旋翼半径 (m)


def uav_power(v: float) -> float:
    """
    计算无人机在速度 v (m/s) 下的瞬时功耗 (W)。

    功率公式 (旋翼无人机标准模型)：
        P(v) = P_b * (1 + 3*v^2 / V_tip^2)
             + P_i * sqrt( sqrt(1 + v^4/(4*u_0^4)) - v^2/(2*u_0^2) )
             + f_0 * a * n * R * v^3 / 2

    注意：旋翼无人机的功率-速度曲线呈 U 型，
    悬停功率 (v=0) 较高，低速飞行时功率先下降后上升。

    Parameters
    ----------
    v : float
        飞行速度 (m/s)，非负

    Returns
    -------
    power : float
        瞬时功耗 (W)
    """
    # 第一项：叶片剖面功率（随 v^2 增长）
    blade = P_B * (1.0 + 3.0 * v ** 2 / V_TIP ** 2)
    # 第二项：感应功率（悬停时最大，飞行时下降）
    inner = np.sqrt(1.0 + v ** 4 / (4.0 * U_0 ** 4)) - v ** 2 / (2.0 * U_0 ** 2)
    induced = P_I * np.sqrt(max(inner, 0.0))
    # 第三项：寄生/机身阻力功率（随 v^3 增长）
    parasite = F_0 * A_RHO * N_ROTOR * R_ROTOR * v ** 3 / 2.0
    return blade + induced + parasite


def uav_energy_for_distance(dist: float, speed: float) -> float:
    """
    计算无人机以恒定速度 speed 飞行 dist 距离所需的能量 (J)。

    Parameters
    ----------
    dist : float
        飞行距离 (m)
    speed : float
        飞行速度 (m/s)

    Returns
    -------
    energy : float
        能量消耗 (J) = P(speed) * (dist / speed)
    """
    if speed <= 0 or dist <= 0:
        return 0.0
    flight_time = dist / speed
    return uav_power(speed) * flight_time


# ============================================================================
#  轻量级适应度模拟器
# ============================================================================
class LightweightSimulator:
    """
    轻量级模拟器：给定染色体（多 UAV 路径列表），快速估算总奖励。

    与真实环境不同之处：
    - 不维护完整的 obs / mask 状态
    - 按时间顺序逐动作执行，更新位置、时间、能量、AoI
    - 当无人机返回 BS 时，为缓冲区中每个 POI 计算奖励：
        reward_k = w_k * max(aoi_k - freshness_k, 0) * (1 + (T - t_upload) / T)
      其中 aoi_k = t_upload - last_reset[k], freshness_k = t_upload - t_collect
    - 对未覆盖的 POI 施加惩罚

    这足以在种群评估中区分好坏方案。

    优化：
    - 预计算所有目标点之间的距离矩阵，避免反复调用 np.linalg.norm
    """

    def __init__(self, env):
        """
        从环境中提取静态参数，构建轻量级模拟器。

        Parameters
        ----------
        env : MultiDroneAoIEnv
            真实环境实例（仅用于读取参数，不会调用 step）
        """
        self.M = env.M                         # 无人机数量
        self.N = env.N                         # 基站数量
        self.K = env.K                         # POI 数量
        self.T = env.T                         # 时间上限（秒）
        self.map_size = env.map_size           # 地图尺寸

        # 位置信息 (numpy 数组)
        self.sensor_pos = np.array(env.sensor_pos, dtype=np.float64)  # (K, 2)
        self.base_pos = np.array(env.base_pos, dtype=np.float64)      # (N, 2)
        self.poi_weights = np.array(env.poi_weights, dtype=np.float64) # (K,)

        # 默认飞行速度
        if hasattr(env, 'speed_levels') and env.speed_levels is not None:
            self.default_speed = env.speed_levels[env.default_speed_idx]
        else:
            self.default_speed = 10.0  # 后备默认值

        # 初始能量
        self.init_energy = np.array(env.init_uav_energy_list, dtype=np.float64)

        # ---------- 预计算距离矩阵以加速适应度评估 ----------
        # 索引布局: 0..K-1 = POI, K..K+N-1 = BS, K+N = 原点
        n_targets = self.K + self.N
        all_pos = np.vstack([self.sensor_pos, self.base_pos])  # (K+N, 2)
        origin = np.zeros((1, 2), dtype=np.float64)
        all_with_origin = np.vstack([all_pos, origin])          # (K+N+1, 2)
        diff = all_with_origin[:, np.newaxis, :] - all_with_origin[np.newaxis, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))   # (K+N+1, K+N+1)
        self.origin_idx = n_targets  # 原点在距离矩阵中的行/列索引

    def _get_dist(self, from_idx: int, to_action: int) -> float:
        """
        查表获取 from_idx 到 to_action 的欧氏距离。
        from_idx / to_action 可以是 0..K+N-1 (目标点) 或 origin_idx (原点)。
        """
        return self.dist_matrix[from_idx, to_action]

    def evaluate(self, chromosome: List[List[int]]) -> float:
        """
        评估染色体适应度（越大越好）。委托给 _evaluate_precise。
        """
        return self._evaluate_precise(chromosome)

    def _evaluate_precise(self, chromosome: List[List[int]]) -> float:
        """
        精确版适应度评估：事件驱动异步模拟。

        模拟流程：
        1. 所有 UAV 从原点 (0,0) 出发
        2. 每一步选完成时间最早的 UAV 推进
        3. 访问 POI -> 加入缓冲区；返回 BS -> 上传计算奖励
        4. 能量耗尽或超时的 UAV 自动停止
        5. 对未覆盖 / 未上传的 POI 施加惩罚

        Returns
        -------
        fitness : float
            总奖励（含惩罚项，越大越好）
        """
        M, K, T = self.M, self.K, self.T
        speed = self.default_speed

        # ---- UAV 状态初始化 ----
        cur_pos_idx = [self.origin_idx] * M   # 当前位置索引
        timing = np.zeros(M, dtype=np.float64)
        energy = self.init_energy.copy()
        # buffers[u] = [(poi_idx, collect_time), ...]  采集时刻记录
        buffers: List[List[Tuple[int, float]]] = [[] for _ in range(M)]

        # 每个 POI 的上次上传（AoI 重置）时刻，初始为 0
        last_reset = np.zeros(K, dtype=np.float64)

        route_ptrs = [0] * M
        finish_times = np.full(M, np.inf)
        total_reward = 0.0

        # 预计算各 UAV 第一步的完成时间
        for u in range(M):
            self._calc_finish_time(u, chromosome, route_ptrs, cur_pos_idx,
                                   timing, energy, finish_times, speed)

        # ---- 主循环 ----
        max_iters = sum(len(r) for r in chromosome) + 10
        for _ in range(max_iters):
            if np.all(np.isinf(finish_times)):
                break

            # 选完成时间最早的 UAV
            actor = int(np.argmin(finish_times))
            action = chromosome[actor][route_ptrs[actor]]
            dist = self._get_dist(cur_pos_idx[actor], action)
            e_cost = uav_energy_for_distance(dist, speed)
            arrive_time = finish_times[actor]

            # 更新 actor 状态
            timing[actor] = arrive_time
            cur_pos_idx[actor] = action
            energy[actor] -= e_cost

            if action < K:
                # ---- 访问 POI：存入缓冲区 ----
                buffers[actor].append((action, arrive_time))
            else:
                # ---- 返回基站：上传缓冲区，计算奖励 ----
                upload_time = arrive_time
                for poi_idx, collect_time in buffers[actor]:
                    aoi_val = upload_time - last_reset[poi_idx]
                    freshness = upload_time - collect_time
                    aoi_reduction = max(aoi_val - freshness, 0.0)
                    remaining_ratio = max((T - upload_time) / T, 0.0)
                    r = self.poi_weights[poi_idx] * aoi_reduction * (1.0 + remaining_ratio)
                    total_reward += r
                    last_reset[poi_idx] = upload_time
                buffers[actor] = []

            # 推进路径指针并计算下一步完成时间
            route_ptrs[actor] += 1
            finish_times[actor] = np.inf
            if route_ptrs[actor] < len(chromosome[actor]):
                self._calc_finish_time(actor, chromosome, route_ptrs, cur_pos_idx,
                                       timing, energy, finish_times, speed)

        # ---- 惩罚项 ----
        penalty = 0.0
        buffered_pois = set()
        for u in range(M):
            for poi_idx, _ in buffers[u]:
                buffered_pois.add(poi_idx)

        for k in range(K):
            remaining_aoi = T - last_reset[k]
            if k in buffered_pois:
                # 采集了但未上传：中等惩罚
                penalty += self.poi_weights[k] * remaining_aoi * 0.5
            elif last_reset[k] == 0.0:
                # 从未访问过：最大惩罚
                penalty += self.poi_weights[k] * T * 1.0

        return total_reward - penalty

    def _calc_finish_time(self, u: int, chromosome, route_ptrs, cur_pos_idx,
                          timing, energy, finish_times, speed: float):
        """
        计算 UAV u 执行其路径中下一个动作的完成时间。
        如果不可行（能量不足 / 超时 / 路径耗尽），设为 inf。
        """
        if route_ptrs[u] >= len(chromosome[u]):
            finish_times[u] = np.inf
            return
        action = chromosome[u][route_ptrs[u]]
        dist = self._get_dist(cur_pos_idx[u], action)
        if speed <= 0:
            finish_times[u] = np.inf
            return
        flight_time = dist / speed
        e_cost = uav_energy_for_distance(dist, speed)
        if energy[u] >= e_cost and timing[u] + flight_time <= self.T:
            finish_times[u] = timing[u] + flight_time
        else:
            finish_times[u] = np.inf


# ============================================================================
#  遗传算法规划器
# ============================================================================
class GeneticAlgorithmPlanner:
    """
    遗传算法规划器：为多架无人机离线规划访问序列。

    算法流程：
    1. 初始化种群（含启发式种子 + 随机个体）
    2. 用 LightweightSimulator 评估适应度
    3. 进化循环：锦标赛选择 -> 交叉 -> 变异 -> 评估 -> 精英保留
    4. 返回全局最佳染色体

    Parameters
    ----------
    env : MultiDroneAoIEnv
        环境实例（用于读取参数）
    pop_size : int
        种群大小（默认 50）
    generations : int
        进化代数（默认 100）
    mutation_rate : float
        变异概率（默认 0.3，即 30% 的后代会被变异）
    buffer_threshold : int or None
        每访问多少个 POI 后插入一次 BS 返回（默认自动计算）
    seed : int
        随机种子，用于可复现性
    """

    def __init__(self, env, pop_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.3, buffer_threshold: int = None,
                 seed: int = 42):
        self.env = env
        self.M = env.M
        self.N = env.N
        self.K = env.K
        self.T = env.T
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 缓冲区阈值
        if buffer_threshold is not None:
            self.buffer_threshold = buffer_threshold
        else:
            self.buffer_threshold = max(2, self.K // (self.M * 2))

        # 构建轻量级模拟器
        self.simulator = LightweightSimulator(env)

        # 位置信息缓存
        self.sensor_pos = np.array(env.sensor_pos, dtype=np.float64)
        self.base_pos = np.array(env.base_pos, dtype=np.float64)

    # ----------------------------------------------------------------
    #  染色体生成方法
    # ----------------------------------------------------------------
    def _random_chromosome(self) -> List[List[int]]:
        """
        随机生成一条染色体：
        1. 将 K 个 POI 随机打乱
        2. 均匀分配给 M 架 UAV
        3. 每架 UAV 内部再打乱
        4. 按 buffer_threshold 插入 BS 动作
        """
        pois = list(range(self.K))
        self.rng.shuffle(pois)

        routes: List[List[int]] = [[] for _ in range(self.M)]
        for i, poi in enumerate(pois):
            routes[i % self.M].append(poi)

        for u in range(self.M):
            self.rng.shuffle(routes[u])

        return self._insert_bs_visits(routes)

    def _nearest_neighbor_chromosome(self) -> List[List[int]]:
        """
        启发式种子 1：最近邻策略。
        轮流为每架 UAV 从当前位置选择距离最近的未访问 POI。
        """
        visited = set()
        routes: List[List[int]] = [[] for _ in range(self.M)]
        positions = np.zeros((self.M, 2), dtype=np.float64)

        for _ in range(self.K):
            for u in range(self.M):
                if len(visited) >= self.K:
                    break
                best_poi = -1
                best_dist = np.inf
                for k in range(self.K):
                    if k not in visited:
                        d = np.linalg.norm(self.sensor_pos[k] - positions[u])
                        if d < best_dist:
                            best_dist = d
                            best_poi = k
                if best_poi >= 0:
                    visited.add(best_poi)
                    routes[u].append(best_poi)
                    positions[u] = self.sensor_pos[best_poi].copy()
            if len(visited) >= self.K:
                break

        return self._insert_bs_visits(routes)

    def _aoi_greedy_chromosome(self) -> List[List[int]]:
        """
        启发式种子 2：AoI 权重贪心。
        按 POI 权重从大到小轮流分配给各架 UAV，
        保证高权重 POI 尽早被访问。
        """
        sorted_pois = np.argsort(-self.simulator.poi_weights).tolist()
        routes: List[List[int]] = [[] for _ in range(self.M)]
        for i, poi in enumerate(sorted_pois):
            routes[i % self.M].append(poi)
        return self._insert_bs_visits(routes)

    def _insert_bs_visits(self, routes: List[List[int]]) -> List[List[int]]:
        """
        在每条路径中按 buffer_threshold 间隔插入 BS 返回动作。
        BS 选择距最后一个 POI 最近的基站。

        Parameters
        ----------
        routes : list of list of int
            每架 UAV 的纯 POI 序列（不含 BS，或混合序列中 BS 会被过滤）

        Returns
        -------
        routes_with_bs : list of list of int
        """
        result: List[List[int]] = []
        for u in range(self.M):
            new_route = []
            poi_count = 0
            for action in routes[u]:
                if action >= self.K:
                    # 过滤掉已有的 BS 动作，防止嵌套调用时产生重复
                    continue
                new_route.append(action)
                poi_count += 1
                if poi_count >= self.buffer_threshold:
                    bs_idx = self._find_nearest_bs(action)
                    new_route.append(self.K + bs_idx)
                    poi_count = 0
            # 路径末尾如果还有未上传的 POI，追加一个 BS
            if poi_count > 0 and len(new_route) > 0:
                last_poi = new_route[-1]
                if last_poi < self.K:
                    bs_idx = self._find_nearest_bs(last_poi)
                else:
                    bs_idx = 0
                new_route.append(self.K + bs_idx)
            result.append(new_route)
        return result

    def _find_nearest_bs(self, poi_idx: int) -> int:
        """
        找到距指定 POI 最近的基站索引 (0..N-1)。
        """
        if poi_idx >= self.K:
            return poi_idx - self.K
        pos = self.sensor_pos[poi_idx]
        dists = np.linalg.norm(self.base_pos - pos, axis=1)
        return int(np.argmin(dists))

    # ----------------------------------------------------------------
    #  初始种群
    # ----------------------------------------------------------------
    def _init_population(self) -> List[List[List[int]]]:
        """
        初始化种群：
        - 2 个启发式种子（最近邻 + AoI 贪心）
        - 其余随机生成
        """
        population = []
        population.append(self._nearest_neighbor_chromosome())
        population.append(self._aoi_greedy_chromosome())
        while len(population) < self.pop_size:
            population.append(self._random_chromosome())
        return population

    # ----------------------------------------------------------------
    #  适应度
    # ----------------------------------------------------------------
    def _fitness(self, chromosome: List[List[int]]) -> float:
        """用轻量级模拟器评估染色体适应度。"""
        return self.simulator._evaluate_precise(chromosome)

    # ----------------------------------------------------------------
    #  选择算子：锦标赛选择
    # ----------------------------------------------------------------
    def _tournament_select(self, population: List[List[List[int]]],
                           fitnesses: List[float],
                           tournament_size: int = 3) -> List[List[int]]:
        """
        锦标赛选择：随机抽 tournament_size 个个体，返回适应度最高者（深拷贝）。
        """
        pop_size = len(population)
        ts = min(tournament_size, pop_size)
        indices = self.rng.choice(pop_size, size=ts, replace=False)
        best_idx = indices[0]
        for idx in indices[1:]:
            if fitnesses[idx] > fitnesses[best_idx]:
                best_idx = idx
        return copy.deepcopy(population[best_idx])

    # ----------------------------------------------------------------
    #  辅助：提取纯 POI 序列 / 修复覆盖
    # ----------------------------------------------------------------
    def _extract_pois(self, chromosome: List[List[int]]) -> List[List[int]]:
        """从染色体中提取每架 UAV 的纯 POI 序列（移除 BS 动作）。"""
        return [[a for a in route if a < self.K] for route in chromosome]

    def _repair_poi_coverage(self, child_pois: List[List[int]]) -> List[List[int]]:
        """
        修复 POI 覆盖：确保每个 POI 恰好出现一次。
        1. 去重：保留每个 POI 的首次出现
        2. 补漏：将遗漏的 POI 分配给最短路径的 UAV

        Parameters
        ----------
        child_pois : list of list of int
            可能有重复或遗漏的 POI 分配

        Returns
        -------
        fixed_pois : list of list of int
            修复后的分配（每个 POI 恰好出现一次）
        """
        # 去重
        seen = set()
        for u in range(self.M):
            deduped = []
            for p in child_pois[u]:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            child_pois[u] = deduped

        # 补漏
        all_assigned = set()
        for u in range(self.M):
            all_assigned.update(child_pois[u])
        missing = [k for k in range(self.K) if k not in all_assigned]
        for poi in missing:
            shortest_u = min(range(self.M), key=lambda uu: len(child_pois[uu]))
            child_pois[shortest_u].append(poi)

        return child_pois

    # ----------------------------------------------------------------
    #  交叉算子
    # ----------------------------------------------------------------
    def _crossover(self, parent1: List[List[int]],
                   parent2: List[List[int]]) -> List[List[int]]:
        """
        交叉算子：以 50% 概率选择 OX 或路径交换。
        """
        if self.rng.random() < 0.5:
            return self._ox_crossover(parent1, parent2)
        else:
            return self._route_swap_crossover(parent1, parent2)

    def _ox_crossover(self, parent1: List[List[int]],
                      parent2: List[List[int]]) -> List[List[int]]:
        """
        顺序交叉 (Order Crossover, OX)：
        1. 随机选一架 UAV u
        2. 对 UAV u 的 POI 序列做 OX（从 parent2 取一段嵌入 parent1）
        3. 修复 POI 唯一性
        4. 重新插入 BS 动作

        OX 细节：
        - 在 route_a (来自 parent1) 上选两个切点 cx1, cx2
        - 从 route_b (来自 parent2) 提取 [cx1:cx2+1] 段
        - 从 route_a 中移除该段包含的 POI
        - 将段插入到 cx1 位置
        """
        pois1 = self._extract_pois(parent1)
        pois2 = self._extract_pois(parent2)

        u = self.rng.randint(self.M)
        route_a = pois1[u]
        route_b = pois2[u]

        if len(route_a) < 2 or len(route_b) < 2:
            return copy.deepcopy(parent1)

        size = len(route_a)
        cx1, cx2 = sorted(self.rng.choice(size, size=2, replace=False))

        seg_end = min(cx2 + 1, len(route_b))
        segment = route_b[cx1:seg_end] if cx1 < len(route_b) else []
        segment_set = set(segment)

        remaining = [p for p in route_a if p not in segment_set]
        child_u = remaining[:cx1] + segment + remaining[cx1:]

        child_pois = copy.deepcopy(pois1)
        child_pois[u] = child_u

        child_pois = self._repair_poi_coverage(child_pois)
        return self._insert_bs_visits(child_pois)

    def _route_swap_crossover(self, parent1: List[List[int]],
                              parent2: List[List[int]]) -> List[List[int]]:
        """
        路径交换交叉：
        1. 随机选一架 UAV u
        2. child 的 UAV u 使用 parent2 的路径
        3. child 的其他 UAV 使用 parent1 的路径
        4. 修复 POI 唯一性
        5. 重新插入 BS
        """
        pois1 = self._extract_pois(parent1)
        pois2 = self._extract_pois(parent2)

        u = self.rng.randint(self.M)
        child_pois = copy.deepcopy(pois1)
        child_pois[u] = copy.deepcopy(pois2[u])

        child_pois = self._repair_poi_coverage(child_pois)
        return self._insert_bs_visits(child_pois)

    # ----------------------------------------------------------------
    #  变异算子
    # ----------------------------------------------------------------
    def _mutate(self, chromosome: List[List[int]]) -> List[List[int]]:
        """
        变异算子：以 mutation_rate 概率对染色体施加一种随机变异。

        变异类型：
        0 -- 段内交换：同一 UAV 路径内交换两个 POI 的位置
        1 -- 跨 UAV 迁移：将一个 POI 从一架 UAV 移到另一架
        2 -- 2-opt 段反转：反转路径中的一段 POI 顺序
        3 -- 段内打乱：随机打乱路径中一小段的顺序
        """
        if self.rng.random() > self.mutation_rate:
            return chromosome

        pois = self._extract_pois(chromosome)

        mut_type = self.rng.randint(4)
        if mut_type == 0:
            pois = self._mutate_swap_within(pois)
        elif mut_type == 1:
            pois = self._mutate_migrate(pois)
        elif mut_type == 2:
            pois = self._mutate_2opt(pois)
        else:
            pois = self._mutate_shuffle_segment(pois)

        return self._insert_bs_visits(pois)

    def _mutate_swap_within(self, pois: List[List[int]]) -> List[List[int]]:
        """段内交换：随机选一架有 >= 2 个 POI 的 UAV，交换两个 POI 位置。"""
        candidates = [u for u in range(self.M) if len(pois[u]) >= 2]
        if not candidates:
            return pois
        u = self.rng.choice(candidates)
        n = len(pois[u])
        i, j = self.rng.choice(n, size=2, replace=False)
        pois[u][i], pois[u][j] = pois[u][j], pois[u][i]
        return pois

    def _mutate_migrate(self, pois: List[List[int]]) -> List[List[int]]:
        """跨 UAV 迁移：随机从一架 UAV 移动一个 POI 到另一架。"""
        donors = [u for u in range(self.M) if len(pois[u]) >= 1]
        if not donors or self.M < 2:
            return pois
        src = self.rng.choice(donors)
        other_uavs = [u for u in range(self.M) if u != src]
        if not other_uavs:
            return pois
        dst = self.rng.choice(other_uavs)

        idx = self.rng.randint(len(pois[src]))
        poi = pois[src].pop(idx)
        insert_pos = self.rng.randint(len(pois[dst]) + 1)
        pois[dst].insert(insert_pos, poi)
        return pois

    def _mutate_2opt(self, pois: List[List[int]]) -> List[List[int]]:
        """2-opt 段反转：随机选一架有 >= 3 个 POI 的 UAV，反转一段。"""
        candidates = [u for u in range(self.M) if len(pois[u]) >= 3]
        if not candidates:
            return pois
        u = self.rng.choice(candidates)
        n = len(pois[u])
        i = self.rng.randint(0, n - 1)
        j = self.rng.randint(i + 1, n)
        pois[u][i:j + 1] = pois[u][i:j + 1][::-1]
        return pois

    def _mutate_shuffle_segment(self, pois: List[List[int]]) -> List[List[int]]:
        """段内打乱：随机选一架有 >= 3 个 POI 的 UAV，打乱一小段。"""
        candidates = [u for u in range(self.M) if len(pois[u]) >= 3]
        if not candidates:
            return pois
        u = self.rng.choice(candidates)
        n = len(pois[u])
        seg_len = min(self.rng.randint(2, max(3, n // 2 + 1)), n)
        start = self.rng.randint(0, n - seg_len + 1)
        segment = pois[u][start:start + seg_len]
        self.rng.shuffle(segment)
        pois[u][start:start + seg_len] = segment
        return pois

    # ----------------------------------------------------------------
    #  进化主循环
    # ----------------------------------------------------------------
    def plan(self, env=None) -> List[List[int]]:
        """
        运行遗传算法，返回最佳染色体（多 UAV 路径规划）。

        Parameters
        ----------
        env : MultiDroneAoIEnv, optional
            如果提供，则用该环境重建模拟器（用于新 episode 开始时）

        Returns
        -------
        best_chromosome : list of list of int
            最佳路径规划。best_chromosome[u] 是 UAV u 的动作序列。
        """
        if env is not None:
            self.simulator = LightweightSimulator(env)
            self.sensor_pos = np.array(env.sensor_pos, dtype=np.float64)
            self.base_pos = np.array(env.base_pos, dtype=np.float64)

        start_time = time.time()

        # ---- 初始化种群 ----
        population = self._init_population()
        fitnesses = [self._fitness(chrom) for chrom in population]

        # 全局最佳
        best_idx = int(np.argmax(fitnesses))
        best_fitness = fitnesses[best_idx]
        best_chromosome = copy.deepcopy(population[best_idx])

        elite_count = 2  # 精英保留数量

        # ---- 进化循环 ----
        for gen in range(self.generations):
            new_population = []

            # 精英保留
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(min(elite_count, len(population))):
                new_population.append(copy.deepcopy(population[sorted_indices[i]]))

            # 生成新个体
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses, tournament_size=3)
                p2 = self._tournament_select(population, fitnesses, tournament_size=3)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            fitnesses = [self._fitness(chrom) for chrom in population]

            # 更新全局最佳
            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chromosome = copy.deepcopy(population[gen_best_idx])

            # 进度日志
            if (gen + 1) % 20 == 0 or gen == 0:
                elapsed = time.time() - start_time
                print(f"[GA] 第 {gen + 1}/{self.generations} 代 | "
                      f"最佳适应度: {best_fitness:.4f} | "
                      f"当代最佳: {fitnesses[gen_best_idx]:.4f} | "
                      f"当代均值: {np.mean(fitnesses):.4f} | "
                      f"用时: {elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"[GA] 进化完成! 总用时 {total_time:.1f}s | "
              f"最终最佳适应度: {best_fitness:.4f}")

        return best_chromosome


# ============================================================================
#  GA 策略类
# ============================================================================
class GAPolicy:
    """
    基于遗传算法规划的策略类。

    工作模式：
    - setup() 阶段：调用 planner.plan() 做 GA 离线规划（或使用已有染色体）
    - choose_action() 阶段：按规划序列输出动作
    - 回退机制：规划用尽或动作被 mask 时，切换到贪心策略

    Parameters
    ----------
    env : MultiDroneAoIEnv
        环境实例
    planner_or_chromosome : GeneticAlgorithmPlanner or list or None
        - GeneticAlgorithmPlanner 实例 -> setup() 时调用 plan()
        - list (染色体) -> 直接使用
        - None -> 自动创建默认 planner (参数由 **planner_kwargs 指定)
    """

    def __init__(self, env, planner_or_chromosome=None, **planner_kwargs):
        self.env = env
        self.M = env.M
        self.K = env.K
        self.N = env.N

        if isinstance(planner_or_chromosome, GeneticAlgorithmPlanner):
            self.planner = planner_or_chromosome
            self.routes = None
        elif isinstance(planner_or_chromosome, list):
            self.planner = None
            self.routes = planner_or_chromosome
        else:
            self.planner = GeneticAlgorithmPlanner(env, **planner_kwargs)
            self.routes = None

        self.route_ptrs: List[int] = [0] * self.M

    def setup(self, env):
        """
        为当前 episode 进行规划（或重置执行指针）。

        - 如果有 planner -> 调用 planner.plan(env)
        - 如果只有静态 chromosome -> 重置指针即可

        Parameters
        ----------
        env : MultiDroneAoIEnv
            当前 episode 的环境实例（已 reset）
        """
        self.env = env
        self.M = env.M
        self.K = env.K
        self.N = env.N

        if self.planner is not None:
            print("[GAPolicy] 正在运行遗传算法规划...")
            self.routes = self.planner.plan(env)
            print("[GAPolicy] 规划完成。")
        elif self.routes is None:
            raise ValueError("GAPolicy: 必须提供 planner 或 chromosome!")

        self.route_ptrs = [0] * self.M

    def choose_action(self, env, uav_id: int,
                      target_mask: np.ndarray) -> int:
        """
        为指定 UAV 选择下一个动作。

        优先级：
        1. 按规划路径执行（跳过被 mask 的动作）
        2. 路径用尽或全部被 mask -> 回退到贪心策略

        Parameters
        ----------
        env : MultiDroneAoIEnv
            当前环境状态
        uav_id : int
            无人机编号 (0..M-1)
        target_mask : np.ndarray
            动作掩码 (K+N,)，1 表示可选

        Returns
        -------
        action : int
            选中的动作索引 (0..K+N-1)
        """
        # 尝试从规划路径中获取动作
        if self.routes is not None and uav_id < len(self.routes):
            route = self.routes[uav_id]
            ptr = self.route_ptrs[uav_id]

            while ptr < len(route):
                action = route[ptr]
                if 0 <= action < len(target_mask) and target_mask[action] == 1:
                    self.route_ptrs[uav_id] = ptr + 1
                    return action
                ptr += 1

            self.route_ptrs[uav_id] = ptr

        # ---- 回退到贪心策略 ----
        return self._greedy_fallback(env, uav_id, target_mask)

    def _greedy_fallback(self, env, uav_id: int,
                         target_mask: np.ndarray) -> int:
        """
        贪心回退策略：
        - 有可选 POI 时：选 (权重 * AoI / 距离) 最大的 POI
        - 无可选 POI 时：选最近的可达 BS
        - 都不行：返回第一个 BS 作为保底

        Parameters
        ----------
        env : MultiDroneAoIEnv
        uav_id : int
        target_mask : np.ndarray

        Returns
        -------
        action : int
        """
        K = self.K
        N = self.N
        drone_pos = env.drone_position_now[uav_id]

        valid_pois = [a for a in range(K) if target_mask[a] == 1]
        valid_bs = [a for a in range(K, K + N) if target_mask[a] == 1]

        if valid_pois:
            # 综合评分：权重 * AoI / 距离
            best_action = valid_pois[0]
            best_score = -np.inf
            for a in valid_pois:
                dist = np.linalg.norm(env.sensor_pos[a] - drone_pos)
                aoi_val = env.aoi[a] if hasattr(env, 'aoi') else 1.0
                weight = env.poi_weights[a] if hasattr(env, 'poi_weights') else 1.0
                score = weight * aoi_val / (dist + 1e-6)
                if score > best_score:
                    best_score = score
                    best_action = a
            return best_action
        elif valid_bs:
            # 选最近的 BS
            best_action = valid_bs[0]
            best_dist = np.inf
            for a in valid_bs:
                bs_idx = a - K
                dist = np.linalg.norm(env.base_pos[bs_idx] - drone_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_action = a
            return best_action
        else:
            # 保底：返回第一个 BS
            return K


# ============================================================================
#  运行单个 episode（异步执行模式）
# ============================================================================
def run_episode(env, policy, max_steps: int = 5000) -> dict:
    """
    使用异步执行模式运行一个完整 episode。

    异步执行逻辑（与环境接口规范严格一致）：
    1. 为每架空闲 UAV 调用 policy.choose_action() 选择动作
    2. 计算各 UAV 完成当前动作的时间 = drone_timing_now[u] + time_cost(u, action)
    3. 选最早完成的 UAV (actor) 执行 env.step(actor, action)
    4. 更新 target_mask 和 action_queue，循环直到全部 done 或无法继续

    Parameters
    ----------
    env : MultiDroneAoIEnv
        环境实例
    policy : GAPolicy (或任何有 setup/choose_action 接口的策略)
        策略实例
    max_steps : int
        最大执行步数（防止死循环）

    Returns
    -------
    result : dict
        {
          "total_reward": float,         # 累计奖励
          "steps": int,                  # 实际执行步数
          "final_aoi_mean": float,       # 最终各 POI 的平均 AoI
          "final_aoi_weighted": float,   # 最终加权 AoI 之和
          "final_aoi_array": ndarray,    # 最终各 POI 的 AoI 向量
        }
    """
    M = env.M

    # 重置环境
    obs = env.reset()

    # 设置策略（触发 GA 离线规划）
    policy.setup(env)

    # 初始化异步执行状态
    action_queue = [None] * M
    done_flag = [False] * M
    target_masks = [env.get_action_masks(i)["target"] for i in range(M)]

    total_reward = 0.0
    step_count = 0

    for _ in range(max_steps):
        # 1. 为空闲 UAV 选择动作
        for uav in range(M):
            if not done_flag[uav] and action_queue[uav] is None:
                action_queue[uav] = policy.choose_action(
                    env, uav, target_masks[uav])

        # 2. 计算完成时间
        times = []
        for u in range(M):
            if not done_flag[u] and action_queue[u] is not None:
                t = env.drone_timing_now[u] + env.time_cost(u, action_queue[u])
                times.append(t)
            else:
                times.append(np.inf)

        # 3. 全部无法继续则退出
        if all(t == np.inf for t in times):
            break

        # 4. 执行最早完成的 UAV
        actor = int(np.argmin(times))
        obs, reward, done, info = env.step(actor, action_queue[actor])
        total_reward += reward
        step_count += 1

        # 5. 更新状态
        target_masks[actor] = info["target"]
        action_queue[actor] = None
        if done:
            done_flag[actor] = True

        # 6. 全部结束则退出
        if all(done_flag):
            break

    # 计算最终统计
    final_aoi = np.mean(env.aoi) if hasattr(env, 'aoi') else 0.0
    weighted_aoi = np.sum(env.poi_weights * env.aoi) if hasattr(env, 'aoi') else 0.0

    return {
        "total_reward": total_reward,
        "steps": step_count,
        "final_aoi_mean": final_aoi,
        "final_aoi_weighted": weighted_aoi,
        "final_aoi_array": env.aoi.copy() if hasattr(env, 'aoi') else None,
    }


# ============================================================================
#  多 episode 评估
# ============================================================================
def evaluate(env, n_episodes: int = 5, pop_size: int = 50,
             generations: int = 100, mutation_rate: float = 0.3,
             buffer_threshold: int = None, seed: int = 42,
             max_steps: int = 5000) -> dict:
    """
    在多个 episode 上评估遗传算法策略。

    每个 episode 流程：env.reset() -> GA 规划 -> 异步执行 -> 记录结果

    Parameters
    ----------
    env : MultiDroneAoIEnv
        环境实例
    n_episodes : int
        评估 episode 数
    pop_size : int
        GA 种群大小
    generations : int
        GA 进化代数
    mutation_rate : float
        变异率
    buffer_threshold : int or None
        缓冲区阈值
    seed : int
        随机种子
    max_steps : int
        每个 episode 的最大步数

    Returns
    -------
    stats : dict
        包含所有 episode 的奖励、AoI 等统计数据
    """
    planner = GeneticAlgorithmPlanner(
        env,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        buffer_threshold=buffer_threshold,
        seed=seed,
    )
    policy = GAPolicy(env, planner)

    rewards = []
    aoi_means = []
    aoi_weighteds = []
    step_counts = []

    for ep in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"  Episode {ep + 1}/{n_episodes}")
        print(f"{'='*60}")

        ep_start = time.time()
        result = run_episode(env, policy, max_steps=max_steps)
        ep_time = time.time() - ep_start

        rewards.append(result["total_reward"])
        aoi_means.append(result["final_aoi_mean"])
        aoi_weighteds.append(result["final_aoi_weighted"])
        step_counts.append(result["steps"])

        print(f"  总奖励: {result['total_reward']:.4f}")
        print(f"  执行步数: {result['steps']}")
        print(f"  最终平均 AoI: {result['final_aoi_mean']:.4f}")
        print(f"  最终加权 AoI: {result['final_aoi_weighted']:.4f}")
        print(f"  Episode 用时: {ep_time:.1f}s")

        # 每个 episode 使用不同的随机种子
        planner.seed = seed + ep + 1
        planner.rng = np.random.RandomState(planner.seed)

    # ---- 汇总 ----
    stats = {
        "n_episodes": n_episodes,
        "rewards": rewards,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "aoi_mean_list": aoi_means,
        "aoi_mean_avg": float(np.mean(aoi_means)),
        "aoi_weighted_list": aoi_weighteds,
        "aoi_weighted_avg": float(np.mean(aoi_weighteds)),
        "steps_list": step_counts,
        "steps_avg": float(np.mean(step_counts)),
    }

    print(f"\n{'='*60}")
    print(f"  评估汇总 ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  平均奖励: {stats['reward_mean']:.4f} +/- {stats['reward_std']:.4f}")
    print(f"  平均 AoI (均值): {stats['aoi_mean_avg']:.4f}")
    print(f"  平均 AoI (加权): {stats['aoi_weighted_avg']:.4f}")
    print(f"  平均步数: {stats['steps_avg']:.1f}")

    return stats


# ============================================================================
#  主函数入口
# ============================================================================
if __name__ == "__main__":
    """
    用法示例
    --------

    1. 基本用法 -- 创建环境，运行 GA 规划并执行一个 episode：

        from multi_drone_aoi_env import MultiDroneAoIEnv
        env = MultiDroneAoIEnv(config)
        planner = GeneticAlgorithmPlanner(env, pop_size=50, generations=100)
        policy = GAPolicy(env, planner)
        result = run_episode(env, policy)
        print(f"总奖励: {result['total_reward']}")

    2. 多 episode 评估：

        stats = evaluate(env, n_episodes=5, pop_size=50, generations=100)

    3. 使用已有染色体直接执行（跳过 GA 进化）：

        chromosome = [[3, 7, 1, K, 5, 9, K+0], [2, 4, K+1, 6, 8, K+0]]
        policy = GAPolicy(env, chromosome)
        result = run_episode(env, policy)

    4. 自定义参数创建策略（无需手动创建 planner）：

        policy = GAPolicy(env, pop_size=30, generations=50, seed=123)
        result = run_episode(env, policy)
    """

    print("=" * 70)
    print("  遗传算法 (GA) 基线 -- 多无人机 AoI 最小化路径规划")
    print("=" * 70)
    print()
    print("本模块提供以下核心组件：")
    print()
    print("  1. LightweightSimulator    -- 轻量级适应度模拟器（事件驱动）")
    print("  2. GeneticAlgorithmPlanner  -- 遗传算法规划器")
    print("     - 初始化：启发式种子（最近邻 + AoI 贪心）+ 随机个体")
    print("     - 选择：锦标赛选择（size=3）")
    print("     - 交叉：OX 顺序交叉 / 路径交换")
    print("     - 变异：段内交换 / 跨 UAV 迁移 / 2-opt / 段内打乱")
    print("     - 精英保留：top-2 直接进入下一代")
    print("  3. GAPolicy                -- 基于规划的策略（带贪心回退）")
    print("  4. run_episode()           -- 单 episode 异步执行")
    print("  5. evaluate()              -- 多 episode 评估")
    print()

    # ---- 尝试导入环境并运行 ----
    try:
        try:
            from env import MultiDroneAoIEnv
        except ImportError:
            try:
                from multi_drone_aoi_env import MultiDroneAoIEnv
            except ImportError:
                from aoi_env import MultiDroneAoIEnv

        print("[INFO] 成功导入环境，开始运行 GA 基线...")
        print()

        env = MultiDroneAoIEnv()
        obs = env.reset()

        print(f"环境参数：")
        print(f"  无人机数量 M = {env.M}")
        print(f"  基站数量   N = {env.N}")
        print(f"  POI 数量   K = {env.K}")
        print(f"  时间上限   T = {env.T}")
        print(f"  地图尺寸     = {env.map_size}")
        print()

        # 创建 GA 规划器
        planner = GeneticAlgorithmPlanner(
            env,
            pop_size=50,       # 种群大小
            generations=100,   # 进化代数
            mutation_rate=0.3, # 变异率
            seed=42,           # 随机种子
        )

        # 创建策略
        policy = GAPolicy(env, planner)

        # 运行单个 episode
        print("[INFO] 运行单个 episode ...")
        result = run_episode(env, policy, max_steps=5000)

        print(f"\n{'='*60}")
        print(f"  单 episode 结果")
        print(f"{'='*60}")
        print(f"  总奖励:     {result['total_reward']:.4f}")
        print(f"  执行步数:   {result['steps']}")
        print(f"  平均 AoI:   {result['final_aoi_mean']:.4f}")
        print(f"  加权 AoI:   {result['final_aoi_weighted']:.4f}")
        print()

        # 多 episode 评估（可选）
        # print("[INFO] 开始多 episode 评估 ...")
        # stats = evaluate(env, n_episodes=3, pop_size=30, generations=50, seed=42)

    except ImportError:
        print("[INFO] 未找到环境模块，仅展示模块结构。")
        print("       请确保环境文件在 Python 路径中，然后运行：")
        print()
        print("       python genetic_algorithm.py")
        print()
        print("  或在代码中导入：")
        print()
        print("       from genetic_algorithm import (")
        print("           GeneticAlgorithmPlanner, GAPolicy,")
        print("           run_episode, evaluate")
        print("       )")
