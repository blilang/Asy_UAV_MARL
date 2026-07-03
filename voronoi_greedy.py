"""
Voronoi Partition + Weighted AoI Greedy 基线算法
================================================

用于多无人机 AoI（Age of Information）最小化环境的贪心基线策略。

算法核心思想：
1. **Voronoi 分区**：利用加权 K-means 聚类将所有 POI 划分为 M 个簇，
   每个簇分配给距离簇质心最近的 UAV，从而实现任务分工。
2. **加权 AoI 贪心决策**：每架 UAV 在自己负责的 POI 集合内，
   选择「权重 × 本地AoI / 距离」得分最高的 POI 进行访问。
3. **缓冲区卸载策略**：当缓冲区达到阈值或已完成所有分配任务时，
   飞往最近的基站卸载数据。
4. **速度选择**：始终使用最大速度以最小化移动时间。

依赖：numpy（必须），scipy（可选，用于空间计算加速）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
#  工具函数
# ============================================================================

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个二维点之间的欧氏距离。"""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _euclidean_distance_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    计算两组点之间的欧氏距离矩阵。

    参数：
        points_a: shape (n, 2)
        points_b: shape (m, 2)
    返回：
        dist_matrix: shape (n, m)，dist_matrix[i, j] = ||a_i - b_j||
    """
    # 利用广播机制高效计算距离矩阵
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]  # (n, m, 2)
    return np.sqrt(np.sum(diff ** 2, axis=-1))  # (n, m)


def _weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加权 K-means 聚类算法。

    与标准 K-means 的区别：在更新质心时，每个点的贡献按其权重加权。
    这确保高权重（高重要性）的 POI 对簇质心有更大影响，
    使得 UAV 更倾向于靠近重要 POI。

    参数：
        points:     shape (K, 2)，POI 坐标
        weights:    shape (K,)，POI 权重（重要性）
        n_clusters: 簇数量（等于 UAV 数量 M）
        max_iter:   最大迭代次数
        seed:       随机种子，确保可复现

    返回：
        labels:    shape (K,)，每个 POI 的簇编号 (0 ~ n_clusters-1)
        centroids: shape (n_clusters, 2)，每个簇的加权质心
    """
    rng = np.random.RandomState(seed)
    K = points.shape[0]

    # --- 初始化：K-means++ 风格选取初始质心 ---
    # 优先选择距离已有质心较远的点，避免初始质心重叠
    centroids = np.empty((n_clusters, 2), dtype=np.float64)
    # 第一个质心：按权重概率采样
    prob = weights / weights.sum()
    first_idx = rng.choice(K, p=prob)
    centroids[0] = points[first_idx]

    for c in range(1, n_clusters):
        # 计算每个点到已有质心的最小距离
        dist_to_existing = _euclidean_distance_matrix(
            points, centroids[:c]
        )  # (K, c)
        min_dist = dist_to_existing.min(axis=1)  # (K,)
        # 距离越远、权重越大的点被选中概率越高
        prob = min_dist * weights
        prob_sum = prob.sum()
        if prob_sum < 1e-12:
            # 所有点都与已有质心重合，随机选一个
            centroids[c] = points[rng.randint(K)]
        else:
            prob /= prob_sum
            centroids[c] = points[rng.choice(K, p=prob)]

    # --- 迭代优化 ---
    labels = np.zeros(K, dtype=np.int32)
    for iteration in range(max_iter):
        # E-step: 将每个 POI 分配到最近的质心
        dist_matrix = _euclidean_distance_matrix(points, centroids)  # (K, n_clusters)
        new_labels = np.argmin(dist_matrix, axis=1)

        # 检查收敛（标签不再变化）
        if np.array_equal(new_labels, labels) and iteration > 0:
            break
        labels = new_labels

        # M-step: 按权重更新质心
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                w = weights[mask]
                w_sum = w.sum()
                if w_sum > 1e-12:
                    # 加权质心 = Σ(w_i * p_i) / Σ(w_i)
                    centroids[c] = (points[mask] * w[:, np.newaxis]).sum(axis=0) / w_sum
                else:
                    centroids[c] = points[mask].mean(axis=0)
            # 如果该簇为空，质心保持不变（惰性处理）

    return labels, centroids


# ============================================================================
#  Voronoi Greedy 策略类
# ============================================================================

class VoronoiGreedyPolicy:
    """
    Voronoi 分区 + 加权 AoI 贪心策略。

    工作流程：
    ┌─────────────────────────────────────────────────┐
    │  初始化阶段 (setup)                              │
    │  1. 对所有 POI 执行加权 K-means 聚类 → M 个簇    │
    │  2. 将每个簇分配给最近的 UAV → 任务分区           │
    └─────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │  决策阶段 (choose_action)，每次一个 UAV          │
    │  ├─ 缓冲区满/任务完成 → 飞往最近基站卸载         │
    │  ├─ 有可访问的分配 POI → 选 AoI加权得分最高的    │
    │  └─ 无可访问的分配 POI → 回退到全局最优 POI      │
    └─────────────────────────────────────────────────┘

    属性：
        partition:       dict, UAV_id → list of assigned POI indices
        poi_to_uav:      ndarray (K,), POI_index → assigned UAV_id
        buffer_threshold: int, 缓冲区卸载阈值
        epsilon:         float, 距离计算中的防零除小量
    """

    def __init__(self, env, buffer_threshold: Optional[int] = None, seed: int = 42):
        """
        初始化策略。

        参数：
            env:              MultiDroneAoIEnv 环境实例
            buffer_threshold: 缓冲区卸载阈值。None 时使用 env.bs_random_factor（默认 5）
            seed:             K-means 随机种子
        """
        self.M = env.M  # UAV 数量
        self.N = env.N  # 基站数量
        self.K = env.K  # POI 数量
        self.seed = seed
        self.epsilon = 1e-6  # 防零除

        # 缓冲区阈值：优先从环境获取，否则使用默认值 5
        if buffer_threshold is not None:
            self.buffer_threshold = buffer_threshold
        elif hasattr(env, 'bs_random_factor'):
            self.buffer_threshold = int(env.bs_random_factor)
        else:
            self.buffer_threshold = 5

        # 分区信息（在 setup 中填充）
        self.partition: Dict[int, List[int]] = {}    # UAV_id → [POI indices]
        self.poi_to_uav: np.ndarray = np.zeros(self.K, dtype=np.int32)  # POI → UAV
        self.centroids: np.ndarray = np.zeros((self.M, 2))  # 簇质心

        # 标记是否已完成分区设置
        self._is_setup = False

    def setup(self, env) -> None:
        """
        执行 Voronoi 分区。应在 env.reset() 之后调用。

        分区策略：
        - 使用加权 K-means 将 K 个 POI 聚为 M 个簇
        - 每个簇分配给距离其质心最近的 UAV
        - 如果多个簇竞争同一 UAV，使用匈牙利算法风格的贪心分配

        注意：所有 UAV 初始位置通常在原点 (0,0)，因此不能简单地用
        初始位置做 Voronoi 划分（所有 UAV 位置相同）。K-means 的质心
        天然地分散在 POI 分布区域，可以合理地进行分配。
        """
        sensor_pos = np.array(env.sensor_pos, dtype=np.float64)   # (K, 2)
        poi_weights = np.array(env.poi_weights, dtype=np.float64)  # (K,)
        drone_pos = np.array(env.drone_position_now, dtype=np.float64)  # (M, 2)

        # --- Step 1: 加权 K-means 聚类 ---
        # 当 POI 数量少于 UAV 数量时的特殊处理
        actual_clusters = min(self.M, self.K)
        labels, centroids = _weighted_kmeans(
            sensor_pos, poi_weights, actual_clusters, max_iter=200, seed=self.seed
        )
        self.centroids = centroids

        # --- Step 2: 将簇分配给 UAV ---
        # 计算每个 UAV 到每个簇质心的距离
        # 使用贪心分配：按距离从小到大逐个匹配，避免冲突
        dist_uav_to_centroid = _euclidean_distance_matrix(
            drone_pos, centroids
        )  # (M, actual_clusters)

        # 初始化分配
        uav_assigned = set()       # 已分配的 UAV
        cluster_assigned = set()   # 已分配的簇
        cluster_to_uav = {}        # 簇 → UAV 映射

        # 构建 (距离, uav_id, cluster_id) 列表并排序
        assignments = []
        for u in range(self.M):
            for c in range(actual_clusters):
                assignments.append((dist_uav_to_centroid[u, c], u, c))
        assignments.sort(key=lambda x: x[0])

        # 贪心匹配：距离最短的优先
        for _, u, c in assignments:
            if u in uav_assigned or c in cluster_assigned:
                continue
            cluster_to_uav[c] = u
            uav_assigned.add(u)
            cluster_assigned.add(c)
            if len(cluster_to_uav) == actual_clusters:
                break

        # 处理未被分配到簇的 UAV（当 M > K 时可能出现）
        # 这些 UAV 的分区为空，它们将在 choose_action 中回退到全局贪心
        unassigned_uavs = [u for u in range(self.M) if u not in uav_assigned]

        # --- Step 3: 构建分区映射 ---
        self.partition = {u: [] for u in range(self.M)}
        self.poi_to_uav = np.full(self.K, -1, dtype=np.int32)

        for c, u in cluster_to_uav.items():
            poi_indices = np.where(labels == c)[0].tolist()
            self.partition[u] = poi_indices
            for k in poi_indices:
                self.poi_to_uav[k] = u

        # 对于未分配的 UAV，如果有未覆盖的 POI（理论上不应出现），
        # 将它们均匀分配
        uncovered_pois = [k for k in range(self.K) if self.poi_to_uav[k] == -1]
        if uncovered_pois and unassigned_uavs:
            # 将未覆盖 POI 平均分给未分配的 UAV
            for i, k in enumerate(uncovered_pois):
                u = unassigned_uavs[i % len(unassigned_uavs)]
                self.partition[u].append(k)
                self.poi_to_uav[k] = u

        self._is_setup = True

    def choose_action(self, env, uav_id: int, target_mask: np.ndarray) -> int:
        """
        为指定 UAV 选择下一步动作（目标 POI 或基站）。

        决策逻辑（按优先级排列）：
        ┌───────────────────────────────────────────────────────┐
        │ 1. 缓冲区达到阈值 → 飞往最近可达基站                   │
        │ 2. 分配区域内所有 POI 都已访问 → 飞往最近基站           │
        │ 3. 分配区域内有可访问 POI → 选加权得分最高的            │
        │ 4. 分配区域无可访问但全局有 → 回退全局贪心              │
        │ 5. 无 POI 可访问但缓冲区非空 → 飞往最近基站             │
        │ 6. 完全无可用动作 → 返回第一个合法目标                  │
        └───────────────────────────────────────────────────────┘

        参数：
            env:         环境实例
            uav_id:      UAV 编号
            target_mask: shape (K+N,)，1 表示可选，0 表示被掩码

        返回：
            action: int，目标动作编号 (0..K-1 为 POI, K..K+N-1 为 BS)
        """
        if not self._is_setup:
            raise RuntimeError("必须先调用 setup() 完成 Voronoi 分区初始化")

        K = self.K
        N = self.N
        drone_pos = np.array(env.drone_position_now[uav_id], dtype=np.float64)  # (2,)
        sensor_pos = np.array(env.sensor_pos, dtype=np.float64)  # (K, 2)
        base_pos = np.array(env.base_pos, dtype=np.float64)      # (N, 2)
        poi_weights = np.array(env.poi_weights, dtype=np.float64)  # (K,)
        local_aoi = np.array(env.drone_local_aoi[uav_id], dtype=np.float64)  # (K,)
        buffer = env.drone_buffer[uav_id]
        buffer_size = len(buffer)

        # --- 辅助函数：选择最近的可达基站 ---
        def _nearest_bs_action() -> Optional[int]:
            """返回最近可达基站的动作编号，若无可达基站返回 None。"""
            best_action = None
            best_dist = np.inf
            for j in range(N):
                action_idx = K + j
                if target_mask[action_idx] > 0.5:
                    d = _euclidean_distance(drone_pos, base_pos[j])
                    if d < best_dist:
                        best_dist = d
                        best_action = action_idx
            return best_action

        # --- 辅助函数：计算 POI 的贪心得分 ---
        def _poi_score(k: int) -> float:
            """
            计算 POI k 的访问优先级得分。

            得分 = poi_weight[k] * local_aoi[uav_id, k] / (distance + ε)

            直觉：
            - 权重越大的 POI 越应优先访问（对全局 AoI 影响大）
            - 本地 AoI 越高的 POI 越紧急（信息越过时）
            - 距离越近的 POI 性价比越高（移动时间短）
            """
            d = _euclidean_distance(drone_pos, sensor_pos[k])
            return poi_weights[k] * local_aoi[k] / (d + self.epsilon)

        # --- 辅助函数：从候选 POI 集合中选得分最高的 ---
        def _best_poi_from(candidates: List[int]) -> Optional[int]:
            """从候选列表中选出得分最高的合法 POI。"""
            best_k = None
            best_score = -np.inf
            for k in candidates:
                if target_mask[k] > 0.5:  # 必须是合法目标（未在缓冲区中）
                    s = _poi_score(k)
                    if s > best_score:
                        best_score = s
                        best_k = k
            return best_k

        # ============================================================
        #  决策主逻辑
        # ============================================================

        # 获取该 UAV 负责的 POI 列表
        assigned_pois = self.partition.get(uav_id, [])

        # 检查分配区域内有多少 POI 是可访问的（未被掩码）
        valid_assigned = [k for k in assigned_pois if target_mask[k] > 0.5]

        # --- 条件 1: 缓冲区达到阈值 → 卸载 ---
        if buffer_size >= self.buffer_threshold:
            bs_action = _nearest_bs_action()
            if bs_action is not None:
                return bs_action
            # 如果所有基站都被掩码（不太可能，但防御性处理），继续选 POI

        # --- 条件 2: 分配区域内所有 POI 都已访问 → 卸载 ---
        if len(assigned_pois) > 0 and len(valid_assigned) == 0 and buffer_size > 0:
            bs_action = _nearest_bs_action()
            if bs_action is not None:
                return bs_action

        # --- 条件 3: 在分配区域内贪心选择 POI ---
        if valid_assigned:
            best_k = _best_poi_from(valid_assigned)
            if best_k is not None:
                return best_k

        # --- 条件 4: 回退 — 分配区域无可访问 POI，尝试全局贪心 ---
        # 这处理了两种情况：
        #   a) 该 UAV 没有被分配任何 POI（M > K 时）
        #   b) 所有分配的 POI 都在缓冲区中但缓冲区未满
        all_valid_pois = [k for k in range(K) if target_mask[k] > 0.5]
        if all_valid_pois:
            best_k = _best_poi_from(all_valid_pois)
            if best_k is not None:
                return best_k

        # --- 条件 5: 无 POI 可访问，缓冲区非空 → 飞往基站 ---
        if buffer_size > 0:
            bs_action = _nearest_bs_action()
            if bs_action is not None:
                return bs_action

        # --- 条件 6: 兜底 — 选择第一个合法目标 ---
        # 理论上不应到达这里，但作为安全保障
        for idx in range(K + N):
            if target_mask[idx] > 0.5:
                return idx

        # 完全没有合法动作（环境异常），返回 0
        return 0


# ============================================================================
#  Episode 运行器
# ============================================================================

def run_episode(
    env,
    policy: VoronoiGreedyPolicy,
    max_steps: int = 5000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    使用给定策略运行一个完整的 episode。

    严格遵循异步执行模式：
    1. 为每架 UAV 维护一个动作队列槽 (action_queue)
    2. 每个时间步，选择「最早完成当前动作」的 UAV 执行
    3. 执行后更新掩码，清空该 UAV 的动作槽

    这种模式确保 UAV 按实际时间顺序依次行动，模拟真实的异步并行飞行。

    参数：
        env:       MultiDroneAoIEnv 环境实例
        policy:    VoronoiGreedyPolicy 策略实例
        max_steps: 最大步数限制（防止死循环）
        verbose:   是否打印调试信息

    返回：
        metrics: dict，包含以下字段：
            - "total_reward":    累计奖励（越大越好，通常为负数）
            - "total_steps":     总执行步数
            - "mean_aoi":        最终平均 AoI
            - "weighted_aoi":    最终加权平均 AoI
            - "max_aoi":         最终最大 AoI
            - "episode_done":    是否所有 UAV 都完成
            - "per_uav_steps":   每架 UAV 的执行步数
    """
    M = env.M
    K = env.K

    # --- 重置环境并初始化策略 ---
    obs = env.reset()
    policy.setup(env)

    # --- 异步执行的核心数据结构 ---
    action_queue = [None] * M        # 每架 UAV 的待执行动作（None 表示空闲）
    done_flag = [False] * M          # 每架 UAV 是否已结束
    total_reward = 0.0               # 累计奖励
    step_count = 0                   # 总步数
    per_uav_steps = [0] * M          # 每架 UAV 的步数

    # 初始化动作掩码
    target_masks = [
        env.get_action_masks(i)["target"] for i in range(M)
    ]

    if verbose:
        print(f"[Episode Start] M={M}, K={K}, N={env.N}, T={env.T}")
        for u in range(M):
            pois = policy.partition.get(u, [])
            print(f"  UAV {u}: 负责 {len(pois)} 个 POI → {pois[:10]}{'...' if len(pois) > 10 else ''}")

    # --- 主循环 ---
    for iteration in range(max_steps):
        # ========================================
        # Phase 1: 为空闲 UAV 填充动作
        # ========================================
        candidate_times = [np.inf] * M

        for uav in range(M):
            if done_flag[uav]:
                # 该 UAV 已结束，不参与调度
                continue

            if action_queue[uav] is None:
                # 该 UAV 空闲，需要决策
                action = policy.choose_action(env, uav, target_masks[uav])
                action_queue[uav] = action

            # 计算该 UAV 完成当前动作的预计时刻
            # = 当前本地时钟 + 执行该动作所需时间
            candidate_times[uav] = (
                env.drone_timing_now[uav] + env.time_cost(uav, action_queue[uav])
            )

        # ========================================
        # Phase 2: 选择最早完成的 UAV
        # ========================================
        if all(t == np.inf for t in candidate_times):
            # 所有 UAV 都已结束或无法行动
            if verbose:
                print(f"[Step {step_count}] 所有 UAV 空闲或已结束，提前退出")
            break

        actor = int(np.argmin(candidate_times))

        # ========================================
        # Phase 3: 执行动作
        # ========================================
        action = action_queue[actor]
        obs, reward, done, info = env.step(actor, action)

        # 累计统计
        total_reward += reward
        step_count += 1
        per_uav_steps[actor] += 1

        if verbose and step_count % 100 == 0:
            aoi_now = np.array(env.aoi)
            print(
                f"  [Step {step_count}] UAV {actor} → action {action}, "
                f"reward={reward:.4f}, mean_aoi={aoi_now.mean():.2f}, "
                f"time={env.drone_timing_now[actor]:.2f}"
            )

        # 更新该 UAV 的动作掩码
        target_masks[actor] = info["target"]

        # 清空该 UAV 的动作槽，下一轮重新决策
        action_queue[actor] = None

        # 检查该 UAV 是否结束（能量耗尽或时间到期）
        if done:
            done_flag[actor] = True

        # 所有 UAV 都结束则退出
        if all(done_flag):
            if verbose:
                print(f"[Step {step_count}] 所有 UAV 结束")
            break

    # --- 计算最终指标 ---
    final_aoi = np.array(env.aoi, dtype=np.float64)        # (K,)
    poi_weights = np.array(env.poi_weights, dtype=np.float64)  # (K,)

    mean_aoi = float(final_aoi.mean())
    max_aoi = float(final_aoi.max())

    # 加权平均 AoI = Σ(w_k * aoi_k) / Σ(w_k)
    weight_sum = poi_weights.sum()
    if weight_sum > 1e-12:
        weighted_aoi = float((poi_weights * final_aoi).sum() / weight_sum)
    else:
        weighted_aoi = mean_aoi

    metrics = {
        "total_reward": total_reward,
        "total_steps": step_count,
        "mean_aoi": mean_aoi,
        "weighted_aoi": weighted_aoi,
        "max_aoi": max_aoi,
        "episode_done": all(done_flag),
        "per_uav_steps": per_uav_steps,
    }

    if verbose:
        print(f"\n[Episode End] 总步数={step_count}, 总奖励={total_reward:.4f}")
        print(f"  平均 AoI={mean_aoi:.4f}, 加权 AoI={weighted_aoi:.4f}, 最大 AoI={max_aoi:.4f}")
        print(f"  各 UAV 步数: {per_uav_steps}")

    return metrics


# ============================================================================
#  批量评估工具
# ============================================================================

def evaluate(
    env,
    n_episodes: int = 10,
    buffer_threshold: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    多轮评估 Voronoi Greedy 策略的性能。

    参数：
        env:              环境实例
        n_episodes:       评估轮数
        buffer_threshold: 缓冲区阈值（None 使用默认值）
        seed:             随机种子
        verbose:          是否打印每轮信息

    返回：
        summary: dict，包含所有指标的均值和标准差
    """
    all_metrics = []

    for ep in range(n_episodes):
        policy = VoronoiGreedyPolicy(env, buffer_threshold=buffer_threshold, seed=seed + ep)
        metrics = run_episode(env, policy, verbose=verbose)
        all_metrics.append(metrics)

        if verbose:
            print(f"--- Episode {ep + 1}/{n_episodes} ---")
            print(f"  reward={metrics['total_reward']:.4f}, "
                  f"weighted_aoi={metrics['weighted_aoi']:.4f}")

    # 汇总统计
    keys = ["total_reward", "total_steps", "mean_aoi", "weighted_aoi", "max_aoi"]
    summary = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))

    summary["n_episodes"] = n_episodes
    summary["all_metrics"] = all_metrics

    return summary


# ============================================================================
#  主程序入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Voronoi Partition + Weighted AoI Greedy 基线算法")
    print("=" * 70)
    print()
    print("模块组件：")
    print()
    print("  1. VoronoiGreedyPolicy 类")
    print("     - __init__(env, buffer_threshold=None, seed=42)")
    print("       初始化策略参数，准备数据结构")
    print()
    print("     - setup(env)")
    print("       执行加权 K-means 聚类，构建 Voronoi 分区")
    print("       在 env.reset() 之后调用")
    print()
    print("     - choose_action(env, uav_id, target_mask) -> int")
    print("       为指定 UAV 选择动作（POI 或基站）")
    print("       决策优先级：缓冲区满→卸载 > 分区内贪心 > 全局回退")
    print()
    print("  2. run_episode(env, policy, max_steps=5000, verbose=False) -> dict")
    print("     运行一个完整 episode，遵循异步执行模式")
    print("     返回: total_reward, mean_aoi, weighted_aoi, max_aoi 等")
    print()
    print("  3. evaluate(env, n_episodes=10, ...) -> dict")
    print("     多轮评估，返回均值和标准差")
    print()
    print("使用示例：")
    print("  ```python")
    print("  from multi_drone_aoi_env import MultiDroneAoIEnv")
    print("  from voronoi_greedy import VoronoiGreedyPolicy, run_episode")
    print()
    print("  env = MultiDroneAoIEnv(config)")
    print("  policy = VoronoiGreedyPolicy(env, buffer_threshold=5)")
    print("  metrics = run_episode(env, policy, verbose=True)")
    print("  print(f'Weighted AoI: {metrics[\"weighted_aoi\"]:.4f}')")
    print("  ```")
    print()
    print("算法特点：")
    print("  - 加权 K-means 确保高权重 POI 集中在簇中心附近")
    print("  - 贪心得分 = weight * local_aoi / distance 兼顾紧迫性和效率")
    print("  - 始终使用最大速度，最小化移动时间开销")
    print("  - 分区失败时自动回退到全局贪心，保证鲁棒性")
