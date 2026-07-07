"""
Env.py — 多无人机异步 AoI 环境 + P2P 邻近通信

核心机制:
  1. 异步执行: step(uav_id, action) 只推进一架 UAV
  2. BS 同步: UAV 回到基站时上传 / 下载 transition tokens (延迟信息)
  3. P2P 通信 (NEW): 每次步进后，距离 < R_comm 的 UAV 对交换实时状态快照
  4. 优化目标不变: 最小化加权 AoI

通信消息 (comm_dim = 2K + 4):
  position(2) + buffer_bitmap(K) + local_aoi(K) + timing(1) + last_target(1)

观测中的通信字段 (每邻居 comm_dim + 2 维):
  availability_mask(1) + staleness(1) + comm_msg(2K+4)
  - mask=1 表示邻居在通信范围内, 0 表示不在
  - staleness = 当前 UAV 时间 - 消息生成时间, 越大越陈旧

BS 中继改进:
  UAV 卸载基站时, 将 P2P 获知的邻居 buffer 状态一并上传
  其他 UAV 下载时可获得 P2P 中继信息 (多跳传播)
"""
import gym
import numpy as np
from gym import spaces
from scipy.spatial.distance import euclidean
import os


def UAV_Energy(v):
    """旋翼无人机功耗模型 (W)"""
    P_b = 79.86
    P_i = 88.63
    V_tip = 120.0
    u_0 = 4.03
    f_0 = 0.6
    a = 1.225
    n = 0.05
    R = 0.503
    energy = (
        P_b * (1 + (3 * v * v) / (V_tip * V_tip))
        + P_i * np.sqrt(
            np.sqrt(1 + (v ** 4) / (4 * u_0 ** 4)) - v * v / (2 * u_0 * u_0)
        )
        + f_0 * a * n * R * v ** 3 / 2
    )
    return float(energy)


class MultiDroneAoIEnv(gym.Env):
    def __init__(self, M=3, N=1, K=15, T=600.0, map_size=500.0,
                 args=None, position_file=None):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.T = float(T)
        self.map_size = float(map_size)
        self.args = args

        # ---- 速度 ----
        self.speed_levels = self._parse_speed_levels()
        self.speed_action_dim = len(self.speed_levels)
        self.default_speed_idx = int(np.argmax(self.speed_levels))
        self.nominal_speed = float(np.mean(self.speed_levels))

        # ---- 能量 ----
        self.init_uav_energy_list = self._parse_init_uav_energies()
        self.init_uav_energy = float(np.max(self.init_uav_energy_list))
        self.reward_divisor = float(getattr(args, "reward_divisor", 10.0))

        # ---- 位置数据 ----
        fname = position_file or getattr(args, "position_file", None)
        if fname is None:
            fname = f"./data/poi_{self.K}_map_{int(self.map_size)}x{int(self.map_size)}.npy"
        data = np.load(fname, allow_pickle=True).item()
        self.sensor_pos = np.asarray(data["poi_positions"], dtype=np.float32)
        self.base_pos = np.asarray(data["bs_positions"], dtype=np.float32)
        self.K = int(self.sensor_pos.shape[0])
        self.N = int(self.base_pos.shape[0])

        raw_w = data.get("poi_weights", None)
        if raw_w is None:
            self.poi_weights = np.ones(self.K, dtype=np.float32)
        else:
            self.poi_weights = np.asarray(raw_w, dtype=np.float32).reshape(-1)
        self.weight_norm = max(float(np.max(self.poi_weights)), 1e-6)

        # ---- P2P 通信参数 ----
        self.R_comm = float(getattr(args, "R_comm", 150.0))
        self.comm_dim = 2 * self.K + 4  # pos(2) + buf(K) + aoi(K) + time(1) + target(1)
        # 每邻居在观测中占 comm_dim + 2 (mask + staleness + msg)
        self.comm_obs_per_neighbor = self.comm_dim + 2

        # ---- 动作空间 ----
        self.action_space = spaces.Discrete(self.K + self.N)
        self.speed_action_space = spaces.Discrete(self.speed_action_dim)
        self.target_action_dim = self.K + self.N

        # ---- 观测空间 ----
        # aoi(K) + dis(K) + rew_poi(K) + rew_bs(1) + buf(K) + time(1)
        # + hist(K*M) + last_rew(M) + buf_len(1) + energy(1) + weights(K)
        # + comm((M-1) * (comm_dim + 2))  ← mask(1) + staleness(1) + msg(comm_dim)
        obs_dim = (5 * self.K + self.K * self.M + self.M + 4
                   + (self.M - 1) * self.comm_obs_per_neighbor)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.token_dim = obs_dim + 2  # +action +reward

        # ---- Transformer 长度 ----
        self.history_horizon = int(getattr(args, "history_horizon", 10))
        self.max_other_agents = max(0, self.M - 1)
        self.encoder_token_len = max(1, (self.M - 1) * self.history_horizon)
        self.decoder_token_len = max(1, self.history_horizon + 1)
        self.critic_token_len = max(1, self.M * (self.history_horizon + 1))

        # ---- 其他参数 ----
        self.bs_random_factor = int(getattr(args, "BS_back_times", 5))
        self.fly_speed = self.nominal_speed

        self.reset()

    # ================================================================
    #  解析辅助
    # ================================================================
    def _parse_speed_levels(self):
        raw = getattr(self.args, "speed_levels", "6-20")
        if isinstance(raw, str):
            raw = raw.strip()
            if "-" in raw and "," not in raw:
                if ":" in raw:
                    range_part, step_part = raw.split(":", 1)
                    step = max(float(step_part.strip()), 1e-3)
                else:
                    range_part = raw
                    step = 1.0
                lo, hi = [v.strip() for v in range_part.split("-", 1)]
                lo_v, hi_v = float(lo), float(hi)
                if hi_v < lo_v:
                    lo_v, hi_v = hi_v, lo_v
                speeds = list(np.arange(lo_v, hi_v + 1e-9, step))
            else:
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                speeds = [float(p) for p in parts] if parts else [15.0]
        elif isinstance(raw, (list, tuple, np.ndarray)):
            speeds = [float(v) for v in raw]
        else:
            speeds = list(np.arange(6.0, 20.0 + 1e-9, 1.0))
        speeds = [max(1e-3, s) for s in speeds]
        return np.array(sorted(speeds), dtype=np.float32)

    def _parse_init_uav_energies(self):
        raw_list = getattr(self.args, "init_uav_energies", None)
        raw_scalar = float(getattr(self.args, "init_uav_energy", 2.0e5))
        if raw_list is None:
            return np.full(self.M, raw_scalar, dtype=np.float32)
        if isinstance(raw_list, (list, tuple, np.ndarray)):
            vals = [float(v) for v in raw_list]
        else:
            text = str(raw_list).strip().strip("[]()")
            text = text.replace(",", " ")
            parts = [p for p in text.split() if p]
            vals = [float(p) for p in parts] if parts else [raw_scalar]
        if len(vals) == 1:
            vals = vals * self.M
        if len(vals) != self.M:
            raise ValueError(f"init_uav_energies len {len(vals)} != M={self.M}")
        return np.asarray([max(0.0, v) for v in vals], dtype=np.float32)

    def _parse_action(self, action):
        if isinstance(action, (tuple, list, np.ndarray)):
            target_action = int(action[0])
            speed_idx = int(action[1]) if len(action) > 1 else self.default_speed_idx
        else:
            target_action = int(action)
            speed_idx = self.default_speed_idx
        speed_idx = int(np.clip(speed_idx, 0, self.speed_action_dim - 1))
        speed = float(self.speed_levels[speed_idx])
        return target_action, speed_idx, speed

    # ================================================================
    #  Token 构造
    # ================================================================
    def _compose_transition_token(self, state_vec, target_action, speed_idx, reward):
        combined = target_action * self.speed_action_dim + speed_idx
        denom = max(1, self.target_action_dim * self.speed_action_dim - 1)
        action_norm = float(combined / denom)
        return self._compose_state_prev_token(state_vec, action_norm, reward)

    def _compose_state_prev_token(self, state_vec, prev_action, prev_reward):
        token = np.concatenate([
            np.asarray(state_vec, dtype=np.float32),
            np.array([prev_action, prev_reward], dtype=np.float32)
        ])
        return token.astype(np.float32)

    def _build_prev_aligned_tokens(self, sar_tokens):
        aligned = []
        prev_action, prev_reward = -1.0, -1.0
        for sar in sar_tokens:
            state_vec = np.array(sar[:-2], dtype=np.float32)
            aligned.append(self._compose_state_prev_token(state_vec, prev_action, prev_reward))
            prev_action = float(sar[-2])
            prev_reward = float(sar[-1])
        return aligned

    def _get_last_action_reward(self, sar_tokens):
        if len(sar_tokens) == 0:
            return -1.0, -1.0
        last = sar_tokens[-1]
        return float(last[-2]), float(last[-1])

    def _pad_tokens(self, token_list, seq_len):
        seq = np.zeros((seq_len, self.token_dim), dtype=np.float32)
        pad_mask = np.ones(seq_len, dtype=np.bool_)
        if len(token_list) == 0:
            return seq, pad_mask
        tail = token_list[-seq_len:]
        start = seq_len - len(tail)
        seq[start:] = np.array(tail, dtype=np.float32)
        pad_mask[start:] = False
        return seq, pad_mask

    # ================================================================
    #  BS 同步 (延迟信息交换)
    # ================================================================
    def _sync_with_bs(self, uav_id):
        """UAV 回到基站时上传自己的 tokens + P2P 中继知识，下载其他 UAV 的信息"""
        self.bs_uploaded_tokens[uav_id] = [
            tok.copy() for tok in self.local_transition_tokens[uav_id]
        ]
        self.bs_uploaded_versions[uav_id] += 1
        self.bs_uploaded_times[uav_id] = self.drone_timing_now[uav_id]

        self.local_known_bs_timing[uav_id, uav_id] = \
            self.drone_visited_history_timing_at_BS[uav_id].astype(np.float32)
        self.local_known_last_reward[uav_id, uav_id] = \
            float(self.drone_last_reward[uav_id])

        # ---- 上传 P2P 中继知识到 BS ----
        # uav_id 将自己通过 P2P 获知的所有邻居 buffer 状态上传到 BS
        # 存储在 bs_relay_buffer 供其他 UAV 下载
        for other_id in range(self.M):
            if other_id == uav_id:
                continue
            # 取 P2P 获知和 BS 已有的中继信息的并集 (取最新)
            merged = np.maximum(self.p2p_known_buffer[uav_id, other_id],
                                self.bs_relay_buffer[uav_id, other_id])
            self.bs_relay_buffer[uav_id, other_id] = merged

        for other_id in range(self.M):
            if other_id == uav_id:
                continue
            latest_ver = self.bs_uploaded_versions[other_id]
            if latest_ver > self.downloaded_versions[uav_id, other_id]:
                self.synced_other_tokens[uav_id][other_id] = [
                    tok.copy() for tok in self.bs_uploaded_tokens[other_id]
                ]
                self.downloaded_versions[uav_id, other_id] = latest_ver
                self.downloaded_times[uav_id, other_id] = self.drone_timing_now[uav_id]
                self.local_known_bs_timing[uav_id, other_id] = \
                    self.drone_visited_history_timing_at_BS[other_id].astype(np.float32)
                self.local_known_last_reward[uav_id, other_id] = \
                    float(self.drone_last_reward[other_id])
                # 下载 other_id 上传的 P2P 中继知识
                # uav_id 现在也知道了 other_id 通过 P2P 获知的第三方的 buffer 状态
                for third_id in range(self.M):
                    if third_id == uav_id or third_id == other_id:
                        continue
                    self.bs_relay_buffer[uav_id, third_id] = np.maximum(
                        self.bs_relay_buffer[uav_id, third_id],
                        self.bs_relay_buffer[other_id, third_id]
                    )

    def _build_segmented_encoder_tokens(self, uav_id):
        tokens = np.zeros((self.encoder_token_len, self.token_dim), dtype=np.float32)
        pad = np.ones(self.encoder_token_len, dtype=np.bool_)
        segment_ids = np.zeros(self.encoder_token_len, dtype=np.int64)

        if self.max_other_agents == 0:
            return tokens, pad, segment_ids

        other_ids = [idx for idx in range(self.M) if idx != uav_id]
        for seg_idx, other_id in enumerate(other_ids, start=1):
            seg_start = (seg_idx - 1) * self.history_horizon
            seg_end = seg_start + self.history_horizon
            other_aligned = self._build_prev_aligned_tokens(
                self.synced_other_tokens[uav_id][other_id]
            )
            seg_history = other_aligned[-self.history_horizon:]
            if len(seg_history) == 0:
                continue
            fill_start = seg_end - len(seg_history)
            tokens[fill_start:seg_end] = np.array(seg_history, dtype=np.float32)
            pad[fill_start:seg_end] = False
            segment_ids[fill_start:seg_end] = seg_idx

        if np.all(pad):
            tokens[-1] = self._compose_state_prev_token(
                np.zeros(self.observation_space.shape[0], dtype=np.float32), -1.0, -1.0
            )
            pad[-1] = False
        return tokens, pad, segment_ids

    # ================================================================
    #  P2P 邻近通信 (实时信息交换)  —— 核心新增
    # ================================================================
    def _build_comm_msg(self, uav_id):
        """构建 UAV uav_id 的通信消息向量 (comm_dim,)"""
        pos = self.drone_position_now[uav_id] / self.map_size             # (2,)
        buffer_bm = np.zeros(self.K, dtype=np.float32)                    # (K,)
        for poi in self.drone_buffer[uav_id]:
            if 0 <= int(poi) < self.K:
                buffer_bm[int(poi)] = 1.0
        aoi = self.drone_local_aoi[uav_id] * 10.0 / self.T               # (K,)
        timing = np.array([self.drone_timing_now[uav_id] / self.T])       # (1,)
        last_target = np.array([self.drone_last_target[uav_id]])          # (1,)
        return np.concatenate([pos, buffer_bm, aoi, timing, last_target]).astype(np.float32)

    def _update_proximity_comm(self):
        """每次步进后更新通信邻接矩阵、消息矩阵和时间戳"""
        for i in range(self.M):
            for j in range(self.M):
                if i == j:
                    continue
                dist = float(np.linalg.norm(
                    self.drone_position_now[i] - self.drone_position_now[j]
                ))
                if dist < self.R_comm and self.R_comm > 0:
                    self.comm_adjacency[i, j] = 1.0
                    self.comm_messages[i, j] = self._build_comm_msg(j)
                    self.comm_msg_timestamp[i, j] = float(self.drone_timing_now[j])
                    # 更新 P2P 中继知识: i 得知 j 的 buffer 状态
                    self.p2p_known_buffer[i, j] = self.comm_messages[i, j, 2:2 + self.K]
                else:
                    self.comm_adjacency[i, j] = 0.0
                    self.comm_messages[i, j] = 0.0
                    # 不清零 p2p_known_buffer: 保留最后一次通信获知的信息
                    # 但标记时间戳为 -inf 表示不再实时
                    self.comm_msg_timestamp[i, j] = -np.inf

    # ================================================================
    #  观测 & 动作掩码
    # ================================================================
    def _get_obs(self, uav_id):
        # AoI
        aoi_obs = self.drone_local_aoi[uav_id].copy() * 10.0 / self.T

        # 距离 & 预期奖励
        dis_2_pois = np.linalg.norm(
            self.sensor_pos - self.drone_position_now[uav_id], axis=-1
        )
        move_t = dis_2_pois / self.fly_speed
        next_aoi = self.drone_local_aoi[uav_id] + move_t
        next_time = move_t + self.drone_timing_now[uav_id]

        mask = np.ones(self.K)
        mask[self.drone_buffer[uav_id]] = 0
        rewards_2_pois = next_aoi * (self.T - next_time) / (2 * self.args.pre_reward_ratio) * mask
        rewards_2_pois *= self.poi_weights
        rewards_2_pois /= (self.args.reward_scale_size * self.reward_divisor)
        rewards_2_pois *= 60
        rewards_2_pois /= (move_t + 1)

        # 回基站预期奖励
        rewards_2_bs = 0.0
        for i in range(len(self.drone_buffer[uav_id])):
            coll_t = self.buffer_timing[uav_id][i]
            tp = self.drone_buffer[uav_id][i]
            curr_aoi = self.drone_timing_now[uav_id] - coll_t
            rewards_2_bs += self.poi_weights[tp] * (
                self.drone_local_aoi[uav_id, tp] - curr_aoi
            ) * (self.T - self.drone_timing_now[uav_id]) / 2
        rewards_2_bs -= sum(self.drone_step_reward[uav_id])
        rewards_2_bs = np.array([rewards_2_bs]) * 5 / (
            self.args.reward_scale_size * self.reward_divisor
        )

        # buffer bitmap
        buffer_obs = np.zeros(self.K, dtype=np.float32)
        for poi in self.drone_buffer[uav_id]:
            if 0 <= int(poi) < self.K:
                buffer_obs[int(poi)] = 1.0

        # BS-synced 历史 (延迟)
        local_bs_history = self.local_known_bs_timing[uav_id].copy()
        own_row = np.zeros(self.K, dtype=np.float32)
        for idx, poi in enumerate(self.drone_buffer[uav_id]):
            if 0 <= int(poi) < self.K and idx < len(self.buffer_timing[uav_id]):
                own_row[int(poi)] = float(self.buffer_timing[uav_id][idx])
        local_bs_history[uav_id] = own_row
        history_visited = local_bs_history.flatten() / self.T

        # 时间
        time_obs = np.array([self.drone_timing_now[uav_id] / self.T])

        # 缓存大小
        buffer_len = np.array([len(self.drone_step_reward[uav_id])])

        # 上次奖励 (从 BS 同步得到)
        local_last_reward = self.local_known_last_reward[uav_id].copy()
        local_last_reward[uav_id] = float(self.drone_last_reward[uav_id])

        # 能量
        init_e = max(float(self.init_uav_energy_list[uav_id]), 1e-6)
        energy_obs = np.array([self.drone_energy_now[uav_id] / init_e], dtype=np.float32)

        # POI 权重
        poi_weight_obs = self.poi_weights / self.weight_norm

        # ---- P2P 邻居实时信息 (改进: mask + staleness + msg) ----
        # 每邻居: [availability_mask(1), staleness(1), comm_msg(comm_dim)]
        comm_info = np.zeros((self.M - 1) * self.comm_obs_per_neighbor, dtype=np.float32)
        n_idx = 0
        for j in range(self.M):
            if j == uav_id:
                continue
            base = n_idx * self.comm_obs_per_neighbor
            if self.comm_adjacency[uav_id, j] > 0:
                # 可用性掩码 = 1
                comm_info[base] = 1.0
                # 时延 = 当前 UAV 时间 - 消息生成时间
                msg_time = float(self.comm_msg_timestamp[uav_id, j])
                if msg_time > -np.inf:
                    staleness = float(self.drone_timing_now[uav_id] - msg_time) / max(self.T, 1e-6)
                    comm_info[base + 1] = max(staleness, 0.0)
                # 消息内容
                comm_info[base + 2: base + 2 + self.comm_dim] = self.comm_messages[uav_id, j]
            else:
                # 不在通信范围内: mask=0, staleness=0, msg=0
                # 但填入 BS 中继的 P2P 知识 (如果有)
                relay_buf = self.bs_relay_buffer[uav_id, j]
                if np.any(relay_buf > 0):
                    # 标记为延迟中继信息 (mask=0.5)
                    comm_info[base] = 0.5
                    # 中继信息的 staleness 设为 1.0 (最大陈旧度)
                    comm_info[base + 1] = 1.0
                    # 只填入 buffer_bitmap 部分, 其余为零
                    comm_info[base + 2 + 2: base + 2 + 2 + self.K] = relay_buf
            n_idx += 1

        return np.concatenate([
            aoi_obs,                          # K
            dis_2_pois / (self.map_size / 2), # K
            rewards_2_pois,                   # K
            rewards_2_bs,                     # 1
            buffer_obs,                       # K
            time_obs,                         # 1
            history_visited,                  # K*M
            local_last_reward / (self.args.reward_scale_size * self.reward_divisor),  # M
            buffer_len,                       # 1
            energy_obs,                       # 1
            poi_weight_obs,                   # K
            comm_info,                        # (M-1)*(comm_dim+2)  ← 改进
        ]).astype(np.float32)

    def get_action_masks(self, uav_id):
        target_mask = np.ones(self.target_action_dim, dtype=np.float32)
        for poi in self.drone_buffer[uav_id]:
            if 0 <= int(poi) < self.K:
                target_mask[int(poi)] = 0.0
        if len(self.drone_buffer[uav_id]) == 0:
            target_mask[self.K:self.K + self.N] = 0.0
        speed_mask = np.ones(self.speed_action_dim, dtype=np.float32)
        return {"target": target_mask, "speed": speed_mask}

    # ================================================================
    #  Critic 全局状态
    # ================================================================
    def get_global_critic_state(self):
        time_scale = max(self.T, 1e-6)
        map_scale = max(self.map_size, 1e-6)

        buffer_mask = np.zeros((self.M, self.K), dtype=np.float32)
        buffer_age = np.zeros((self.M, self.K), dtype=np.float32)
        for uid in range(self.M):
            curr_t = float(self.drone_timing_now[uid])
            for idx, poi in enumerate(self.drone_buffer[uid]):
                pi = int(poi)
                if 0 <= pi < self.K:
                    buffer_mask[uid, pi] = 1.0
                    if idx < len(self.buffer_timing[uid]):
                        ct = float(self.buffer_timing[uid][idx])
                        buffer_age[uid, pi] = max(curr_t - ct, 0.0) / time_scale

        init_e = np.maximum(self.init_uav_energy_list.astype(np.float32), 1e-6)
        max_e = max(float(np.max(init_e)), 1e-6)

        return np.concatenate([
            buffer_mask.reshape(-1),                                    # M*K
            buffer_age.reshape(-1),                                     # M*K
            self.drone_position_now.astype(np.float32).reshape(-1) / map_scale,  # M*2
            self.sensor_pos.astype(np.float32).reshape(-1) / map_scale,           # K*2
            self.base_pos.astype(np.float32).reshape(-1) / map_scale,             # N*2
            self.aoi.astype(np.float32) / time_scale,                              # K
            self.poi_weights.astype(np.float32) / max(self.weight_norm, 1e-6),     # K
            self.drone_timing_now.astype(np.float32) / time_scale,                 # M
            self.drone_energy_now.astype(np.float32) / max_e,                      # M
            self.drone_energy_now.astype(np.float32) / init_e,                     # M
            self.comm_adjacency.flatten().astype(np.float32),                      # M*M ← NEW
        ], axis=0).astype(np.float32)

    def get_transformer_inputs(self, uav_id):
        current_obs = self._get_obs(uav_id).astype(np.float32)
        prev_a, prev_r = self._get_last_action_reward(self.local_transition_tokens[uav_id])
        current_token = self._compose_state_prev_token(current_obs, prev_a, prev_r)

        self_aligned = self._build_prev_aligned_tokens(self.local_transition_tokens[uav_id])
        decoder_tokens_raw = self_aligned + [current_token]
        decoder_tokens, decoder_pad = self._pad_tokens(decoder_tokens_raw, self.decoder_token_len)

        encoder_tokens, encoder_pad, encoder_seg = self._build_segmented_encoder_tokens(uav_id)

        if self.critic_token_len > 1:
            history_tail = self.global_critic_history_tokens[-(self.critic_token_len - 1):]
            critic_tokens_raw = [tok.copy() for tok in history_tail] + [current_token]
        else:
            critic_tokens_raw = [current_token]
        critic_tokens, critic_pad = self._pad_tokens(critic_tokens_raw, self.critic_token_len)

        return {
            "encoder_tokens": encoder_tokens,
            "encoder_pad": encoder_pad,
            "encoder_segment_ids": encoder_seg,
            "decoder_tokens": decoder_tokens,
            "decoder_pad": decoder_pad,
            "critic_tokens": critic_tokens,
            "critic_pad": critic_pad,
            "critic_state": self.get_global_critic_state(),
            "obs": current_obs,
        }

    def get_critic_agent_order(self, uav_id):
        if self.critic_token_len > 1:
            tail = self.global_critic_history_agent_ids[-(self.critic_token_len - 1):]
            seq = list(tail) + [int(uav_id)]
        else:
            seq = [int(uav_id)]
        return seq[-self.critic_token_len:]

    # ================================================================
    #  统计
    # ================================================================
    def _reset_stats(self):
        self.comm_active_steps = 0
        self.comm_total_steps = 0
        self.comm_total_pairs = 0
        self.repeat_visits = 0
        self.total_poi_visits = 0

    def get_comm_stats(self):
        ts = max(self.comm_total_steps, 1)
        tp = max(self.total_poi_visits, 1)
        return {
            "comm_active_ratio": self.comm_active_steps / ts,
            "avg_neighbors": self.comm_total_pairs / ts,
            "repeat_visit_rate": self.repeat_visits / tp,
            "comm_active_steps": self.comm_active_steps,
            "comm_total_steps": self.comm_total_steps,
            "repeat_visits": self.repeat_visits,
            "total_poi_visits": self.total_poi_visits,
        }

    # ================================================================
    #  Reset & Step
    # ================================================================
    def reset(self):
        self.aoi = np.zeros(self.K, dtype=np.float32)
        self.global_timing = 0.0
        self.drone_last_visited_history_at_BS = np.zeros((self.M, self.K))
        self.drone_visited_history_timing_at_BS = np.zeros((self.M, self.K))

        self.drone_buffer = [[] for _ in range(self.M)]
        self.buffer_timing = [[] for _ in range(self.M)]
        self.drone_timing_now = np.zeros(self.M)
        self.drone_local_aoi = np.zeros((self.M, self.K))
        self.drone_last_reward = np.zeros(self.M)
        self.drone_step_reward = [[] for _ in range(self.M)]

        # UAV 初始位置 = 基站位置
        self.drone_position_now = np.tile(self.base_pos[0], (self.M, 1)).astype(np.float32)
        self.drone_energy_now = self.init_uav_energy_list.copy()
        self.drone_last_target = np.zeros(self.M, dtype=np.float32)

        self.local_transition_tokens = [[] for _ in range(self.M)]
        self.global_critic_history_tokens = []
        self.global_critic_history_agent_ids = []

        # BS sync
        self.bs_uploaded_tokens = [[] for _ in range(self.M)]
        self.synced_other_tokens = [[[] for _ in range(self.M)] for _ in range(self.M)]
        self.bs_uploaded_versions = np.zeros(self.M, dtype=np.int64)
        self.bs_uploaded_times = np.full(self.M, -np.inf, dtype=np.float32)
        self.downloaded_versions = np.zeros((self.M, self.M), dtype=np.int64)
        self.downloaded_times = np.full((self.M, self.M), -np.inf, dtype=np.float32)
        self.local_known_bs_timing = np.zeros((self.M, self.M, self.K), dtype=np.float32)
        self.local_known_last_reward = np.zeros((self.M, self.M), dtype=np.float32)

        # P2P comm
        self.comm_adjacency = np.zeros((self.M, self.M), dtype=np.float32)
        self.comm_messages = np.zeros((self.M, self.M, self.comm_dim), dtype=np.float32)
        self.comm_msg_timestamp = np.full((self.M, self.M), -np.inf, dtype=np.float32)

        # P2P 中继知识: p2p_known_buffer[i][j][k] = UAV i 通过 P2P 得知 UAV j 的 buffer 中有 POI k
        self.p2p_known_buffer = np.zeros((self.M, self.M, self.K), dtype=np.float32)
        # BS 中继的 P2P 知识: bs_relay_buffer[i][j][k] = UAV i 从 BS 下载的关于 UAV j 的 P2P 中继信息
        self.bs_relay_buffer = np.zeros((self.M, self.M, self.K), dtype=np.float32)

        self._update_proximity_comm()

        # stats
        self._reset_stats()

        return self._get_obs(0)

    def step(self, uav_id, action):
        assert 0 <= uav_id < self.M
        target_action, speed_idx, selected_speed = self._parse_action(action)
        assert 0 <= target_action < self.target_action_dim

        is_bs_action = self.K <= target_action <= self.K + self.N - 1
        state_before = self._get_obs(uav_id).astype(np.float32)

        if is_bs_action:
            target_position = self.base_pos[target_action - self.K]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis / selected_speed

            self.drone_position_now[uav_id] = target_position
            self.drone_timing_now[uav_id] += move_time

            for i in range(self.K):
                self.drone_local_aoi[uav_id, i] += move_time
                self.aoi[i] += self.drone_timing_now[uav_id] - self.global_timing
            self.global_timing = self.drone_timing_now[uav_id]

            reward = 0.0
            if self.drone_buffer[uav_id]:
                for i in range(len(self.drone_buffer[uav_id])):
                    coll_t = self.buffer_timing[uav_id][i]
                    tp = self.drone_buffer[uav_id][i]
                    curr_aoi = self.drone_timing_now[uav_id] - coll_t
                    reward += self.poi_weights[tp] * max(
                        min(self.aoi[tp], self.drone_local_aoi[uav_id, tp]) - curr_aoi, 0
                    ) * (self.T - self.drone_timing_now[uav_id]) / 2
                    self.aoi[tp] = min(curr_aoi, self.aoi[tp])

                self.drone_local_aoi[uav_id] = self.aoi
                self.drone_visited_history_timing_at_BS[uav_id] = np.zeros(self.K)
                for i in range(len(self.drone_buffer[uav_id])):
                    self.drone_visited_history_timing_at_BS[uav_id, self.drone_buffer[uav_id][i]] = \
                        self.buffer_timing[uav_id][i]
                self.drone_last_reward[uav_id] = reward

                step_reward = np.array(self.drone_step_reward[uav_id])
                neg_idx = np.argwhere(step_reward <= 0)
                step_reward[neg_idx] *= -1
                reward -= sum(step_reward)

                self.drone_step_reward[uav_id] = []
                self.drone_buffer[uav_id] = []
                self.buffer_timing[uav_id] = []

                self._sync_with_bs(uav_id)
        else:
            target_poi = target_action
            target_position = self.sensor_pos[target_poi]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis / selected_speed

            self.drone_timing_now[uav_id] += move_time
            self.drone_buffer[uav_id].append(target_poi)
            self.buffer_timing[uav_id].append(self.drone_timing_now[uav_id])
            self.drone_position_now[uav_id] = target_position

            for i in range(self.K):
                self.drone_local_aoi[uav_id, i] += move_time

            reward = self.poi_weights[target_poi] * \
                self.drone_local_aoi[uav_id, target_poi] * \
                (self.T - self.drone_timing_now[uav_id]) / (2 * self.args.pre_reward_ratio)

            if getattr(self.args, "buffer_punishment", False) and \
               len(self.drone_step_reward[uav_id]) >= self.bs_random_factor:
                reward = -float(getattr(self.args, "punishment_value", 4.0)) * self.args.reward_scale_size

            self.drone_step_reward[uav_id].append(reward)

            # ---- 统计：重复访问检测 ----
            self.total_poi_visits += 1
            for j in range(self.M):
                if j == uav_id:
                    continue
                if target_poi in self.drone_buffer[j]:
                    self.repeat_visits += 1
                    break

        # 能量
        energy_cost = UAV_Energy(selected_speed) * move_time
        self.drone_energy_now[uav_id] = max(
            self.drone_energy_now[uav_id] - energy_cost, 0.0
        )

        # 记录 last target
        self.drone_last_target[uav_id] = float(target_action) / max(self.target_action_dim - 1, 1)

        done = (self.drone_timing_now[uav_id] >= self.T) or (self.drone_energy_now[uav_id] <= 0)

        if self.drone_timing_now[uav_id] > self.T:
            reward = 0.0

        reward_scaled = reward / (self.args.reward_scale_size * self.reward_divisor)

        # ---- Tokens ----
        prev_a, prev_r = self._get_last_action_reward(self.local_transition_tokens[uav_id])
        critic_history_token = self._compose_state_prev_token(state_before, prev_a, prev_r)
        transition_token = self._compose_transition_token(
            state_before, target_action, speed_idx, reward_scaled
        )
        self.local_transition_tokens[uav_id].append(transition_token)
        self.global_critic_history_tokens.append(critic_history_token)
        self.global_critic_history_agent_ids.append(int(uav_id))

        # ---- P2P 通信更新 ----
        self._update_proximity_comm()

        # ---- 统计 ----
        self.comm_total_steps += 1
        n_pairs = int(np.sum(self.comm_adjacency) / 2)
        self.comm_total_pairs += n_pairs
        if n_pairs > 0:
            self.comm_active_steps += 1

        info = self.get_action_masks(uav_id)
        info["move_time"] = float(move_time)
        info["energy_cost"] = float(energy_cost)
        info["remaining_energy"] = float(self.drone_energy_now[uav_id])
        info["comm_active"] = n_pairs > 0
        info["comm_neighbors"] = int(np.sum(self.comm_adjacency[uav_id]))

        obs = self._get_obs(uav_id)
        return obs, reward_scaled, done, info

    def time_cost(self, uav_id, action):
        target_action, _, selected_speed = self._parse_action(action)
        if self.K <= target_action <= self.K + self.N - 1:
            target_position = self.base_pos[target_action - self.K]
        else:
            target_position = self.sensor_pos[target_action]
        move_dis = euclidean(target_position, self.drone_position_now[uav_id])
        return move_dis / selected_speed

    def get_pos(self, action):
        all_pos = np.concatenate([self.sensor_pos, self.base_pos], axis=0)
        ta, _, _ = self._parse_action(action)
        return all_pos[ta]

    def visualize_routes(self, test_actions, output_dir="output", dpi=150, file="", nums=10):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm

        poi_pos = self.sensor_pos
        bs_pos = self.base_pos
        n_uavs = len(test_actions)

        trimmed = []
        for i in range(n_uavs):
            route = np.asarray(test_actions[i][:nums], dtype=np.float32)
            trimmed.append(route)

        fig = plt.figure(figsize=(8, 8))
        if len(poi_pos) > 0:
            plt.scatter(poi_pos[:, 0], poi_pos[:, 1], c="blue", marker="o", label="POIs", alpha=0.6)
        if len(bs_pos) > 0:
            plt.scatter(bs_pos[:, 0], bs_pos[:, 1], c="red", marker="^", s=100, label="BS", alpha=0.8)
        colors = cm.rainbow(np.linspace(0, 1, n_uavs))
        for i in range(n_uavs):
            route = trimmed[i]
            if route.size == 0:
                continue
            plt.plot(route[:, 0], route[:, 1], c=colors[i], marker="o", markersize=5,
                     label=f"UAV {i}", alpha=0.7, linewidth=2)
        plt.xlim(-10, self.map_size + 10)
        plt.ylim(-10, self.map_size + 10)
        plt.xlabel("X"); plt.ylabel("Y")
        plt.title(f"UAV Routes ({self.K} POIs, {self.N} BS, {n_uavs} UAVs)")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{file}routes.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close()
