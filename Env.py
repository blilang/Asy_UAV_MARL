import gym
import numpy as np
from gym import spaces
from scipy.spatial.distance import euclidean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from matplotlib import cm


def UAV_Energy(v):
    P_b = 79.86
    P_i = 88.63
    V_tip = 120
    u_0 = 4.03
    f_0 = 0.6
    a = 1.225
    n = 0.05
    R = 0.503
    energy = (
        P_b * (1 + (3 * v * v) / (V_tip * V_tip))
        + P_i * np.sqrt(np.sqrt(1 + (v * v * v * v) / (4 * (u_0**4))) - v * v / (2 * u_0 * u_0))
        + f_0 * a * n * R * v * v * v / 2
    )
    return float(energy)


class MultiDroneAoIEnv(gym.Env):
    def __init__(
        self,
        M=3,
        N=2,
        K=5,
        T=100.0,
        map_size=100.0,
        drone_speeds=None,
        sense_times=None,
        args=None,
        position_file=None,
        base_positions=None,
        sensor_positions=None
    ):
        """
        多无人机数据感知环境，目标：最小化K个感知点的平均信息年龄 (AoI)。
        异步执行：step接受uav_id和action，返回最早完成动作的无人机的obs、reward等。
        记录每架无人机在回传前的感知点访问历史，用于奖励计算。

        参数:
            M (int): 无人机数量
            N (int): 基站数量
            K (int): 感知点数量
            T (float): 时间限制（秒）
            map_size (float): 地图大小（正方形边长）
            drone_speeds (list): 每架无人机的速度（m/s），默认 [1.0, ...]
            sense_times (list): 每个感知点的感知时间（秒），默认 [1.0, ...]
            position_file (str): CSV文件路径，包含基站和感知点位置
            base_positions (np.array): 基站位置数组，形状 (N, 2)
            sensor_positions (np.array): 感知点位置数组，形状 (K, 2)
        """
        super(MultiDroneAoIEnv, self).__init__()
        self.M = M
        self.N = N
        self.K = K
        self.T = T
        self.map_size = map_size
        self.args = args

        # 无人机速度（支持异构性）
        self.drone_speeds = drone_speeds if drone_speeds is not None else [1.0] * M
        self.speed_levels = self._parse_speed_levels()
        self.speed_action_dim = len(self.speed_levels)
        self.default_speed_idx = int(np.argmax(self.speed_levels))
        self.nominal_speed = float(np.mean(self.speed_levels))
        self.init_uav_energy_list = self._parse_init_uav_energies()
        # 保留标量字段用于兼容旧打印与逻辑
        self.init_uav_energy = float(np.max(self.init_uav_energy_list))
        self.reward_divisor = float(getattr(self.args, "reward_divisor", 10.0))
        # 位置：无人机、基站、感知点
        self.drone_pos = np.zeros((M, 2))  # 初始位置 (0, 0)
        self.base_pos = np.zeros((self.N, 2))
        self.sensor_pos = np.zeros((self.K, 2))

        # !数据文件：支持指定位置文件，并读取 poi 权重
        position_filename = position_file or getattr(self.args, "position_file", None)
        if position_filename is None:
            new_name = f"./data/poi_{self.K}_map_{int(self.map_size)}x{int(self.map_size)}.npy"
            old_name = f"./data/poi_{self.K}_bs_{self.N}_map_{int(self.map_size)}x{int(self.map_size)}.npy"
            position_filename = new_name if os.path.exists(new_name) else old_name
        data = np.load(position_filename, allow_pickle=True).item()
        self.sensor_pos = np.asarray(data["poi_positions"], dtype=np.float32)
        self.base_pos = np.asarray(data["bs_positions"], dtype=np.float32)
        self.K = int(self.sensor_pos.shape[0])
        self.N = int(self.base_pos.shape[0])

        raw_weights = data.get("poi_weights", None)
        if raw_weights is None:
            self.poi_weights = np.ones(self.K, dtype=np.float32)
        else:
            self.poi_weights = np.asarray(raw_weights, dtype=np.float32).reshape(-1)
            if self.poi_weights.shape[0] != self.K:
                raise ValueError(
                    f"poi_weights length {self.poi_weights.shape[0]} does not match K={self.K} in {position_filename}"
                )
        self.weight_norm = max(float(np.max(self.poi_weights)), 1e-6)

        # ! 看如何计算
        if sense_times is None:
            self.sense_times = [1.0] * self.K
        else:
            if len(sense_times) != self.K:
                raise ValueError(f"sense_times length {len(sense_times)} must match K={self.K}")
            self.sense_times = sense_times

        # !全局信息
        # 信息年龄 (AoI)
        self.aoi = np.zeros(self.K)
        self.global_timing = 0
        # !后期需要进行修改
        self.fly_speed = self.nominal_speed
        self.drone_last_visited_history_at_BS = np.zeros((M, self.K))
        self.drone_visited_history_timing_at_BS = np.zeros((M, self.K))
        # self.drone_last_decision_at_BS = np.zeros(M)
        # self.drone_last_decision_time_at_BS = np.zeros(M)
        
        # !局部信息
        # 当前时刻无人机的缓存列表
        self.drone_buffer = [[] for _ in range(M)]
        # 每个缓存对应的时刻
        self.buffer_timing = [[] for _ in range(M)]
        # 每个无人机单独的时间
        self.drone_timing_now = np.zeros(M)
        self.drone_local_aoi = np.zeros((M, self.K))
        self.drone_last_reward = np.zeros(M)
        self.drone_step_reward = [[] for i in range(M)]
        self.drone_position_now = np.zeros((self.M, 2))
        self.other_drones_delay_rewards = [[] for i in range(M)]
        self.drone_energy_now = self.init_uav_energy_list.copy()
        self.drone_task_time_log = [[] for _ in range(M)]
        self.drone_task_energy_log = [[] for _ in range(M)]
        
        # !观测和动作空间
        # 动作空间：目标点（POI/BS） + 速度档位
        self.action_space = spaces.Discrete(self.K + self.N)
        self.speed_action_space = spaces.Discrete(self.speed_action_dim)
        self.target_action_dim = self.K + self.N
        # 观测空间：单个无人机
        # obs_dim = AoI(K) + distance(K) + weighted_est_reward(K) + reward2bs(1)
        #         + buffer(K) + time(1) + bs_history(K*M) + last_reward(M)
        #         + buffer_len(1) + energy(1) + poi_weight(K)
        obs_dim = self.K + self.K + self.K + 1 + self.K + 1 + self.K * M + M + 1 + 1 + self.K
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.token_dim = obs_dim + 2
        self.history_horizon = int(getattr(self.args, "history_horizon", 10))
        self.max_other_agents = max(0, self.M - 1)
        self.encoder_token_len = max(1, (self.M - 1) * self.history_horizon)
        self.decoder_token_len = max(1, self.history_horizon + 1)  # +1 for current state token
        self.critic_token_len = max(1, self.M * (self.history_horizon + 1))
        
        self.left = 4
        self.right = 8
        # self.bs_random_factor = np.random.randint(self.left, self.right)
        self.bs_random_factor = self.args.BS_back_times

        # 初始化环境
        self.reset()

    def reset(self):
        """重置环境"""
        # !全局信息
        # 信息年龄 (AoI)
        self.aoi = np.zeros(self.K)
        self.global_timing = 0
        self.drone_last_visited_history_at_BS = np.zeros((self.M, self.K))
        self.drone_visited_history_timing_at_BS = np.zeros((self.M, self.K))
        
        # !局部信息
        # 当前时刻无人机的缓存列表
        self.drone_buffer = [[] for _ in range(self.M)]
        # 每个缓存对应的时刻
        self.buffer_timing = [[] for _ in range(self.M)]
        # 每个无人机单独的时间
        self.drone_timing_now = np.zeros(self.M)
        self.drone_local_aoi = np.zeros((self.M, self.K))
        self.drone_last_reward = np.zeros(self.M)
        self.drone_step_reward = [[] for i in range(self.M)]
        self.drone_position_now = np.zeros((self.M, 2))
        self.other_drones_delay_rewards = [[] for i in range(self.M)]
        self.drone_energy_now = self.init_uav_energy_list.copy()
        self.drone_task_time_log = [[] for _ in range(self.M)]
        self.drone_task_energy_log = [[] for _ in range(self.M)]
        self.local_transition_tokens = [[] for _ in range(self.M)]
        self.global_critic_history_tokens = []
        self.global_critic_history_agent_ids = []
        self.bs_uploaded_tokens = [[] for _ in range(self.M)]
        self.synced_other_tokens = [[[] for _ in range(self.M)] for _ in range(self.M)]
        self.bs_uploaded_versions = np.zeros(self.M, dtype=np.int64)
        self.bs_uploaded_times = np.full(self.M, -np.inf, dtype=np.float32)
        self.downloaded_versions = np.zeros((self.M, self.M), dtype=np.int64)
        self.downloaded_times = np.full((self.M, self.M), -np.inf, dtype=np.float32)
        self.bs_reward_logs = [[] for _ in range(self.M)]  # (version, coop_upload_reward_scaled, upload_time)
        self.downloaded_reward_versions = np.zeros((self.M, self.M), dtype=np.int64)
        # observer_uav x source_uav x poi: what observer has learned at BS about each source UAV.
        self.local_known_bs_timing = np.zeros((self.M, self.M, self.K), dtype=np.float32)
        # observer_uav x source_uav: latest downloaded reward snapshot per source UAV.
        self.local_known_last_reward = np.zeros((self.M, self.M), dtype=np.float32)
        return self._get_obs(0)

    def _parse_speed_levels(self):
        raw = getattr(self.args, "speed_levels", "6-20")
        if isinstance(raw, str):
            raw = raw.strip()
            if "-" in raw and "," not in raw:
                # 支持区间格式: "6-20" 或 "6-20:1"
                if ":" in raw:
                    range_part, step_part = raw.split(":", 1)
                    step = max(float(step_part.strip()), 1e-3)
                else:
                    range_part = raw
                    step = 1.0
                low_s, high_s = [v.strip() for v in range_part.split("-", 1)]
                low_v = float(low_s)
                high_v = float(high_s)
                if high_v < low_v:
                    low_v, high_v = high_v, low_v
                speeds = list(np.arange(low_v, high_v + 1e-9, step))
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
        """Parse per-UAV initial energies from args.init_uav_energies or fallback to scalar init_uav_energy."""
        raw_list = getattr(self.args, "init_uav_energies", None)
        raw_scalar = float(getattr(self.args, "init_uav_energy", 2.0e5))

        if raw_list is None:
            return np.full(self.M, raw_scalar, dtype=np.float32)

        if isinstance(raw_list, (list, tuple, np.ndarray)):
            vals = [float(v) for v in raw_list]
        else:
            text = str(raw_list).strip()
            text = text.strip("[]()")
            text = text.replace(",", " ")
            parts = [p for p in text.split() if p]
            vals = [float(p) for p in parts] if parts else [raw_scalar]

        if len(vals) == 1:
            vals = vals * self.M
        if len(vals) != self.M:
            raise ValueError(
                f"init_uav_energies length {len(vals)} does not match M={self.M}. "
                f"Use one value or exactly M values."
            )
        vals = [max(0.0, float(v)) for v in vals]
        return np.asarray(vals, dtype=np.float32)

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

    def _compose_transition_token(self, state_vec, target_action, speed_idx, reward):
        combined_action = target_action * self.speed_action_dim + speed_idx
        denom = max(1, self.target_action_dim * self.speed_action_dim - 1)
        action_norm = float(combined_action / denom)
        return self._compose_state_prev_token(state_vec, action_norm, reward)

    def _compose_state_prev_token(self, state_vec, prev_action, prev_reward):
        token = np.concatenate([state_vec, np.array([prev_action, prev_reward], dtype=np.float32)])
        return token.astype(np.float32)

    def _build_prev_aligned_tokens(self, sar_tokens):
        """
        将历史 [s_t, a_t, r_t] 转为 [s_t, a_{t-1}, r_{t-1}]。
        第一个token没有上一步动作和奖励，用 -1 填充。
        """
        aligned = []
        prev_action = -1.0
        prev_reward = -1.0
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

    def _sync_with_bs(self, uav_id, coop_upload_reward_scaled):
        # UAV uploads its newest local history to BS, BS version increments.
        self.bs_uploaded_tokens[uav_id] = [tok.copy() for tok in self.local_transition_tokens[uav_id]]
        self.bs_uploaded_versions[uav_id] += 1
        self.bs_uploaded_times[uav_id] = self.drone_timing_now[uav_id]
        curr_version = int(self.bs_uploaded_versions[uav_id])
        self.bs_reward_logs[uav_id].append(
            (curr_version, float(coop_upload_reward_scaled), float(self.drone_timing_now[uav_id]))
        )
        # Uploader always knows its own latest BS-side summary after returning to BS.
        self.local_known_bs_timing[uav_id, uav_id] = self.drone_visited_history_timing_at_BS[uav_id].astype(np.float32)
        self.local_known_last_reward[uav_id, uav_id] = float(self.drone_last_reward[uav_id])

        # Downloader can only refresh others' histories at BS.
        # Only newer BS versions are downloaded, avoiding stale overwrite.
        for other_id in range(self.M):
            if other_id == uav_id:
                continue
            latest_version = self.bs_uploaded_versions[other_id]
            if latest_version > self.downloaded_versions[uav_id, other_id]:
                self.synced_other_tokens[uav_id][other_id] = [
                    tok.copy() for tok in self.bs_uploaded_tokens[other_id]
                ]
                self.downloaded_versions[uav_id, other_id] = latest_version
                self.downloaded_times[uav_id, other_id] = self.drone_timing_now[uav_id]
                self.local_known_bs_timing[uav_id, other_id] = self.drone_visited_history_timing_at_BS[
                    other_id
                ].astype(np.float32)
                self.local_known_last_reward[uav_id, other_id] = float(self.drone_last_reward[other_id])

    def _collect_pending_coop_rewards(self, uav_id):
        coop_reward = 0.0
        for other_id in range(self.M):
            if other_id == uav_id:
                continue
            last_downloaded = int(self.downloaded_reward_versions[uav_id, other_id])
            latest_version = int(self.bs_uploaded_versions[other_id])
            if latest_version <= last_downloaded:
                continue

            for version, reward_scaled, _ in self.bs_reward_logs[other_id]:
                if last_downloaded < version <= latest_version:
                    coop_reward += float(reward_scaled)
            self.downloaded_reward_versions[uav_id, other_id] = latest_version
        return coop_reward

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
            other_aligned = self._build_prev_aligned_tokens(self.synced_other_tokens[uav_id][other_id])
            seg_history = other_aligned[-self.history_horizon:]
            if len(seg_history) == 0:
                continue

            fill_start = seg_end - len(seg_history)
            tokens[fill_start:seg_end] = np.array(seg_history, dtype=np.float32)
            pad[fill_start:seg_end] = False
            segment_ids[fill_start:seg_end] = seg_idx

        # 防止 encoder 全padding导致 Transformer 注意力出现 NaN
        if np.all(pad):
            tokens[-1] = self._compose_state_prev_token(
                np.zeros(self.observation_space.shape[0], dtype=np.float32), -1.0, -1.0
            )
            pad[-1] = False
            segment_ids[-1] = 0

        return tokens, pad, segment_ids

    def get_action_masks(self, uav_id):
        target_mask = np.ones(self.target_action_dim, dtype=np.float32)
        target_mask[self.drone_buffer[uav_id]] = 0.0
        if len(self.drone_buffer[uav_id]) == 0:
            target_mask[self.K:self.K + self.N] = 0.0
        speed_mask = np.ones(self.speed_action_dim, dtype=np.float32)
        return {"target": target_mask, "speed": speed_mask}

    def get_global_critic_state(self):
        buffer_mask = np.zeros((self.M, self.K), dtype=np.float32)
        buffer_age = np.zeros((self.M, self.K), dtype=np.float32)
        time_scale = max(float(self.T), 1e-6)
        map_scale = max(float(self.map_size), 1e-6)

        for uav_id in range(self.M):
            curr_time = float(self.drone_timing_now[uav_id])
            for idx, poi in enumerate(self.drone_buffer[uav_id]):
                poi_idx = int(poi)
                if not (0 <= poi_idx < self.K):
                    continue
                buffer_mask[uav_id, poi_idx] = 1.0
                if idx < len(self.buffer_timing[uav_id]):
                    coll_time = float(self.buffer_timing[uav_id][idx])
                    buffer_age[uav_id, poi_idx] = max(curr_time - coll_time, 0.0) / time_scale

        init_energy = np.maximum(self.init_uav_energy_list.astype(np.float32), 1e-6)
        max_init_energy = max(float(np.max(init_energy)), 1e-6)

        return np.concatenate(
            [
                buffer_mask.reshape(-1),
                buffer_age.reshape(-1),
                self.drone_position_now.astype(np.float32).reshape(-1) / map_scale,
                self.sensor_pos.astype(np.float32).reshape(-1) / map_scale,
                self.base_pos.astype(np.float32).reshape(-1) / map_scale,
                self.aoi.astype(np.float32) / time_scale,
                self.poi_weights.astype(np.float32) / max(float(self.weight_norm), 1e-6),
                self.drone_timing_now.astype(np.float32) / time_scale,
                self.drone_energy_now.astype(np.float32) / max_init_energy,
                self.drone_energy_now.astype(np.float32) / init_energy,
            ],
            axis=0,
        ).astype(np.float32)

    def get_transformer_inputs(self, uav_id):
        current_obs = self._get_obs(uav_id).astype(np.float32)
        prev_action, prev_reward = self._get_last_action_reward(self.local_transition_tokens[uav_id])
        current_token = self._compose_state_prev_token(current_obs, prev_action, prev_reward)

        self_aligned = self._build_prev_aligned_tokens(self.local_transition_tokens[uav_id])
        decoder_tokens_raw = self_aligned + [current_token]
        decoder_tokens, decoder_pad = self._pad_tokens(decoder_tokens_raw, self.decoder_token_len)

        encoder_tokens, encoder_pad, encoder_segment_ids = self._build_segmented_encoder_tokens(uav_id)

        # Centralized critic tokens follow the global async action completion order.
        # Reserve the tail position for the currently evaluated UAV token.
        if self.critic_token_len > 1:
            history_tail = self.global_critic_history_tokens[-(self.critic_token_len - 1):]
            critic_tokens_raw = [tok.copy() for tok in history_tail] + [current_token]
        else:
            critic_tokens_raw = [current_token]
        critic_tokens, critic_pad = self._pad_tokens(critic_tokens_raw, self.critic_token_len)

        return {
            "encoder_tokens": encoder_tokens,
            "encoder_pad": encoder_pad,
            "encoder_segment_ids": encoder_segment_ids,
            "decoder_tokens": decoder_tokens,
            "decoder_pad": decoder_pad,
            "critic_tokens": critic_tokens,
            "critic_pad": critic_pad,
            "critic_state": self.get_global_critic_state(),
            "obs": current_obs,
        }

    def get_critic_agent_order(self, uav_id):
        """Return agent-id order used by centralized critic input for current UAV."""
        if self.critic_token_len > 1:
            history_tail = self.global_critic_history_agent_ids[-(self.critic_token_len - 1):]
            seq = list(history_tail) + [int(uav_id)]
        else:
            seq = [int(uav_id)]
        if len(seq) > self.critic_token_len:
            seq = seq[-self.critic_token_len:]
        return seq

    def _get_obs(self, uav_id):
        """获取指定无人机的观测"""
        
        # *AoI信息
        aoi_obs = self.drone_local_aoi[uav_id].copy()*10/self.T
        
        # *位置信息
        dis_2_pois = np.linalg.norm(self.sensor_pos-self.drone_position_now[uav_id], axis=-1)
        move_t_2_pois = dis_2_pois/self.fly_speed
        next_aoi_2_poi = self.drone_local_aoi[uav_id] + move_t_2_pois
        next_time_2_poi = move_t_2_pois + self.drone_timing_now[uav_id]
        
        mask = np.ones(self.K)
        mask[self.drone_buffer[uav_id]] = 0
        rewards_2_pois = next_aoi_2_poi * (self.T-next_time_2_poi)/(2*self.args.pre_reward_ratio)*mask
        rewards_2_pois *= self.poi_weights
        rewards_2_pois /= (self.args.reward_scale_size * self.reward_divisor)
        
        # 转换为效率，rewards/time
        rewards_2_pois *= 60
        rewards_2_pois /= (move_t_2_pois+1)
        
        rewards_2_bs = 0
        for i in range(len(self.drone_buffer[uav_id])):
            coll_timing = self.buffer_timing[uav_id][i]
            target_poi = self.drone_buffer[uav_id][i]
            curr_aoi = self.drone_timing_now[uav_id] - coll_timing
            
            rewards_2_bs += self.poi_weights[target_poi] * (self.drone_local_aoi[uav_id, target_poi] - curr_aoi) * (self.T - self.drone_timing_now[uav_id]) / 2
        rewards_2_bs -= sum(self.drone_step_reward[uav_id])
        rewards_2_bs = np.array([rewards_2_bs]) * 5 / (self.args.reward_scale_size * self.reward_divisor)
        
        
        
        pos_obs = np.concatenate([
            self.base_pos.flatten(),
            self.sensor_pos.flatten()
        ])/self.map_size
        
        buffer_obs = np.zeros(self.K, dtype=np.float32)
        for poi in self.drone_buffer[uav_id]:
            poi_idx = int(poi)
            if 0 <= poi_idx < self.K:
                buffer_obs[poi_idx] = 1.0

        # *无人机缓存信息
        # *自己缓存的为1，别人缓存的为2，没有缓存的为0
        local_bs_history = self.local_known_bs_timing[uav_id].copy()
        own_history_row = np.zeros(self.K, dtype=np.float32)
        for idx, poi in enumerate(self.drone_buffer[uav_id]):
            poi_idx = int(poi)
            if 0 <= poi_idx < self.K:
                own_history_row[poi_idx] = float(self.buffer_timing[uav_id][idx])
        local_bs_history[uav_id] = own_history_row
        history_visited_sensors_and_timing = local_bs_history.flatten() / self.T
            
        # *时间信息
        time_obs = np.array([self.drone_timing_now[uav_id]])/self.T
        
        # *缓存大小
        buffer_len = np.array([len(self.drone_step_reward[uav_id])])
        local_last_reward = self.local_known_last_reward[uav_id].copy()
        local_last_reward[uav_id] = float(self.drone_last_reward[uav_id])
        # *剩余能量（归一化）
        init_energy_uav = max(float(self.init_uav_energy_list[uav_id]), 1e-6)
        energy_obs = np.array([self.drone_energy_now[uav_id] / init_energy_uav], dtype=np.float32)
        # *poi权重（归一化）
        poi_weight_obs = self.poi_weights / self.weight_norm
        
        # !注意需要进行归一化 K+K+K+1+K+1+K*M+M+1+1
        # print(aoi_obs, dis_2_pois/500)
        return np.concatenate([
            aoi_obs,
            dis_2_pois / 500,
            rewards_2_pois,
            rewards_2_bs,
            buffer_obs,
            time_obs,
            history_visited_sensors_and_timing,
            local_last_reward / (self.args.reward_scale_size * self.reward_divisor),
            buffer_len,
            energy_obs,
            poi_weight_obs,
        ])

    def step(self, uav_id, action):
        """
        异步步进：接受uav_id和action，处理最早完成动作的无人机，返回其obs、reward等。

        参数:
            uav_id (int): 执行动作的无人机ID
            action (int): 动作（0~K+N-1）

        返回:
            obs (np.array): 下一无人机的观测
            reward (float): 奖励
            done (bool): 是否终止
            info (dict): 包含drone_id、avg_aoi、current_time等
        """
        assert 0 <= uav_id < self.M, f"Invalid uav_id: {uav_id}"
        target_action, speed_idx, selected_speed = self._parse_action(action)
        assert self.action_space.contains(target_action), f"Invalid target action: {target_action}"
        is_bs_action = self.K <= target_action <= self.K + self.N - 1

        state_before = self._get_obs(uav_id).astype(np.float32)

        # K是感知点
        # N是基站
        # *如果是基站
        coop_upload_reward = 0.0
        if is_bs_action:
            target_position = self.base_pos[target_action - self.K]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis / selected_speed

            # 根据计算结果更新（位置、时间）
            self.drone_position_now[uav_id] = target_position
            self.drone_timing_now[uav_id] += move_time

            for i in range(self.K):
                self.drone_local_aoi[uav_id, i] += move_time
                self.aoi[i] += self.drone_timing_now[uav_id] - self.global_timing

            self.global_timing = self.drone_timing_now[uav_id]

            reward = 0
            if self.drone_buffer[uav_id]:
                for i in range(len(self.drone_buffer[uav_id])):
                    coll_timing = self.buffer_timing[uav_id][i]
                    target_poi = self.drone_buffer[uav_id][i]
                    curr_aoi = self.drone_timing_now[uav_id] - coll_timing

                    reward += self.poi_weights[target_poi] * max(
                        (min(self.aoi[target_poi], self.drone_local_aoi[uav_id, target_poi]) - curr_aoi), 0
                    ) * (self.T - self.drone_timing_now[uav_id]) / 2
                    self.aoi[target_poi] = min(curr_aoi, self.aoi[target_poi])

                self.drone_local_aoi[uav_id] = self.aoi
                self.drone_visited_history_timing_at_BS[uav_id] = np.zeros(self.K)
                for i in range(len(self.drone_buffer[uav_id])):
                    self.drone_visited_history_timing_at_BS[uav_id, self.drone_buffer[uav_id][i]] = self.buffer_timing[uav_id][i]

                # 这一轮总的reward
                self.drone_last_reward[uav_id] = reward

                # 将自己当前的总奖励分给其他uav
                for i in range(self.M):
                    if i != uav_id:
                        self.other_drones_delay_rewards[i].append(reward)

                # BS结算前的原始奖励（不减去预分配奖励），用于coop同步
                coop_upload_reward = float(reward)

                # 首先减去之前为了鼓励无人机飞行而产生的reward
                # 清空自己之前的reward记录，进行下一轮收集
                step_reward = np.array(self.drone_step_reward[uav_id])
                negative_reward_indx = np.argwhere(step_reward <= 0)
                step_reward[negative_reward_indx] *= -1
                reward -= sum(step_reward)

                self.drone_step_reward[uav_id] = []
                self.other_drones_delay_rewards[uav_id] = []

                # 清空buffer
                self.drone_buffer[uav_id] = []
                self.buffer_timing[uav_id] = []

        # *如果不是基站，只需要维护自己的内容，不知道别人的内容
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

            # *三分之一的reward
            reward = self.poi_weights[target_poi] * self.drone_local_aoi[uav_id, target_poi] * (self.T - self.drone_timing_now[uav_id]) / (2 * self.args.pre_reward_ratio)
            if self.args.buffer_punishment and len(self.drone_step_reward[uav_id]) >= self.bs_random_factor:
                reward = -self.args.punishment_value * self.args.reward_scale_size

            self.drone_step_reward[uav_id].append(reward)

        # 动作完成后更新能量：能耗功率 * 飞行时长
        energy_cost = UAV_Energy(selected_speed) * move_time
        self.drone_task_time_log[uav_id].append(float(move_time))
        self.drone_task_energy_log[uav_id].append(float(energy_cost))
        self.drone_energy_now[uav_id] -= energy_cost
        self.drone_energy_now[uav_id] = max(self.drone_energy_now[uav_id], 0.0)

        done = (self.drone_timing_now[uav_id] >= self.T) or (self.drone_energy_now[uav_id] <= 0)

        # 注意时间超出时的波动很大
        if self.drone_timing_now[uav_id] > self.T:
            reward = 0
            coop_upload_reward = 0.0

        reward_scaled = reward / (self.args.reward_scale_size * self.reward_divisor)
        if is_bs_action:
        #     # 上传BS结算前奖励用于coop，再领取其他无人机未领取的BS奖励（合作奖励）
            coop_upload_reward_scaled = coop_upload_reward / (self.args.reward_scale_size * self.reward_divisor)
            self._sync_with_bs(uav_id, coop_upload_reward_scaled=coop_upload_reward_scaled)

        prev_action_hist, prev_reward_hist = self._get_last_action_reward(self.local_transition_tokens[uav_id])
        critic_history_token = self._compose_state_prev_token(state_before, prev_action_hist, prev_reward_hist)
        transition_token = self._compose_transition_token(
            state_before, target_action=target_action, speed_idx=speed_idx, reward=reward_scaled
        )
        self.local_transition_tokens[uav_id].append(transition_token)
        self.global_critic_history_tokens.append(critic_history_token)
        self.global_critic_history_agent_ids.append(int(uav_id))

        info = self.get_action_masks(uav_id)
        info["move_time"] = float(move_time)
        info["energy_cost"] = float(energy_cost)
        info["remaining_energy"] = float(self.drone_energy_now[uav_id])
        obs = self._get_obs(uav_id)

        if self.args.print_info:
            print(
                "Action UAV: {}, action: ({}, {}), speed: {:.2f}, time now: {}, time cost: {}, local AoI: {}, position: {}, buffer: {}".format(
                    uav_id,
                    target_action,
                    speed_idx,
                    selected_speed,
                    self.drone_timing_now[uav_id],
                    move_time,
                    # self.drone_local_aoi[uav_id],
                    reward_scaled,
                    self.drone_position_now[uav_id],
                    self.drone_buffer[uav_id],
                )
            )
            print(
                "UAV {} energy: {:.2f}/{:.2f}, energy_cost: {:.2f}".format(
                    uav_id,
                    self.drone_energy_now[uav_id],
                    float(self.init_uav_energy_list[uav_id]),
                    energy_cost,
                )
            )

        return obs, reward_scaled, done, info

    def time_cost(self, uav_id, action):
        target_action, _, selected_speed = self._parse_action(action)
        if self.K <= target_action <= self.K + self.N - 1:
            target_position = self.base_pos[target_action - self.K]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis / selected_speed
        else:
            target_poi = target_action
            target_position = self.sensor_pos[target_poi]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis / selected_speed
        return move_time

    def render(self, mode="human"):
        """渲染环境"""
        print(f"Time: {self.global_timing:.2f}/{self.T}")
        print(f"Average AoI: {np.mean(self.aoi):.2f}")
        print("Drone Energy:", self.drone_energy_now)
        print("Drone Positions:", self.drone_pos)
        print("Drone Buffers:", self.drone_buffer)
        print("Drone Visit History:", self.buffer_timing)
        print("Drone Action Times:", self.drone_timing_now)

    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        
    def get_pos(self, action):
        all_positions = np.concatenate([self.sensor_pos, self.base_pos], axis=0)
        target_action, _, _ = self._parse_action(action)
        return all_positions[target_action]
        
    def visualize_routes(self, test_actions, output_dir="output", dpi=150, file="", nums=10):
        """
        Visualize POI positions, base station positions, and UAV routes (first 10 steps).
        
        Parameters:
        - self: Object containing sensor_pos (POI) and base_pos (base stations).
        - test_actions: Array of shape (n_uavs, n_steps, 2) representing UAV positions over time.
        - output_dir: Directory to save the visualization (default: 'output').
        - dpi: Resolution of the output image (default: 300).
        """
        # Validate inputs


        # Extract data
        poi_positions = self.sensor_pos  # Shape: (n_poi, 2)
        bs_positions = self.base_pos     # Shape: (n_bs, 2)
        uav_positions = test_actions     # Shape: (n_uavs, n_steps, 2)

        n_poi = poi_positions.shape[0]
        n_bs = bs_positions.shape[0]
        n_uavs = len(uav_positions)

        # Each UAV keeps its own first-N tasks; lengths can be different in async rollout.
        trimmed_routes = []
        for i in range(n_uavs):
            route = np.asarray(uav_positions[i][:nums], dtype=np.float32)
            trimmed_routes.append(route)

        # Compute map boundaries
        boundary_points = [poi_positions, bs_positions]
        route_points = [r for r in trimmed_routes if r.size > 0]
        if route_points:
            boundary_points.append(np.concatenate(route_points, axis=0))
        all_positions = np.concatenate(boundary_points, axis=0)
        x_min, y_min = np.min(all_positions, axis=0) - 10  # Add padding
        x_max, y_max = np.max(all_positions, axis=0) + 10
        map_width = x_max - x_min
        map_height = y_max - y_min

        # Create figure (non-interactive backend + explicit close avoids GDI/bitmap leak on Windows)
        fig = plt.figure(figsize=(8, 8))

        # Plot POIs (blue circles)
        if n_poi > 0:
            plt.scatter(poi_positions[:, 0], poi_positions[:, 1], c='blue', marker='o', label='POIs', alpha=0.6)
            for i in range(n_poi):
                plt.text(poi_positions[i, 0] + 1, poi_positions[i, 1] + 1, str(i),
                            fontsize=10, color='black', weight='bold', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'))

        # Plot base stations (red triangles)
        if n_bs > 0:
            plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker='^', s=100, label='Base Stations', alpha=0.8)

        # Plot UAV routes (different colors for each UAV)
        colors = cm.rainbow(np.linspace(0, 1, n_uavs))  # Generate distinct colors
        for i in range(n_uavs):
            route = trimmed_routes[i]
            if route.size == 0:
                continue
            plt.plot(route[:, 0], route[:, 1], c=colors[i], marker='o', markersize=5, 
                    label=f'UAV {i+1}', alpha=0.7, linewidth=2)
            # Mark start and end points
            plt.scatter(route[0, 0], route[0, 1], c=colors[i], marker='s', s=100, edgecolors='black')  # Start
            plt.scatter(route[-1, 0], route[-1, 1], c=colors[i], marker='*', s=150, edgecolors='black')  # End

        # Set plot properties
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'POI, Base Stations, and UAV Routes\n({n_poi} POIs, {n_bs} BS, {n_uavs} UAVs, Map: {int(map_width)}x{int(map_height)})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Save plot
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = file + f"poi_{n_poi}_bs_{n_bs}_map_{int(map_width)}x{int(map_height)}_uav_routes.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
