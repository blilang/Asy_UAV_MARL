import gym
import numpy as np
from gym import spaces
from scipy.spatial.distance import euclidean


class CommunicationModel:
    def __init__(self):
        """
        无人机通信模型参数设置
        基于文档中的通信模型，考虑视距(LoS)和非视距(NLoS)通信
        """
        # === 基本环境参数 ===
        self.H = 100.0  # 无人机飞行高度 (m)

        # === 信道模型参数 ===
        self.a = 10.0  # 环境参数a
        self.b = 0.6  # 环境参数b
        self.alpha = 2.2  # 路径损耗指数
        self.kappa = 0.2  # 非视距通信附加衰减因子
        self.rho0_dB = -60  # 参考距离1m时的信道增益 (dB)
        self.rho0 = 10 ** (self.rho0_dB / 10)  # 转换为线性值

        # === 噪声功率 ===
        self.sigma2_uav_dB = -110  # 无人机接收端噪声功率 (dBm)
        self.sigma2_station_dB = -110  # 基站接收端噪声功率 (dBm)
        self.sigma2_uav = 10 ** (self.sigma2_uav_dB / 10) * 1e-3  # 转换为W
        self.sigma2_station = 10 ** (self.sigma2_station_dB / 10) * 1e-3  # 转换为W

        # === 传输功率 ===
        self.G_device_dB = 20  # 地面设备传输功率 (dBm)
        self.G_station_dB = 40  # 无人机到基站传输功率 (dBm)
        self.G_device = 10 ** (self.G_device_dB / 10) * 1e-3  # 转换为W
        self.G_station = 10 ** (self.G_station_dB / 10) * 1e-3  # 转换为W

        # === 其他参数 ===
        self.Gamma_dB = 8.2  # 调制编码损失 (dB)
        self.Gamma = 10 ** (self.Gamma_dB / 10)
        self.B = 1e6  # 通信带宽 (Hz) = 1 MHz
        self.SNR_min_dB = 16  # 最小信噪比要求 (dB)
        self.SNR_min = 10 ** (self.SNR_min_dB / 10)

        # === 数据相关参数 ===
        self.D_device = 5e7  # 单个设备数据大小 (bits) = 10 Mbits

    def calculate_distance_3d(self, uav_pos, target_pos):
        """计算无人机与目标之间的3D距离"""
        horizontal_dist = np.linalg.norm(np.array(uav_pos) - np.array(target_pos))
        return np.sqrt(horizontal_dist ** 2 + self.H ** 2)

    def calculate_elevation_angle(self, distance_3d):
        """计算仰角（度）"""
        return (180 / np.pi) * np.arcsin(self.H / distance_3d)

    def calculate_los_probability(self, elevation_angle):
        """计算视距传输概率"""
        return 1 / (1 + self.a * np.exp(-self.b * (elevation_angle - self.a)))

    def calculate_path_loss(self, distance_3d, is_los=True):
        """计算路径损耗"""
        if is_los:
            return self.rho0 * (distance_3d ** (-self.alpha))
        else:
            return self.kappa * self.rho0 * (distance_3d ** (-self.alpha))

    def calculate_average_channel_gain(self, distance_3d):
        """计算平均信道增益"""
        elevation_angle = self.calculate_elevation_angle(distance_3d)
        p_los = self.calculate_los_probability(elevation_angle)

        # 计算视距和非视距路径损耗
        path_loss_los = self.calculate_path_loss(distance_3d, is_los=True)
        path_loss_nlos = self.calculate_path_loss(distance_3d, is_los=False)

        # 平均信道增益
        avg_gain = p_los * path_loss_los + (1 - p_los) * path_loss_nlos
        return avg_gain

    def calculate_snr_uplink(self, uav_pos, device_pos):
        """计算上行链路SNR（设备到无人机）"""
        distance_3d = self.calculate_distance_3d(uav_pos, device_pos)
        avg_gain = self.calculate_average_channel_gain(distance_3d)

        # SNR计算
        snr_linear = (self.G_device * avg_gain) / (self.sigma2_uav * self.Gamma)
        snr_dB = 10 * np.log10(snr_linear)

        return snr_linear, snr_dB

    def calculate_snr_downlink(self, uav_pos, station_pos):
        """计算下行链路SNR（无人机到基站）"""
        distance_3d = self.calculate_distance_3d(uav_pos, station_pos)
        avg_gain = self.calculate_average_channel_gain(distance_3d)

        # SNR计算
        snr_linear = (self.G_station * avg_gain) / (self.sigma2_station * self.Gamma)
        snr_dB = 10 * np.log10(snr_linear)

        return snr_linear, snr_dB

    def calculate_data_rate(self, snr_linear):
        """计算数据传输速率"""
        return self.B * np.log2(1 + snr_linear)

    def calculate_transmission_time(self, data_size, data_rate):
        """计算传输时间"""
        return data_size / data_rate

    def get_data_collection_time(self, uav_pos, poi_pos):
        snr_linear, snr_dB = self.calculate_snr_uplink(uav_pos, poi_pos)
        data_rate = self.calculate_data_rate(snr_linear)
        transmission_time = self.calculate_transmission_time(self.D_device, data_rate)

        return transmission_time

    def get_data_offload_time(self, uav_pos, bs_pos, data_amount):
        snr_linear, snr_dB = self.calculate_snr_downlink(uav_pos, bs_pos)
        data_rate = self.calculate_data_rate(snr_linear)
        transmission_time = self.calculate_transmission_time(data_amount, data_rate)

        return transmission_time

    def check_communication_feasibility(self, uav_pos, target_pos, link_type='uplink'):
        if link_type == 'uplink':
            snr_linear, _ = self.calculate_snr_uplink(uav_pos, target_pos)
        else:
            snr_linear, _ = self.calculate_snr_downlink(uav_pos, target_pos)

        return snr_linear >= self.SNR_min


class MultiDroneAoIEnv(gym.Env):
    def __init__(
            self,
            M=3,
            N=2,
            K=5,
            T=1000.0,
            map_size=1000.0,
            drone_speeds=None,
            sense_times=None,
            args=None
    ):
        super(MultiDroneAoIEnv, self).__init__()
        self.M = M
        self.N = N
        self.K = K
        self.T = T
        self.map_size = map_size
        self.args = args
        self.max_hover_radius = args.max_hover_radius

        # 初始化通信模型
        self.comm_model = CommunicationModel()

        # 数据文件
        position_filename = f"./data/poi_{self.K}_bs_{self.N}_map_{int(self.map_size)}x{int(self.map_size)}.npy"
        data = np.load(position_filename, allow_pickle=True).item()
        self.sensor_pos = data['poi_positions']
        self.base_pos = data['bs_positions']
        self.weights = data['weights']

        if hasattr(args, 'uav_energies') and args.uav_energies:
            self.uav_energies = args.uav_energies[:M]
        else:
            self.uav_energies = [args.Energy] * M

        if hasattr(args, 'uav_max_speeds') and args.uav_max_speeds:
            self.uav_max_speeds = args.uav_max_speeds[:M]
        else:
            self.uav_max_speeds = [args.max_speed] * M

        while len(self.uav_energies) < M:
            self.uav_energies.append(args.Energy)
        while len(self.uav_max_speeds) < M:
            self.uav_max_speeds.append(args.max_speed)

        # 全局信息
        self.aoi = np.zeros(K)
        self.global_timing = 0
        self.drone_last_visited_history_at_BS = np.zeros((M, K))
        self.drone_visited_history_timing_at_BS = np.zeros((M, K))
        self.aoi_gain_total = 0
        self.drone_alive = np.ones(M)

        # 局部信息
        self.drone_buffer = [[] for _ in range(M)]
        self.buffer_timing = [[] for _ in range(M)]
        self.drone_timing_now = np.zeros(M)
        self.drone_local_aoi = np.zeros((M, K))
        self.drone_last_reward = np.zeros(M)
        self.drone_step_reward = [[] for _ in range(M)]
        # 无人机位置
        self.drone_position_now = np.zeros((self.M, 2))
        # 无人机的能量
        self.drone_energys = np.array(self.uav_energies, dtype=np.float32)  # 能量记录（会变）
        self.initial_energys = self.drone_energys.copy()  # 初始能量（不变）
        self.drone_aoi_gain = [[] for i in range(M)]  # 记录每个UAV对信息年龄增益的贡献度
        # 通信传输时间
        self.step_transmission_times = [[] for _ in range(M)]

        # 上层动作空间与观测空间
        self.action_space = spaces.Discrete(K + N)
        obs_dim = 4*K + M*K + 2*M + N + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 下层观测空间
        self.speed_range = max(self.uav_max_speeds) - self.args.min_speed + 1
        low_obs_dim = K + 3 * self.speed_range + 7 + M
        self.lower_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(low_obs_dim,), dtype=np.float32)
        self.bs_random_factor = self.args.BS_back_times

        # 初始化环境
        self.reset()

    def calculate_hover_position(self, target_pos, angle, normalized_distance):
        """
        根据角度和归一化距离计算悬停位置
        angle: 0 到 2π
        normalized_distance: 0 到 1，1代表距离目标水平距离30米
        """
        actual_distance = normalized_distance * self.max_hover_radius
        hover_x = target_pos[0] + actual_distance * np.cos(angle)
        hover_y = target_pos[1] + actual_distance * np.sin(angle)

        # 确保悬停位置在地图范围内
        hover_x = np.clip(hover_x, 0, self.map_size)
        hover_y = np.clip(hover_y, 0, self.map_size)

        return np.array([hover_x, hover_y])

    def reset(self):
        """重置环境"""
        # 全局信息
        self.aoi = np.zeros(self.K)
        self.global_timing = 0
        self.drone_last_visited_history_at_BS = np.zeros((self.M, self.K))
        self.drone_visited_history_timing_at_BS = np.zeros((self.M, self.K))
        self.aoi_gain_total = 0
        self.drone_alive = np.ones(self.M)

        # 局部信息
        self.drone_buffer = [[] for _ in range(self.M)]
        self.buffer_timing = [[] for _ in range(self.M)]
        self.drone_timing_now = np.zeros(self.M)
        self.drone_local_aoi = np.zeros((self.M, self.K))
        self.drone_last_reward = np.zeros(self.M)
        self.drone_step_reward = [[] for i in range(self.M)]
        # 无人机位置
        self.drone_position_now = np.zeros((self.M, 2))
        offsets = np.array([[10, 0], [-10, 0], [0, 10], [0, -10]])
        self.drone_position_now = np.zeros((self.M, 2))
        for i in range(self.M):
            self.drone_position_now[i] = self.base_pos[0] + offsets[i % len(offsets)]
        # 无人机能量
        self.drone_energys = np.array(self.uav_energies, dtype=np.float32)
        self.initial_energys = self.drone_energys.copy()
        self.drone_aoi_gain = [[] for i in range(self.M)]
        # 通信传输时间
        self.step_transmission_times = [[] for _ in range(self.M)]

        return self._get_obs(0)

    def _get_obs(self, uav_id):
        """获取指定无人机的观测"""
        aoi_obs = self.drone_local_aoi[uav_id].copy() * 10 / self.T

        # 假设速度
        fly_speed = 18.0
        dis_2_pois = np.linalg.norm(self.sensor_pos - self.drone_position_now[uav_id], axis=-1)
        move_t_2_pois = dis_2_pois / fly_speed
        next_aoi_2_poi = self.drone_local_aoi[uav_id] + (move_t_2_pois * self.weights)  # 权重
        next_time_2_poi = move_t_2_pois + self.drone_timing_now[uav_id]
        mask = np.ones(self.K)
        mask[self.drone_buffer[uav_id]] = 0
        rewards_2_pois = next_aoi_2_poi * (self.T - next_time_2_poi) / (2 * self.args.pre_reward_ratio) * mask
        rewards_2_pois /= self.args.reward_scale_size
        rewards_2_pois *= 60
        rewards_2_pois /= (move_t_2_pois + 1)

        rewards_2_bs = 0
        for i in range(len(self.drone_buffer[uav_id])):
            coll_timing = self.buffer_timing[uav_id][i]
            target_poi = self.drone_buffer[uav_id][i]
            curr_aoi = (self.drone_timing_now[uav_id] - coll_timing) * self.weights[target_poi]  # 权重
            if self.T > self.drone_timing_now[uav_id]:
                rewards_2_bs += (self.drone_local_aoi[uav_id, target_poi] - curr_aoi) * (
                    self.T - self.drone_timing_now[uav_id]) / 2
        rewards_2_bs -= sum(self.drone_step_reward[uav_id])
        rewards_2_bs = np.array([rewards_2_bs]) * 4 / self.args.reward_scale_size

        buffer_obs = np.zeros(self.K)
        for poi in self.drone_buffer[uav_id]:
            buffer_obs[poi] = 1

        # 历史缓存时间信息
        alive_info = self.drone_alive
        temp = self.drone_visited_history_timing_at_BS.copy()
        temp[self.drone_alive == 0] = -1        # 退出服务的无人机历史记录置-1
        temp[self.drone_alive == 1] /= self.T
        history_visited_sensors_and_timing = temp.flatten()

        # 时间信息
        time_obs = np.array([self.drone_timing_now[uav_id]]) / self.T

        # 缓存大小
        buffer_len = np.array([len(self.drone_step_reward[uav_id])])

        # 剩余能量（归一化使用各自的初始能量）
        rest_energy = np.array([self.drone_energys[uav_id] / self.initial_energys[uav_id]])

        # 注意需要进行归一化
        return np.concatenate([
            aoi_obs,  # K
            dis_2_pois / (self.map_size / 2),  # K
            rewards_2_pois,  # K
            rewards_2_bs,  # N
            buffer_obs,  # K
            time_obs,  # 1
            alive_info,  # M
            history_visited_sensors_and_timing,  # M*K
            self.drone_last_reward / self.args.reward_scale_size,  # M
            buffer_len,  # 1
            rest_energy,  # 1
        ])

    def get_lower_obs(self, uav_id, target_action):
        # 1. 基本信息
        aoi_info = self.drone_local_aoi[uav_id].copy() * 10 / self.T
        time_progress = np.array([self.drone_timing_now[uav_id] / self.T])
        energy_ratio = np.array([self.drone_energys[uav_id] / self.initial_energys[uav_id]])
        target_info = np.array([target_action])

        # 2. 目标相关信息
        if target_action < self.K:
            target_position = self.sensor_pos[target_action]
            is_poi = np.array([1.0])
        else:
            target_position = self.base_pos[target_action - self.K]
            is_poi = np.array([0.0])
        distance_to_target = np.array([euclidean(target_position, self.drone_position_now[uav_id])])
        distance_normalized = distance_to_target / (self.map_size / 2)

        # 3. 速度相关的关键特征 - 使用当前无人机的最大速度
        current_max_speed = self.uav_max_speeds[uav_id]
        test_speeds = list(range(self.args.min_speed, current_max_speed + 1))

        # 为了保持观测空间维度一致，我们需要填充到最大可能的速度范围
        max_speed_range = max(self.uav_max_speeds) - self.args.min_speed + 1
        time_costs = []
        energy_costs = []
        reward_estimates = []

        current_remaining_time = max(0, self.T - self.drone_timing_now[uav_id])
        current_energy = self.drone_energys[uav_id]

        for i in range(max_speed_range):
            if i < len(test_speeds):
                speed = test_speeds[i]
                time_cost = distance_to_target[0] / speed
                energy_cost = UAV_Energy(speed) * time_cost
                time_costs.append(time_cost / self.T)  # 归一化时间成本
                energy_costs.append(energy_cost / self.initial_energys[uav_id])  # 归一化能量成本

                if target_action < self.K:  # 访问设备
                    aoi_of_target = self.drone_local_aoi[uav_id, target_action]
                    future_aoi = aoi_of_target + time_cost * self.weights[target_action]
                    remaining_time_after = current_remaining_time - time_cost
                    potential_reward = future_aoi * max(remaining_time_after, 0) / (
                            2 * self.args.pre_reward_ratio * self.args.reward_scale_size)
                    remaining_energy_after = current_energy - energy_cost
                    min_return_energy = float('inf')
                    for bs_idx in range(self.N):
                        bs_pos = self.base_pos[bs_idx]
                        dist_to_bs = euclidean(target_position, bs_pos)  # 从目标位置到基站的距离
                        return_energy = UAV_Energy(15.0) * (dist_to_bs / 15.0)
                        min_return_energy = min(min_return_energy, return_energy)
                    if remaining_energy_after >= min_return_energy:
                        reward_estimate = potential_reward
                    else:
                        energy_deficit_ratio = (min_return_energy - remaining_energy_after) / self.initial_energys[
                            uav_id]
                        reward_estimate = potential_reward * (1 - energy_deficit_ratio * 2)
                else:  # 访问基站
                    rewards_2_bs = 0
                    for j in range(len(self.drone_buffer[uav_id])):
                        coll_timing = self.buffer_timing[uav_id][j]
                        target_poi = self.drone_buffer[uav_id][j]
                        curr_aoi = (self.drone_timing_now[uav_id] - coll_timing) * self.weights[target_poi]

                        rewards_2_bs += (self.drone_local_aoi[uav_id, target_poi] - curr_aoi) * (
                                self.T - self.drone_timing_now[uav_id] - time_cost) / 2
                    rewards_2_bs -= sum(self.drone_step_reward[uav_id])
                    reward_estimate = rewards_2_bs * 5 / self.args.reward_scale_size
                reward_estimates.append(reward_estimate)
            else:
                # 填充无效速度的默认值
                time_costs.append(0.0)
                energy_costs.append(0.0)
                reward_estimates.append(0.0)

        # 4. 缓存相关信息 (对速度决策很重要)
        buffer_size = np.array([len(self.drone_buffer[uav_id]) / self.K])  # 归一化缓存大小
        buffer_value = np.array([sum(self.drone_step_reward[uav_id]) / self.args.reward_scale_size])  # 缓存价值

        # 增添信息
        alive_info = self.drone_alive

        obs = np.concatenate([
            aoi_info,  # K
            time_progress,  # 1
            energy_ratio,  # 1
            target_info,  # 1
            is_poi,  # 1
            distance_normalized,  # 1
            time_costs,  # max_speed_range
            energy_costs,  # max_speed_range
            reward_estimates,  # max_speed_range
            buffer_size,  # 1
            buffer_value,  # 1
            alive_info,     # M
        ])
        return obs

    def step(self, uav_id, action):
        fly_speed = action[1]
        action = int(action[0])

        assert 0 <= uav_id < self.M, f"Invalid uav_id: {uav_id}"
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 确保速度不超过该无人机的最大速度
        fly_speed = min(fly_speed, self.uav_max_speeds[uav_id])

        if self.K <= action <= self.K + self.N - 1:
            # 前往基站
            target_position = self.base_pos[action - self.K]
            hover_position = self.calculate_hover_position(target_position, 0.0, 0.0)
            move_dis = euclidean(hover_position, self.drone_position_now[uav_id])
            move_time = move_dis / fly_speed

            # 根据计算结果更新（位置、时间、能量）
            self.drone_position_now[uav_id] = hover_position
            self.drone_timing_now[uav_id] += move_time
            energy_move_cost = move_time * UAV_Energy(fly_speed)
            self.drone_energys[uav_id] -= energy_move_cost

            for i in range(self.K):
                self.drone_local_aoi[uav_id, i] += move_time * self.weights[i]
                self.aoi[i] += (self.drone_timing_now[uav_id] - self.global_timing) * self.weights[i]

            self.global_timing = self.drone_timing_now[uav_id]  # 更新飞行阶段后的时间

            reward = 0
            if self.drone_buffer[uav_id]:
                total_data_amount = len(self.drone_buffer[uav_id]) * self.comm_model.D_device

                offload_time = self.comm_model.get_data_offload_time(hover_position, target_position, total_data_amount) / 2
                # print("数据卸载时间：", offload_time)
                energy_hover_cost = offload_time * UAV_Energy(0)
                self.drone_energys[uav_id] -= energy_hover_cost
                self.step_transmission_times[uav_id].append(offload_time)
                self.drone_timing_now[uav_id] += offload_time
                self.global_timing += offload_time

                for i in range(self.K):
                    self.drone_local_aoi[uav_id, i] += offload_time * self.weights[i]
                    self.aoi[i] += offload_time * self.weights[i]

                # 进行AOI更新
                for i in range(len(self.drone_buffer[uav_id])):
                    coll_timing = self.buffer_timing[uav_id][i]
                    target_poi = self.drone_buffer[uav_id][i]
                    curr_aoi = (self.drone_timing_now[uav_id] - coll_timing) * self.weights[target_poi]

                    aoi_decline_gain = max(
                        (min(self.aoi[target_poi], self.drone_local_aoi[uav_id, target_poi]) - curr_aoi),
                        0) * (self.T - self.drone_timing_now[uav_id])

                    reward += aoi_decline_gain / 2

                    self.aoi_gain_total += aoi_decline_gain
                    self.drone_aoi_gain[uav_id].append(aoi_decline_gain)
                    self.aoi[target_poi] = min(curr_aoi, self.aoi[target_poi])

                self.drone_local_aoi[uav_id] = self.aoi      # 与环境进行同步
                self.drone_visited_history_timing_at_BS[uav_id] = np.zeros(self.K)
                for i in range(len(self.drone_buffer[uav_id])):
                    self.drone_visited_history_timing_at_BS[uav_id, self.drone_buffer[uav_id][i]] = self.buffer_timing[uav_id][i]

                self.drone_last_reward[uav_id] = reward

                step_reward = np.array(self.drone_step_reward[uav_id])
                negative_reward_indx = np.argwhere(step_reward <= 0)
                step_reward[negative_reward_indx] *= -1
                reward -= sum(step_reward)
                reward -= (energy_move_cost / self.initial_energys[uav_id]) * self.args.reward_scale_size  # 考虑能耗

            # 清空buffer
            self.drone_step_reward[uav_id] = []
            self.drone_buffer[uav_id] = []
            self.buffer_timing[uav_id] = []

        else:
            # 前往POI收集数据
            target_poi = int(action)
            target_position = self.sensor_pos[target_poi]
            hover_position = self.calculate_hover_position(target_position, 0.0, 0.0)
            move_dis = euclidean(hover_position, self.drone_position_now[uav_id])
            move_time = move_dis / fly_speed

            # 飞行时间与能耗
            self.drone_position_now[uav_id] = hover_position
            energy_move_cost = move_time * UAV_Energy(fly_speed)
            self.drone_energys[uav_id] -= energy_move_cost

            # 收集时间与能耗
            data_collection_time = self.comm_model.get_data_collection_time(hover_position, target_position)
            # print("数据收集时间", data_collection_time)
            energy_hover_cost = data_collection_time * UAV_Energy(0)
            self.drone_energys[uav_id] -= energy_hover_cost
            self.step_transmission_times[uav_id].append(data_collection_time)
            total_time_cost = move_time + data_collection_time
            self.drone_timing_now[uav_id] += total_time_cost

            # 将数据添加到缓存
            self.drone_buffer[uav_id].append(target_poi)
            self.buffer_timing[uav_id].append(self.drone_timing_now[uav_id])  # 收集完成的时刻开始算信息年龄
            # 更新AoI（无论数据收集成功与否，时间都在流逝）
            for i in range(self.K):
                self.drone_local_aoi[uav_id, i] += total_time_cost * self.weights[i]

            reward = self.drone_local_aoi[uav_id, target_poi] * (self.T - self.drone_timing_now[uav_id]) / (
                    2 * self.args.pre_reward_ratio)
            reward -= (energy_move_cost / self.initial_energys[uav_id]) * self.args.reward_scale_size  # 考虑能耗
            if self.args.buffer_punishment:
                if len(self.drone_step_reward[uav_id]) > self.bs_random_factor:
                    reward = -self.args.punishment_value * self.args.reward_scale_size

            self.drone_step_reward[uav_id].append(reward)

        done = False
        # 能量或者时间很少的时候，也可以为True
        if self.drone_timing_now[uav_id] >= self.T or self.drone_energys[uav_id] <= 10:
            done = True
            self.drone_alive[uav_id] = 0
            self.drone_visited_history_timing_at_BS[uav_id] = -1
            self.drone_last_visited_history_at_BS[uav_id] = -1

        # !之后加上mask
        info = np.ones(self.K + self.N)
        info[self.drone_buffer[uav_id]] = 0

        if len(self.drone_buffer[uav_id]) == 0:
            info[self.K:self.K + self.N] = 0

        if self.drone_timing_now[uav_id] >= self.T and action < self.K:  # 未回归基站的惩罚
            reward = -self.args.punishment_value_2 * self.args.reward_scale_size

        if self.drone_timing_now[uav_id] >= self.T and action >= self.K:  # 回到基站但是超过给定时间了，还是无法更新
            reward = 0.0

        if self.drone_energys[uav_id] <= 10 and action < self.K:  # 未回归基站的惩罚
            reward = -self.args.punishment_value_2 * self.args.reward_scale_size

        if self.args.print_info:
            print(
                "Action UAV: {}, action: {}, time now: {}, time cost: {}, local AoI: {}, position: {}, buffer: {}".format(
                    uav_id, action, self.drone_timing_now[uav_id], move_time, self.drone_local_aoi[uav_id],
                    self.drone_position_now[uav_id], self.drone_buffer[uav_id]))

        return self._get_obs(uav_id), reward / self.args.reward_scale_size, done, info

    def time_cost(self, uav_id, action):
        """计算动作的时间成本，现在包括通信时间和悬停位置优化"""
        fly_speed = action[1]
        action = int(action[0])

        # 确保速度不超过该无人机的最大速度
        fly_speed = min(fly_speed, self.uav_max_speeds[uav_id])

        if self.K <= action <= self.K + self.N - 1:
            # 前往基站
            target_position = self.base_pos[action - self.K]
            hover_position = self.calculate_hover_position(target_position, 0.0, 0.0)
            move_dis = euclidean(hover_position, self.drone_position_now[uav_id])
            move_time = move_dis / fly_speed

            # 如果有缓存数据，加上卸载时间
            if self.drone_buffer[uav_id]:
                total_data_amount = len(self.drone_buffer[uav_id]) * self.comm_model.D_device
                offload_time = self.comm_model.get_data_offload_time(hover_position, target_position, total_data_amount)
                move_time += offload_time
        else:
            # 前往POI
            target_poi = action
            target_position = self.sensor_pos[target_poi]
            hover_position = self.calculate_hover_position(target_position, 0.0, 0.0)
            move_dis = euclidean(hover_position, self.drone_position_now[uav_id])
            move_time = move_dis / fly_speed

            # 加上数据收集时间
            data_collection_time = self.comm_model.get_data_collection_time(hover_position, target_position)
            move_time += data_collection_time

        return move_time

    def get_pos(self, action):
        all_positions = np.concatenate([self.sensor_pos, self.base_pos], axis=0)
        return all_positions[action]


def UAV_Energy(v):
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

