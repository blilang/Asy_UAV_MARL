import numpy as np
from Env import MultiDroneAoIEnv
import argparse


class PollingAlgorithm:
    def __init__(self, env, uav_id=0, order = []):
        self.env = env
        self.uav_id = uav_id
        self.K = env.K
        self.N = env.N
        self.T = env.T
        self.reset()
        
        self.count = 0
        self.order = order

    def reset(self):
        self.env.reset()
        self.path = []
        self.total_reward = 0
        self.total_aoi = 0
        self.current_time = 0
        self.steps = 0

    def get_next_poi(self):
        # 优先选择本地 AoI 最高的未缓存 POI
        aoi = self.env.drone_local_aoi[self.uav_id].copy()
        aoi[self.env.drone_buffer[self.uav_id]] = -np.inf  # 屏蔽已缓存 POI
        # 考虑其他无人机的最近访问时间
        for i in range(self.K):
            if self.env.drone_visited_history_timing_at_BS[:, i].max() > 0:
                aoi[i] = min(aoi[i], self.env.drone_timing_now[self.uav_id] - self.env.drone_visited_history_timing_at_BS[:, i].max())
        if np.any(aoi > -np.inf):
            return np.argmax(aoi)
        return None

    def get_nearest_bs(self):
        distances = np.linalg.norm(self.env.base_pos - self.env.drone_position_now[self.uav_id], axis=1)
        return self.K + np.argmin(distances)

    def step(self):
        if self.current_time >= self.T:
            return None, 0, True

        # 如果缓冲区未满且有未缓存 POI，选择 AoI 最高的 POI
        if len(self.env.drone_buffer[self.uav_id]) >= 7:
            action = self.N+self.K-1
        else:
            action = self.order[self.count%len(self.order)]
            self.count+=1

        # 执行动作
        obs, reward, done, info = self.env.step(self.uav_id, action)
        self.current_time = self.env.drone_timing_now[self.uav_id]
        self.total_reward += reward
        self.total_aoi += np.mean(self.env.aoi)
        self.steps += 1
        self.path.append(action)

        return action, reward, done

    def run(self):
        while self.current_time < self.T:
            action, reward, done = self.step()
            if done:
                break
        return self.path, self.total_reward, self.total_aoi / self.steps

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for PPO algorithm on RoboschoolWalker2d-v1")
    
    # Environment hyperparameters
    parser.add_argument('--env_name', type=str, default="A-MAPPO", 
                        help="Name of the gym environment")
    parser.add_argument('--has_continuous_action_space', action='store_true', 
                        help="Whether the action space is continuous (default: discrete)")
    parser.add_argument('--max_ep_len', type=int, default=1000, 
                        help="Maximum timesteps in one episode")
    parser.add_argument('--max_training_timesteps', type=int, default=int(1e7), 
                        help="Maximum total timesteps for training")
    parser.add_argument('--print_freq', type=int, default=None, 
                        help="Print average reward interval (in timesteps, default: max_ep_len * 10)")
    parser.add_argument('--log_freq', type=int, default=None, 
                        help="Log average reward interval (in timesteps, default: max_ep_len * 2)")
    parser.add_argument('--save_model_freq', type=int, default=int(1e5), 
                        help="Model saving frequency (in timesteps)")
    
    # !Continuous action space parameters 不用管
    parser.add_argument('--action_std', type=float, default=0.6, 
                        help="Starting std for action distribution (continuous action space)")
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05, 
                        help="Decay rate for action std")
    parser.add_argument('--min_action_std', type=float, default=0.1, 
                        help="Minimum action std (stop decay when reached)")
    parser.add_argument('--action_std_decay_freq', type=int, default=int(2.5e5), 
                        help="Action std decay frequency (in timesteps)")
    
    # PPO hyperparameters
    parser.add_argument('--update_timestep', type=int, default=None, 
                        help="Policy update frequency (in timesteps, default: max_ep_len * 4)")
    parser.add_argument('--K_epochs', type=int, default=10, 
                        help="Number of epochs for policy update in one PPO update(每次更新使用多少次数据)")
    parser.add_argument('--eps_clip', type=float, default=0.2, 
                        help="Clip parameter for PPO")
    parser.add_argument('--gamma', type=float, default=0.97, 
                        help="Discount factor")
    parser.add_argument('--lr_actor', type=float, default=0.0003, 
                        help="Learning rate for actor network")
    parser.add_argument('--lr_critic', type=float, default=0.001, 
                        help="Learning rate for critic network")
    parser.add_argument('--random_seed', type=int, default=0, 
                        help="Random seed (0 = no random seed)")
    parser.add_argument('--entropy_ratio', type=float, default=float(0.01), 
                        help="Ratio of pre reward")
    parser.add_argument('--gae-lambda', type=float, default=0.97,
                        help='lambda parameter for GAE (default: 1.00)')
    parser.add_argument('--gae_flag', action='store_true',
                        help="Use gae?")
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default="PPO_logs", 
                        help="Directory for log files")
    parser.add_argument('--checkpoint_dir', type=str, default="PPO_preTrained", 
                        help="Directory for model checkpoints")
    parser.add_argument('--run_num_pretrained', type=int, default=0, 
                        help="Run number for pretrained model to prevent overwriting")

    # 环境参数
    parser.add_argument('--M', type=int, default=2, 
                        help="Num of UAVs")
    parser.add_argument('--N', type=int, default=1, 
                        help="Num of Bases")
    parser.add_argument('--K', type=int, default=20, 
                        help="Num of PoIs")
    parser.add_argument('--T', type=int, default=1800,
                        help="Time limits")
    parser.add_argument('--map_size', type=int, default=1000, 
                        help="Map size")
    parser.add_argument('--pre_reward_ratio', type=float, default=float(4), 
                        help="Ratio of pre reward")
    parser.add_argument('--buffer_punishment', action='store_true',
                        help="Whether the action space is continuous (default: discrete)")
    parser.add_argument('--reward_scale_size', type=float, default=float(50000), 
                        help="Ratio of pre reward")
    parser.add_argument('--print_info', action='store_true',
                        help="print_info?")
    
    parser.add_argument('--BS_back_times', type=int, default=10,
                        help="Time limits")
    args = parser.parse_args()
    return args

# 示例
if __name__ == "__main__":
    
    args = parse_args()
    
    order = [[1, 17, 6, 18, 9, 20, 4,3,5,7,20],
              [2,8,12,19,11,20,16,14,10,0,15,13,20]]
    
    env = MultiDroneAoIEnv(args.M, args.N, args.K, args.T, args.map_size, args=args)
    sum_rewards = 0
    for i in range(args.M):
        algo = PollingAlgorithm(env, uav_id=i, order=order[i])
        path, total_reward, avg_aoi = algo.run()
        sum_rewards += total_reward
        print(f"Path: {path}")
    print(f"Total Reward: {sum_rewards}")
    print(f"Average AoI: {avg_aoi}")
    
    visual_num = 30

    # 可视化路径
    # test_actions = [[] for _ in range(env.M)]
    # test_actions[0].append([0, 0])
    # for action in path:
    #     pos = env.get_pos(action)
    #     test_actions[0].append(pos)
    # test_actions = np.array(test_actions)
    # env.visualize_routes(test_actions[:, :visual_num], "./test_figs", file="rolling_", nums=visual_num)