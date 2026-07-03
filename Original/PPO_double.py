import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 6
np.random.seed(SEED)
torch.manual_seed(SEED)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.masks = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.masks[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim

        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Mish(),
                nn.Linear(128, 128),
                nn.Mish(),
                nn.Linear(128, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Mish(),
                nn.Linear(128, 128),
                nn.Mish(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask=None, deter_action=None):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

            if deter_action is not None:
                action = torch.tensor(deter_action, dtype=torch.float32).to(device)
                if action.dim() == 0:
                    action = action.unsqueeze(0)
            else:
                action = dist.sample()

            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)

            return action.detach(), action_logprob.detach(), state_val.detach()
        else:
            action_probs = self.actor(state)

            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
                if mask.shape != (self.action_dim,):
                    raise ValueError(f"Mask shape {mask.shape} does not match action_dim {self.action_dim}")
                if mask.sum() == 0:
                    raise ValueError("Mask cannot be all zeros (no valid actions)")

                masked_probs = action_probs * mask
                masked_probs = masked_probs + 1e-25 * (1 - mask)
                masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            else:
                masked_probs = action_probs

            if deter_action is not None:
                dist = Categorical(probs=masked_probs)
                action = torch.tensor(deter_action, dtype=torch.long).to(device)
                action_logprob = dist.log_prob(action)
                state_val = self.critic(state)
            else:
                dist = Categorical(probs=masked_probs)
                action = dist.sample()
                action_logprob = dist.log_prob(action)
                state_val = self.critic(state)

            return action.detach(), action_logprob.detach(), state_val.detach()

    def act_test(self, state, mask=None):
        with torch.no_grad():
            if self.has_continuous_action_space:
                action_mean = self.actor(state)
                return action_mean.cpu().numpy()
            else:
                action_probs = self.actor(state)

                if mask is not None:
                    mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
                    if mask.shape != (self.action_dim,):
                        raise ValueError(f"Mask shape {mask.shape} does not match action_dim {self.action_dim}")
                    if mask.sum() == 0:
                        raise ValueError("Mask cannot be all zeros (no valid actions)")

                    masked_probs = action_probs.masked_fill(mask == 0, float('-inf'))
                else:
                    masked_probs = action_probs

                max_prob_action = masked_probs.argmax(dim=-1)
                return max_prob_action.item()

    def evaluate(self, state, action, mask=None):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

            if action.dim() == 1:
                action = action.unsqueeze(-1)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_values = self.critic(state)

            return action_logprobs, state_values, dist_entropy
        else:
            action_probs = self.actor(state)

            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
                if mask.sum() == 0:
                    raise ValueError("Mask cannot be all zeros (no valid actions)")

                masked_probs = action_probs * mask
                masked_probs = masked_probs + 1e-8 * (1 - mask)
                masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            else:
                masked_probs = action_probs

            dist = Categorical(probs=masked_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_values = self.critic(state)

            return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, agent_id, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, summaryWriter, entropy_ratio, gae_lambda, gae_flag, n_step_td,
                 action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_ratio = entropy_ratio
        self.gae_lambda = gae_lambda
        self.gae_flag = gae_flag
        self.n_step_td = n_step_td

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.writer = summaryWriter
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.update_times = 0

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            self.set_action_std(self.action_std)

    def select_action(self, state, mask=None, deter_action=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state, mask, deter_action)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def action_test(self, state, mask=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action = self.policy_old.act_test(state, mask)
        return action

    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        print_reward = 0
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            print_reward += reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        len_batch = len(rewards)

        masks = torch.squeeze(torch.stack(self.buffer.masks[:len_batch], dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(self.buffer.states[:len_batch], dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions[:len_batch], dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:len_batch], dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values[:len_batch], dim=0)).detach().to(device)

        if self.gae_flag:
            advantages = self.compute_gae(self.buffer.rewards, old_state_values, self.buffer.is_terminals)
        else:
            advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, masks)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, rewards).mean()
            entropy_loss = dist_entropy.mean()
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values,
                                                                 rewards) - self.entropy_ratio * dist_entropy

            if self.id == 0:
                layer_type = "upper" if not self.has_continuous_action_space else "lower"
                self.writer.add_scalar(f'loss/policy_{layer_type}', policy_loss.detach().item(), self.update_times)
                self.writer.add_scalar(f'loss/critic_{layer_type}', critic_loss.detach().item(), self.update_times)
                self.writer.add_scalar(f'stats/critic_{layer_type}', state_values.mean(), self.update_times)
                self.writer.add_scalar(f'stats/entropy_{layer_type}', entropy_loss.detach().item(), self.update_times)
                self.writer.add_scalar(f'reward/train_{layer_type}', print_reward, self.update_times)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.update_times += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def compute_gae(self, rewards, state_values, is_terminals):
        gae = 0
        advantages = []
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(is_terminals, dtype=torch.float32).to(device)
        state_values = torch.squeeze(state_values).detach()

        for t in reversed(range(len(rewards))):
            if is_terminals[t]:
                delta = rewards[t] - state_values[t]
                gae = delta
            else:
                # 修复索引越界问题
                if t < len(rewards) - 1:
                    next_value = state_values[t + 1]
                else:
                    next_value = 0

                delta = rewards[t] + self.gamma * next_value - state_values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        return advantages


class LowerPPO:
    def __init__(self, agent_id, state_dim_lower, min_speed, max_speed, lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip, summaryWriter, entropy_ratio,
                 gae_lambda, gae_flag, n_step_td, action_std_init=0.6):
        self.has_continuous_action_space = True
        self.id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_ratio = entropy_ratio
        self.gae_lambda = gae_lambda
        self.gae_flag = gae_flag
        self.n_step_td = n_step_td
        self.action_std = action_std_init

        self.state_dim_lower = state_dim_lower
        self.action_dim_lower = 1
        self.min_speed = min_speed
        self.max_speed = max_speed

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(self.state_dim_lower, self.action_dim_lower, self.has_continuous_action_space,
                                  action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.writer = summaryWriter
        self.policy_old = ActorCritic(self.state_dim_lower, self.action_dim_lower, self.has_continuous_action_space,
                                      action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.update_times = 0

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def select_action(self, state_lower, deter_action=None):
        with torch.no_grad():
            state_lower = torch.FloatTensor(state_lower).to(device)
            action, action_logprob, state_val = self.policy_old.act(state_lower, deter_action=deter_action)

        self.buffer.states.append(state_lower)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if isinstance(action, torch.Tensor):
            if action.dim() > 0:
                speed = self._scale_action(action.item())
            else:
                speed = self._scale_action(action.item())
        else:
            speed = self._scale_action(action)
        return speed

    def action_test(self, state_lower, mask=None):
        with torch.no_grad():
            state_lower = torch.FloatTensor(state_lower).to(device)
            action = self.policy_old.act_test(state_lower)

        if isinstance(action, np.ndarray):
            if action.size > 1:
                speed = self._scale_action(action[0])
            else:
                speed = self._scale_action(action.item())
        else:
            speed = self._scale_action(action)
        return speed

    def _scale_action(self, action):
        """将动作从[-1,1]缩放到[min_speed, max_speed]"""
        action = np.clip(action, -1.0, 1.0)
        speed = self.min_speed + (action + 1.0) * (self.max_speed - self.min_speed) / 2.0
        return np.clip(speed, self.min_speed, self.max_speed)

    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        print_reward = 0
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)
        len_batch = len(rewards)

        old_states = torch.squeeze(torch.stack(self.buffer.states[:len_batch], dim=0)).detach().to(device)
        old_actions = torch.stack(self.buffer.actions[:len_batch], dim=0).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:len_batch], dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values[:len_batch], dim=0)).detach().to(device)

        if old_actions.dim() == 3 and old_actions.size(1) == 1:
            old_actions = old_actions.squeeze(1)
        elif old_actions.dim() == 1:
            old_actions = old_actions.unsqueeze(-1)

        advantages = self.compute_n_step_td_advantages(rewards, old_state_values, is_terminals, self.n_step_td)
        targets = self.compute_td_targets(rewards, old_state_values, is_terminals)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.detach()

            policy_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, targets.detach()).mean()
            entropy_loss = dist_entropy.mean()

            loss = policy_loss + 0.5 * critic_loss - self.entropy_ratio * entropy_loss

            if self.id == 0:
                self.writer.add_scalar('loss/policy_lower', policy_loss.detach().item(), self.update_times)
                self.writer.add_scalar('loss/critic_lower', critic_loss.detach().item(), self.update_times)
                self.writer.add_scalar('stats/critic_lower', state_values.mean(), self.update_times)
                self.writer.add_scalar('stats/entropy_lower', entropy_loss.detach().item(), self.update_times)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.update_times += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def compute_td_targets(self, rewards, state_values, is_terminals):
        targets = []
        for t in range(len(rewards)):
            if t == len(rewards) - 1 or is_terminals[t]:
                target = rewards[t]
            else:
                target = rewards[t] + self.gamma * state_values[t + 1]
            targets.append(target)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        return targets

    def compute_n_step_td_advantages(self, rewards, state_values, is_terminals, n_steps=3):
        advantages = []
        for t in range(len(rewards)):
            td_error = 0
            discount = 1

            for k in range(min(n_steps, len(rewards) - t)):
                if t + k < len(rewards):
                    td_error += discount * rewards[t + k]
                    discount *= self.gamma
                    if is_terminals[t + k]:
                        break

            if t + n_steps < len(rewards) and not any(is_terminals[t:t + n_steps]):
                td_error += discount * state_values[t + n_steps]

            td_error -= state_values[t]
            advantages.append(td_error)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages


class PPO_Hierarchical:
    def __init__(self, agent_id, state_dim, action_dim, state_dim_lower, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, summary_dir, entropy_ratio_upper, entropy_ratio_lower, gae_lambda,
                 gae_flag, min_speed, max_speed, n_step_td_upper, n_step_td_lower, action_std_init=0.6):

        if agent_id == 0:
            self.summary_dir = summary_dir
            self.writer = SummaryWriter(log_dir=self.summary_dir)
        else:
            self.writer = None

        self.agent_id = agent_id

        # 直接联合训练，不分阶段
        self.upper_ppo = PPO(agent_id, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                             has_continuous_action_space, self.writer, entropy_ratio_upper, gae_lambda, gae_flag,
                             n_step_td_upper, action_std_init)
        self.lower_ppo = LowerPPO(agent_id, state_dim_lower, min_speed, max_speed, lr_actor, lr_critic, gamma,
                                  K_epochs, eps_clip, self.writer, entropy_ratio_lower, gae_lambda, gae_flag,
                                  n_step_td_lower, action_std_init)

        self.env = None

    def set_env(self, env):
        self.env = env

    def set_action_std(self, new_action_std):
        """设置动作标准差（仅对下层连续动作有效）"""
        self.lower_ppo.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """衰减动作标准差"""
        current_std = self.lower_ppo.action_std
        new_std = current_std - action_std_decay_rate
        new_std = round(new_std, 4)
        if new_std <= min_action_std:
            new_std = min_action_std
        self.set_action_std(new_std)

    def select_action(self, state, mask, uav_id, deter_action=None):
        # 上层选择目标
        if deter_action is not None:
            action_upper = self.upper_ppo.select_action(state, mask, deter_action)
        else:
            action_upper = self.upper_ppo.select_action(state, mask)

        # 下层选择速度
        state_lower = self.env.get_lower_obs(uav_id, action_upper)
        action_lower = self.lower_ppo.select_action(state_lower)

        return [action_upper, action_lower]

    def action_test(self, state, mask, uav_id):
        action_upper = self.upper_ppo.action_test(state, mask)
        state_lower = self.env.get_lower_obs(uav_id, action_upper)
        action_lower = self.lower_ppo.action_test(state_lower)
        return [action_upper, action_lower]

    def update(self):
        """同时更新上层和下层"""
        self.upper_ppo.update()
        self.lower_ppo.update()

    def add_reward_to_buffer(self, reward, is_terminal):
        """向两个buffer添加reward和terminal信息"""
        self.upper_ppo.buffer.rewards.append(reward)
        self.upper_ppo.buffer.is_terminals.append(is_terminal)
        self.lower_ppo.buffer.rewards.append(reward)
        self.lower_ppo.buffer.is_terminals.append(is_terminal)

    def add_mask_to_buffer(self, mask):
        """向上层buffer添加mask信息"""
        self.upper_ppo.buffer.masks.append(torch.FloatTensor(mask).to(device))

    def save(self, checkpoint_path):
        torch.save({
            'upper_policy_old_state_dict': self.upper_ppo.policy_old.state_dict(),
            'lower_policy_old_state_dict': self.lower_ppo.policy_old.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.upper_ppo.policy_old.load_state_dict(checkpoint['upper_policy_old_state_dict'])
        self.lower_ppo.policy_old.load_state_dict(checkpoint['lower_policy_old_state_dict'])
        self.upper_ppo.policy.load_state_dict(checkpoint['upper_policy_old_state_dict'])
        self.lower_ppo.policy.load_state_dict(checkpoint['lower_policy_old_state_dict'])
        print(f"模型成功加载: {checkpoint_path}")

    def call_2_record(self, name, steps, value):
        if self.writer:
            self.writer.add_scalar(name, value, steps)