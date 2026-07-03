import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


print("============================================================================================")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def _build_encoder(layer, num_layers):
    # Keep train/eval behavior stable by avoiding nested tensor auto-conversion.
    try:
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


class ActorRolloutBuffer:
    def __init__(self):
        self.encoder_tokens = []
        self.encoder_pad = []
        self.encoder_segment_ids = []
        self.decoder_tokens = []
        self.decoder_pad = []
        self.target_masks = []

        self.target_actions = []
        self.speed_actions = []
        self.logprobs = []

    def clear(self):
        self.encoder_tokens.clear()
        self.encoder_pad.clear()
        self.encoder_segment_ids.clear()
        self.decoder_tokens.clear()
        self.decoder_pad.clear()
        self.target_masks.clear()

        self.target_actions.clear()
        self.speed_actions.clear()
        self.logprobs.clear()


class TransformerActorNet(nn.Module):
    def __init__(
        self,
        token_dim,
        target_action_dim,
        speed_action_dim,
        max_encoder_len,
        max_decoder_len,
        max_other_agents=0,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.target_action_dim = target_action_dim
        self.speed_action_dim = speed_action_dim
        self.max_other_agents = max(0, int(max_other_agents))

        self.token_proj = nn.Linear(token_dim, d_model)
        self.encoder_pos = nn.Parameter(torch.zeros(max_encoder_len, d_model))
        self.decoder_pos = nn.Parameter(torch.zeros(max_decoder_len, d_model))
        self.encoder_segment_embedding = nn.Embedding(self.max_other_agents + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.actor_encoder = _build_encoder(enc_layer, num_layers)
        self.actor_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.target_head = nn.Linear(d_model, target_action_dim)
        self.speed_head = nn.Linear(d_model, speed_action_dim)

    @staticmethod
    def _last_valid(h, pad_mask):
        if pad_mask is None:
            return h[:, -1, :]
        valid_len = (~pad_mask).long().sum(dim=1) - 1
        valid_len = torch.clamp(valid_len, min=0)
        batch_idx = torch.arange(h.size(0), device=h.device)
        return h[batch_idx, valid_len, :]

    def _add_positional(self, x, pos_table):
        return x + pos_table[: x.size(1)].unsqueeze(0)

    @staticmethod
    def _masked_target_logits(target_logits, target_mask):
        target_mask = torch.as_tensor(target_mask, dtype=torch.float32, device=target_logits.device)
        if target_mask.dim() == 1:
            target_mask = target_mask.unsqueeze(0)
        if torch.any(target_mask.sum(dim=1) <= 0):
            raise ValueError("Target mask cannot be all zeros.")
        return target_logits.masked_fill(target_mask <= 0, -1e9)

    def actor_forward(self, encoder_tokens, decoder_tokens, encoder_pad=None, decoder_pad=None, encoder_segment_ids=None):
        if encoder_pad is not None:
            encoder_pad = encoder_pad.to(dtype=torch.bool)
        if decoder_pad is not None:
            decoder_pad = decoder_pad.to(dtype=torch.bool)

        src = self._add_positional(self.token_proj(encoder_tokens), self.encoder_pos)
        if encoder_segment_ids is not None:
            encoder_segment_ids = torch.clamp(encoder_segment_ids.long(), min=0, max=self.max_other_agents)
            src = src + self.encoder_segment_embedding(encoder_segment_ids)
        tgt = self._add_positional(self.token_proj(decoder_tokens), self.decoder_pos)

        src = torch.nan_to_num(src, nan=0.0, posinf=1e4, neginf=-1e4)
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e4, neginf=-1e4)

        memory = self.actor_encoder(src, src_key_padding_mask=encoder_pad)
        if encoder_pad is not None and encoder_pad.size(1) != memory.size(1):
            if encoder_pad.size(1) > memory.size(1):
                encoder_pad = encoder_pad[:, -memory.size(1):]
            else:
                pad_extra = torch.zeros(
                    (encoder_pad.size(0), memory.size(1) - encoder_pad.size(1)),
                    dtype=encoder_pad.dtype,
                    device=encoder_pad.device,
                )
                encoder_pad = torch.cat([pad_extra, encoder_pad], dim=1)

        dec_h = self.actor_decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=decoder_pad,
            memory_key_padding_mask=encoder_pad,
        )
        actor_h = self._last_valid(dec_h, decoder_pad)

        target_logits = self.target_head(actor_h)
        speed_logits = self.speed_head(actor_h)
        target_logits = torch.nan_to_num(target_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        speed_logits = torch.nan_to_num(speed_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return target_logits, speed_logits

    def act(self, inputs, masks, deter_action=None):
        target_logits, speed_logits = self.actor_forward(
            inputs["encoder_tokens"],
            inputs["decoder_tokens"],
            inputs.get("encoder_pad"),
            inputs.get("decoder_pad"),
            inputs.get("encoder_segment_ids"),
        )
        target_logits = self._masked_target_logits(target_logits, masks["target"])

        dist_target = Categorical(logits=target_logits)
        dist_speed = Categorical(logits=speed_logits)

        if deter_action is not None:
            target_action = torch.tensor([int(deter_action[0])], dtype=torch.long, device=target_logits.device)
            speed_action = torch.tensor([int(deter_action[1])], dtype=torch.long, device=target_logits.device)
        else:
            target_action = dist_target.sample()
            speed_action = dist_speed.sample()

        logprob = dist_target.log_prob(target_action) + dist_speed.log_prob(speed_action)
        return target_action.detach(), speed_action.detach(), logprob.detach()

    def act_test(self, inputs, masks):
        with torch.no_grad():
            target_logits, speed_logits = self.actor_forward(
                inputs["encoder_tokens"],
                inputs["decoder_tokens"],
                inputs.get("encoder_pad"),
                inputs.get("decoder_pad"),
                inputs.get("encoder_segment_ids"),
            )
            target_logits = self._masked_target_logits(target_logits, masks["target"])

            target_action = torch.argmax(target_logits, dim=-1)
            speed_action = torch.argmax(speed_logits, dim=-1)

        return int(target_action.item()), int(speed_action.item())

    def evaluate(self, inputs, target_actions, speed_actions, target_mask):
        target_logits, speed_logits = self.actor_forward(
            inputs["encoder_tokens"],
            inputs["decoder_tokens"],
            inputs.get("encoder_pad"),
            inputs.get("decoder_pad"),
            inputs.get("encoder_segment_ids"),
        )
        target_logits = self._masked_target_logits(target_logits, target_mask)

        dist_target = Categorical(logits=target_logits)
        dist_speed = Categorical(logits=speed_logits)

        action_logprobs = dist_target.log_prob(target_actions) + dist_speed.log_prob(speed_actions)
        dist_entropy = dist_target.entropy() + dist_speed.entropy()
        return action_logprobs, dist_entropy


class TransformerCriticNet(nn.Module):
    def __init__(
        self,
        token_dim,
        max_critic_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, d_model)
        self.critic_pos = nn.Parameter(torch.zeros(max_critic_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.critic_encoder = _build_encoder(enc_layer, num_layers)
        self.critic_head = nn.Linear(d_model, 1)

    def _add_positional(self, x):
        return x + self.critic_pos[: x.size(1)].unsqueeze(0)

    def forward(self, critic_tokens, critic_pad=None):
        critic_x = self._add_positional(self.token_proj(critic_tokens))
        critic_x = torch.nan_to_num(critic_x, nan=0.0, posinf=1e4, neginf=-1e4)
        critic_h = self.critic_encoder(critic_x, src_key_padding_mask=critic_pad)

        if critic_pad is None:
            pooled = critic_h.mean(dim=1)
        else:
            valid = (~critic_pad).float().unsqueeze(-1)
            denom = torch.clamp(valid.sum(dim=1), min=1.0)
            pooled = (critic_h * valid).sum(dim=1) / denom
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)

        value = self.critic_head(pooled)
        return torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)


class MLPCriticNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, critic_state):
        critic_state = torch.nan_to_num(critic_state, nan=0.0, posinf=1e4, neginf=-1e4)
        value = self.net(critic_state)
        return torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)


class PPO:
    def __init__(
        self,
        agent_id,
        token_dim,
        target_action_dim,
        speed_action_dim,
        max_encoder_len,
        max_decoder_len,
        max_critic_len,
        max_other_agents,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        summary_dir,
        entropy_ratio,
        gae_lambda,
        gae_flag,
        decoupled_clip=False,
        eps_clip_pos=0.3,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        # Keep signature compatible with previous calls; actor update uses only actor-side args.
        self.id = agent_id
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_ratio = entropy_ratio
        self.decoupled_clip = bool(decoupled_clip)
        self.eps_clip_pos = max(float(eps_clip_pos), float(self.eps_clip))

        self.buffer = ActorRolloutBuffer()

        self.policy = TransformerActorNet(
            token_dim=token_dim,
            target_action_dim=target_action_dim,
            speed_action_dim=speed_action_dim,
            max_encoder_len=max_encoder_len,
            max_decoder_len=max_decoder_len,
            max_other_agents=max_other_agents,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.policy_old = TransformerActorNet(
            token_dim=token_dim,
            target_action_dim=target_action_dim,
            speed_action_dim=speed_action_dim,
            max_encoder_len=max_encoder_len,
            max_decoder_len=max_decoder_len,
            max_other_agents=max_other_agents,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.update_times = 0
        self.summary_dir = summary_dir
        self.writer = SummaryWriter(log_dir=self.summary_dir)

    def _to_actor_inputs(self, obs_pack):
        encoder_tokens = torch.as_tensor(obs_pack["encoder_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
        encoder_tokens = torch.nan_to_num(encoder_tokens, nan=0.0, posinf=1e4, neginf=-1e4)

        encoder_segment_ids = obs_pack.get("encoder_segment_ids", None)
        if encoder_segment_ids is None:
            encoder_segment_ids = torch.zeros(encoder_tokens.size(1), dtype=torch.long, device=device).unsqueeze(0)
        else:
            encoder_segment_ids = torch.as_tensor(encoder_segment_ids, dtype=torch.long, device=device).unsqueeze(0)

        decoder_tokens = torch.as_tensor(obs_pack["decoder_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
        decoder_tokens = torch.nan_to_num(decoder_tokens, nan=0.0, posinf=1e4, neginf=-1e4)

        return {
            "encoder_tokens": encoder_tokens,
            "encoder_pad": torch.as_tensor(obs_pack["encoder_pad"], dtype=torch.bool, device=device).unsqueeze(0),
            "encoder_segment_ids": encoder_segment_ids,
            "decoder_tokens": decoder_tokens,
            "decoder_pad": torch.as_tensor(obs_pack["decoder_pad"], dtype=torch.bool, device=device).unsqueeze(0),
        }

    def select_action(self, obs_pack, masks, deter_action=None):
        inputs = self._to_actor_inputs(obs_pack)
        target_mask = torch.as_tensor(masks["target"], dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            target_action, speed_action, action_logprob = self.policy_old.act(
                inputs,
                {"target": target_mask},
                deter_action=deter_action,
            )

        transition = {
            "encoder_tokens": inputs["encoder_tokens"].squeeze(0).detach().cpu(),
            "encoder_pad": inputs["encoder_pad"].squeeze(0).detach().cpu(),
            "encoder_segment_ids": inputs["encoder_segment_ids"].squeeze(0).detach().cpu(),
            "decoder_tokens": inputs["decoder_tokens"].squeeze(0).detach().cpu(),
            "decoder_pad": inputs["decoder_pad"].squeeze(0).detach().cpu(),
            "target_mask": target_mask.squeeze(0).detach().cpu(),
            "target_action": target_action.squeeze(0).detach().cpu(),
            "speed_action": speed_action.squeeze(0).detach().cpu(),
            "logprob": action_logprob.squeeze(0).detach().cpu(),
        }

        return (int(target_action.item()), int(speed_action.item())), transition

    def store_transition(self, transition):
        self.buffer.encoder_tokens.append(transition["encoder_tokens"])
        self.buffer.encoder_pad.append(transition["encoder_pad"])
        self.buffer.encoder_segment_ids.append(transition["encoder_segment_ids"])
        self.buffer.decoder_tokens.append(transition["decoder_tokens"])
        self.buffer.decoder_pad.append(transition["decoder_pad"])
        self.buffer.target_masks.append(transition["target_mask"])

        self.buffer.target_actions.append(transition["target_action"])
        self.buffer.speed_actions.append(transition["speed_action"])
        self.buffer.logprobs.append(transition["logprob"])

    def last_transition_index(self):
        return len(self.buffer.target_actions) - 1

    def action_test(self, obs_pack, masks):
        inputs = self._to_actor_inputs(obs_pack)
        target_mask = torch.as_tensor(masks["target"], dtype=torch.float32, device=device).unsqueeze(0)
        was_training = self.policy_old.training
        self.policy_old.eval()
        action = self.policy_old.act_test(inputs, {"target": target_mask})
        if was_training:
            self.policy_old.train()
        return action

    def update(self, advantages):
        n_samples = len(self.buffer.target_actions)
        if n_samples == 0:
            return

        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device).view(-1)
        if advantages.numel() != n_samples:
            raise ValueError(
                f"Advantage size mismatch for actor {self.id}: got {advantages.numel()}, expected {n_samples}"
            )

        old_inputs = {
            "encoder_tokens": torch.stack(self.buffer.encoder_tokens, dim=0).detach().to(device),
            "encoder_pad": torch.stack(self.buffer.encoder_pad, dim=0).detach().to(device),
            "encoder_segment_ids": torch.stack(self.buffer.encoder_segment_ids, dim=0).detach().to(device),
            "decoder_tokens": torch.stack(self.buffer.decoder_tokens, dim=0).detach().to(device),
            "decoder_pad": torch.stack(self.buffer.decoder_pad, dim=0).detach().to(device),
        }

        old_target_masks = torch.stack(self.buffer.target_masks, dim=0).detach().to(device)
        old_target_actions = torch.stack(self.buffer.target_actions, dim=0).detach().to(device)
        old_speed_actions = torch.stack(self.buffer.speed_actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, dist_entropy = self.policy.evaluate(
                old_inputs,
                old_target_actions,
                old_speed_actions,
                old_target_masks,
            )

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2_default = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            clipped_surr = torch.min(surr1, surr2_default)
            if self.decoupled_clip:
                surr2_pos = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip_pos) * advantages
                clipped_surr_pos = torch.min(surr1, surr2_pos)
                clipped_surr = torch.where(advantages > 0, clipped_surr_pos, clipped_surr)

            policy_loss = -clipped_surr.mean()
            entropy_loss = dist_entropy.mean()
            loss = policy_loss - self.entropy_ratio * entropy_loss

            if self.id == 0:
                self.writer.add_scalar("loss/policy", policy_loss.detach().item(), self.update_times)
                self.writer.add_scalar("stats/entropy", entropy_loss.detach().item(), self.update_times)
                self.writer.add_scalar("stats/decoupled_clip_enabled", float(self.decoupled_clip), self.update_times)
                self.writer.add_scalar("stats/eps_clip_pos", float(self.eps_clip_pos), self.update_times)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.update_times += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def call_2_record(self, steps, value):
        self.writer.add_scalar("reward/test", value, steps)


class CentralizedCritic:
    def __init__(
        self,
        critic_state_dim,
        lr_critic,
        gamma,
        K_epochs,
        gae_lambda,
        gae_flag,
        summary_writer=None,
        hidden_dim=128,
    ):
        self.gamma = float(gamma)
        self.K_epochs = int(K_epochs)
        self.gae_lambda = float(gae_lambda)
        self.gae_flag = bool(gae_flag)
        self.writer = summary_writer

        self.policy = MLPCriticNet(
            in_dim=critic_state_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        self.policy_old = MLPCriticNet(
            in_dim=critic_state_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_critic)
        self.mse_loss = nn.MSELoss()
        self.update_times = 0

    def snapshot(self, obs_pack):
        critic_state = torch.as_tensor(obs_pack["critic_state"], dtype=torch.float32)
        critic_state = torch.nan_to_num(critic_state, nan=0.0, posinf=1e4, neginf=-1e4)

        with torch.no_grad():
            value = self.policy_old(critic_state.unsqueeze(0).to(device))

        return {
            "critic_state": critic_state.detach().cpu(),
            "value": float(value.item()),
        }

    def compute_advantages_and_returns(self, rewards, dones, old_values):
        if len(rewards) == 0:
            empty = torch.empty(0, dtype=torch.float32)
            return empty, empty

        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)
        values_t = torch.as_tensor(old_values, dtype=torch.float32, device=device)

        if self.gae_flag:
            advantages = torch.zeros_like(rewards_t)
            gae = torch.tensor(0.0, dtype=torch.float32, device=device)
            next_value = torch.tensor(0.0, dtype=torch.float32, device=device)

            for t in reversed(range(len(rewards_t))):
                mask = 1.0 - dones_t[t]
                delta = rewards_t[t] + self.gamma * next_value * mask - values_t[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[t] = gae
                next_value = values_t[t]

            returns = advantages + values_t
        else:
            returns = torch.zeros_like(rewards_t)
            discounted_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
            for t in reversed(range(len(rewards_t))):
                if dones_t[t] > 0.5:
                    discounted_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
                discounted_reward = rewards_t[t] + self.gamma * discounted_reward
                returns[t] = discounted_reward
            advantages = returns - values_t

        return advantages.detach().cpu(), returns.detach().cpu()

    def update(self, critic_snapshots, returns):
        if len(critic_snapshots) == 0:
            return

        critic_states = torch.stack([snap["critic_state"] for snap in critic_snapshots], dim=0).detach().to(device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device).view(-1)

        for _ in range(self.K_epochs):
            values = self.policy(critic_states).view(-1)
            critic_loss = self.mse_loss(values, returns)

            self.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.writer is not None:
                self.writer.add_scalar("loss/critic", critic_loss.detach().item(), self.update_times)
                self.writer.add_scalar("stats/critic", values.mean().detach().item(), self.update_times)

            self.update_times += 1

        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
