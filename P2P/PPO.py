"""
PPO.py — Transformer Actor (encoder-decoder) + MLP Centralized Critic

Actor 结构:
  - Encoder: 处理其他 UAV 的 BS-synced 历史 tokens (segmented)
  - Decoder: 处理自身历史 tokens + 当前 token
  - 输出: target_logits (K+N) + speed_logits (speed_action_dim)

Critic 结构:
  - MLP: 输入全局状态 (含 comm_adjacency), 输出 V(s)
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"Device: {device}")
print("=" * 60)


def _build_encoder(layer, num_layers):
    try:
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


# ================================================================
#  Rollout Buffer
# ================================================================
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


# ================================================================
#  Transformer Actor
# ================================================================
class TransformerActorNet(nn.Module):
    def __init__(self, token_dim, target_action_dim, speed_action_dim,
                 max_encoder_len, max_decoder_len, max_other_agents=0,
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.target_action_dim = target_action_dim
        self.speed_action_dim = speed_action_dim
        self.max_other_agents = max(0, int(max_other_agents))

        self.token_proj = nn.Linear(token_dim, d_model)
        self.encoder_pos = nn.Parameter(torch.zeros(max_encoder_len, d_model))
        self.decoder_pos = nn.Parameter(torch.zeros(max_decoder_len, d_model))
        self.encoder_segment_embedding = nn.Embedding(self.max_other_agents + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
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
        idx = torch.arange(h.size(0), device=h.device)
        return h[idx, valid_len, :]

    def _add_pos(self, x, pos_table):
        return x + pos_table[:x.size(1)].unsqueeze(0)

    @staticmethod
    def _mask_logits(logits, mask):
        mask = torch.as_tensor(mask, dtype=torch.float32, device=logits.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if torch.any(mask.sum(dim=1) <= 0):
            mask = mask + 1e-8
        return logits.masked_fill(mask <= 0, -1e9)

    def actor_forward(self, enc_tokens, dec_tokens, enc_pad=None, dec_pad=None, enc_seg=None):
        if enc_pad is not None:
            enc_pad = enc_pad.to(dtype=torch.bool)
        if dec_pad is not None:
            dec_pad = dec_pad.to(dtype=torch.bool)

        src = self._add_pos(self.token_proj(enc_tokens), self.encoder_pos)
        if enc_seg is not None:
            enc_seg = torch.clamp(enc_seg.long(), min=0, max=self.max_other_agents)
            src = src + self.encoder_segment_embedding(enc_seg)
        tgt = self._add_pos(self.token_proj(dec_tokens), self.decoder_pos)

        src = torch.nan_to_num(src, nan=0.0, posinf=1e4, neginf=-1e4)
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e4, neginf=-1e4)

        memory = self.actor_encoder(src, src_key_padding_mask=enc_pad)
        if enc_pad is not None and enc_pad.size(1) != memory.size(1):
            if enc_pad.size(1) > memory.size(1):
                enc_pad = enc_pad[:, -memory.size(1):]
            else:
                pad_extra = torch.zeros(
                    (enc_pad.size(0), memory.size(1) - enc_pad.size(1)),
                    dtype=enc_pad.dtype, device=enc_pad.device,
                )
                enc_pad = torch.cat([pad_extra, enc_pad], dim=1)

        dec_h = self.actor_decoder(
            tgt=tgt, memory=memory,
            tgt_key_padding_mask=dec_pad,
            memory_key_padding_mask=enc_pad,
        )
        h = self._last_valid(dec_h, dec_pad)

        target_logits = self.target_head(h)
        speed_logits = self.speed_head(h)
        target_logits = torch.nan_to_num(target_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        speed_logits = torch.nan_to_num(speed_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return target_logits, speed_logits

    def act(self, inputs, masks, deter_action=None):
        t_logits, s_logits = self.actor_forward(
            inputs["encoder_tokens"], inputs["decoder_tokens"],
            inputs.get("encoder_pad"), inputs.get("decoder_pad"),
            inputs.get("encoder_segment_ids"),
        )
        t_logits = self._mask_logits(t_logits, masks["target"])
        dist_t = Categorical(logits=t_logits)
        dist_s = Categorical(logits=s_logits)

        if deter_action is not None:
            ta = torch.tensor([int(deter_action[0])], dtype=torch.long, device=t_logits.device)
            sa = torch.tensor([int(deter_action[1])], dtype=torch.long, device=t_logits.device)
        else:
            ta = dist_t.sample()
            sa = dist_s.sample()
        logprob = dist_t.log_prob(ta) + dist_s.log_prob(sa)
        return ta.detach(), sa.detach(), logprob.detach()

    def act_test(self, inputs, masks):
        with torch.no_grad():
            t_logits, s_logits = self.actor_forward(
                inputs["encoder_tokens"], inputs["decoder_tokens"],
                inputs.get("encoder_pad"), inputs.get("decoder_pad"),
                inputs.get("encoder_segment_ids"),
            )
            t_logits = self._mask_logits(t_logits, masks["target"])
            ta = torch.argmax(t_logits, dim=-1)
            sa = torch.argmax(s_logits, dim=-1)
        return int(ta.item()), int(sa.item())

    def evaluate(self, inputs, target_actions, speed_actions, target_mask):
        t_logits, s_logits = self.actor_forward(
            inputs["encoder_tokens"], inputs["decoder_tokens"],
            inputs.get("encoder_pad"), inputs.get("decoder_pad"),
            inputs.get("encoder_segment_ids"),
        )
        t_logits = self._mask_logits(t_logits, target_mask)
        dist_t = Categorical(logits=t_logits)
        dist_s = Categorical(logits=s_logits)
        logprobs = dist_t.log_prob(target_actions) + dist_s.log_prob(speed_actions)
        entropy = dist_t.entropy() + dist_s.entropy()
        return logprobs, entropy


# ================================================================
#  MLP Critic
# ================================================================
class MLPCriticNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        v = self.net(x)
        return torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)


# ================================================================
#  PPO Agent (Actor)
# ================================================================
class PPO:
    def __init__(self, agent_id, token_dim, target_action_dim, speed_action_dim,
                 max_encoder_len, max_decoder_len, max_critic_len, max_other_agents,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, summary_dir,
                 entropy_ratio, gae_lambda, gae_flag, decoupled_clip=False,
                 eps_clip_pos=0.3, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        self.id = agent_id
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_ratio = entropy_ratio
        self.decoupled_clip = bool(decoupled_clip)
        self.eps_clip_pos = max(float(eps_clip_pos), float(eps_clip))
        self.buffer = ActorRolloutBuffer()

        kw = dict(token_dim=token_dim, target_action_dim=target_action_dim,
                   speed_action_dim=speed_action_dim, max_encoder_len=max_encoder_len,
                   max_decoder_len=max_decoder_len, max_other_agents=max_other_agents,
                   d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.policy = TransformerActorNet(**kw).to(device)
        self.policy_old = TransformerActorNet(**kw).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.update_times = 0
        self.writer = SummaryWriter(log_dir=summary_dir)

    def _to_inputs(self, obs_pack):
        et = torch.as_tensor(obs_pack["encoder_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
        et = torch.nan_to_num(et, nan=0.0, posinf=1e4, neginf=-1e4)
        esi = obs_pack.get("encoder_segment_ids", None)
        if esi is None:
            esi = torch.zeros(et.size(1), dtype=torch.long, device=device).unsqueeze(0)
        else:
            esi = torch.as_tensor(esi, dtype=torch.long, device=device).unsqueeze(0)
        dt = torch.as_tensor(obs_pack["decoder_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
        dt = torch.nan_to_num(dt, nan=0.0, posinf=1e4, neginf=-1e4)
        return {
            "encoder_tokens": et,
            "encoder_pad": torch.as_tensor(obs_pack["encoder_pad"], dtype=torch.bool, device=device).unsqueeze(0),
            "encoder_segment_ids": esi,
            "decoder_tokens": dt,
            "decoder_pad": torch.as_tensor(obs_pack["decoder_pad"], dtype=torch.bool, device=device).unsqueeze(0),
        }

    def select_action(self, obs_pack, masks, deter_action=None):
        inputs = self._to_inputs(obs_pack)
        tm = torch.as_tensor(masks["target"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            ta, sa, logprob = self.policy_old.act(inputs, {"target": tm}, deter_action=deter_action)
        transition = {
            "encoder_tokens": inputs["encoder_tokens"].squeeze(0).detach().cpu(),
            "encoder_pad": inputs["encoder_pad"].squeeze(0).detach().cpu(),
            "encoder_segment_ids": inputs["encoder_segment_ids"].squeeze(0).detach().cpu(),
            "decoder_tokens": inputs["decoder_tokens"].squeeze(0).detach().cpu(),
            "decoder_pad": inputs["decoder_pad"].squeeze(0).detach().cpu(),
            "target_mask": tm.squeeze(0).detach().cpu(),
            "target_action": ta.squeeze(0).detach().cpu(),
            "speed_action": sa.squeeze(0).detach().cpu(),
            "logprob": logprob.squeeze(0).detach().cpu(),
        }
        return (int(ta.item()), int(sa.item())), transition

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
        inputs = self._to_inputs(obs_pack)
        tm = torch.as_tensor(masks["target"], dtype=torch.float32, device=device).unsqueeze(0)
        was_training = self.policy_old.training
        self.policy_old.eval()
        action = self.policy_old.act_test(inputs, {"target": tm})
        if was_training:
            self.policy_old.train()
        return action

    def update(self, advantages):
        n = len(self.buffer.target_actions)
        if n == 0:
            return
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device).view(-1)
        if advantages.numel() != n:
            raise ValueError(f"Advantage size mismatch: {advantages.numel()} != {n}")

        old_inputs = {
            "encoder_tokens": torch.stack(self.buffer.encoder_tokens, dim=0).detach().to(device),
            "encoder_pad": torch.stack(self.buffer.encoder_pad, dim=0).detach().to(device),
            "encoder_segment_ids": torch.stack(self.buffer.encoder_segment_ids, dim=0).detach().to(device),
            "decoder_tokens": torch.stack(self.buffer.decoder_tokens, dim=0).detach().to(device),
            "decoder_pad": torch.stack(self.buffer.decoder_pad, dim=0).detach().to(device),
        }
        old_tm = torch.stack(self.buffer.target_masks, dim=0).detach().to(device)
        old_ta = torch.stack(self.buffer.target_actions, dim=0).detach().to(device)
        old_sa = torch.stack(self.buffer.speed_actions, dim=0).detach().to(device)
        old_lp = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, entropy = self.policy.evaluate(old_inputs, old_ta, old_sa, old_tm)
            ratios = torch.exp(logprobs - old_lp)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            clipped = torch.min(surr1, surr2)
            if self.decoupled_clip:
                surr2_pos = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip_pos) * advantages
                clipped = torch.where(advantages > 0, torch.min(surr1, surr2_pos), clipped)

            loss = -clipped.mean() - self.entropy_ratio * entropy.mean()
            if self.id == 0:
                self.writer.add_scalar("loss/policy", float(-clipped.mean().item()), self.update_times)
                self.writer.add_scalar("stats/entropy", float(entropy.mean().item()), self.update_times)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.update_times += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path):
        torch.save({
            "policy": self.policy_old.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_times": self.update_times,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=lambda s, l: s, weights_only=False)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            self.policy_old.load_state_dict(ckpt["policy"])
            self.policy.load_state_dict(ckpt["policy"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "update_times" in ckpt:
                self.update_times = ckpt["update_times"]
        else:
            # 兼容旧格式 (纯 state_dict)
            self.policy_old.load_state_dict(ckpt)
            self.policy.load_state_dict(ckpt)

    def call_2_record(self, step, value):
        self.writer.add_scalar("reward/test", value, step)


# ================================================================
#  Centralized Critic
# ================================================================
class CentralizedCritic:
    def __init__(self, critic_state_dim, lr_critic, gamma, K_epochs,
                 gae_lambda, gae_flag, summary_writer=None, hidden_dim=128):
        self.gamma = float(gamma)
        self.K_epochs = int(K_epochs)
        self.gae_lambda = float(gae_lambda)
        self.gae_flag = bool(gae_flag)
        self.writer = summary_writer

        self.policy = MLPCriticNet(critic_state_dim, hidden_dim).to(device)
        self.policy_old = MLPCriticNet(critic_state_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_critic)
        self.mse_loss = nn.MSELoss()
        self.update_times = 0

    def snapshot(self, obs_pack):
        cs = torch.as_tensor(obs_pack["critic_state"], dtype=torch.float32)
        cs = torch.nan_to_num(cs, nan=0.0, posinf=1e4, neginf=-1e4)
        with torch.no_grad():
            v = self.policy_old(cs.unsqueeze(0).to(device))
        return {"critic_state": cs.detach().cpu(), "value": float(v.item())}

    def compute_advantages_and_returns(self, rewards, dones, old_values):
        if len(rewards) == 0:
            e = torch.empty(0, dtype=torch.float32)
            return e, e
        r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        d = torch.as_tensor(dones, dtype=torch.float32, device=device)
        v = torch.as_tensor(old_values, dtype=torch.float32, device=device)

        if self.gae_flag:
            adv = torch.zeros_like(r)
            gae = torch.tensor(0.0, device=device)
            next_v = torch.tensor(0.0, device=device)
            for t in reversed(range(len(r))):
                mask = 1.0 - d[t]
                delta = r[t] + self.gamma * next_v * mask - v[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                adv[t] = gae
                next_v = v[t]
            returns = adv + v
        else:
            returns = torch.zeros_like(r)
            dr = torch.tensor(0.0, device=device)
            for t in reversed(range(len(r))):
                if d[t] > 0.5:
                    dr = torch.tensor(0.0, device=device)
                dr = r[t] + self.gamma * dr
                returns[t] = dr
            adv = returns - v
        return adv.detach().cpu(), returns.detach().cpu()

    def update(self, snapshots, returns):
        if len(snapshots) == 0:
            return
        cs = torch.stack([s["critic_state"] for s in snapshots], dim=0).detach().to(device)
        rt = torch.as_tensor(returns, dtype=torch.float32, device=device).view(-1)

        for _ in range(self.K_epochs):
            v = self.policy(cs).view(-1)
            loss = self.mse_loss(v, rt)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar("loss/critic", float(loss.item()), self.update_times)
                self.writer.add_scalar("stats/critic", float(v.mean().item()), self.update_times)
            self.update_times += 1

        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, path):
        torch.save({
            "policy": self.policy_old.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_times": self.update_times,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=lambda s, l: s, weights_only=False)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            self.policy_old.load_state_dict(ckpt["policy"])
            self.policy.load_state_dict(ckpt["policy"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "update_times" in ckpt:
                self.update_times = ckpt["update_times"]
        else:
            self.policy_old.load_state_dict(ckpt)
            self.policy.load_state_dict(ckpt)
