# P2P_Comm_MARL — 多无人机异步协同 + P2P 邻近通信

## 项目概述

本项目在多无人机异步 AoI（Age of Information）最小化任务上，引入 **P2P 邻近通信机制**（Proximity State Broadcast）。当两架 UAV 的物理距离小于通信半径 `R_comm` 时，它们自动交换一个紧凑的状态快照向量，帮助各 UAV 在异步决策中获得邻居的实时信息，减少重复访问、提升协同效率。

训练框架为 **CTDE（Centralized Training, Decentralized Execution）**，Actor 使用 Transformer encoder-decoder 架构，Critic 使用集中式 MLP。优化目标不变（最小化加权 AoI），P2P 通信作为环境的一部分，不引入额外的可训练通信模块。

---

## 文件结构

```
P2P_Comm_MARL/
├── data_generate.py    # 生成 POI / BS 位置数据 (.npy + .png)
├── Env.py              # 多无人机异步 AoI 环境 + P2P 通信
├── PPO.py              # Transformer Actor + MLP Critic + PPO 更新
├── train.py            # 训练入口 (CTDE + 监控 + 测试评估)
├── verify.py           # 消融验证 (COMM_OFF vs COMM_ON)
├── smoke_test.py       # 冒烟测试 (7 项检查)
├── README.md           # 本文件
├── data/               # 数据文件目录
│   └── poi_8_map_300x300.npy   # 冒烟测试用数据 (8 POI, 300x300)
├── PPO_logs/           # 训练日志 (CSV)
├── PPO_preTrained/     # 模型 checkpoint
└── logs/               # TensorBoard 日志
```

---

## 快速开始

### 1. 环境依赖

```
Python >= 3.10
torch >= 2.0
numpy
scipy
gym
tensorboard
matplotlib (可选, 用于可视化)
```

安装:

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install torch numpy scipy gym tensorboard matplotlib
```

### 2. 冒烟测试

```bash
python smoke_test.py
```

预期输出: `ALL SMOKE TESTS PASSED ✓`

### 3. 生成数据

```bash
# 生成 15 个 POI, 500x500 地图
python data_generate.py --K 15 --map_size 500

# 或自定义参数
python data_generate.py --K 20 --map_size 800 --output ./data/poi_20_map_800x800.npy
```

### 4. 训练

```bash
# 默认配置: 3 UAV, 15 POI, 500x500 地图, R_comm=150
python train.py

# 自定义参数
python train.py --K 20 --map_size 800 --R_comm 200 --T 800 --max_training_timesteps 5000000

# 关闭 P2P 通信 (消融对照组)
python train.py --R_comm 0
```

### 5. TensorBoard 监控

```bash
tensorboard --logdir logs/
```

### 6. 消融验证

```bash
# 对比 P2P 通信开启 vs 关闭
python verify.py --K 15 --map_size 500 --R_comm_on 150 --episodes 20
```

---

## 模型结构

### Actor (Transformer Encoder-Decoder)

```
输入:
  Encoder tokens:  其他 UAV 的 BS-synced 历史 tokens (segmented)
                   shape: (max_encoder_len, token_dim)
                   max_encoder_len = (M-1) * history_horizon
  Decoder tokens:  自身历史 tokens + 当前 token
                   shape: (max_decoder_len, token_dim)
                   max_decoder_len = history_horizon + 1

结构:
  Token Projection:   Linear(token_dim → d_model)
  Positional Encoding: 可学习参数 (max_encoder_len, d_model) / (max_decoder_len, d_model)
  Segment Embedding:   Embedding(max_other_agents+1, d_model)
  Encoder:             TransformerEncoder(d_model, nhead, num_layers, dim_feedforward)
  Decoder:             TransformerDecoder(d_model, nhead, num_layers, dim_feedforward)
  Output Heads:
    target_head: Linear(d_model → K+N)      # 目标选择 (POI 或 BS)
    speed_head:  Linear(d_model → n_speeds)  # 速度选择

  取 decoder 最后一个有效 token 的隐状态 → 两个输出头
```

### Critic (集中式 MLP)

```
输入: 全局状态向量 (critic_state_dim)
  包含: buffer_mask(M*K) + buffer_age(M*K) + UAV位置(M*2) + POI位置(K*2)
        + BS位置(N*2) + 全局AoI(K) + POI权重(K) + UAV时间(M)
        + UAV能量(M) + UAV能量比率(M) + comm邻接矩阵(M*M)

结构:
  Linear(critic_state_dim → hidden_dim) → Tanh
  Linear(hidden_dim → hidden_dim) → Tanh
  Linear(hidden_dim → 1)
```

### 观测空间 (obs_dim)

每个 UAV 的观测向量组成:

| 分量 | 维度 | 说明 |
|------|------|------|
| aoi | K | 本地 AoI 估计 (归一化) |
| distance_to_pois | K | 到各 POI 距离 (归一化) |
| rewards_to_pois | K | 到各 POI 的预期奖励 |
| rewards_to_bs | 1 | 回基站预期奖励 |
| buffer_bitmap | K | buffer 中已有哪些 POI |
| time | 1 | 当前时间 (归一化) |
| history_visited | K*M | BS 同步的延迟历史 |
| last_rewards | M | 各 UAV 上次奖励 (BS 同步) |
| buffer_len | 1 | buffer 大小 |
| energy | 1 | 剩余能量比率 |
| poi_weights | K | POI 权重 (归一化) |
| **comm_info** | **(M-1)*(comm_dim+2)** | **P2P 邻居信息 (mask+staleness+msg)** |

总计: `5K + KM + M + 4 + (M-1)*(comm_dim+2)`

### P2P 通信消息 (comm_dim = 2K + 4)

| 分量 | 维度 | 说明 |
|------|------|------|
| position | 2 | UAV 归一化位置 |
| buffer_bitmap | K | UAV 当前 buffer 中有哪些 POI |
| local_aoi | K | UAV 的本地 AoI 估计 |
| timing | 1 | UAV 当前时间 |
| last_target | 1 | UAV 上一个目标动作 |

### 观测中的通信字段 (每邻居 comm_dim + 2 维)

| 分量 | 维度 | 说明 |
|------|------|------|
| availability_mask | 1 | 1.0=实时P2P, 0.5=BS中继, 0.0=无信息 |
| staleness | 1 | 信息时延 (归一化), 0=最新, 1=最陈旧 |
| comm_msg | comm_dim | 邻居状态快照 (实时时完整, 中继时仅buffer_bitmap) |

三种信息状态:
- **mask=1.0**: 邻居在 P2P 通信范围内, 信息实时, staleness 接近 0
- **mask=0.5**: 邻居不在范围内, 但 BS 中继了 P2P 知识, 仅 buffer_bitmap 可用, staleness=1.0
- **mask=0.0**: 完全没有该邻居的信息

---

## 关键参数

### 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--M` | 3 | UAV 数量 |
| `--N` | 1 | 基站数量 |
| `--K` | 15 | POI 数量 |
| `--T` | 600.0 | 时间上限 (秒) |
| `--map_size` | 500.0 | 地图边长 (m) |
| `--R_comm` | 150.0 | **P2P 通信半径 (m), 0=关闭** |
| `--position_file` | None | 数据文件路径 (None 则自动查找) |

### 速度 & 能量参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--speed_levels` | "6-20" | 速度档位: 6,7,...,20 m/s (共 15 档) |
| `--init_uav_energy` | 2e5 | 初始能量 (J), 约 200kJ |
| `--init_uav_energies` | None | 各 UAV 独立能量, 如 "[1.5e5,2e5,2.5e5]" |
| `--reward_divisor` | 10.0 | 奖励缩放因子 |

能量模型为旋翼无人机功耗模型:

```
P(v) = P_b(1 + 3v²/V_tip²) + P_i·sqrt(sqrt(1 + v⁴/(4u₀⁴)) - v²/(2u₀²)) + f₀·a·n·R·v³/2
```

其中: P_b=79.86W, P_i=88.63W, V_tip=120m/s, u_0=4.03, f_0=0.6, a=1.225, n=0.05, R=0.503m

### 奖励参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--reward_scale_size` | 10000.0 | 奖励全局缩放 |
| `--pre_reward_ratio` | 3.0 | 预期奖励时间折扣比 |
| `--BS_back_times` | 5 | buffer 满多少步后强制回 BS |
| `--buffer_punishment` | False | 是否启用 buffer 溢出惩罚 |
| `--punishment_value` | 4.0 | 惩罚强度 |

### PPO 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_ep_len` | 300 | 每 episode 最大步数 |
| `--max_training_timesteps` | 3e6 | 总训练步数 |
| `--K_epochs` | 10 | PPO 每次 update 的 epoch 数 |
| `--eps_clip` | 0.2 | PPO clip 参数 |
| `--gamma` | 0.99 | 折扣因子 |
| `--lr_actor` | 1e-4 | Actor 学习率 |
| `--lr_critic` | 1e-4 | Critic 学习率 |
| `--entropy_ratio` | 0.03 | 熵正则系数 |
| `--gae_lambda` | 0.97 | GAE lambda |
| `--gae_flag` | False | 是否启用 GAE (store_true) |
| `--update_every_episodes` | 5 | 每 N 个 episode 更新一次 |
| `--force_return_bs_threshold` | 8 | 连续非 BS 动作多少步后强制回 BS |

### Transformer 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--history_horizon` | 10 | 历史 token 长度 |
| `--transformer_dim` | 128 | d_model |
| `--transformer_heads` | 2 | 注意力头数 |
| `--transformer_layers` | 2 | 层数 |
| `--transformer_dropout` | 0.1 | dropout |

### 日志 & 保存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--log_dir` | "PPO_logs" | CSV 日志目录 |
| `--checkpoint_dir` | "PPO_preTrained" | 模型保存目录 |
| `--save_model_freq` | 1e5 | 每 N 步保存模型 |

---

## 监控指标

### 每 episode 记录 (CSV + TensorBoard)

| 指标 | 说明 |
|------|------|
| `team_reward` | 团队总 reward |
| `mean_aoi` | 平均 AoI |
| `weighted_aoi` | 加权 AoI |
| `max_aoi` | 最大 AoI |
| `comm_active_ratio` | P2P 通信活跃步占比 |
| `avg_neighbors` | 平均邻居数 |
| `repeat_visit_rate` | 重复 POI 访问率 |

### 每 update 记录 (TensorBoard)

| 指标 | 说明 |
|------|------|
| `loss/policy` | Actor policy loss |
| `loss/critic` | Critic MSE loss |
| `stats/entropy` | 策略熵 |
| `stats/critic` | Critic 输出均值 |

### 每 20 episodes 测试评估

| 指标 | 说明 |
|------|------|
| `reward/test` | 测试 team reward |
| `best_test_routes.png` | 最优测试路径可视化 |

---

## 异步执行机制

1. 每个 episode 开始时，所有 UAV 从基站出发
2. 每步选择 `drone_timing_now` 最小的 UAV 执行动作 (最早空闲优先)
3. 该 UAV 飞向目标 POI 或基站，期间其他 UAV 不受影响
4. 动作完成后，更新该 UAV 的位置、时间、能量、buffer
5. **每次步进后自动更新 P2P 通信邻接矩阵** (`_update_proximity_comm`)
6. UAV 回到基站时触发 BS 同步 (`_sync_with_bs`)，上传/下载延迟信息

---

## P2P 通信机制详解

### 通信条件

两架 UAV i 和 j 之间的距离 `||pos_i - pos_j|| < R_comm` 时建立通信链路。

### 通信图

动态邻接矩阵 `A ∈ {0,1}^{M×M}`，每次 step 后更新:

```
A[i][j] = 1  if  ||pos_i - pos_j|| < R_comm
A[i][j] = 0  otherwise
```

### 信息交换内容

当 `A[i][j]=1` 时，UAV i 可以读取 UAV j 的实时状态快照:

```
msg_j = [pos_j(2), buffer_bitmap_j(K), local_aoi_j(K), timing_j(1), last_target_j(1)]
```

### 通信 vs BS 同步

| 特性 | P2P 通信 | BS 同步 | BS 中继 P2P 知识 |
|------|---------|---------|-----------------|
| 触发条件 | 物理距离 < R_comm | UAV 回到基站 | 随 BS 同步一起传播 |
| 延迟 | 实时 (0 步延迟) | 延迟 (下次到 BS 才获取) | 多跳延迟 (P2P→BS→下载) |
| 信息量 | 紧凑快照 (2K+4 维) | 完整 transition tokens | 仅 buffer_bitmap (K 维) |
| 范围 | 仅邻居 | 全部 UAV (通过 BS 中转) | 全部 UAV (多跳传播) |
| mask 值 | 1.0 | — | 0.5 |

信息流动示例:
```
UAV C 采集 POI#5
  → P2P: UAV A (在 C 附近) 实时得知 C 的 buffer 含 POI#5 (mask=1.0)
  → A 回 BS: A 将 C 的 buffer 知识上传到 BS
  → UAV B 从 BS 下载: B 得知 C 的 buffer 含 POI#5 (mask=0.5, 延迟中继)
  → B 避免重复访问 POI#5
```

---

## 消融验证设计

### 实验组

| 组别 | R_comm | 说明 |
|------|--------|------|
| COMM_OFF | 0 | P2P 通信关闭，仅 BS 同步 |
| COMM_ON | 150 | P2P 通信开启 |

### 关键指标

| 指标 | 预期 (COMM_ON vs COMM_OFF) |
|------|---------------------------|
| reward_mean | ↑ (提升) |
| mean_aoi | ↓ (降低) |
| repeat_visit_rate | ↓ (显著降低) |
| comm_active_ratio | > 0 (通信确实在发生) |

### 运行

```bash
# 先训练两组模型 (COMM_OFF 和 COMM_ON)
python train.py --R_comm 0   --env_name P2P-MARL-OFF
python train.py --R_comm 150 --env_name P2P-MARL-ON

# 再用 verify.py 对比
python verify.py --episodes 20
```

---

## 冒烟测试结果

```
[1/7] Generating data...                         ✓
[2/7] Creating environment...                    ✓ (obs_dim=115, token_dim=117)
[3/7] Checking P2P communication...              ✓ (3 active pairs at start)
[4/7] Running random policy episode...           ✓ (66 steps, 3/3 done)
[5/7] Creating PPO agents and selecting actions.. ✓
[6/7] Running CTDE update (2 mini episodes)...   ✓
[7/7] Running test evaluation...                 ✓ (team_reward=1.0366)

ALL SMOKE TESTS PASSED ✓
```

测试环境: 3 UAV, 8 POI, 1 BS, 300x300 地图, R_comm=100, T=300

---

## 注意事项

1. **地图与通信半径**: `R_comm` 应根据地图大小调整。建议 `R_comm ≈ map_size * 0.3` 作为起点。
2. **能量参数**: 默认 2e5 J 适合 500x500 地图、T=600s 的场景。更大地图需要更多能量。
3. **速度档位**: 默认 6-20 m/s 共 15 档。可改为 `--speed_levels "10,15,20"` 指定离散速度。
4. **GPU**: 代码自动检测 CUDA，无 GPU 时使用 CPU。小规模问题 (M=3, K=15) CPU 可训练。
5. **TensorBoard**: `tensorboard --logdir logs/` 查看训练曲线。
6. **数据文件**: 训练前确保 `data/` 目录下有对应 `poi_{K}_map_{map_size}x{map_size}.npy` 文件，可用 `data_generate.py` 生成。
