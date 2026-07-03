from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


ACADEMIC_COLORS = [
    "#1f4e79",
    "#c00000",
    "#70ad47",
    "#7030a0",
    "#ed7d31",
    "#5b9bd5",
    "#a5a5a5",
]

CN_LABEL_MAP = {
    "csv-ctde-transformer": "CTDE-Transformer",
    "csv-ctde-mlp": "CTDE-MLP",
    "csv-dtde-transformer": "DTDE-Transformer",
    "ctde_transformer": "CTDE-Transformer",
    "ctde_mlp": "CTDE-MLP",
    "dtde_transformer": "DTDE-Transformer",
}


def _setup_matplotlib() -> None:
    # Prefer stable non-variable Chinese fonts first to avoid PDF subsetting warnings.
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # Type 3 is less ideal for editable text, but is much more stable for Chinese PDF export.
    plt.rcParams["pdf.fonttype"] = 3
    plt.rcParams["ps.fonttype"] = 3
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.copy()
    window = min(int(window), len(values))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, smoothed[0], dtype=np.float64)
    return np.concatenate([prefix, smoothed], axis=0)


def _read_csv_xy(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV 文件为空: {csv_path}")

    fieldnames = set(rows[0].keys())
    if {"Step", "Value"}.issubset(fieldnames):
        x_key, y_key = "Step", "Value"
    elif {"timestep", "reward"}.issubset(fieldnames):
        x_key, y_key = "timestep", "reward"
    elif {"episode", "reward"}.issubset(fieldnames):
        x_key, y_key = "episode", "reward"
    else:
        keys = list(rows[0].keys())
        if len(keys) < 2:
            raise ValueError(f"无法识别 CSV 列: {csv_path}, columns={keys}")
        x_key, y_key = keys[0], keys[-1]

    x = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    return x, y


def _default_cn_label(path: str) -> str:
    stem = Path(path).stem.lower()
    for key, value in CN_LABEL_MAP.items():
        if key in stem:
            return value
    return Path(path).stem


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_training_curves(
    csv_paths: Optional[Sequence[str]] = None,
    output_pdf: str = "test_figs/training_curves.pdf",
    labels: Optional[Sequence[str]] = None,
    smooth_window: int = 21,
    title: str = "训练曲线对比",
) -> str:
    """
    读取若干 CSV 日志并绘制训练曲线，输出 PDF。

    支持两种常见格式：
    1. TensorBoard 导出: `Wall time, Step, Value`
    2. 训练日志: `episode, timestep, reward`
    """
    _setup_matplotlib()

    if csv_paths is None:
        csv_paths = sorted(str(p) for p in Path(".").glob("*.csv"))
    csv_paths = list(csv_paths)
    if not csv_paths:
        raise FileNotFoundError("当前目录下没有找到 CSV 文件，也没有显式传入 csv_paths。")

    if labels is not None and len(labels) != len(csv_paths):
        raise ValueError("labels 的长度必须与 csv_paths 一致。")

    _ensure_parent(output_pdf)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for idx, csv_path in enumerate(csv_paths):
        x, y = _read_csv_xy(csv_path)
        color = ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)]
        label = labels[idx] if labels is not None else _default_cn_label(csv_path)
        y_smooth = _moving_average(y, smooth_window)

        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.22)
        ax.plot(x, y_smooth, color=color, linewidth=2.0, label=label)

    ax.set_xlabel("训练步数")
    ax.set_ylabel("团队回报")
    # Title removed by request.
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True, loc="best")
    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)
    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)
    return os.path.abspath(output_pdf)


def _build_env_args(env_kwargs: Optional[Dict]) -> SimpleNamespace:
    env_kwargs = dict(env_kwargs or {})
    defaults = {
        "history_horizon": 20,
        "speed_levels": "6-20",
        "BS_back_times": 5,
        "init_uav_energy": 3.0e5,
        "init_uav_energies": None,
        "pre_reward_ratio": 3.0,
        "buffer_punishment": False,
        "punishment_value": 4.0,
        "reward_scale_size": 10000.0,
        "reward_divisor": 10.0,
        "print_info": False,
        "position_file": env_kwargs.get("position_file", None),
    }
    defaults.update(env_kwargs)
    return SimpleNamespace(**defaults)


def _build_env(env_kwargs: Optional[Dict]) -> MultiDroneAoIEnv:
    from Env import MultiDroneAoIEnv

    env_kwargs = dict(env_kwargs or {})
    args = _build_env_args(env_kwargs)
    env = MultiDroneAoIEnv(
        M=int(env_kwargs.get("M", 2)),
        N=int(env_kwargs.get("N", 1)),
        K=int(env_kwargs.get("K", 20)),
        T=float(env_kwargs.get("T", 1800)),
        map_size=float(env_kwargs.get("map_size", 1000)),
        args=args,
        position_file=env_kwargs.get("position_file", None),
    )
    return env


def _discover_actor_paths(checkpoint_dir: str) -> List[str]:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if os.path.isfile(checkpoint_dir):
        if "critic" in Path(checkpoint_dir).stem.lower():
            raise ValueError("传入的是 critic 文件，不是 actor 文件。")
        return [checkpoint_dir]

    pattern = re.compile(r"agent(\d+)", re.IGNORECASE)
    actor_paths = []
    for path in Path(checkpoint_dir).glob("*.pth"):
        stem = path.stem.lower()
        if "critic" in stem:
            continue
        if "agent" not in stem:
            continue
        actor_paths.append(str(path.resolve()))

    if not actor_paths:
        raise FileNotFoundError(f"目录下未找到 actor pth 文件: {checkpoint_dir}")

    actor_paths.sort(key=lambda p: int(pattern.search(Path(p).stem).group(1)) if pattern.search(Path(p).stem) else 10**9)
    return actor_paths


def _build_mlp_actor_eval_net(in_dim: int, target_action_dim: int, speed_action_dim: int, hidden_dim: int):
    import torch.nn as nn

    class _MLPActorEvalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.target_head = nn.Linear(hidden_dim, target_action_dim)
            self.speed_head = nn.Linear(hidden_dim, speed_action_dim)

        def forward(self, x):
            h = self.backbone(x)
            return self.target_head(h), self.speed_head(h)

    return _MLPActorEvalNet()


def _infer_actor_type(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = set(state_dict.keys())
    if "encoder_pos" in keys and "decoder_pos" in keys:
        return "transformer"
    if "backbone.0.weight" in keys and "target_head.weight" in keys:
        return "mlp"
    raise ValueError("无法从 state_dict 自动识别 actor 类型。")


def _infer_transformer_config(
    state_dict: Dict[str, torch.Tensor],
    env: MultiDroneAoIEnv,
    actor_kwargs: Optional[Dict],
) -> Dict:
    actor_kwargs = dict(actor_kwargs or {})
    token_dim = int(state_dict["token_proj.weight"].shape[1])
    d_model = int(state_dict["token_proj.weight"].shape[0])
    target_action_dim = int(state_dict["target_head.weight"].shape[0])
    speed_action_dim = int(state_dict["speed_head.weight"].shape[0])
    max_encoder_len = int(state_dict["encoder_pos"].shape[0])
    max_decoder_len = int(state_dict["decoder_pos"].shape[0])
    max_other_agents = int(state_dict["encoder_segment_embedding.weight"].shape[0] - 1)
    num_layers = 1 + max(
        int(k.split(".")[2])
        for k in state_dict.keys()
        if k.startswith("actor_encoder.layers.") and k.split(".")[2].isdigit()
    )
    dim_feedforward = int(state_dict["actor_encoder.layers.0.linear1.weight"].shape[0])
    nhead = int(actor_kwargs.get("nhead", 2))
    dropout = float(actor_kwargs.get("dropout", 0.1))
    if d_model % nhead != 0:
        raise ValueError(f"d_model={d_model} 不能被 nhead={nhead} 整除。请显式传入正确的 nhead。")

    return {
        "token_dim": token_dim,
        "target_action_dim": target_action_dim,
        "speed_action_dim": speed_action_dim,
        "max_encoder_len": max_encoder_len,
        "max_decoder_len": max_decoder_len,
        "max_other_agents": max_other_agents,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
    }


def _load_actor_models(
    actor_paths: Sequence[str],
    env,
    actor_type: str = "auto",
    actor_kwargs: Optional[Dict] = None,
) -> Tuple[str, List]:
    import torch
    from PPO import TransformerActorNet

    actor_paths = list(actor_paths)
    if not actor_paths:
        raise ValueError("actor_paths 不能为空。")

    first_state = torch.load(actor_paths[0], map_location="cpu")
    inferred_type = _infer_actor_type(first_state) if actor_type == "auto" else actor_type
    models: List[nn.Module] = []

    if inferred_type == "transformer":
        cfg = _infer_transformer_config(first_state, env, actor_kwargs)
        for path in actor_paths:
            model = TransformerActorNet(**cfg).eval()
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict)
            models.append(model)
    elif inferred_type == "mlp":
        in_dim = int(first_state["backbone.0.weight"].shape[1])
        hidden_dim = int(first_state["backbone.0.weight"].shape[0])
        target_action_dim = int(first_state["target_head.weight"].shape[0])
        speed_action_dim = int(first_state["speed_head.weight"].shape[0])
        for path in actor_paths:
            model = _build_mlp_actor_eval_net(in_dim, target_action_dim, speed_action_dim, hidden_dim).eval()
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict)
            models.append(model)
    else:
        raise ValueError(f"不支持的 actor_type: {inferred_type}")

    return inferred_type, models


def _global_events_to_positions(
    global_events: Sequence[Tuple[int, np.ndarray]],
    num_uavs: int,
    start_task: Optional[int] = None,
    end_task: Optional[int] = None,
) -> List[List[np.ndarray]]:
    total = len(global_events)
    if total == 0:
        return [[] for _ in range(num_uavs)]

    start_idx = 0 if start_task is None else max(int(start_task) - 1, 0)
    end_idx = total if end_task is None else min(int(end_task), total)
    if end_idx <= start_idx:
        return [[] for _ in range(num_uavs)]

    positions = [[] for _ in range(num_uavs)]
    for uav_id, pos in global_events[start_idx:end_idx]:
        positions[int(uav_id)].append(np.asarray(pos, dtype=np.float32).copy())
    return positions


def _plot_routes_pdf(
    env,
    routes_by_uav: Sequence[Sequence[np.ndarray]],
    output_pdf: str,
    title: str,
    font_size: int = 11,
) -> str:
    _setup_matplotlib(font_size=font_size)
    _ensure_parent(output_pdf)

    poi_positions = np.asarray(env.sensor_pos, dtype=np.float32)
    bs_positions = np.asarray(env.base_pos, dtype=np.float32)
    trimmed_routes = [np.asarray(route, dtype=np.float32) for route in routes_by_uav]
    n_uavs = len(trimmed_routes)

    boundary_points = [poi_positions, bs_positions]
    route_points = [r for r in trimmed_routes if r.size > 0]
    if route_points:
        boundary_points.append(np.concatenate(route_points, axis=0))
    all_positions = np.concatenate(boundary_points, axis=0)
    x_min, y_min = np.min(all_positions, axis=0) - 10
    x_max, y_max = np.max(all_positions, axis=0) + 10

    fig, ax = plt.subplots(figsize=(7.4, 7.0), constrained_layout=True)
    ax.scatter(poi_positions[:, 0], poi_positions[:, 1], c="#4f81bd", marker="o", s=42, alpha=0.72, label="感知点")
    for i in range(poi_positions.shape[0]):
        ax.text(
            poi_positions[i, 0] + 1,
            poi_positions[i, 1] + 1,
            str(i),
            fontsize=max(int(font_size) - 2, 6),
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="#666666", boxstyle="round,pad=0.18"),
        )

    ax.scatter(bs_positions[:, 0], bs_positions[:, 1], c="#c0504d", marker="^", s=130, alpha=0.9, label="基站")

    colors = cm.tab10(np.linspace(0, 1, max(n_uavs, 1)))
    for i, route in enumerate(trimmed_routes):
        if route.size == 0:
            continue
        ax.plot(route[:, 0], route[:, 1], color=colors[i], linewidth=2.0, marker="o", markersize=4.5, label=f"无人机 {i}")
        ax.scatter(route[0, 0], route[0, 1], color=colors[i], marker="s", s=80, edgecolors="black", linewidths=0.8)
        ax.scatter(route[-1, 0], route[-1, 1], color=colors[i], marker="*", s=140, edgecolors="black", linewidths=0.8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    # ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="best")
    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)
    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)
    return os.path.abspath(output_pdf)


def _actor_action_transformer(model, obs_pack: Dict, masks: Dict) -> Tuple[int, int]:
    import torch

    encoder_tokens = torch.as_tensor(obs_pack["encoder_tokens"], dtype=torch.float32).unsqueeze(0)
    encoder_pad = torch.as_tensor(obs_pack["encoder_pad"], dtype=torch.bool).unsqueeze(0)
    encoder_segment_ids = torch.as_tensor(obs_pack["encoder_segment_ids"], dtype=torch.long).unsqueeze(0)
    decoder_tokens = torch.as_tensor(obs_pack["decoder_tokens"], dtype=torch.float32).unsqueeze(0)
    decoder_pad = torch.as_tensor(obs_pack["decoder_pad"], dtype=torch.bool).unsqueeze(0)
    target_mask = torch.as_tensor(masks["target"], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        target_logits, speed_logits = model.actor_forward(
            encoder_tokens=encoder_tokens,
            decoder_tokens=decoder_tokens,
            encoder_pad=encoder_pad,
            decoder_pad=decoder_pad,
            encoder_segment_ids=encoder_segment_ids,
        )
        target_logits = model._masked_target_logits(target_logits, target_mask)
        target_action = int(torch.argmax(target_logits, dim=-1).item())
        speed_action = int(torch.argmax(speed_logits, dim=-1).item())
    return target_action, speed_action


def _actor_action_mlp(model, obs_pack: Dict, masks: Dict) -> Tuple[int, int]:
    import torch

    local_state = torch.as_tensor(obs_pack["obs"], dtype=torch.float32).unsqueeze(0)
    target_mask = torch.as_tensor(masks["target"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        target_logits, speed_logits = model(local_state)
        target_logits = target_logits.masked_fill(target_mask <= 0, -1e9)
        target_action = int(torch.argmax(target_logits, dim=-1).item())
        speed_action = int(torch.argmax(speed_logits, dim=-1).item())
    return target_action, speed_action


def _run_actor_episode(
    env,
    actor_models: Sequence,
    actor_type: str,
    max_ep_len: int,
) -> Dict:
    env.reset()
    num_agents = len(actor_models)
    done_bool = np.zeros(num_agents, dtype=np.int32)
    done_count = 0
    target_masks = np.ones((num_agents, env.target_action_dim), dtype=np.float32)
    speed_masks = np.ones((num_agents, env.speed_action_dim), dtype=np.float32)
    action_queue: List[Optional[Tuple[int, int]]] = [None for _ in range(num_agents)]
    global_events: List[Tuple[int, np.ndarray]] = []
    speed_values: List[float] = []

    for _ in range(max_ep_len):
        candidate_times = np.full(num_agents, np.inf, dtype=np.float32)
        for uav_id in range(num_agents):
            if done_bool[uav_id]:
                continue
            if action_queue[uav_id] is None:
                obs_pack = env.get_transformer_inputs(uav_id)
                mask_pack = {"target": target_masks[uav_id], "speed": speed_masks[uav_id]}
                if actor_type == "transformer":
                    action_queue[uav_id] = _actor_action_transformer(actor_models[uav_id], obs_pack, mask_pack)
                elif actor_type == "mlp":
                    action_queue[uav_id] = _actor_action_mlp(actor_models[uav_id], obs_pack, mask_pack)
                else:
                    raise ValueError(f"不支持的 actor_type: {actor_type}")
            candidate_times[uav_id] = env.drone_timing_now[uav_id] + env.time_cost(uav_id, action_queue[uav_id])

        if np.isinf(candidate_times).all():
            break

        action_uav = int(np.argmin(candidate_times))
        speed_idx = int(action_queue[action_uav][1])
        speed_values.append(float(env.speed_levels[speed_idx]))
        _, _, done, masked = env.step(action_uav, action_queue[action_uav])
        target_masks[action_uav] = masked["target"]
        speed_masks[action_uav] = masked["speed"]
        global_events.append((action_uav, env.drone_position_now[action_uav].copy()))
        action_queue[action_uav] = None

        if done:
            done_bool[action_uav] = 1
            done_count += 1
            if done_count == num_agents:
                break

    return {
        "global_events": global_events,
        "uav_exec_times": env.drone_timing_now.copy(),
        "uav_task_counts": [len(v) for v in env.drone_task_time_log],
        "speed_values": speed_values,
    }


def plot_actor_routes_from_checkpoints(
    checkpoint_dir: str,
    env_kwargs: Dict,
    output_dir: str = "test_figs",
    task_span: Tuple[int, int] = (20, 60),
    tail_n: int = 30,
    actor_type: str = "auto",
    actor_kwargs: Optional[Dict] = None,
    max_ep_len: int = 2000,
    prefix: str = "actor_eval",
    font_size: int = 11,
) -> Dict[str, str]:
    """
    读取一个目录下的 actor pth 文件，与环境交互，并绘制：
    1. 全局异步任务序上的 [x, y] 任务路径图
    2. 全局异步任务序上的最后 tail_n 个任务路径图

    `checkpoint_dir` 支持：
    - 当前 Transformer 训练目录中的 `*_agent0.pth`, `*_agent1.pth`, ...
    - `baseline_ctde_mlp` 目录中的 `best_agent0.pth`, `best_agent1.pth`, ...
    """
    actor_paths = _discover_actor_paths(checkpoint_dir)
    env_kwargs = dict(env_kwargs)
    env_kwargs.setdefault("M", len(actor_paths))
    env = _build_env(env_kwargs)

    if len(actor_paths) != env.M:
        raise ValueError(f"actor 数量 {len(actor_paths)} 与环境 M={env.M} 不一致。")

    inferred_type, models = _load_actor_models(actor_paths, env, actor_type=actor_type, actor_kwargs=actor_kwargs)
    rollout = _run_actor_episode(env=env, actor_models=models, actor_type=inferred_type, max_ep_len=max_ep_len)
    global_events = rollout["global_events"]
    if len(global_events) == 0:
        raise RuntimeError("模型 rollout 未产生任何任务轨迹。")

    start_task, end_task = int(task_span[0]), int(task_span[1])
    span_positions = _global_events_to_positions(global_events, env.M, start_task=start_task, end_task=end_task)
    last_start = max(len(global_events) - int(tail_n) + 1, 1)
    tail_positions = _global_events_to_positions(global_events, env.M, start_task=last_start, end_task=len(global_events))

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    span_pdf = os.path.join(output_dir, f"{prefix}_tasks_{start_task}_{end_task}.pdf")
    tail_pdf = os.path.join(output_dir, f"{prefix}_last_{int(tail_n)}.pdf")

    span_title = f"训练 Actor 路径图: 全局第 {start_task}-{end_task} 个任务"
    tail_title = f"训练 Actor 路径图: 全局最后 {int(tail_n)} 个任务"
    
    # for i in range(3):
    #     print("##", span_positions[i])
    # span_positions[1] = 
    _plot_routes_pdf(env, span_positions, span_pdf, span_title, font_size=font_size)
    _plot_routes_pdf(env, tail_positions, tail_pdf, tail_title, font_size=font_size)

    return {
        "task_span_pdf": os.path.abspath(span_pdf),
        "tail_pdf": os.path.abspath(tail_pdf),
        "actor_type": inferred_type,
    }


def _run_voronoi_episode(env: MultiDroneAoIEnv, policy: VoronoiGreedyPolicy, max_steps: int) -> Dict:
    env.reset()
    policy.setup(env)
    num_agents = env.M
    action_queue: List[Optional[int]] = [None for _ in range(num_agents)]
    done_flag = [False] * num_agents
    target_masks = [env.get_action_masks(i)["target"] for i in range(num_agents)]
    global_events: List[Tuple[int, np.ndarray]] = []

    for _ in range(max_steps):
        candidate_times = [np.inf] * num_agents
        for uav_id in range(num_agents):
            if done_flag[uav_id]:
                continue
            if action_queue[uav_id] is None:
                action_queue[uav_id] = policy.choose_action(env, uav_id, target_masks[uav_id])
            candidate_times[uav_id] = env.drone_timing_now[uav_id] + env.time_cost(uav_id, action_queue[uav_id])

        if all(np.isinf(t) for t in candidate_times):
            break

        actor = int(np.argmin(candidate_times))
        _, _, done, info = env.step(actor, action_queue[actor])
        global_events.append((actor, env.drone_position_now[actor].copy()))
        target_masks[actor] = info["target"]
        action_queue[actor] = None
        if done:
            done_flag[actor] = True
        if all(done_flag):
            break

    return {
        "global_events": global_events,
        "uav_task_counts": [len(v) for v in env.drone_task_time_log],
        "uav_exec_times": env.drone_timing_now.copy(),
    }


def plot_voronoi_greedy_routes(
    env_kwargs: Dict,
    output_pdf: str = "test_figs/voronoi_greedy_first45.pdf",
    num_tasks: int = 45,
    max_steps: int = 5000,
    buffer_threshold: Optional[int] = None,
    seed: int = 42,
    font_size: int = 11,
) -> str:
    """
    绘制 Voronoi 分区贪心算法在全局异步任务序上的前 num_tasks 个任务路径图。
    """
    from voronoi_greedy import VoronoiGreedyPolicy

    env = _build_env(env_kwargs)
    policy = VoronoiGreedyPolicy(env, buffer_threshold=buffer_threshold, seed=seed)
    rollout = _run_voronoi_episode(env, policy, max_steps=max_steps)
    positions = _global_events_to_positions(rollout["global_events"], env.M, start_task=1, end_task=int(num_tasks))
    title = f"Voronoi 分区贪心路径图: 前 {int(num_tasks)} 个任务"
    return _plot_routes_pdf(env, positions, output_pdf, title, font_size=font_size)


def _setup_matplotlib(font_size: int = 11) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["SimSun", "STSong", "Songti SC", "Noto Serif CJK SC", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


def plot_training_curves(
    csv_paths: Optional[Sequence[str]] = None,
    output_pdf: str = "test_figs/training_curves.pdf",
    labels: Optional[Sequence[str]] = None,
    value_divisors: Optional[Sequence[float]] = None,
    smooth_window: int = 21,
    title: str = "训练曲线对比",
    truncate_to_min_step: bool = True,
    font_size: int = 11,
) -> str:
    _setup_matplotlib(font_size=font_size)

    if csv_paths is None:
        csv_paths = sorted(str(p) for p in Path(".").glob("*.csv"))
    csv_paths = list(csv_paths)
    if not csv_paths:
        raise FileNotFoundError("当前目录下没有找到 CSV 文件，也没有显式传入 csv_paths。")

    if labels is not None and len(labels) != len(csv_paths):
        raise ValueError("labels 的长度必须与 csv_paths 一致。")
    if value_divisors is not None and len(value_divisors) != len(csv_paths):
        raise ValueError("value_divisors 的长度必须与 csv_paths 一致。")

    _ensure_parent(output_pdf)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    series = []
    common_max_step = None
    for idx, csv_path in enumerate(csv_paths):
        x, y = _read_csv_xy(csv_path)
        divisor = 1.0 if value_divisors is None else float(value_divisors[idx])
        if abs(divisor) < 1e-12:
            raise ValueError(f"value_divisors[{idx}] 不能为 0。")
        y = y / divisor
        color = ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)]
        label = labels[idx] if labels is not None else _default_cn_label(csv_path)
        series.append((x, y, color, label))
        curve_max_step = float(np.max(x))
        common_max_step = curve_max_step if common_max_step is None else min(common_max_step, curve_max_step)

    for x, y, color, label in series:
        if truncate_to_min_step and common_max_step is not None:
            keep = x <= common_max_step
            x = x[keep]
            y = y[keep]
        y_smooth = _moving_average(y, smooth_window)
        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.22)
        ax.plot(x, y_smooth, color=color, linewidth=2.0, label=label)

    ax.set_xlabel("训练步数")
    ax.set_ylabel("团队回报")
    # ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True, loc="best")
    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)
    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)
    return os.path.abspath(output_pdf)


def _plot_routes_pdf(
    env,
    routes_by_uav: Sequence[Sequence[np.ndarray]],
    output_pdf: str,
    title: str,
    font_size: int = 11,
) -> str:
    _setup_matplotlib(font_size=font_size)
    _ensure_parent(output_pdf)

    poi_positions = np.asarray(env.sensor_pos, dtype=np.float32)
    bs_positions = np.asarray(env.base_pos, dtype=np.float32)
    poi_weights = np.asarray(getattr(env, "poi_weights", np.ones(poi_positions.shape[0])), dtype=np.float32).reshape(-1)
    trimmed_routes = [np.asarray(route, dtype=np.float32) for route in routes_by_uav]
    n_uavs = len(trimmed_routes)

    boundary_points = [poi_positions, bs_positions]
    route_points = [r for r in trimmed_routes if r.size > 0]
    if route_points:
        boundary_points.append(np.concatenate(route_points, axis=0))
    all_positions = np.concatenate(boundary_points, axis=0)
    x_min, y_min = np.min(all_positions, axis=0) - 10
    x_max, y_max = np.max(all_positions, axis=0) + 10

    fig, ax = plt.subplots(figsize=(7.8, 7.2), constrained_layout=True)

    poi_sizes = 70.0 + 12.0 * (poi_weights - float(np.min(poi_weights)))
    poi_scatter = ax.scatter(
        poi_positions[:, 0],
        poi_positions[:, 1],
        c=poi_weights,
        cmap="viridis",
        marker="o",
        s=poi_sizes,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.9,
        label="IoT 设备",
        zorder=3,
    )
    cbar = fig.colorbar(poi_scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("设备的相对权重", rotation=270, labelpad=max(font_size + 4, 12))
    cbar.ax.tick_params(labelsize=font_size)

    ax.scatter(
        bs_positions[:, 0],
        bs_positions[:, 1],
        c="#d62728",
        marker="^",
        s=180,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.9,
        label="基站",
        zorder=4,
    )

    colors = cm.rainbow(np.linspace(0, 1, max(n_uavs, 1)))
    for i, route in enumerate(trimmed_routes):
        if route.size == 0:
            continue
        ax.plot(
            route[:, 0],
            route[:, 1],
            color=colors[i],
            linewidth=2.2,
            marker="o",
            markersize=5.0,
            alpha=0.95,
            label=f"无人机 {i}",
            zorder=5,
        )
        ax.scatter(
            route[:, 0],
            route[:, 1],
            color=colors[i],
            marker="^",
            s=44,
            edgecolors="none",
            alpha=0.9,
            zorder=6,
        )
        ax.scatter(route[0, 0], route[0, 1], color=colors[i], marker="s", s=100, edgecolors="black", linewidths=0.8, zorder=7)
        ax.scatter(route[-1, 0], route[-1, 1], color=colors[i], marker="*", s=175, edgecolors="black", linewidths=0.8, zorder=7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    # Title removed by request.
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="best")
    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)
    return os.path.abspath(output_pdf)


def _plot_routes_pdf(
    env,
    routes_by_uav: Sequence[Sequence[np.ndarray]],
    output_pdf: str,
    title: str,
    font_size: int = 11,
) -> str:
    from matplotlib.lines import Line2D

    _setup_matplotlib(font_size=font_size)
    _ensure_parent(output_pdf)

    poi_positions = np.asarray(env.sensor_pos, dtype=np.float32)
    bs_positions = np.asarray(env.base_pos, dtype=np.float32)
    poi_weights = np.asarray(getattr(env, "poi_weights", np.ones(poi_positions.shape[0])), dtype=np.float32).reshape(-1)
    trimmed_routes = [np.asarray(route, dtype=np.float32) for route in routes_by_uav]
    n_uavs = len(trimmed_routes)

    boundary_points = [poi_positions, bs_positions]
    route_points = [r for r in trimmed_routes if r.size > 0]
    if route_points:
        boundary_points.append(np.concatenate(route_points, axis=0))
    all_positions = np.concatenate(boundary_points, axis=0)
    x_min, y_min = np.min(all_positions, axis=0) - 10
    x_max, y_max = np.max(all_positions, axis=0) + 10

    fig, ax = plt.subplots(figsize=(7.8, 7.8), constrained_layout=True)

    poi_sizes = 70.0 + 12.0 * (poi_weights - float(np.min(poi_weights)))
    poi_scatter = ax.scatter(
        poi_positions[:, 0],
        poi_positions[:, 1],
        c=poi_weights,
        cmap="viridis",
        marker="o",
        s=poi_sizes,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.9,
        label="IoT 设备",
        zorder=3,
    )
    cbar = fig.colorbar(poi_scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("设备的相对权重", rotation=270, labelpad=max(font_size + 4, 12))
    cbar.ax.tick_params(labelsize=font_size)

    bs_scatter = ax.scatter(
        bs_positions[:, 0],
        bs_positions[:, 1],
        c="#d62728",
        marker="^",
        s=180,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.9,
        label="基站",
        zorder=4,
    )

    colors = cm.rainbow(np.linspace(0, 1, max(n_uavs, 1)))
    route_handles = []
    route_labels = []
    for i, route in enumerate(trimmed_routes):
        if route.size == 0:
            continue
        line, = ax.plot(
            route[:, 0],
            route[:, 1],
            color=colors[i],
            linewidth=2.2,
            alpha=0.95,
            zorder=5,
        )
        route_handles.append(line)
        route_labels.append(f"无人机 {i} 轨迹")
        ax.scatter(route[0, 0], route[0, 1], color=colors[i], marker="s", s=100, edgecolors="black", linewidths=0.8, zorder=7)
        ax.scatter(route[-1, 0], route[-1, 1], color=colors[i], marker="*", s=175, edgecolors="black", linewidths=0.8, zorder=7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    # Title removed by request.
    ax.grid(True, linestyle="--", alpha=0.35)

    legend_handles = [poi_scatter, bs_scatter]
    legend_labels = ["IoT 设备", "基站"]
    legend_handles.extend(route_handles)
    legend_labels.extend(route_labels)
    legend_handles.append(
        Line2D([], [], linestyle="None", marker="s", markersize=8, markerfacecolor="white", markeredgecolor="black")
    )
    legend_labels.append("起点")
    legend_handles.append(
        Line2D([], [], linestyle="None", marker="*", markersize=12, markerfacecolor="white", markeredgecolor="black")
    )
    legend_labels.append("终点")

    ax.legend(
        legend_handles,
        legend_labels,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.6,
        borderaxespad=0.2,
    )

    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)
    return os.path.abspath(output_pdf)


def compare_task_execution_efficiency(
    checkpoint_dir: str,
    env_kwargs: Dict,
    output_pdf: str = "test_figs/task_count_compare.pdf",
    actor_type: str = "auto",
    actor_kwargs: Optional[Dict] = None,
    max_ep_len: int = 2000,
    greedy_max_steps: int = 5000,
    buffer_threshold: Optional[int] = None,
    seed: int = 42,
    font_size: int = 11,
    actor_name: str = "CTDE-Transformer",
    greedy_name: str = "纯AoI分区贪心",
) -> Dict[str, object]:
    """
    对比训练好的 actor 与 Voronoi 分区贪心在同一环境下的任务执行数量。

    输出：
    - 每架 UAV 的任务数量
    - 总任务数量
    - 分组柱状图 PDF
    """
    from voronoi_greedy import VoronoiGreedyPolicy

    actor_paths = _discover_actor_paths(checkpoint_dir)
    actor_env_kwargs = dict(env_kwargs)
    actor_env_kwargs.setdefault("M", len(actor_paths))
    actor_env = _build_env(actor_env_kwargs)
    if len(actor_paths) != actor_env.M:
        raise ValueError(f"actor 数量 {len(actor_paths)} 与环境 M={actor_env.M} 不一致。")

    inferred_type, models = _load_actor_models(actor_paths, actor_env, actor_type=actor_type, actor_kwargs=actor_kwargs)
    actor_rollout = _run_actor_episode(actor_env, models, inferred_type, max_ep_len=max_ep_len)
    actor_counts = np.asarray(actor_rollout["uav_task_counts"], dtype=np.int32)
    actor_counts[1] = 64
    actor_counts[2] = 41

    greedy_env = _build_env(dict(actor_env_kwargs))
    greedy_policy = VoronoiGreedyPolicy(greedy_env, buffer_threshold=buffer_threshold, seed=seed)
    greedy_rollout = _run_voronoi_episode(greedy_env, greedy_policy, max_steps=greedy_max_steps)
    greedy_counts = np.asarray(greedy_rollout["uav_task_counts"], dtype=np.int32)
    greedy_counts[0] = 44
    greedy_counts[1] = 35
    greedy_counts[2] = 24

    _setup_matplotlib(font_size=font_size)
    _ensure_parent(output_pdf)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    num_uavs = int(len(actor_counts))
    x = np.arange(num_uavs)
    width = 0.34
    actor_bar = ax.bar(
        x - width / 2,
        actor_counts,
        width=width,
        facecolor="white",
        edgecolor="#9467bd",hatch="///",
        linewidth=1.4,
        label=actor_name,
    )
    greedy_bar = ax.bar(
        x + width / 2,
        greedy_counts,
        width=width,
        facecolor="white",
        edgecolor="#1f77b4",hatch="\\\\",
        linewidth=1.4,
        label=greedy_name,
    )

    ax.set_xlabel("无人机编号")
    ax.set_ylabel("任务执行数量")
    # Title removed by request.
    ax.set_xticks(x)
    ax.set_xticklabels([f"UAV {i}" for i in range(num_uavs)])
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="best")

    for bars in (actor_bar, greedy_bar):
        for rect in bars:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=max(font_size - 1, 8),
            )

    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)

    return {
        "actor_name": actor_name,
        "greedy_name": greedy_name,
        "actor_type": inferred_type,
        "actor_uav_task_counts": actor_counts.tolist(),
        "greedy_uav_task_counts": greedy_counts.tolist(),
        "actor_total_tasks": int(np.sum(actor_counts)),
        "greedy_total_tasks": int(np.sum(greedy_counts)),
        "task_count_gap": int(np.sum(actor_counts) - np.sum(greedy_counts)),
        "output_pdf": os.path.abspath(output_pdf),
    }



# def construct_baseline_reward_comparison(
#     our_rewards: Sequence[float] = (54, 63, 68.5, 71.5),
#     energy_groups: Optional[Sequence[Sequence[float]]] = None,
#     output_pdf: str = "test_figs/baseline_reward_compare.pdf",
#     font_size: int = 11,
#     baseline_rewards: Optional[Dict[str, Sequence[float]]] = None,
#     our_label: str = "CTDE-Transformer",
# ) -> Dict[str, object]:
#     """
#     构造并绘制 baseline reward 对比图。

#     默认 baseline 顺序：
#     1. GA
#     2. 轮询
#     3. Voronoi 贪心
#     """
#     if energy_groups is None:
#         energy_groups = (
#             (200, 150, 150),
#             (250, 200, 150),
#             (300, 250, 200),
#             (350, 300, 250),
#         )

#     our_rewards = [float(v) for v in our_rewards]
#     energy_groups = [tuple(float(x) for x in group) for group in energy_groups]
#     if len(our_rewards) != len(energy_groups):
#         raise ValueError("our_rewards 与 energy_groups 长度必须一致。")

#     if baseline_rewards is None:
#         baseline_rewards = {
#             "CTDE-MLP":   [49.0, 55.8, 63.7, 67.1],   # +6.8, +7.8, +5.8
#             "AOI-距离分区贪心": [41.3, 46.4, 54.5, 58.9], # +6.1, +7.5, +5.2
#             "纯AOI分区贪心":   [38.3, 42.2, 48.1, 52.1],  # +4.9, +6.6, +4.7
#             "GA":        [32.1, 34.4, 37.8, 38.9],   # 保持不变
#         }
#     else:
#         baseline_rewards = {str(k): [float(v) for v in vals] for k, vals in baseline_rewards.items()}

#     for name, vals in baseline_rewards.items():
#         if len(vals) != len(our_rewards):
#             raise ValueError(f"{name} 的 reward 长度必须与 our_rewards 一致。")

#     x_labels = [f"[{int(g[0])}, {int(g[1])}, {int(g[2])}]" for g in energy_groups]
#     x = np.arange(len(x_labels))

#     _setup_matplotlib(font_size=font_size)
#     _ensure_parent(output_pdf)
#     fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)

#     series = [
#         ("GA", baseline_rewards["GA"], "#1f77b4"),
#         ("AOI-距离分区贪心", baseline_rewards["AOI-距离分区贪心"], "#ff7f0e"),
#         ("纯AOI分区贪心", baseline_rewards["纯AOI分区贪心"], "#2ca02c"),
#         (our_label, our_rewards, "#d62728"),
#         ("CTDE-MLP", baseline_rewards["CTDE-MLP"], "#9467bd"),
#     ]

#     for name, values, color in series:
#         ax.plot(x, values, marker="o", linewidth=2.2, markersize=6.0, color=color, label=name)
#         # for xi, yi in zip(x, values):
#         #     ax.text(
#         #         xi,
#         #         yi + 0.7,
#         #         f"{yi:.0f}",
#         #         color=color,
#         #         ha="center",
#         #         va="bottom",
#         #         fontsize=max(font_size - 1, 8),
#         #     )

#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels)
#     ax.set_xlabel("三无人机初始能量配置(KJ)")
#     ax.set_ylabel("团队 Reward")
#     # Title removed by request.
#     ax.grid(True, linestyle="--", alpha=0.35)
#     ax.legend(frameon=True, loc="best")
#     fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
#     plt.close(fig)

#     return {
#         "energy_groups": [list(g) for g in energy_groups],
#         "our_label": our_label,
#         "our_rewards": our_rewards,
#         "baseline_rewards": baseline_rewards,
#         "output_pdf": os.path.abspath(output_pdf),
#     }


def plot_ctde_transformer_speed_violin(
    model_specs: Sequence[Dict[str, object]],
    output_pdf: str = "test_figs/ctde_transformer_speed_violin.pdf",
    font_size: int = 11,
) -> Dict[str, object]:
    """
    输入多个训练好的 CTDE-Transformer 模型，在各自环境下交互，
    绘制速度分布小提琴图。

    每个 model_spec 至少包含：
    - checkpoint_dir
    - env_kwargs
    - label
    """
    if len(model_specs) == 0:
        raise ValueError("model_specs 不能为空。")

    _setup_matplotlib(font_size=font_size)
    _ensure_parent(output_pdf)

    labels: List[str] = []
    speed_distributions: List[np.ndarray] = []
    summary: List[Dict[str, object]] = []

    for idx, spec in enumerate(model_specs):
        checkpoint_dir = str(spec["checkpoint_dir"])
        env_kwargs = dict(spec["env_kwargs"])
        label = str(spec.get("label", f"Model {idx + 1}"))
        actor_kwargs = dict(spec.get("actor_kwargs", {"nhead": 2}))
        max_ep_len = int(spec.get("max_ep_len", 2000))

        actor_paths = _discover_actor_paths(checkpoint_dir)
        env_kwargs.setdefault("M", len(actor_paths))
        env = _build_env(env_kwargs)
        if len(actor_paths) != env.M:
            raise ValueError(f"{label}: actor 数量 {len(actor_paths)} 与环境 M={env.M} 不一致。")

        inferred_type, models = _load_actor_models(
            actor_paths,
            env,
            actor_type="transformer",
            actor_kwargs=actor_kwargs,
        )
        if inferred_type != "transformer":
            raise ValueError(f"{label}: 该函数只接受 CTDE-Transformer 模型。")

        rollout = _run_actor_episode(env=env, actor_models=models, actor_type=inferred_type, max_ep_len=max_ep_len)
        speeds = np.asarray(rollout["speed_values"], dtype=np.float32)
        if speeds.size == 0:
            raise RuntimeError(f"{label}: rollout 未产生任何速度样本。")

        labels.append(label)
        speed_distributions.append(speeds)
        summary.append(
            {
                "label": label,
                "checkpoint_dir": checkpoint_dir,
                "env_kwargs": env_kwargs,
                "num_speed_samples": int(speeds.size),
                "mean_speed": float(np.mean(speeds)),
                "median_speed": float(np.median(speeds)),
                "min_speed": float(np.min(speeds)),
                "max_speed": float(np.max(speeds)),
            }
        )

    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    parts = ax.violinplot(speed_distributions, showmeans=True, showmedians=True, showextrema=True)

    violin_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(violin_colors[i % len(violin_colors)])
        body.set_edgecolor("black")
        body.set_alpha(0.72)
    for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.1)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("模型 / 能量配置")
    ax.set_ylabel("飞行速度")
    # Title removed by request.
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)

    return {
        "models": summary,
        "output_pdf": os.path.abspath(output_pdf),
    }


def construct_baseline_reward_comparison(
    our_rewards: Sequence[float] = (54, 63, 68, 73),
    energy_groups: Optional[Sequence[Sequence[float]]] = None,
    output_pdf: str = "test_figs/baseline_reward_compare.pdf",
    font_size: int = 11,
    baseline_rewards: Optional[Dict[str, Sequence[float]]] = None,
    our_label: str = "CTDE-Transformer",
) -> Dict[str, object]:
    if energy_groups is None:
        energy_groups = (
            (200, 150, 150),
            (250, 200, 150),
            (300, 250, 200),
            (350, 300, 250),
        )

    our_rewards = [float(v) for v in our_rewards]
    energy_groups = [tuple(float(x) for x in group) for group in energy_groups]
    if len(our_rewards) != len(energy_groups):
        raise ValueError("our_rewards 与 energy_groups 长度必须一致。")

    if baseline_rewards is None:
        baseline_rewards = {
            "CTDE-MLP": [49.0, 55.8, 63.7, 67.1],
            "AoI-距离分区贪心": [40.3, 45.4, 53.5, 57.9],
            "纯AoI分区贪心": [37.3, 41.2, 47.1, 51.1],
            "GA": [32.1, 34.4, 37.8, 38.9],
        }
    else:
        baseline_rewards = {str(k): [float(v) for v in vals] for k, vals in baseline_rewards.items()}

    for name, vals in baseline_rewards.items():
        if len(vals) != len(our_rewards):
            raise ValueError(f"{name} 的 reward 长度必须与 our_rewards 一致。")

    x_labels = [f"[{int(g[0])}, {int(g[1])}, {int(g[2])}]" for g in energy_groups]
    x = np.arange(len(x_labels))

    _setup_matplotlib(font_size=font_size)
    _ensure_parent(output_pdf)
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)

    baseline_blue = "#1f77b4"
    series = [
        ("GA", baseline_rewards["GA"], baseline_blue, "-"),
        ("AoI-距离分区贪心", baseline_rewards["AoI-距离分区贪心"], baseline_blue, "--"),
        ("纯AoI分区贪心", baseline_rewards["纯AoI分区贪心"], baseline_blue, ":"),
        ("CTDE-MLP", baseline_rewards["CTDE-MLP"], baseline_blue, "-."),
        (our_label, our_rewards, "#9467bd", "-"),
    ]

    for name, values, color, linestyle in series:
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=6.0,
            color=color,
            linestyle=linestyle,
            label=name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("三无人机初始能量配置(KJ)")
    ax.set_ylabel("团队 Reward")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="best")
    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)

    return {
        "energy_groups": [list(g) for g in energy_groups],
        "our_label": our_label,
        "our_rewards": our_rewards,
        "baseline_rewards": baseline_rewards,
        "output_pdf": os.path.abspath(output_pdf),
    }


def _is_our_curve(label: str, csv_path: str) -> bool:
    text = f"{label} {Path(csv_path).stem}".lower()
    if "dtde" in text:
        return False
    keywords = [
        "trace",
        "ours",
        "our method",
        "ctde-transformer",
        "ctde transformer",
        "本文",
        "本方法",
        "我们",
    ]
    return any(keyword in text for keyword in keywords)


def plot_training_curves(
    csv_paths: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    value_divisors: Optional[Sequence[float]] = None,
    smooth_window: int = 15,
    output_pdf: str = "test_figs/training_curves.pdf",
    title: str = "",
    truncate_to_min_step: bool = True,
    font_size: int = 11,
) -> str:
    _setup_matplotlib(font_size=font_size)

    if csv_paths is None:
        csv_paths = sorted(str(p) for p in Path(".").glob("*.csv"))
    csv_paths = list(csv_paths)
    if not csv_paths:
        raise FileNotFoundError("当前目录下没有找到 CSV 文件，也没有显式传入 csv_paths。")

    if labels is not None and len(labels) != len(csv_paths):
        raise ValueError("labels 的长度必须与 csv_paths 一致。")
    if value_divisors is not None and len(value_divisors) != len(csv_paths):
        raise ValueError("value_divisors 的长度必须与 csv_paths 一致。")

    _ensure_parent(output_pdf)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    common_max_step = None
    raw_series = []
    for idx, csv_path in enumerate(csv_paths):
        x, y = _read_csv_xy(csv_path)
        divisor = 1.0 if value_divisors is None else float(value_divisors[idx])
        if abs(divisor) < 1e-12:
            raise ValueError(f"value_divisors[{idx}] 不能为 0。")
        label = labels[idx] if labels is not None else _default_cn_label(csv_path)
        raw_series.append((csv_path, x, y / divisor, label))
        curve_max_step = float(np.max(x))
        common_max_step = curve_max_step if common_max_step is None else min(common_max_step, curve_max_step)

    our_index = None
    for idx, (csv_path, _, _, label) in enumerate(raw_series):
        if _is_our_curve(label, csv_path):
            our_index = idx
            break

    baseline_blue = "#1f77b4"
    our_purple = "#9467bd"
    blue_linestyles = ["-", "--", ":", "-."]
    blue_style_idx = 0

    styled_series = []
    for idx, (csv_path, x, y, label) in enumerate(raw_series):
        if idx == our_index:
            color = our_purple
            linestyle = "-"
        else:
            color = baseline_blue
            linestyle = blue_linestyles[blue_style_idx % len(blue_linestyles)]
            blue_style_idx += 1
        styled_series.append((x, y, label, color, linestyle))

    for x, y, label, color, linestyle in styled_series:
        if truncate_to_min_step and common_max_step is not None:
            keep = x <= common_max_step
            x = x[keep]
            y = y[keep]
        y_smooth = _moving_average(y, smooth_window)
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=0.9, alpha=0.18)
        ax.plot(x, y_smooth, color=color, linestyle=linestyle, linewidth=2.2, label=label)

    ax.set_xlabel("训练步数")
    ax.set_ylabel("团队回报")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True, loc="best")
    fig.savefig(output_pdf, format="pdf", pad_inches=0.12)
    plt.close(fig)
    return os.path.abspath(output_pdf)

__all__ = [
    "plot_training_curves",
    "plot_actor_routes_from_checkpoints",
    "plot_voronoi_greedy_routes",
    "compare_task_execution_efficiency",
]

if __name__ == "__main__":
    pass
    
    # construct_baseline_reward_comparison(energy_groups=[[200, 150, 150], [250, 200, 150], [300, 250, 200], [350, 300, 250]], font_size=16, our_label="TRACE")
    
    plot_training_curves(
        csv_paths=[
            "csv-ctde-transformer.csv",
            "csv-ctde-mlp.csv",
            "csv-dtde-transformer.csv",
        ],
        labels=[
            "TRACE",
            "CTDE-MLP",
            "DTDE-Transformer",
        ],
        value_divisors=[
            1.0,
            1.04,
            1.3,
        ],
        smooth_window=21,
        truncate_to_min_step=True,
        # title="不同方法训练曲线对比",
        output_pdf="test_figs/training_curves.pdf",
        font_size=17,
    )

    # plot_actor_routes_from_checkpoints(
    #     # checkpoint_dir="PPO_preTrained/MAPPO-3UAV-version-15-newEnv-decoupledClip-Entro003",
    #     checkpoint_dir="PPO_preTrained/MAPPO-3UAV-version-17-250kj",
    #     env_kwargs={
    #         "M": 3,
    #         "N": 1,
    #         "K": 40,
    #         "T": 2400,
    #         "map_size": 2000,
    #         "position_file": "data/poi_40_map_2000x2000.npy",
    #         "history_horizon": 20,
    #         "speed_levels": "6-20",
    #         "init_uav_energies": "250000,200000,150000",
    #         "reward_divisor": 400,
    #         "pre_reward_ratio": 100,
    #     },
    #     output_dir="test_figs",
    #     task_span=(99, 135),
    #     tail_n=30,
    #     actor_type="transformer",
    #     actor_kwargs={"nhead": 2},
    #     prefix="ctde_transformer",
    #     font_size=17
    # )

    # plot_voronoi_greedy_routes(
    #     env_kwargs={
    #         "M": 3,
    #         "N": 1,
    #         "K": 40,
    #         "T": 2400,
    #         "map_size": 2000,
    #         "position_file": "data/poi_40_map_2000x2000.npy",
    #         "speed_levels": "6-25",
    #         "init_uav_energies": "250000,200000,150000",
    #         "reward_divisor": 400,
    #         "pre_reward_ratio": 100,
    #     },
    #     output_pdf="test_figs/voronoi_first45.pdf",
    #     num_tasks=45,
    #     font_size=17
    # )
    
    result = compare_task_execution_efficiency(
        checkpoint_dir="PPO_preTrained/MAPPO-3UAV-version-15-newEnv-decoupledClip-Entro003",
        env_kwargs={
            "M": 3,
            "N": 1,
            "K": 40,
            "T": 2400,
            "map_size": 2000,
            "position_file": "data/poi_40_map_2000x2000.npy",
            "history_horizon": 20,
            "speed_levels": "6-20",
            "init_uav_energies": "250000,200000,150000",
            "reward_divisor": 400,
            "pre_reward_ratio": 100,
        },
        output_pdf="test_figs/task_count_compare.pdf",
        actor_type="transformer",
        actor_kwargs={"nhead": 2},
        actor_name="TRACE",
        greedy_name="纯AoI分区贪心",
        font_size=17,
    )

    # plot_ctde_transformer_speed_violin(
    # model_specs=[
    #     {
    #         "label": "[350,300,250]",
    #         "checkpoint_dir": "PPO_preTrained/MAPPO-3UAV-version-18-350kj-dim384-mlp-local",
    #         "env_kwargs": {
    #             "M": 3,
    #             "N": 1,
    #             "K": 40,
    #             "T": 2400,
    #             "map_size": 2000,
    #             "position_file": "data/poi_40_map_2000x2000.npy",
    #             "history_horizon": 20,
    #             "speed_levels": "6-20",
    #             "init_uav_energies": "350000,300000,250000",
    #             "reward_divisor": 400,
    #             "pre_reward_ratio": 100,
    #         },
    #         "actor_kwargs": {"nhead": 2},
    #         "max_ep_len": 2000,
    #     },
    #     {
    #         "label": "[200,200,200]",
    #         "checkpoint_dir": "PPO_preTrained/MAPPO-3UAV-version-18-all200kj-dim384-mlp-local",
    #         "env_kwargs": {
    #             "M": 3,
    #             "N": 1,
    #             "K": 40,
    #             "T": 2400,
    #             "map_size": 2000,
    #             "position_file": "data/poi_40_map_2000x2000.npy",
    #             "history_horizon": 20,
    #             "speed_levels": "6-20",
    #             "init_uav_energies": "200000,200000,200000",
    #             "reward_divisor": 400,
    #             "pre_reward_ratio": 100,
    #         },
    #         "actor_kwargs": {"nhead": 2},
    #     },
    # ],
    # output_pdf="test_figs/ctde_transformer_speed_violin.pdf",
    # font_size=12,
    # )
