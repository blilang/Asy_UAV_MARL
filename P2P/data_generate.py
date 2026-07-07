"""
data_generate.py — 生成 POI / BS 位置数据
用法:
    python data_generate.py --K 15 --map_size 500
    python data_generate.py --K 8  --map_size 300 --output ./data/poi_8_map_300x300.npy
"""
import numpy as np
import os
import argparse


def generate_data(K, N, map_size, output_path, seed=42):
    np.random.seed(seed)

    # 基站放在地图中心
    bs_pos = np.array([[map_size / 2, map_size / 2]], dtype=np.float32)
    if N > 1:
        extra = np.random.uniform(
            map_size * 0.2, map_size * 0.8, (N - 1, 2)
        ).astype(np.float32)
        bs_pos = np.concatenate([bs_pos, extra], axis=0)

    # POI 随机分布
    poi_pos = np.random.uniform(0, map_size, (K, 2)).astype(np.float32)

    # POI 权重：[0.5, 2.0]
    poi_weights = np.random.uniform(0.5, 2.0, K).astype(np.float32)

    data = {
        "poi_positions": poi_pos,
        "bs_positions": bs_pos,
        "poi_weights": poi_weights,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, data)
    print(f"[data_generate] Saved → {output_path}")
    print(f"  POIs={K}  BS={N}  Map={map_size}x{map_size}")

    # ---- 可视化 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(poi_pos[:, 0], poi_pos[:, 1], c="blue", marker="o",
                   s=60, alpha=0.6, label="POI")
        for i in range(K):
            ax.annotate(str(i), (poi_pos[i, 0] + 3, poi_pos[i, 1] + 3), fontsize=7)
        ax.scatter(bs_pos[:, 0], bs_pos[:, 1], c="red", marker="^",
                   s=120, label="BS")
        ax.set_xlim(-10, map_size + 10)
        ax.set_ylim(-10, map_size + 10)
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{K} POIs, {N} BS, {map_size}x{map_size}")
        ax.grid(True, alpha=0.3)
        img_path = output_path.replace(".npy", ".png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Visualization → {img_path}")
    except Exception:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--N", type=int, default=1)
    p.add_argument("--map_size", type=int, default=500)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output = args.output or f"./data/poi_{args.K}_map_{args.map_size}x{args.map_size}.npy"
    generate_data(args.K, args.N, args.map_size, output, args.seed)
