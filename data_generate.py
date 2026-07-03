import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate POI/BS map data with configurable ranges and POI weights."
    )
    parser.add_argument("--n_poi", type=int, default=40, help="Number of POIs")
    parser.add_argument("--n_bs", type=int, default=1, help="Number of base stations")
    parser.add_argument("--map_width", type=float, default=2000.0, help="Map width")
    parser.add_argument("--map_height", type=float, default=2000.0, help="Map height")

    parser.add_argument("--poi_x_min", type=float, default=None, help="POI x min")
    parser.add_argument("--poi_x_max", type=float, default=None, help="POI x max")
    parser.add_argument("--poi_y_min", type=float, default=None, help="POI y min")
    parser.add_argument("--poi_y_max", type=float, default=None, help="POI y max")

    parser.add_argument("--bs_x_min", type=float, default=None, help="BS x min")
    parser.add_argument("--bs_x_max", type=float, default=None, help="BS x max")
    parser.add_argument("--bs_y_min", type=float, default=None, help="BS y min")
    parser.add_argument("--bs_y_max", type=float, default=None, help="BS y max")

    parser.add_argument("--weight_min", type=float, default=1.0, help="POI weight min")
    parser.add_argument("--weight_max", type=float, default=10.0, help="POI weight max")

    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--output_name", type=str, default="", help="Output .npy name (optional)")
    parser.add_argument("--seed", type=int, default=4, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Save visualization image")
    parser.add_argument("--dpi", type=int, default=150, help="Visualization DPI")
    args = parser.parse_args()

    if args.n_poi <= 0:
        raise ValueError("--n_poi must be > 0")
    if args.n_bs <= 0:
        raise ValueError("--n_bs must be > 0")
    if args.map_width <= 0 or args.map_height <= 0:
        raise ValueError("Map size must be positive")
    if args.weight_max < args.weight_min:
        raise ValueError("--weight_max must be >= --weight_min")

    if args.poi_x_min is None:
        args.poi_x_min = 0.0
    if args.poi_x_max is None:
        args.poi_x_max = args.map_width
    if args.poi_y_min is None:
        args.poi_y_min = 0.0
    if args.poi_y_max is None:
        args.poi_y_max = args.map_height

    if args.bs_x_min is None:
        args.bs_x_min = 0.0
    if args.bs_x_max is None:
        args.bs_x_max = args.map_width
    if args.bs_y_min is None:
        args.bs_y_min = 0.0
    if args.bs_y_max is None:
        args.bs_y_max = args.map_height

    return args


def _validate_range(name, low, high, upper_bound):
    if low < 0 or high < 0:
        raise ValueError(f"{name} range cannot be negative")
    if high <= low:
        raise ValueError(f"{name} high must be greater than low")
    if high > upper_bound:
        raise ValueError(f"{name} high cannot exceed map boundary {upper_bound}")


def generate_data(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    _validate_range("poi_x", args.poi_x_min, args.poi_x_max, args.map_width)
    _validate_range("poi_y", args.poi_y_min, args.poi_y_max, args.map_height)
    _validate_range("bs_x", args.bs_x_min, args.bs_x_max, args.map_width)
    _validate_range("bs_y", args.bs_y_min, args.bs_y_max, args.map_height)

    poi_x = np.random.uniform(args.poi_x_min, args.poi_x_max, size=(args.n_poi, 1))
    poi_y = np.random.uniform(args.poi_y_min, args.poi_y_max, size=(args.n_poi, 1))
    poi_positions = np.concatenate([poi_x, poi_y], axis=1).astype(np.float32)

    # 基站固定在地图正中心
    center_x = args.map_width / 2.0
    center_y = args.map_height / 2.0
    bs_positions = np.tile(np.array([[center_x, center_y]], dtype=np.float32), (args.n_bs, 1))

    poi_weights = np.random.uniform(args.weight_min, args.weight_max, size=(args.n_poi,)).astype(np.float32)

    data = {
        "poi_positions": poi_positions,
        "bs_positions": bs_positions,
        "poi_weights": poi_weights,
        "map_width": float(args.map_width),
        "map_height": float(args.map_height),
        "poi_range": np.array([args.poi_x_min, args.poi_x_max, args.poi_y_min, args.poi_y_max], dtype=np.float32),
        "bs_range": np.array([args.bs_x_min, args.bs_x_max, args.bs_y_min, args.bs_y_max], dtype=np.float32),
    }
    return data


def save_data(data, args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name:
        npy_name = args.output_name
        if not npy_name.endswith(".npy"):
            npy_name = npy_name + ".npy"
    else:
        npy_name = f"poi_{args.n_poi}_map_{int(args.map_width)}x{int(args.map_height)}.npy"
    npy_path = os.path.join(args.output_dir, npy_name)
    np.save(npy_path, data)
    print(f"Saved map data: {npy_path}")
    return npy_path


def visualize_data(data, npy_path, args):
    poi_positions = data["poi_positions"]
    bs_positions = data["bs_positions"]
    poi_weights = data["poi_weights"]

    fig = plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        poi_positions[:, 0],
        poi_positions[:, 1],
        c=poi_weights,
        cmap="viridis",
        marker="o",
        alpha=0.8,
        label="POIs",
    )
    plt.colorbar(sc, label="POI weight")
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c="red", marker="^", s=120, label="BS")

    plt.xlim(0, args.map_width)
    plt.ylim(0, args.map_height)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"POI/BS map ({args.n_poi} POIs, {args.n_bs} BS)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    png_path = os.path.splitext(npy_path)[0] + ".png"
    plt.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {png_path}")


def main():
    args = parse_args()
    data = generate_data(args)
    npy_path = save_data(data, args)
    # if args.visualize:
    visualize_data(data, npy_path, args)


if __name__ == "__main__":
    main()
