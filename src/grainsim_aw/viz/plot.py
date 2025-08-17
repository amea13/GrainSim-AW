from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse


# 用于可视化仿真快照的绘图实用程序
# 查找最新快照
def find_latest_snapshot(out_dir: Path) -> Path:
    snaps = sorted(out_dir.glob("step_*.npz"))
    if not snaps:
        raise FileNotFoundError(f"没有找到快照：{out_dir}")
    return snaps[-1]


# 绘制快照中的场
def plot_field(npz_path: Path, field: str = "fs", out_png: Path | None = None) -> Path:
    z = np.load(npz_path)
    if field not in z.files:
        raise KeyError(f"'{field}' 不在快照里，包含的键有：{sorted(z.files)}")
    A = z[field]
    # 简单热图
    plt.figure()
    plt.imshow(A, origin="lower")
    plt.colorbar(label=field)
    plt.title(f"{field} | step={int(z['step'])} t={float(z['t']):.4g}")
    out_png = out_png or (
        npz_path.with_suffix("").with_name(npz_path.stem + f"_{field}.png")
    )
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Plot a field from latest snapshot.")
    ap.add_argument("out_dir", type=Path, help="输出目录，例如 data/output/run-minimal")
    ap.add_argument(
        "--field", choices=["fs", "grain_id", "theta", "CL", "CS"], default="fs"
    )
    args = ap.parse_args()

    snap = find_latest_snapshot(args.out_dir)
    png = plot_field(snap, field=args.field)
    print("已保存：", png)


if __name__ == "__main__":
    main()
