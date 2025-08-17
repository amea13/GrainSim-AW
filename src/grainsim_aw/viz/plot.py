from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse


# 查找最新快照
def find_latest_snapshot(out_dir: Path) -> Path:
    snaps = sorted(out_dir.glob("step_*.npz"))
    if not snaps:
        raise FileNotFoundError(f"没有找到快照：{out_dir}")
    return snaps[-1]


def _core_slice(arr: np.ndarray, nghost: int | None):
    if not nghost or nghost <= 0:
        return arr
    g = int(nghost)
    return arr[g:-g, g:-g]


# 绘制快照中的场
def plot_field(
    npz_path: Path,
    field: str = "fs",
    out_png: Path | None = None,
    core_nghost: int | None = None,
) -> Path:
    z = np.load(npz_path)
    keys = list(z.files)
    if field not in keys:
        raise KeyError(f"'{field}' 不在快照里，包含的键有：{sorted(keys)}")
    A = z[field]
    step = int(z["step"]) if "step" in keys else -1
    t = float(z["t"]) if "t" in keys else float("nan")

    # 可选裁掉 ghost
    A_plot = _core_slice(A, core_nghost)

    # 绘图：整数场用离散显示，其它用默认
    plt.figure()
    if np.issubdtype(A_plot.dtype, np.integer) or field == "grain_id":
        plt.imshow(A_plot, origin="lower", interpolation="nearest")
    else:
        plt.imshow(A_plot, origin="lower")
    plt.colorbar(label=field)
    title_step = f"step={step}" if step >= 0 else "step=?"
    title_time = f" t={t:.4g}" if np.isfinite(t) else ""
    plt.title(f"{field} | {title_step}{title_time}")
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
        "--field",
        choices=["fs", "grain_id", "theta", "CL", "CS", "T"],  # ← 加入 T
        default="fs",
    )
    ap.add_argument(
        "--core-nghost",
        type=int,
        default=None,
        help="若提供，则裁掉 ghost（例如 3 表示显示 g:-g, g:-g 的核心区）",
    )
    args = ap.parse_args()

    snap = find_latest_snapshot(args.out_dir)
    png = plot_field(snap, field=args.field, core_nghost=args.core_nghost)
    print("已保存：", png)


if __name__ == "__main__":
    main()
