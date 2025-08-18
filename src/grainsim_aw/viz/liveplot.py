# src/grainsim_aw/viz/liveplot.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap, NoNorm
from matplotlib.figure import Figure


class LivePlotter:
    """
    实时显示：T、CL、grain_id、fs
    - 只创建一次窗口，循环内仅 set_data + draw_idle
    - 自动裁掉 ghost：依赖 grid.core 提供的切片
    - 支持 stride 节流，稳健百分位色标避免闪烁
    - 可通过 cfg['ranges'] 为连续场固定色标范围
    """

    def __init__(self, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.stride: int = int(cfg.get("stride", 10))
        self.fields: Sequence[str] = tuple(
            cfg.get("fields", ("T", "CL", "grain_id", "fs"))
        )
        self.percentile: float = float(cfg.get("percentile", 99.0))
        self.figsize = tuple(cfg.get("figsize", (10, 8)))
        self.fixed_ranges: Dict[str, Tuple[float, float]] = dict(cfg.get("ranges", {}))

        self._fig: Optional[Figure] = None
        self.keep_open: bool = bool(cfg.get("keep_open", False))
        self._fig: Optional[Figure] = None
        self._axes: Dict[str, Axes] = {}
        self._images: Dict[str, AxesImage] = {}
        self._colorbars: Dict[str, Colorbar] = {}

    # -------- 工具 -------- #
    @staticmethod
    def _core(arr: np.ndarray, grid) -> np.ndarray:
        ys, xs = grid.core  # grid.core 返回核心区切片
        return arr[ys, xs]

    def _robust_range(self, data: np.ndarray) -> Tuple[float, float]:
        d = np.asarray(data)
        m = np.isfinite(d)
        if not m.any():
            return 0.0, 1.0
        hi_p = self.percentile
        lo = np.percentile(d[m], 100 - hi_p)
        hi = np.percentile(d[m], hi_p)
        if not np.isfinite(lo):
            lo = float(np.min(d[m]))
        if not np.isfinite(hi):
            hi = float(np.max(d[m]))
        if hi <= lo:
            hi = lo + (1e-12 if lo != 0.0 else 1.0)
        return float(lo), float(hi)

    def _extract(self, grid, field: str) -> np.ndarray:
        if field == "T":
            return self._core(grid.T, grid)
        if field == "CL":
            return self._core(grid.CL, grid)
        if field == "fs":
            return self._core(grid.fs, grid)
        if field == "grain_id":
            return self._core(grid.grain_id, grid)
        raise KeyError(f"Unknown field: {field}")

    @staticmethod
    def _make_gid_cmap(max_gid: int) -> ListedColormap:
        # 0 号为灰色，其余随机色
        ncol = max(1024, max_gid + 1)
        rng = np.random.default_rng(12345)
        colors = np.ones((ncol, 4), dtype=float)
        colors[0, :3] = 0.5
        colors[1:, :3] = rng.random((ncol - 1, 3))
        return ListedColormap(colors)

    # -------- 生命周期 -------- #
    def start(self, grid) -> None:
        if not self.enabled or self._fig is not None:
            return

        plt.ion()
        self._fig = plt.figure("GrainSim-AW Live", figsize=self.figsize)
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        used = list(self.fields[:4])
        while len(used) < 4:
            used.append("")

        for idx, field in enumerate(used):
            ax = self._fig.add_subplot(gs[idx])
            if not field:
                ax.axis("off")
                continue
            ax.set_title(field)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            self._axes[field] = ax

        self._init_images(grid)
        self._fig.tight_layout()
        plt.show(block=False)

    def _init_images(self, grid) -> None:
        # 用 fs 的核心区尺寸得到 extent，避免依赖 grid.nx/ny
        fs_core = self._extract(grid, "fs")
        ny, nx = fs_core.shape
        extent = (0, grid.dx * nx, 0, grid.dy * ny)

        for field, ax in self._axes.items():
            data = self._extract(grid, field)

            if field == "grain_id":
                gid_max = int(np.max(data)) if data.size else 1
                cmap = self._make_gid_cmap(gid_max)
                im = ax.imshow(
                    data,
                    origin="lower",
                    interpolation="nearest",
                    extent=extent,
                    cmap=cmap,
                    norm=NoNorm(),
                )
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[])
            else:
                if field in self.fixed_ranges:
                    vmin, vmax = self.fixed_ranges[field]
                else:
                    vmin, vmax = self._robust_range(data)
                im = ax.imshow(
                    data,
                    origin="lower",
                    interpolation="nearest",
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                )
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            self._images[field] = im
            self._colorbars[field] = cb

    def update(self, grid, t: float, step: int) -> None:
        if not self.enabled:
            return
        if self._fig is None:
            self.start(grid)
        if step % self.stride != 0:
            return

        for field, im in self._images.items():
            data = self._extract(grid, field)
            im.set_data(data)
            if field != "grain_id" and field not in self.fixed_ranges:
                vmin, vmax = self._robust_range(data)
                im.set_clim(vmin, vmax)

        if self._fig is not None:
            self._fig.suptitle(f"t = {t:.6f} s   step = {step}", fontsize=11)
            self._fig.canvas.draw_idle()
            plt.pause(0.001)

    def hold(self) -> None:
        """阻塞显示，直到用户手动关闭窗口。"""
        if self._fig is not None:
            plt.ioff()  # 关闭交互式，使 show 阻塞
            self._fig.canvas.draw_idle()
            plt.show()  # 阻塞到手动关闭

    def close(self) -> None:
        if self._fig is not None:
            if self.keep_open:
                # 结束时保留窗口，由用户手动关闭
                self.hold()
            else:
                plt.close(self._fig)
            self._fig = None
