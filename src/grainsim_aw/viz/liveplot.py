from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.colors import BoundaryNorm, ListedColormap


class LivePlotter:
    def __init__(
        self, field: str = "fs", *, autoscale: bool = False, show_ghosts: bool = False
    ):
        self.field = field
        self.autoscale = autoscale
        self.show_ghosts = show_ghosts
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.im: Optional[AxesImage] = None
        self.txt: Optional[Text] = None
        self.vmin: Optional[float] = None
        self.vmax: Optional[float] = None
        self._gid_cmap: Optional[ListedColormap] = None
        self._gid_norm: Optional[BoundaryNorm] = None

    def start(self, grid):
        A = self._extract(grid)
        self.fig, self.ax = plt.subplots()
        # 不同后端安全设标题
        mgr = getattr(self.fig.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            try:
                mgr.set_window_title(f"Live: {self.field}")
            except Exception:
                pass
        assert self.ax is not None  # 给类型检查器吃一个确定性

        self.ax.set_axis_off()

        if self.field == "grain_id":
            gid_max = int(np.nanmax(A))
            self._set_gid_cmap(gid_max)
            self.im = self.ax.imshow(
                A,
                origin="lower",
                cmap=self._gid_cmap,
                norm=self._gid_norm,
                interpolation="nearest",
            )
        else:
            if self.autoscale:
                vmin = float(np.nanmin(A))
                vmax = float(np.nanmax(A))
                if not np.isfinite(vmax) or vmax == vmin:
                    vmin, vmax = 0.0, 1.0
            else:
                if self.field == "fs":
                    vmin, vmax = 0.0, 1.0
                else:
                    vmin = float(np.nanmin(A))
                    vmax = float(np.nanmax(A))
                    if not np.isfinite(vmax) or vmax == vmin:
                        vmin, vmax = 0.0, 1.0
            self.vmin, self.vmax = vmin, vmax
            self.im = self.ax.imshow(
                A, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            try:
                self.fig.colorbar(
                    self.im, ax=self.ax, fraction=0.046, pad=0.04, label=self.field
                )
            except Exception:
                pass

        self.txt = self.ax.text(
            0.02,
            0.98,
            "step=?\nt=?",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
        )
        plt.ion()
        plt.show(block=False)
        plt.pause(0.001)

    def update(self, grid, t: float, step: int) -> None:
        if not self.alive:
            return
        assert (
            self.im is not None and self.ax is not None and self.txt is not None
        )  # 消除“可能为 None”的告警

        A = self._extract(grid)

        if self.field == "grain_id":
            gid_max_now = int(np.nanmax(A))
            need_recolor = (self._gid_cmap is None) or (
                gid_max_now + 1 > int(self._gid_cmap.N)
            )
            if need_recolor:
                self._set_gid_cmap(gid_max_now)
                # 判空后再改色图
                if self._gid_cmap is not None and self._gid_norm is not None:
                    self.im.set_cmap(self._gid_cmap)
                    self.im.set_norm(self._gid_norm)
        else:
            if self.autoscale:
                vmin = float(np.nanmin(A))
                vmax = float(np.nanmax(A))
                if np.isfinite(vmax) and vmax > vmin:
                    self.im.set_clim(vmin=vmin, vmax=vmax)

        self.im.set_data(A)
        self.txt.set_text(f"step={step}\nt={t:.4g}")
        self.ax.set_title(self.field)
        plt.pause(0.001)

    def close(self) -> None:
        if self.fig is not None:
            try:
                plt.close(self.fig)
            finally:
                self.fig = None
                self.ax = None
                self.im = None
                self.txt = None

    @property
    def alive(self) -> bool:
        return (self.fig is not None) and plt.fignum_exists(self.fig.number)

    def _extract(self, grid) -> np.ndarray:
        if self.show_ghosts:
            sl = (slice(0, grid.Ny), slice(0, grid.Nx))
        else:
            sl = grid.core
        A = getattr(grid, self.field)
        return A[sl]

    def _set_gid_cmap(self, gid_max: int) -> None:
        gid_max = max(1, int(gid_max))
        rng = np.random.default_rng(42)

        # 背景色 1 行（RGBA）
        bg = np.array([[0.95, 0.95, 0.95, 1.0]], dtype=float)

        # 为每个晶粒生成随机 RGB，并补上 alpha=1，组成 RGBA
        rgb = rng.random((gid_max, 3))  # (gid_max, 3)
        a = np.ones((gid_max, 1), dtype=float)  # (gid_max, 1)
        rgba = np.hstack([rgb, a])  # (gid_max, 4)

        # 叠在一起：第 0 号颜色用于背景，1..gid_max 对应晶粒 1..gid_max
        colors = np.vstack([bg, rgba])  # (gid_max+1, 4)

        self._gid_cmap = ListedColormap(colors)
        bounds = np.arange(-0.5, gid_max + 1.5, 1.0)
        self._gid_norm = BoundaryNorm(bounds, self._gid_cmap.N)

    def block(self) -> None:
        """阻塞直到用户关闭窗口。"""
        if self.fig is None:
            return
        plt.ioff()  # 关闭交互式，进入阻塞 show
        try:
            # 有的后端支持把窗口置顶，不行就忽略
            mgr = getattr(self.fig.canvas, "manager", None)
            if mgr is not None and hasattr(mgr, "window"):
                try:
                    mgr.window.activateWindow()
                    mgr.window.raise_()
                except Exception:
                    pass
        except Exception:
            pass
        plt.show()  # 阻塞直到窗口被关闭
