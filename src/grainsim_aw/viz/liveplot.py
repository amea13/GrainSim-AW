# src/grainsim_aw/viz/liveplot.py
from __future__ import annotations
import os, time
from typing import List, Optional, Dict, Union
import numpy as np
from pathlib import Path

import matplotlib

if "DISPLAY" not in os.environ and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _core_view(a: np.ndarray, nghost: int) -> np.ndarray:
    if nghost <= 0:
        return a
    return a[nghost:-nghost, nghost:-nghost]


class LivePlotter:
    """
    轻量实时可视化（限帧 + 结束阻塞）：
    - fields: ["fs", "T", "CL"] 等
    - every: 步频采样
    - max_fps: 最大刷新帧率（默认 5）
    - block_on_finish: 结束后是否阻塞等待手动关闭窗口
    - save_frames: 可选导出帧
    """

    def __init__(
        self,
        grid,
        fields: Optional[List[str]] = None,
        every: int = 10,
        figsize=(10, 4),
        dpi: int = 100,
        save_frames: bool = False,
        outdir: Optional[Union[str, Path]] = None,
        clim: Optional[Dict[str, tuple]] = None,
        with_colorbar: bool = False,
        title_prefix: str = "Live",
        max_fps: float = 5.0,
        block_on_finish: bool = True,
    ):
        self.grid = grid
        self.fields = fields or ["fs", "T", "CL"]
        self.every = max(1, int(every))
        self.save_frames = bool(save_frames)
        self.title_prefix = title_prefix
        self.clim = clim or {}
        self.with_colorbar = with_colorbar
        self.max_fps = max(0.1, float(max_fps))
        self.block_on_finish = bool(block_on_finish)

        self.enabled = True
        self.images = []
        self.axes = []
        self.cbars = []

        # 交互模式：允许非阻塞刷新
        try:
            plt.ion()
        except Exception:
            pass

        n = len(self.fields)
        self.fig, self.axes = plt.subplots(
            1,
            n,
            figsize=figsize if n > 1 else (figsize[0] / 2, figsize[1]),
            dpi=dpi,
            squeeze=False,
        )
        self.axes = self.axes[0]

        # 输出目录
        self.frame_dir = None
        if self.save_frames:
            base = Path(outdir) if outdir is not None else Path(".")
            self.frame_dir = base / "live"
            self.frame_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 imshow
        for i, name in enumerate(self.fields):
            ax = self.axes[i] if n > 1 else self.axes
            arr = getattr(self.grid, name, None)
            if arr is None:
                data = np.zeros((4, 4))
                ax.set_title(f"{name} (missing)")
            else:
                data = _core_view(arr, self.grid.nghost)
                ax.set_title(name)

            im = ax.imshow(data, origin="lower", interpolation="nearest")
            if name in self.clim:
                vmin, vmax = self.clim[name]
                im.set_clim(vmin, vmax)

            ax.set_xticks([])
            ax.set_yticks([])
            self.images.append(im)

            if self.with_colorbar:
                cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                self.cbars.append(cb)
            else:
                self.cbars.append(None)

        self.fig.tight_layout()

        # 限帧计时器
        self._last_draw = 0.0
        self._min_interval = 1.0 / self.max_fps

    def update(self, grid, step: int, t: float):
        if step % self.every != 0:
            return

        # 限帧：若没到时间，跳过一次刷新
        now = time.perf_counter()
        if now - self._last_draw < self._min_interval:
            return
        self._last_draw = now

        # 刷数据
        for name, im in zip(self.fields, self.images):
            arr = getattr(grid, name, None)
            if arr is None:
                continue
            data = _core_view(arr, grid.nghost)
            im.set_data(data)
            if name not in self.clim:
                vmin = float(np.nanmin(data))
                vmax = float(np.nanmax(data))
                if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                    im.set_clim(vmin, vmax)

        self.fig.suptitle(f"{self.title_prefix}  t={t:.5g}  step={step}")
        try:
            self.fig.canvas.draw_idle()
            # 这里给一个更长一点的 pause，避免看起来“闪”
            plt.pause(0.03)
        except Exception:
            self.enabled = False

        # 可选保存帧
        if self.save_frames and self.frame_dir is not None:
            fname = self.frame_dir / f"frame_{step:08d}.png"
            self.fig.savefig(fname)

    def close(self):
        # 结束时固定最后一帧；若配置了阻塞，就等待你手动关闭窗口
        try:
            self.fig.canvas.draw()
        except Exception:
            pass

        if (
            self.block_on_finish
            and self.enabled
            and matplotlib.get_backend().lower() != "agg"
        ):
            try:
                plt.ioff()
                plt.show()  # 阻塞到关窗为止
            except Exception:
                pass
        else:
            plt.close(self.fig)
