from __future__ import annotations
from typing import Dict, Any
import numpy as np


def sample(grid, t: float, cfg_temp: Dict[str, Any]) -> np.ndarray:
    """
    生成与 grid.fs 同形状(含 ghost)的温度场缓冲区 Tbuf（不修改 grid.T）。

    支持模式：
      - constant:          T = T0
      - constant_cooling:  T = T0 - cooling_rate * t
      - linear_y:          T = T0 + Gy * (y - y_center) - cooling_rate * t(可选)

    配置字段（按需）：
      T0: float 1794.15
      cooling_rate: float [K/s], 默认 0
      mode: "constant" | "constant_cooling" | "linear_y"
      Gy: float [K/m], 仅 linear_y 使用，默认 0
    """
    mode = str(cfg_temp.get("mode", "constant")).lower()
    T0 = float(cfg_temp.get("T0", 933.47))
    R = float(cfg_temp.get("cooling_rate", 0.0))

    # 网格尺寸（含 ghost）
    Ny, Nx = grid.fs.shape

    if mode == "constant":
        return np.full((Ny, Nx), T0, dtype=float)

    if mode == "constant_cooling":
        return np.full((Ny, Nx), T0 - R * t, dtype=float)

    if mode == "linear_y":
        Gy = float(cfg_temp.get("Gy", 0.0))  # 温度梯度 K/m
        g = int(grid.nghost)
        # 以 core 的几何中心为零位，构造每一行的 y 坐标（元胞中心，单位 m）
        y_idx = np.arange(Ny) - (g + grid.ny / 2.0)
        y = (y_idx + 0.5) * float(grid.dy)
        T_line = T0 + Gy * (y[:, None]) - R * t  # (Ny,1)
        return np.broadcast_to(T_line, (Ny, Nx)).copy()

    # 未知模式：退化为常量
    return np.full((Ny, Nx), T0, dtype=float)


def update(*, grid, cfg: Dict[str, Any], t: float) -> None:
    """
    就地更新 grid.T，内部调用 sample(...)。
    """
    grid.T[...] = sample(grid, t, cfg)
