# src/grainsim_aw/multiphysics/temperature_adapter.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np


def sample(grid, t: float, cfg_temp: Dict[str, Any]) -> np.ndarray:
    """
    返回与 grid.fs 同形状(含 ghost)的温度场 Tbuf。
    支持模式:
      - constant:          T = T0
      - constant_cooling:  T = T0 - cooling_rate * t
      - linear_y:          T = T0 + Gy * (y - y_center) - cooling_rate * t(可选)
    cfg_temp 字段（按需）:
      T0: float, 默认 933.47
      cooling_rate: float [K/s], 默认 0
      mode: "constant" | "constant_cooling" | "linear_y"
      Gy: float [K/m], 仅 linear_y 使用，默认 0
    """
    mode = str(cfg_temp.get("mode", "constant")).lower()
    T0 = float(cfg_temp.get("T0", 933.47))
    R = float(cfg_temp.get("cooling_rate", 0.0))
    Ny, Nx = grid.Ny, grid.Nx

    if mode == "constant":
        return np.full((Ny, Nx), T0, dtype=float)

    if mode == "constant_cooling":
        return np.full((Ny, Nx), T0 - R * t, dtype=float)

    if mode == "linear_y":
        Gy = float(cfg_temp.get("Gy", 0.0))  # 温度梯度 K/m
        # 以 core 的几何中心为零位，含 ghost 的绝对 y 坐标（元胞中心）
        g = grid.nghost
        y_idx = np.arange(Ny) - (g + grid.ny / 2.0)  # 以 core 中心为 0
        y = (y_idx + 0.5) * grid.dy  # 单位 m
        T_line = T0 + Gy * (y[:, None] - 0.0) - R * t
        return np.broadcast_to(T_line, (Ny, Nx)).copy()

    # 未知模式：退化为常量
    return np.full((Ny, Nx), T0, dtype=float)
