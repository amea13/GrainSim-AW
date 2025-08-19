# -*- coding: utf-8 -*-
"""
Stefan 守恒（含固相扩散）的界面速度
- 面通量法：把两侧扩散通量按 x/y 面离散得到 Vx, Vy，再由 Vn = Vx*nx + Vy*ny
- 只在界面带 mask_int 上赋值，其它位置置零
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..core import Dl_from_T, Ds_from_T


def compute_velocity(
    cfg_if: Dict[str, Any],
    mask_int: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    *,
    grid=None,
    CLs: Optional[np.ndarray] = None,
    CSs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 缺依赖则返回零
    if grid is None or CLs is None or CSs is None:
        z = np.zeros_like(nx, dtype=float)
        return z, z, z

    CL = grid.CL
    CS = grid.CS
    T = grid.T
    dx = float(grid.dx)
    dy = float(grid.dy)

    k0 = float(cfg_if.get("k0", 1.0))

    # 扩散率（元胞中心）
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    # 中心差分梯度
    CL_E = np.roll(CL, -1, 1)
    CL_W = np.roll(CL, 1, 1)
    CL_N = np.roll(CL, -1, 0)
    CL_S = np.roll(CL, 1, 0)
    CS_E = np.roll(CS, -1, 1)
    CS_W = np.roll(CS, 1, 1)
    CS_N = np.roll(CS, -1, 0)
    CS_S = np.roll(CS, 1, 0)

    dCLdx = (CL_E - CL_W) / (2.0 * dx)
    dCLdy = (CL_N - CL_S) / (2.0 * dy)
    dCSdx = (CS_E - CS_W) / (2.0 * dx)
    dCSdy = (CS_N - CS_S) / (2.0 * dy)

    # 分子（向量形式）：液侧驱动 - 固侧抵消
    num_x = DL * dCLdx - DS * dCSdx
    num_y = DL * dCLdy - DS * dCSdy

    # 分母（带稳健保护）
    den = (1.0 - k0) * CLs
    eps = max(1e-12, float(np.nanmax(np.abs(den[mask_int]))) * 1e-12 + 1e-18)
    sign = np.where(den >= 0.0, 1.0, -1.0)
    den_safe = np.where(np.abs(den) < eps, sign * eps, den)

    # 只在界面带赋值
    Vx = np.zeros_like(CL, dtype=float)
    Vy = np.zeros_like(CL, dtype=float)
    Vx[mask_int] = num_x[mask_int] / den_safe[mask_int]
    Vy[mask_int] = num_y[mask_int] / den_safe[mask_int]

    Vn = Vx * nx + Vy * ny
    return Vn, Vx, Vy
