# -*- coding: utf-8 -*-
"""
基于 Stefan 守恒（含固相扩散）的界面速度计算
- 面通量法：把两侧法向扩散通量按 x/y 面离散，得到 Vx, Vy，再给出 Vn = Vx*nx + Vy*ny
- 只在界面带 mask_int 上赋值，其它位置置零
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np


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
    """
    Stefan 面通量法

    参数
    ----
    cfg_if : dict
        需要键：
          k0   : 分配系数
          DL   : 液相扩散系数
          DS   : 固相扩散系数
    mask_int : ndarray(bool)
        界面带掩码（0<fs<1）
    nx, ny : ndarray
        界面法向（用于 Vn = Vx*nx + Vy*ny）
    grid : Grid
        提供 fs, CL, CS, dx, dy
    CLs, CSs : ndarray
        界面处 C_L^*, C_S^*（由 equilibrium 反解得到）

    返回
    ----
    Vn, Vx, Vy : ndarray
    """
    # 若缺少必要对象，退化为全零，便于老代码兼容
    if grid is None or CLs is None or CSs is None:
        shape = nx.shape
        z = np.zeros(shape, dtype=float)
        return z, z, z

    fs = grid.fs
    CL = grid.CL
    CS = grid.CS
    dx = float(grid.dx)
    dy = float(grid.dy)

    k0 = float(cfg_if.get("k0", 1.0))
    DL = float(cfg_if.get("DL", 0.0))
    DS = float(cfg_if.get("DS", 0.0))

    shape = fs.shape
    Vx = np.zeros(shape, dtype=float)
    Vy = np.zeros(shape, dtype=float)

    # 面邻居（包含 ghost；核心区域由 mask_int 控制）
    # 邻域值
    CL_W = np.roll(CL, 1, axis=1)
    CL_E = np.roll(CL, -1, axis=1)
    CL_S = np.roll(CL, 1, axis=0)
    CL_N = np.roll(CL, -1, axis=0)
    CS_W = np.roll(CS, 1, axis=1)
    CS_E = np.roll(CS, -1, axis=1)
    CS_S = np.roll(CS, 1, axis=0)
    CS_N = np.roll(CS, -1, axis=0)

    # 面开口：固相用 fs_face，液相用 alpha_face；注意液相开口不是 1-fs_face
    fs_W = np.minimum(fs, np.roll(fs, 1, 1))
    fs_E = np.minimum(fs, np.roll(fs, -1, 1))
    fs_S = np.minimum(fs, np.roll(fs, 1, 0))
    fs_N = np.minimum(fs, np.roll(fs, -1, 0))

    alpha = 1.0 - fs
    a_W = np.minimum(alpha, np.roll(alpha, 1, 1))
    a_E = np.minimum(alpha, np.roll(alpha, -1, 1))
    a_S = np.minimum(alpha, np.roll(alpha, 1, 0))
    a_N = np.minimum(alpha, np.roll(alpha, -1, 0))

    # 分母保护：den = (1-k0)*CLs
    den = (1.0 - k0) * CLs
    # 数值保护，避免尖端或 k0≈1 爆速
    eps = max(1e-12, float(np.nanmax(np.abs(den[mask_int]))) * 1e-12 + 1e-18)
    den_safe = np.where(np.abs(den) < eps, np.sign(den) * eps, den)

    # x 向面通量差额 / (dx * den)
    num_x = DS * ((CSs - CS_W) * fs_W + (CSs - CS_E) * fs_E) + DL * (
        (CLs - CL_W) * a_W + (CLs - CL_E) * a_E
    )
    Vx_loc = num_x / (dx * den_safe)

    # y 向
    num_y = DS * ((CSs - CS_S) * fs_S + (CSs - CS_N) * fs_N) + DL * (
        (CLs - CL_S) * a_S + (CLs - CL_N) * a_N
    )
    Vy_loc = num_y / (dy * den_safe)

    # 只在界面带赋值
    Vx[mask_int] = Vx_loc[mask_int]
    Vy[mask_int] = Vy_loc[mask_int]

    # 法向速度
    Vn = Vx * nx + Vy * ny
    return Vn, Vx, Vy
