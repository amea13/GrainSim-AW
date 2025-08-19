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
    """
    Stefan 面通量法（含固相扩散）
    需要 grid 提供 fs, CL, CS, T, dx, dy；需要 CLs/CSs 为界面浓度
    返回 Vn, Vx, Vy（与 fs 同形状，含 ghost）
    """
    # 缺必需对象则退化为零场（便于兼容）
    if grid is None or CLs is None or CSs is None:
        z = np.zeros(nx.shape, dtype=float)
        return z, z, z

    fs = grid.fs
    CL = grid.CL
    CS = grid.CS
    T = grid.T
    dx = float(grid.dx)
    dy = float(grid.dy)

    k0 = float(cfg_if.get("k0", 1.0))

    shape = fs.shape
    Vx = np.zeros(shape, dtype=float)
    Vy = np.zeros(shape, dtype=float)

    # --- 计算面开口（闸门） ---
    fs_W = np.minimum(fs, np.roll(fs, 1, 1))
    fs_E = np.minimum(fs, np.roll(fs, -1, 1))
    fs_S = np.minimum(fs, np.roll(fs, 1, 0))
    fs_N = np.minimum(fs, np.roll(fs, -1, 0))

    alpha = 1.0 - fs
    a_W = np.minimum(alpha, np.roll(alpha, 1, 1))
    a_E = np.minimum(alpha, np.roll(alpha, -1, 1))
    a_S = np.minimum(alpha, np.roll(alpha, 1, 0))
    a_N = np.minimum(alpha, np.roll(alpha, -1, 0))

    # --- 邻居值（中心差分邻接） ---
    CL_W = np.roll(CL, 1, 1)
    CL_E = np.roll(CL, -1, 1)
    CL_S = np.roll(CL, 1, 0)
    CL_N = np.roll(CL, -1, 0)
    CS_W = np.roll(CS, 1, 1)
    CS_E = np.roll(CS, -1, 1)
    CS_S = np.roll(CS, 1, 0)
    CS_N = np.roll(CS, -1, 0)

    # --- 温度相关扩散系数：面上取算术平均 ---
    DLc = Dl_from_T(T)  # cell-centered
    DSc = Ds_from_T(T)

    DL_W = 0.5 * (DLc + np.roll(DLc, 1, 1))
    DL_E = 0.5 * (DLc + np.roll(DLc, -1, 1))
    DL_S = 0.5 * (DLc + np.roll(DLc, 1, 0))
    DL_N = 0.5 * (DLc + np.roll(DLc, -1, 0))

    DS_W = 0.5 * (DSc + np.roll(DSc, 1, 1))
    DS_E = 0.5 * (DSc + np.roll(DSc, -1, 1))
    DS_S = 0.5 * (DSc + np.roll(DSc, 1, 0))
    DS_N = 0.5 * (DSc + np.roll(DSc, -1, 0))

    # --- 分母与保护 ---
    den = (1.0 - k0) * CLs
    eps = max(1e-12, float(np.nanmax(np.abs(den[mask_int]))) * 1e-12 + 1e-18)
    den_safe = np.where(np.abs(den) < eps, np.sign(den) * eps, den)

    # --- x 向：两侧面通量之和 / (dx * den) ---
    num_x = (
        DS_W * (CSs - CS_W) * fs_W
        + DS_E * (CSs - CS_E) * fs_E
        + DL_W * (CLs - CL_W) * a_W
        + DL_E * (CLs - CL_E) * a_E
    )
    Vx_loc = num_x / (dx * den_safe)

    # --- y 向 ---
    num_y = (
        DS_S * (CSs - CS_S) * fs_S
        + DS_N * (CSs - CS_N) * fs_N
        + DL_S * (CLs - CL_S) * a_S
        + DL_N * (CLs - CL_N) * a_N
    )
    Vy_loc = num_y / (dy * den_safe)

    # 仅在界面带赋值
    Vx[mask_int] = Vx_loc[mask_int]
    Vy[mask_int] = Vy_loc[mask_int]

    Vn = Vx * nx + Vy * ny
    return Vn, Vx, Vy
