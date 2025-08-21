# -*- coding: utf-8 -*-
"""
各向异性与几何量计算工具
- 由固相分数 fs 计算界面法向 (nx, ny) 与曲率 kappa
- 由法向角 phi 与生长取向角 theta 计算各向异性因子 f(phi, theta)
注意：差分基于 ghost，使用前应先调用 update_ghosts 保证边界可用
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def compute_normals_and_curvature(
    fs: np.ndarray,
    dx: float,
    dy: float,
    *,
    mask_int: np.ndarray | None = None,
    smooth: int = 1,  # 对 fs 轻度平滑 1~2 次
    smooth_n: int = 1,  # 对 n 再轻度平滑 1 次
):
    f = np.asarray(fs, dtype=float)

    # 1) 只为“曲率用途”对 fs 做极轻的 3x3 盒式平滑，抑制台阶
    def box3(a):
        return (
            a
            + np.roll(a, 1, 0)
            + np.roll(a, -1, 0)
            + np.roll(a, 1, 1)
            + np.roll(a, -1, 1)
            + np.roll(np.roll(a, 1, 0), 1, 1)
            + np.roll(np.roll(a, 1, 0), -1, 1)
            + np.roll(np.roll(a, -1, 0), 1, 1)
            + np.roll(np.roll(a, -1, 0), -1, 1)
        ) / 9.0

    s = f.copy()
    for _ in range(max(0, smooth)):
        s = box3(s)

    # 2) 只在界面带计算梯度与法向；带外法向置零
    fx = (np.roll(s, -1, 1) - np.roll(s, 1, 1)) / (2.0 * dx)
    fy = (np.roll(s, -1, 0) - np.roll(s, 1, 0)) / (2.0 * dy)
    mag = np.hypot(fx, fy)

    # 自动界面带：若未提供 mask_int，用 0<fs<1 定义
    if mask_int is None:
        band = (fs > 1e-8) & (fs < 1 - 1e-8)
    else:
        band = mask_int.astype(bool)

    inv = np.zeros_like(mag)
    inv[band] = 1.0 / np.maximum(mag[band], 1e-12)  # 带内保护
    nx = fx * inv
    ny = fy * inv
    nx[~band] = 0.0
    ny[~band] = 0.0

    # 3) 法向再平滑一次，避免散度看到硬跳变
    for _ in range(max(0, smooth_n)):
        nx = box3(nx)
        ny = box3(ny)

    # 4) 计算散度，仅报告带内曲率，带外置零
    dnx_dx = (np.roll(nx, -1, 1) - np.roll(nx, 1, 1)) / (2.0 * dx)
    dny_dy = (np.roll(ny, -1, 0) - np.roll(ny, 1, 0)) / (2.0 * dy)
    kappa = dnx_dx + dny_dy
    kappa_out = np.zeros_like(kappa)
    kappa_out[band] = kappa[band]

    return nx, ny, kappa_out


def anisotropy_factor(
    nx: np.ndarray, ny: np.ndarray, theta: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算各向异性因子 f(phi, theta) 以及法向角 phi。
    f(phi, theta) = 1 - 15 * eps * cos(4 * (phi - theta))

    参数
    ----
    nx, ny : ndarray
        法向分量
    theta : ndarray
        晶粒生长取向角（与 x 轴夹角，弧度），来自 grid.theta
    eps : float
        各向异性幅值，建议 0~0.05 起步

    返回
    ----
    ani, phi : ndarray
        ani 为各向异性因子，phi 为法向角
    """
    phi = np.arctan2(ny, nx)
    ani = 1.0 - 15.0 * float(eps) * np.cos(4.0 * (phi - theta))
    return ani, phi

