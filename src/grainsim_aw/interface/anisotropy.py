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
    fs: np.ndarray, dx: float, dy: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    由 fs 场计算：
      - 法向分量 nx, ny 归一化指向“固相增大方向”（与 ∇fs 同向）
      - 曲率 kappa = ∂x nx + ∂y ny

    参数
    ----
    fs : ndarray
        含 ghost 的固相分数场 [0,1]
    dx, dy : float
        网格步长 [m]
    smooth : int
        若 >0 则对 fs 先进行 smooth 次 3x3 盒式平滑，用于抑制锯齿噪声

    返回
    ----
    nx, ny, kappa : ndarray
        与 fs 同形状，含 ghost
    """
    f = fs
    # 中心差分计算梯度
    fx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)
    fy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)

    # 归一化，加入小正数保护，避免除零放大噪声
    mag2 = fx * fx + fy * fy
    guard = max(1e-30, float(np.median(mag2)) * 1e-12 + 1e-18)
    inv_mag = 1.0 / np.sqrt(mag2 + guard)
    nx = fx * inv_mag
    ny = fy * inv_mag

    # 中心差分计算散度，得到曲率
    dnx_dx = (np.roll(nx, -1, 1) - np.roll(nx, 1, 1)) / (2.0 * dx)
    dny_dy = (np.roll(ny, -1, 0) - np.roll(ny, 1, 0)) / (2.0 * dy)
    kappa = dnx_dx + dny_dy

    return nx, ny, kappa


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
