# -*- coding: utf-8 -*-
"""
kernels.py — 圆核权重与圆核质心法法向（首用即建，d=7 固定）
----------------------------------------------------------------
提供:
  centroid_normal(fs) -> (nx, ny)
    - 对整场 fs 计算“圆核质心法”法向，返回单位向量分量
    - 圆核直径固定为 7 个网格；子采样精度固定为 8×8
    - 首次调用时构建并缓存相交面积权重模板 W，后续直接复用

数值/几何约定
- 网格方形，dx=dy；核半径 R=(7-1)/2=3（单位: 格）
- 需要在调用前已更新 ghost，并保证 nghost ≥ 3
- 方向约定： n_c ∥ (A - B)，其中 B 为以 fs·w_ij 加权的局部质心
  （由于后续会单位化，∑fs·w_ij 的分母对方向无影响）
"""

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np

__all__ = ["centroid_normal"]

# -----------------------------
# 全局缓存：权重模板（仅 d=7）
# -----------------------------
_WEIGHTS_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_DEFAULT_D = 7  # 固定直径（格数）
_SUBSAMPLE = 8  # 固定子采样精度（8x8）


def _generate_weights_by_sampling(
    d_cells: int, subsample: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用子单元采样近似“圆-方格相交面积比例”：
      W[di,dj] = 该相对格单元中，sub×sub 子单元中心落在圆内的比例 ∈ [0,1]
    返回：
      DI, DJ, W 形状均为 (K,K)，K=2R+1，R=(d_cells-1)//2
    """
    if d_cells <= 0 or d_cells % 2 == 0:
        raise ValueError("d_cells 必须为正奇数，例如 7")
    if subsample <= 0:
        raise ValueError("subsample 必须为正整数，例如 8")

    R = (d_cells - 1) // 2
    K = 2 * R + 1
    Rc = 0.5 * d_cells  # 圆半径（以“格宽=1”的无量纲单位表示）

    # 子单元中心坐标（均匀分布在 [-0.5, 0.5)）
    u = (np.arange(subsample, dtype=float) + 0.5) / subsample - 0.5

    # 相对索引网格 (Δi, Δj)
    DI = np.arange(-R, R + 1, dtype=int)[:, None] * np.ones((1, K), dtype=int)
    DJ = np.ones((K, 1), dtype=int) * np.arange(-R, R + 1, dtype=int)[None, :]

    # 逐格子单元做子采样，求落在圆内的比例
    W = np.zeros((K, K), dtype=float)
    for a in range(K):
        for b in range(K):
            di = DI[a, b]
            dj = DJ[a, b]
            # 子单元中心（以当前格中心为原点）的相对坐标
            YY = di + u[:, None]  # (sub, 1) → 广播到 (sub, sub)
            XX = dj + u[None, :]  # (1, sub) → 广播到 (sub, sub)
            inside = (XX * XX + YY * YY) <= (Rc * Rc)
            W[a, b] = inside.mean()

    return DI, DJ, W


def _ensure_weights():
    """确保 d=7 的权重模板已在缓存里（首用即建）。"""
    if _DEFAULT_D in _WEIGHTS_CACHE:
        return
    DI, DJ, W = _generate_weights_by_sampling(_DEFAULT_D, _SUBSAMPLE)
    _WEIGHTS_CACHE[_DEFAULT_D] = (DI, DJ, W)


def centroid_normal(fs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    圆核质心法计算整场法向（单位向量）
    参数
    ----
    fs : ndarray，(Ny,Nx)，含 ghost；调用前请保证已更新 ghost，且 nghost ≥ 3

    返回
    ----
    nx, ny : ndarray，与 fs 同形状，单位向量分量；方向 n_c ∥ (A - B)
    """
    _ensure_weights()
    DI, DJ, W = _WEIGHTS_CACHE[_DEFAULT_D]
    K = W.shape[0]

    # 分子：∑ fs(i+di,j+dj) * w_ij * (Δj, Δi)，以“格”为单位累加
    num_x = np.zeros_like(fs, dtype=float)
    num_y = np.zeros_like(fs, dtype=float)

    # 使用 ghost 后，np.roll 可以安全获得邻域；要求 R ≤ nghost
    for a in range(K):
        for b in range(K):
            w = W[a, b]
            if w == 0.0:
                continue
            di = int(DI[a, b])
            dj = int(DJ[a, b])
            fs_nb = np.roll(np.roll(fs, di, axis=0), dj, axis=1)
            num_x += fs_nb * w * dj  # x 坐标对应列偏移 dj
            num_y += fs_nb * w * di  # y 坐标对应行偏移 di

    # 法向：n_c ∥ (A - B) ∥ -(num_x, num_y)；单位化
    mag = np.sqrt(num_x * num_x + num_y * num_y) + 1e-30  # 极小保护，防止纯液区域除零
    nx = -num_x / mag
    ny = -num_y / mag
    return nx, ny
