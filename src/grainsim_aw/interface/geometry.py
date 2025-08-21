# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np

__all__ = ["compute_curvature", "compute_normal"]

# =========================
# 圆核质心法权重缓存
# key: (d_cells, subsample) -> (WX, WY)
# WX, WY 为一阶矩权重，在首次使用时生成并缓存
# =========================
_WEIGHTS_CACHE: dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _generate_first_moment_weights(
    d_cells: int, subsample: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成圆核质心法的一阶矩权重:
      WX[a,b] 为模板格 (a,b) 在圆内子单元的 x 坐标均值
      WY[a,b] 为模板格 (a,b) 在圆内子单元的 y 坐标均值
    """
    if d_cells <= 0 or d_cells % 2 == 0:
        raise ValueError("d_cells 必须为正奇数，例如 7 或 5")
    if subsample <= 0:
        raise ValueError("subsample 必须为正整数，例如 8")

    R = (d_cells - 1) // 2
    K = 2 * R + 1
    Rc = 0.5 * d_cells  # 圆半径（格宽=1）

    # 子单元中心坐标，范围 [-0.5, 0.5)
    u = (np.arange(subsample, dtype=float) + 0.5) / subsample - 0.5
    # 预先生成 (sub, sub) 网格，后续只需平移
    Ux, Uy = np.meshgrid(u, u, indexing="xy")  # 均为 (sub, sub)

    WX = np.zeros((K, K), dtype=float)
    WY = np.zeros((K, K), dtype=float)

    for a in range(K):
        di = a - R
        for b in range(K):
            dj = b - R
            # 平移到模板格 (di, dj) 的子单元坐标
            XX = Ux + dj
            YY = Uy + di
            inside = (XX * XX + YY * YY) <= (Rc * Rc)  # (sub, sub)

            if inside.any():
                # 只对圆内子单元取均值，得到一阶矩权重
                WX[a, b] = XX[inside].mean()
                WY[a, b] = YY[inside].mean()
            else:
                WX[a, b] = 0.0
                WY[a, b] = 0.0

    return WX, WY


def _get_weights(d_cells: int, subsample: int) -> Tuple[np.ndarray, np.ndarray]:
    key = (int(d_cells), int(subsample))
    W = _WEIGHTS_CACHE.get(key)
    if W is None:
        W = _generate_first_moment_weights(*key)
        _WEIGHTS_CACHE[key] = W
    return W


# =========================
# 曲率（中心差分法）
# κ = (f_xx f_y^2 - 2 f_x f_y f_xy + f_yy f_x^2) / (f_x^2 + f_y^2)^(3/2)
# 只在界面带写入 out
# =========================
def compute_curvature(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    用中心差分计算 level-set 形式的曲率 κ，号与 κ = div(∇f/|∇f|) 一致。
    仅对界面带写入，其他位置保持 out 原值或置零。

    cfg 可选项:
      eps_curv: float，分母极小保护，默认 1e-30
    """
    fs = grid.fs
    dx = float(grid.dx)
    dy = float(grid.dy)

    intf: np.ndarray = masks["intf"]
    if intf is None:
        raise KeyError("masks 中缺少 'intf' 或 'mask_int'")
    if intf.dtype != bool:
        intf = intf.astype(bool, copy=False)

    roll = np.roll

    # 一阶导
    fx = (roll(fs, -1, axis=1) - roll(fs, 1, axis=1)) / (2.0 * dx)
    fy = (roll(fs, -1, axis=0) - roll(fs, 1, axis=0)) / (2.0 * dy)

    # 二阶与混合导
    fxx = (roll(fs, -1, axis=1) + roll(fs, 1, axis=1) - 2.0 * fs) / (dx * dx)
    fyy = (roll(fs, -1, axis=0) + roll(fs, 1, axis=0) - 2.0 * fs) / (dy * dy)
    fxy = (
        roll(roll(fs, -1, axis=0), 1, axis=1)
        + roll(roll(fs, 1, axis=0), -1, axis=1)
        - roll(roll(fs, -1, axis=0), -1, axis=1)
        - roll(roll(fs, 1, axis=0), 1, axis=1)
    ) / (4.0 * dx * dy)

    g2 = fx * fx + fy * fy
    num = fxx * (fy * fy) - 2.0 * fx * fy * fxy + fyy * (fx * fx)
    den = np.power(g2, 1.5) + float(cfg.get("eps_curv", 1e-30))

    kappa_full = num / den

    if out is None:
        out = np.zeros_like(fs, dtype=float)
    out[intf] = kappa_full[intf]
    return out


# =========================
# 法向（圆核质心法，一阶矩权重）
# n = - (num_x, num_y) / |(num_x, num_y)|
# 只在界面带写入 out_nx/out_ny
# =========================
def compute_normal(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    out_nx: Optional[np.ndarray] = None,
    out_ny: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用圆核质心法计算法向，方向从固到液（对 fs 取负梯度的几何意义保持一致）。
    仅对界面带写入，其他位置保持 out_* 原值或置零。

    cfg 可选项:
      centroid_d:   int   圆核直径，奇数，默认 7
      centroid_sub: int   子采样数，默认 8
      eps_norm:     float 极小保护，默认 1e-30
    """
    fs = grid.fs
    intf: np.ndarray = masks["intf"]
    if intf is None:
        raise KeyError("masks 中缺少 'intf' 或 'mask_int'")
    if intf.dtype != bool:
        intf = intf.astype(bool, copy=False)

    if out_nx is None:
        out_nx = np.zeros_like(fs, dtype=float)
    if out_ny is None:
        out_ny = np.zeros_like(fs, dtype=float)

    d_cells = int(cfg.get("centroid_d", 7))
    subsample = int(cfg.get("centroid_sub", 8))
    epsn = float(cfg.get("eps_norm", 1e-30))

    WX, WY = _get_weights(d_cells, subsample)

    roll = np.roll
    K = WX.shape[0]
    R = (K - 1) // 2

    num_x = np.zeros_like(fs, dtype=float)
    num_y = np.zeros_like(fs, dtype=float)

    # 卷积式聚合
    for a in range(K):
        di = a - R
        row_wx = WX[a]
        row_wy = WY[a]
        for b in range(K):
            wx = row_wx[b]
            wy = row_wy[b]
            if wx == 0.0 and wy == 0.0:
                continue
            dj = b - R
            nb = roll(roll(fs, di, axis=0), dj, axis=1)
            if wx != 0.0:
                num_x += nb * wx
            if wy != 0.0:
                num_y += nb * wy

    mag = np.sqrt(num_x * num_x + num_y * num_y)
    mag = np.maximum(mag, epsn)

    nx_full = num_x / mag  # 从固到液
    ny_full = num_y / mag

    out_nx[intf] = nx_full[intf]
    out_ny[intf] = ny_full[intf]
    return out_nx, out_ny
