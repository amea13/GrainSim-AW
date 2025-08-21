# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

__all__ = ["compute_curvature", "compute_normal"]

# ========= 圆质心法：权重模板（d=7, sub=8），首用即建 =========
_WEIGHTS_CACHE: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
_D_CELLS = 7
_SUBSAMPLE = 8


def _generate_weights_by_sampling(
    d_cells: int, sub: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d_cells <= 0 or d_cells % 2 == 0:
        raise ValueError("d_cells 必须为正奇数，如 7")
    R = (d_cells - 1) // 2
    K = 2 * R + 1
    Rc = 0.5 * d_cells  # 以“格宽=1”为单位的圆半径
    # 子单元中心坐标（[-0.5, 0.5)）
    u = (np.arange(sub, dtype=float) + 0.5) / sub - 0.5

    DI = np.arange(-R, R + 1, dtype=int)[:, None] * np.ones((1, K), dtype=int)
    DJ = np.ones((K, 1), dtype=int) * np.arange(-R, R + 1, dtype=int)[None, :]
    W = np.zeros((K, K), dtype=float)

    for a in range(K):
        for b in range(K):
            di = DI[a, b]
            dj = DJ[a, b]
            YY = di + u[:, None]  # (sub,sub)
            XX = dj + u[None, :]
            inside = (XX * XX + YY * YY) <= (Rc * Rc)
            W[a, b] = inside.mean()
    return DI, DJ, W


def _ensure_weights():
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is None:
        _WEIGHTS_CACHE = _generate_weights_by_sampling(_D_CELLS, _SUBSAMPLE)


# ========= 曲率（中心差分法） =========
def compute_curvature(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    中心差分法计算曲率 κ（level-set 形式），只在界面带写入。
    κ = (2 fx fy fxy - fxx fy^2 - fyy fx^2) / ( (fx^2 + fy^2)^(3/2) )
    """
    fs = grid.fs
    dx = float(grid.dx)
    dy = float(grid.dy)
    intf: np.ndarray = masks["intf"] if "intf" in masks else masks["mask_int"]
    if intf.dtype != bool:
        intf = intf.astype(bool, copy=False)

    # 一阶导
    fx = (np.roll(fs, -1, axis=1) - np.roll(fs, 1, axis=1)) / (2.0 * dx)
    fy = (np.roll(fs, -1, axis=0) - np.roll(fs, 1, axis=0)) / (2.0 * dy)

    # 二阶与混合导
    fxx = (np.roll(fs, -1, axis=1) + np.roll(fs, 1, axis=1) - 2.0 * fs) / (dx * dx)
    fyy = (np.roll(fs, -1, axis=0) + np.roll(fs, 1, axis=0) - 2.0 * fs) / (dy * dy)
    fxy = (
        np.roll(np.roll(fs, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(fs, 1, axis=0), -1, axis=1)
        - np.roll(np.roll(fs, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(fs, 1, axis=0), 1, axis=1)
    ) / (4.0 * dx * dy)

    num = 2.0 * fx * fy * fxy - fxx * (fy * fy) - fyy * (fx * fx)
    g2 = fx * fx + fy * fy
    den = np.power(g2, 1.5) + 1e-30  # 极小保护

    kappa_full = num / den
    if out is None:
        out = np.zeros_like(fs, dtype=float)
    out[intf] = kappa_full[intf]
    return out


# ========= 法向（圆质心法） =========
def compute_normal(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict,
    out_nx: np.ndarray | None = None,
    out_ny: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    圆质心法计算法向 (nx, ny)，只在界面带写入；方向取 n ∥ (A - B)。
    要求：已更新 ghost，且 nghost >= 3。
    """
    _ensure_weights()
    DI, DJ, W = _WEIGHTS_CACHE  # type: ignore
    fs = grid.fs
    intf: np.ndarray = masks["intf"] if "intf" in masks else masks["mask_int"]
    if intf.dtype != bool:
        intf = intf.astype(bool, copy=False)

    if out_nx is None:
        out_nx = np.zeros_like(fs, dtype=float)
    if out_ny is None:
        out_ny = np.zeros_like(fs, dtype=float)

    num_x = np.zeros_like(fs, dtype=float)
    num_y = np.zeros_like(fs, dtype=float)

    K = W.shape[0]
    for a in range(K):
        for b in range(K):
            w = W[a, b]
            if w == 0.0:
                continue
            di = int(DI[a, b])
            dj = int(DJ[a, b])
            nb = np.roll(np.roll(fs, di, axis=0), dj, axis=1)
            num_x += nb * w * dj
            num_y += nb * w * di

    mag = np.sqrt(num_x * num_x + num_y * num_y) + 1e-30
    nx_full = -num_x / mag
    ny_full = -num_y / mag

    out_nx[intf] = nx_full[intf]
    out_ny[intf] = ny_full[intf]
    return out_nx, out_ny
