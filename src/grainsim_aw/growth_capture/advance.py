# src/grainsim_aw/growth_capture/advance.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np


def L_n(nx: np.ndarray, ny: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """由法向分量计算“界面穿越长度” Ln（dx=dy 时等价于常用式）。"""
    eps = 1e-12
    c = np.maximum(np.abs(nx), eps)
    s = np.maximum(np.abs(ny), eps)
    Ln_c_ge_s = dx * (1.0 / c + s - (s * s) / c)
    Ln_s_gt_c = dy * (1.0 / s + c - (c * c) / s)
    return np.where(c >= s, Ln_c_ge_s, Ln_s_gt_c)


def shape_factor_GF(
    fs: np.ndarray, nx: np.ndarray, ny: np.ndarray, masks: Dict[str, np.ndarray]
) -> np.ndarray:
    Ny, Nx = fs.shape
    GF = np.ones((Ny, Nx), dtype=float)

    # ☆ 关键：用索引而不是 get，保证是 ndarray 而非 Optional
    mask_sol: np.ndarray = masks["mask_sol"] if "mask_sol" in masks else masks["sol"]
    mask_int: np.ndarray = masks["mask_int"] if "mask_int" in masks else masks["intf"]

    # ☆ 保证布尔 dtype（静态检查 + 位运算都更稳）
    if mask_sol.dtype != bool:
        mask_sol = mask_sol.astype(bool, copy=False)
    if mask_int.dtype != bool:
        mask_int = mask_int.astype(bool, copy=False)

    # 轴向邻胞（roll 的参数现在是 ndarray[bool]，Pylance 不再报错）
    solN = np.roll(mask_sol, 1, axis=0)
    solS = np.roll(mask_sol, -1, axis=0)
    solW = np.roll(mask_sol, 1, axis=1)
    solE = np.roll(mask_sol, -1, axis=1)
    has_primary = solN | solS | solW | solE

    # 对角邻胞
    solNE = np.roll(np.roll(mask_sol, 1, axis=0), -1, axis=1)
    solNW = np.roll(np.roll(mask_sol, 1, axis=0), 1, axis=1)
    solSE = np.roll(np.roll(mask_sol, -1, axis=0), -1, axis=1)
    solSW = np.roll(np.roll(mask_sol, -1, axis=0), 1, axis=1)
    diag_count = (
        solNE.astype(np.int8)
        + solNW.astype(np.int8)
        + solSE.astype(np.int8)
        + solSW.astype(np.int8)
    )
    mask_diag_single = (~has_primary) & (diag_count == 1)

    eps = 1e-12
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    cos_plus = np.abs(nx + ny) * inv_sqrt2
    cos_minus = np.abs(nx - ny) * inv_sqrt2

    mask_NE_SW = mask_diag_single & (solNE ^ solSW) & ~(solNW | solSE)
    denom_NE_SW = np.maximum(cos_minus, eps)
    GF_NE_SW = inv_sqrt2 / denom_NE_SW
    GF[mask_NE_SW & mask_int] = GF_NE_SW[mask_NE_SW & mask_int]

    mask_NW_SE = mask_diag_single & (solNW ^ solSE) & ~(solNE | solSW)
    denom_NW_SE = np.maximum(cos_plus, eps)
    GF_NW_SE = inv_sqrt2 / denom_NW_SE
    GF[mask_NW_SE & mask_int] = GF_NW_SE[mask_NW_SE & mask_int]

    GF[~mask_int] = 1.0
    return GF


def update_Ldia(grid, delta_fs: np.ndarray, theta: np.ndarray) -> None:
    """Δf_s 推进偏心正方形半对角线 L_dia：ΔL = Δf_s * (dx / max(|sinθ|,|cosθ|))."""
    dx = float(grid.dx)
    eps = 1e-12
    s = np.abs(np.sin(theta))
    c = np.abs(np.cos(theta))
    denom = np.maximum(np.maximum(s, c), eps)
    Ldia_max = dx / denom
    grid.L_dia += delta_fs * Ldia_max
    np.minimum(grid.L_dia, Ldia_max, out=grid.L_dia)


def advance_interface(
    grid,
    vn: np.ndarray,
    dt: float,
    cfg: Dict[str, Any],
    fields,
):
    """
    界面推进：计算 Ln、GF，得到 Δf_s，更新 fs/CL/L_dia，并写出 fs_dot 到 fields。
    返回 fs_dot（同 fields.fs_dot）。
    """
    fs = grid.fs
    masks = fields.masks
    mask_int = masks.get("intf") or masks.get("mask_int")

    dx = float(grid.dx)
    dy = float(grid.dy)

    # 1) Ln（法向穿越长度）
    Ln = L_n(fields.nx, fields.ny, dx, dy)

    # 2) 形状因子 GF（降低栅格各向异性）
    GF = shape_factor_GF(fs, fields.nx, fields.ny, masks)

    # 3) Δf_s（界面带；单向、限幅）
    eps = 1e-30
    delta_fs = np.zeros_like(fs, dtype=float)
    num = GF[mask_int] * vn[mask_int] * dt
    den = np.maximum(Ln[mask_int], eps)
    df_int = num / den
    df_int = np.maximum(df_int, 0.0)
    np.minimum(df_int, 1.0 - fs[mask_int], out=df_int)
    delta_fs[mask_int] = df_int

    # 4) 原地更新 fs；界面满固后令 CL=0
    fs += delta_fs
    grid.CL[fs == 1.0] = 0.0

    # 5) 更新 ESVC 半对角线
    update_Ldia(grid, delta_fs, grid.theta)

    # 6) 输出给溶质源项
    fs_dot = delta_fs / dt
    fields.fs_dot[...] = fs_dot
    return fs_dot
