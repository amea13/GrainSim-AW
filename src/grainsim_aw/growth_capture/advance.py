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
    fs: np.ndarray, theta_rad: np.ndarray, masks: Dict[str, np.ndarray]
) -> np.ndarray:
    Ny, Nx = fs.shape
    GF = np.ones((Ny, Nx), dtype=float)

    mask_sol = masks["mask_sol"] if "mask_sol" in masks else masks["sol"]
    mask_int = masks["mask_int"] if "mask_int" in masks else masks["intf"]
    if mask_sol.dtype != bool:
        mask_sol = mask_sol.astype(bool, copy=False)
    if mask_int.dtype != bool:
        mask_int = mask_int.astype(bool, copy=False)

    # 一阶轴向邻胞（FNNC）
    solN = np.roll(mask_sol, 1, axis=0)
    solS = np.roll(mask_sol, -1, axis=0)
    solW = np.roll(mask_sol, 1, axis=1)
    solE = np.roll(mask_sol, -1, axis=1)
    has_primary = solN | solS | solW | solE  # NFNNC > 0

    # 二阶对角邻胞（SNNC）
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

    # 分段：
    # 1) NFNNC = 0 且 NSNNC = 0 → GF = 0
    mask_none_sol = (~has_primary) & (diag_count == 0)
    GF[mask_int & mask_none_sol] = 0.0

    # 2) NFNNC > 0 → GF = 1 （默认已是 1）
    # 3) NSNNC ≥ 2 → GF = 1 （默认已是 1）

    # 4) 仅单一对角固相（NFNNC = 0 且 NSNNC = 1）→ GF = 1 / (√2 * cos θ_min)
    mask_single_diag = (~has_primary) & (diag_count == 1)
    GF_single = (1.0 / np.sqrt(2.0)) / np.cos(theta_rad)
    GF[mask_int & mask_single_diag] = GF_single[mask_int & mask_single_diag]

    # 非界面保持 1
    GF[~mask_int] = 1.0
    return GF


def update_Ldia(grid, delta_fs: np.ndarray, theta: np.ndarray) -> None:
    """Δf_s 推进偏心正方形半对角线 L_dia：ΔL = Δf_s * (dx / max(|sinθ|,|cosθ|))."""
    dx = float(grid.dx)
    s = np.abs(np.sin(theta))
    c = np.cos(theta)
    denom = np.maximum(s, c)
    Ldia_max = dx / denom
    grid.L_dia += delta_fs * Ldia_max
    np.minimum(grid.L_dia, Ldia_max, out=grid.L_dia)


def advance_interface(
    grid,
    masks,
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
    CL = grid.CL
    CS = grid.CS
    mask_int = masks.get("intf")
    k0 = float(cfg.get("k0", 0.34))

    dx = float(grid.dx)
    dy = float(grid.dy)

    # 1) Ln（法向穿越长度）
    Ln = L_n(fields.nx, fields.ny, dx, dy)

    # 2) 形状因子 GF（降低栅格各向异性）
    GF = shape_factor_GF(fs, grid.theta, masks)

    # 3) Δf_s（界面带；单向、限幅）
    delta_fs = np.zeros_like(fs, dtype=float)
    num = GF[mask_int] * vn[mask_int] * dt
    den = Ln[mask_int]
    df_int = num / den
    np.minimum(df_int, 1.0 - fs[mask_int], out=df_int)
    delta_fs[mask_int] = df_int

    # 4) 原地更新 fs；界面满固后令 CL=0
    fs_prev = fs.copy()
    CL_prev = CL.copy()
    CS_prev = CS.copy()
    fs += delta_fs

    CL[mask_int] = np.where(fs[mask_int] == 1.0, 0.0, CL[mask_int])
    CS[mask_int] = (
        CS_prev[mask_int] * fs_prev[mask_int]
        + k0 * CL_prev[mask_int] * delta_fs[mask_int]
    ) / (fs_prev[mask_int] + delta_fs[mask_int])

    # 5) 更新 ESVC 半对角线
    update_Ldia(grid, delta_fs, grid.theta)

    # 6) 输出给溶质源项
    fs_dot = delta_fs / dt
    fields.fs_dot[...] = fs_dot
    return fs_dot
