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
    fs: np.ndarray, theta_deg: np.ndarray, masks: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    GF（度制）。θ 取“晶粒主生长方向与对角线族 {±45°} 的最小夹角”。
    规则：有一阶固相或二阶≥2 → GF=1；无固相 → GF=0；
          仅单一对角固相 → GF = 1 / (√2 * |cos θ_min|)。
    """
    Ny, Nx = fs.shape
    GF = np.ones((Ny, Nx), dtype=float)

    mask_sol = masks["mask_sol"] if "mask_sol" in masks else masks["sol"]
    mask_int = masks["mask_int"] if "mask_int" in masks else masks["intf"]
    if mask_sol.dtype != bool:
        mask_sol = mask_sol.astype(bool, copy=False)
    if mask_int.dtype != bool:
        mask_int = mask_int.astype(bool, copy=False)

    # 一阶轴向邻胞
    solN = np.roll(mask_sol, 1, axis=0)
    solS = np.roll(mask_sol, -1, axis=0)
    solW = np.roll(mask_sol, 1, axis=1)
    solE = np.roll(mask_sol, -1, axis=1)
    has_primary = solN | solS | solW | solE

    # 二阶对角邻胞
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

    mask_none_sol = (~has_primary) & (diag_count == 0)
    mask_single_diag = (~has_primary) & (diag_count == 1)

    # 角度：与 {±45°} 的“最小夹角”
    th = np.deg2rad(theta_deg)
    eps = 1e-12
    # inv_sqrt2 = 1.0  # 配合4向上风 凸归一化 去掉sqrt2
    inv_sqrt2 = 1.0 / np.sqrt(2.0)  # 原始
    cos_to_p45 = np.abs(np.cos(th - np.deg2rad(45.0)))
    cos_to_m45 = np.abs(np.cos(th + np.deg2rad(45.0)))
    cos_min_angle = np.maximum(cos_to_p45, cos_to_m45)  # cos(最小夹角) = 两者取大
    denom = np.maximum(cos_min_angle, eps)
    GF_single = inv_sqrt2 / denom

    # 把同一个 GF 应用于“仅单一对角”的两种情形
    tgt_single = mask_single_diag & mask_int
    GF[tgt_single] = GF_single[tgt_single]

    # 无固相：GF=0；非界面：GF=1
    GF[mask_int & mask_none_sol] = 0.0
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
    mask_int = masks.get("intf")

    dx = float(grid.dx)
    dy = float(grid.dy)

    # 1) Ln（法向穿越长度）
    Ln = L_n(fields.nx, fields.ny, dx, dy)

    # 2) 形状因子 GF（降低栅格各向异性）
    GF = shape_factor_GF(fs, grid.theta, masks)

    # 3) Δf_s（界面带；单向、限幅）
    eps = 1e-30
    delta_fs = np.zeros_like(fs, dtype=float)
    num = GF[mask_int] * vn[mask_int] * dt
    den = np.maximum(Ln[mask_int], eps)
    df_int = num / den
    # df_int = num / 1e-6
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
