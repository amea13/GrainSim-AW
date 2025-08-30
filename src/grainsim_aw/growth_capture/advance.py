from __future__ import annotations
from typing import Dict, Any
import numpy as np


def _viewer(a: np.ndarray, pad: int = 1):
    """非环绕平移视图：越界补 0，不产生大临时数组。"""
    Ny, Nx = a.shape
    ap = np.pad(a, pad_width=pad, mode="constant", constant_values=0.0)
    base = pad

    def V(di: int, dj: int) -> np.ndarray:
        return ap[base + di : base + di + Ny, base + dj : base + dj + Nx]  # 视图切片

    return V


def L_n(nx: np.ndarray, ny: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    if |cos(angn)| >= |sin(angn)|:
        Ln = dx*( 1/|cos| + (1 - |tan|)*|sin| )
    else:
        Ln = dx*( 1/|sin| + (1 - |1/tan|)*|cos| )
    其中 cos(angn) = -nx, sin(angn) = ny；取绝对值后等价于 |nx|、|ny|。
    """
    c = np.abs(nx)  # |cos|
    s = np.abs(ny)  # |sin|
    use_c = c >= s

    # c 分支：dx*(1/c + (1 - s/c)*s)
    inv_c = np.divide(1.0, c, out=np.zeros_like(c), where=(c > 0))
    s_over_c = np.divide(s, c, out=np.zeros_like(c), where=(c > 0))
    Ln_c = dx * (inv_c + (1.0 - s_over_c) * s)

    # s 分支：dx*(1/s + (1 - c/s)*c)
    inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=(s > 0))
    c_over_s = np.divide(c, s, out=np.zeros_like(s), where=(s > 0))
    Ln_s = dx * (inv_s + (1.0 - c_over_s) * c)

    Ln = np.where(use_c, Ln_c, Ln_s)
    return Ln


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
    """Δf_s 推进偏心正方形半对角线 L_dia：ΔL = Δf_s * (dx / max(|sinθ|,cosθ))."""
    dx = float(grid.dx)
    s = np.abs(np.sin(theta))
    c = np.cos(theta)  # 保持与你当前实现一致（未取 abs）
    denom = np.maximum(s, c)
    Ldia_max = dx / denom
    grid.L_dia += delta_fs * Ldia_max
    # 若需要限幅，按你注释解开下一行：
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
    fs: np.ndarray = grid.fs
    Cl: np.ndarray = grid.CL
    Cs: np.ndarray = grid.CS
    m = masks.get("intf")  # 保持你原来的掩码使用
    k0 = float(cfg.get("k0", 0.34))
    _ = fields.test  # 保留以不改变外部行为

    dx = float(grid.dx)
    dy = float(grid.dy)  # 按你的版本保留（未使用也不删除）

    # 1) Ln（法向穿越长度）
    Ln = L_n(fields.nx, fields.ny, dx, dy)

    # 2) 形状因子 GF（降低栅格各向异性）
    GF = shape_factor_GF(fs, grid.theta, masks)

    # 3) Δf_s（界面带；单向、限幅）
    delta_fs = np.zeros_like(fs, dtype=np.float64)
    num = GF[m] * vn[m] * dt
    den = Ln[m]
    delta_fs[m] = num / den  # 保持与你当前版本完全相同的除法行为

    # 4) 保存上一次迭代后的元胞状态
    fs_prev = fs.copy()
    Cl_prev = Cl.copy()
    Cs_prev = Cs.copy()

    # 限幅并更新 fs
    delta_fs[m] = np.minimum(delta_fs[m], (1.0 - fs[m]))
    fs[m] = fs[m] + delta_fs[m]

    # 5) 更新偏心正方形半对角线长度（保持你现在的顺序：在 fs 更新之后）
    update_Ldia(grid, delta_fs, grid.theta)

    # 若界面带上 fs==1 → Cl=0
    Cl[m] = np.where(fs[m] == 1.0, 0.0, Cl[m])

    # 更新 Cs（保持你的公式与除法方式）
    Cs[m] = (Cs_prev[m] * fs_prev[m] + k0 * Cl_prev[m] * delta_fs[m]) / (
        fs_prev[m] + delta_fs[m]
    )

    # 6) 输出给溶质源项
    fs_dot = delta_fs / dt
    fields.fs_dot[...] = fs_dot
    return fs_dot
