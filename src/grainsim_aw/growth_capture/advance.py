from __future__ import annotations
from typing import Dict, Any
import numpy as np


def L_n(nx: np.ndarray, ny: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
        if |cos(angn)| >= |sin(angn)|:
            Ln = len*( 1/|cos| + (1 - |tan|)*|sin| )
        else:
            Ln = len*( 1/|sin| + (1 - |1/tan|)*|cos| )
    其中 cos(angn) = -nx, sin(angn) = ny。。
    """
    Ny, Nx = nx.shape
    Ln = np.zeros_like(nx, dtype=np.float64)

    for i in range(Ny):
        for j in range(Nx):
            # |cos(angn)| = | -nx | = |nx|,  |sin(angn)| = |ny|
            c = abs(nx[i, j])
            s = abs(ny[i, j])

            if c >= s:
                # |tan| = s / c
                Ln[i, j] = dx * (1.0 / c + (1.0 - (s / c)) * s)
            else:
                # |1/tan| = c / s
                Ln[i, j] = dx * (1.0 / s + (1.0 - (c / s)) * c)

    return Ln


def shape_factor_GF(
    fs: np.ndarray, theta_rad: np.ndarray, masks: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    - 仅在界面胞上计算（masks['intf'] 为 True）
    - 轴向邻胞有固相 → GF=1
    - 轴向全无且对角固相数≥2 → GF=1
    - 轴向全无且对角固相数<2 → GF=1/√2/ cos(theta)
    - 其余位置默认 0
    """
    Ny, Nx = fs.shape
    gf = np.zeros_like(fs, dtype=np.float64)
    mask_int = masks["intf"]

    for i in range(Ny):
        for j in range(Nx):
            if not mask_int[i, j]:
                continue

            im, ip = i - 1, i + 1
            jm, jp = j - 1, j + 1

            # 轴向固相计数 S1
            S1 = 0.0
            if fs[im, j] == 1.0:
                S1 += 1.0
            if fs[ip, j] == 1.0:
                S1 += 1.0
            if fs[i, jm] == 1.0:
                S1 += 1.0
            if fs[i, jp] == 1.0:
                S1 += 1.0

            # 对角固相计数 S2
            S2 = 0.0
            if fs[im, jm] == 1.0:
                S2 += 1.0
            if fs[im, jp] == 1.0:
                S2 += 1.0
            if fs[ip, jm] == 1.0:
                S2 += 1.0
            if fs[ip, jp] == 1.0:
                S2 += 1.0

            if S1 == 0.0 and S2 == 0.0:
                gf[i, j] = 0.0
            elif S1 > 0.0:
                gf[i, j] = 1.0
            elif S2 >= 2.0:
                gf[i, j] = 1.0
            else:
                gf[i, j] = 1.0 / np.sqrt(2.0) / np.cos(theta_rad[i, j])

    return gf


def update_Ldia(grid, delta_fs: np.ndarray, theta: np.ndarray) -> None:
    """Δf_s 推进偏心正方形半对角线 L_dia：ΔL = Δf_s * (dx / max(|sinθ|,|cosθ|))."""
    dx = float(grid.dx)
    s = np.abs(np.sin(theta))
    c = np.cos(theta)
    denom = np.maximum(s, c)
    Ldia_max = dx / denom
    grid.L_dia += delta_fs * Ldia_max
    # np.minimum(grid.L_dia, Ldia_max, out=grid.L_dia)


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
    Cl = grid.CL
    Cs = grid.CS
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
    delta_fs[mask_int] = num / den

    # 4)更新偏心正方形半对角线长度
    update_Ldia(grid, delta_fs, grid.theta)

    # 5) 保存上一次迭代后的元胞状态
    fs_prev = fs.copy()
    Cl_prev = Cl.copy()
    Cs_prev = Cs.copy()

    if np.any(delta_fs[mask_int] > 1 - fs[mask_int]):
        delta_fs[mask_int] = 1 - fs[mask_int]
    fs[mask_int] = fs[mask_int] + delta_fs[mask_int]

    Cs[mask_int] = (
        Cs_prev[mask_int] * fs_prev[mask_int]
        + k0 * Cl_prev[mask_int] * delta_fs[mask_int]
    ) / (fs_prev[mask_int] + delta_fs[mask_int])

    # 6) 输出给溶质源项
    fs_dot = delta_fs / dt
    fields.fs_dot[...] = fs_dot
    return fs_dot
