from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from ..core.material import Dl_from_T, Ds_from_T


def compute_velocity(
    grid,
    cfg: Dict,
    masks: Dict[str, np.ndarray],
    *,
    normal: Tuple[np.ndarray, np.ndarray],
    eq: Tuple[np.ndarray, np.ndarray],  # (CL_star, CS_star) == (Cl8, Cs8)
    out_vn: Optional[np.ndarray] = None,
    out_vx: Optional[np.ndarray] = None,
    out_vy: Optional[np.ndarray] = None,
):
    nx, ny = normal
    CL_star, CS_star = eq

    fs = grid.fs
    fl = 1.0 - fs
    CL = grid.CL
    CS = grid.CS
    T = grid.T

    # Dl/Ds 按中心元胞温度取值
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    k0 = float(cfg.get("k0", cfg.get("k", 0.34)))
    dx = float(getattr(grid, "dx", cfg.get("dx", 1e-6)))

    mask_int = masks["intf"]

    if out_vx is None:
        out_vx = np.zeros_like(fs, dtype=np.float64)
    if out_vy is None:
        out_vy = np.zeros_like(fs, dtype=np.float64)
    if out_vn is None:
        out_vn = np.zeros_like(fs, dtype=np.float64)

    Ny, Nx = fs.shape

    for i in range(Ny):
        for j in range(Nx):
            if not mask_int[i, j]:
                continue

            im, ip = i - 1, i + 1
            jm, jp = j - 1, j + 1

            # vel_x
            termLx = (1.0 - CL[i, jm] / CL_star[i, j]) * fl[i, jm] + (
                1.0 - CL[i, jp] / CL_star[i, j]
            ) * fl[i, jp]
            termSx = (1.0 - CS[i, jm] / CS_star[i, j]) * fs[i, jm] + (
                1.0 - CS[i, jp] / CS_star[i, j]
            ) * fs[i, jp]
            vx = (
                DL[i, j] / dx / (1.0 - k0) * termLx
                + k0 * DS[i, j] / dx / (1.0 - k0) * termSx
            )

            # vel_y
            termLy = (1.0 - CL[im, j] / CL_star[i, j]) * fl[im, j] + (
                1.0 - CL[ip, j] / CL_star[i, j]
            ) * fl[ip, j]
            termSy = (1.0 - CS[im, j] / CS_star[i, j]) * fs[im, j] + (
                1.0 - CS[ip, j] / CS_star[i, j]
            ) * fs[ip, j]
            vy = (
                DL[i, j] / dx / (1.0 - k0) * termLy
                + k0 * DS[i, j] / dx / (1.0 - k0) * termSy
            )

            out_vx[i, j] = vx
            out_vy[i, j] = vy

            # 合成 vel（按符号分支 + |nx|/|ny| 权重）
            if vx >= 0.0 and vy >= 0.0:
                v = vx * abs(nx[i, j]) + vy * abs(ny[i, j])
            elif vx >= 0.0 and vy < 0.0:
                v = vx * abs(nx[i, j])
            elif vx < 0.0 and vy >= 0.0:
                v = vy * abs(ny[i, j])
            else:
                v = 0.0

            out_vn[i, j] = v

    return out_vn, out_vx, out_vy
