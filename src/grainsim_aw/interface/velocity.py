from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..core.material import Dl_from_T, Ds_from_T


def compute_velocity(
    cfg_if: Dict[str, Any],
    mask_int: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    *,
    grid=None,
    CLs: Optional[np.ndarray] = None,  # C_L^* at interface band
    CSs: Optional[np.ndarray] = None,  # C_S^* at interface band
    forbid_remelt: bool = False,  # True 时把 Vn<0 截为 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid is None or CLs is None or CSs is None or not np.any(mask_int):
        z = np.zeros_like(nx, dtype=float)
        return z, z, z

    CL, CS, fs, T = grid.CL, grid.CS, grid.fs, grid.T
    dx, dy = float(grid.dx), float(grid.dy)
    k0 = float(cfg_if.get("k0", 1.0))

    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    # 邻居值
    roll = np.roll
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # 面开口系数（闸门）
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))

    # 四个面的“通量因子” 𝓝_face
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * (1.0 - fs_W)
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * (1.0 - fs_E)
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * (1.0 - fs_S)
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * (1.0 - fs_N)

    # 法向上风权重
    wx_E = np.maximum(nx, 0.0)
    wx_W = np.maximum(-nx, 0.0)  # 和 |nx| 配套
    wy_N = np.maximum(ny, 0.0)
    wy_S = np.maximum(-ny, 0.0)

    # 分母及稳健保护
    den = (1.0 - k0) * CLs
    eps = max(1e-12, float(np.nanmax(np.abs(den[mask_int]))) * 1e-12 + 1e-18)
    sign = np.where(den >= 0.0, 1.0, -1.0)
    den_safe = np.where(np.abs(den) < eps, sign * eps, den)

    # Vx,Vy 是已经“沿法向上风”的分量贡献
    Vx = (wx_E * N_E + wx_W * N_W) / (dx * den_safe)
    Vy = (wy_N * N_N + wy_S * N_S) / (dy * den_safe)

    Vn = Vx + Vy
    if forbid_remelt:
        Vn = np.maximum(Vn, 0.0)

    # 仅在界面带赋值
    Z = np.zeros_like(Vn)
    Z[mask_int] = Vn[mask_int]
    Vx_out = np.zeros_like(Vx)
    Vx_out[mask_int] = Vx[mask_int]
    Vy_out = np.zeros_like(Vy)
    Vy_out[mask_int] = Vy[mask_int]
    return Z, Vx_out, Vy_out
