from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from ..core.material import Dl_from_T, Ds_from_T


def compute_velocity(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict,
    *,
    normal: Tuple[np.ndarray, np.ndarray],
    eq: Tuple[np.ndarray, np.ndarray],  # (CLs, CSs)
    out_vn: Optional[np.ndarray] = None,
    out_vx: Optional[np.ndarray] = None,
    out_vy: Optional[np.ndarray] = None,
):
    """
    基于 Stefan 条件 + “面闸门”离散计算界面法向速度。
      1) 四个面通量因子 N_face（固/液扩散 + 闸门）
      2) 轴向速度: Vx=(N_W+N_E)/((1-k0)CLs*dx), Vy=(N_S+N_N)/((1-k0)CLs*dy)
      3) 半上风重组: Vn = max(Vx,0)*|nx| + max(Vy,0)*|ny|
      4) 仅在界面带写入 out_vn/out_vx/out_vy

    返回: (Vn, Vx_pos, Vy_pos) —— 返回值主要便于调试；通常你传出数组就地写入即可。
    """
    fs = grid.fs
    CL = grid.CL
    CS = grid.CS
    T = grid.T
    dx = float(grid.dx)
    dy = float(grid.dy)

    # 配置
    k0 = float(cfg.get("k0", 1.0))
    forbid_remelt = bool(cfg.get("forbid_remelt", True))

    # 掩码（不要用 `or`，避免数组真值歧义）
    intf = masks["intf"]

    band = np.asarray(intf, dtype=bool)

    # 若当前没有界面带，安全返回
    if not np.any(band):
        z = np.zeros_like(fs, dtype=float)
        if out_vn is not None:
            out_vn[...] = 0.0
        if out_vx is not None:
            out_vx[...] = 0.0
        if out_vy is not None:
            out_vy[...] = 0.0
        return z, z, z

    # 法向、平衡浓度（与网格同形）
    nx, ny = normal
    CLs, CSs = eq

    # 物性
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    roll = np.roll
    # 邻居中心值
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # 面闸门：f_S,face = min(f_S,P, f_S,邻)
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))

    # 四个面通量因子 N_face
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * (1.0 - fs_W)
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * (1.0 - fs_E)
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * (1.0 - fs_S)
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * (1.0 - fs_N)

    # 分母，仅对界面带做极小值保护，避免除零/NaN
    den_x = (1.0 - k0) * CLs * dx
    den_y = (1.0 - k0) * CLs * dy

    def _safe_den(den: np.ndarray) -> np.ndarray:
        out = den.copy()
        # 带内阈值按量级设定
        amp = float(np.nanmax(np.abs(den[band])))
        eps = max(1e-12, amp * 1e-12 + 1e-18)
        sign = np.where(den >= 0.0, 1.0, -1.0)  # 避免 sign(0)=0
        out = np.where(np.abs(den) < eps, sign * eps, den)
        return out

    den_x_safe = _safe_den(den_x)
    den_y_safe = _safe_den(den_y)

    # 轴向速度（面闸门离散）
    Vx = (N_W + N_E) / den_x_safe
    Vy = (N_S + N_N) / den_y_safe

    # 半上风重组：只取推进方向
    Vx_pos = np.maximum(Vx, 0.0)
    Vy_pos = np.maximum(Vy, 0.0)

    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vx_pos[band] * np.abs(nx[band]) + Vy_pos[band] * np.abs(ny[band])

    if forbid_remelt:
        # 再保证非负
        Vn[band] = np.maximum(Vn[band], 0.0)

    # 写出（仅界面带）
    if out_vn is not None:
        out_vn[band] = Vn[band]
    if out_vx is not None:
        out_vx[band] = Vx_pos[band]
    if out_vy is not None:
        out_vy[band] = Vy_pos[band]

    return Vn, Vx_pos, Vy_pos
