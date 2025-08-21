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
    out_vn: np.ndarray | None = None,
    out_vx: np.ndarray | None = None,
    out_vy: np.ndarray | None = None,
):
    """
    依据 Stefan 守恒的“面闸门”离散得到界面法向速度：
      1) 计算四个面的通量因子 N_face（固/液扩散 + 闸门）
      2) 轴向速度 Vx, Vy = (N_W+N_E)/( (1-k0)CLs*dx ), (N_S+N_N)/( (1-k0)CLs*dy )
      3) 半上风重组：Vn = max(Vx,0)*|nx| + max(Vy,0)*|ny|
      4) 仅在界面带写入 out_vn/out_vx/out_vy
    """
    fs = grid.fs
    CL = grid.CL
    CS = grid.CS
    T = grid.T

    dx = float(grid.dx)
    dy = float(grid.dy)
    k0 = float(cfg.get("k0", 1.0))
    forbid_remelt = bool(cfg.get("forbid_remelt", True))

    intf: np.ndarray = masks["intf"] if "intf" in masks else masks["mask_int"]
    nx, ny = normal  # 由 InterfaceProcess 传入 (fields.nx, fields.ny)
    CLs, CSs = eq  # 由 InterfaceProcess 传入 (fields.cls, fields.css)

    # 物性系数（中心点）
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    roll = np.roll
    # 邻居中心值
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # 面闸门（开口系数）：f_S,face = min(f_S,P, f_S,邻)
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))

    # 四个面的“通量因子” N_face
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * (1.0 - fs_W)
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * (1.0 - fs_E)
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * (1.0 - fs_S)
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * (1.0 - fs_N)

    # 分母（仅界面带用得到）
    den_x = (1.0 - k0) * CLs * dx
    den_y = (1.0 - k0) * CLs * dy

    # 数值保护（按量级给最小阈值），尽量少干预
    band = intf.astype(bool)
    epsx = max(1e-12, float(np.nanmax(np.abs(den_x[band]))) * 1e-12 + 1e-18)
    epsy = max(1e-12, float(np.nanmax(np.abs(den_y[band]))) * 1e-12 + 1e-18)
    den_x_safe = np.where(np.abs(den_x) < epsx, np.sign(den_x) * epsx, den_x)
    den_y_safe = np.where(np.abs(den_y) < epsy, np.sign(den_y) * epsy, den_y)

    # 轴向速度（面闸门离散）
    Vx = (N_W + N_E) / den_x_safe
    Vy = (N_S + N_N) / den_y_safe

    # 半上风重组（只取推进方向）
    Vx_pos = np.maximum(Vx, 0.0)
    Vy_pos = np.maximum(Vy, 0.0)

    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vx_pos[band] * np.abs(nx[band]) + Vy_pos[band] * np.abs(ny[band])

    if forbid_remelt:
        # 正向推进已保证非负；此句仅为稳妥
        np.maximum(Vn, 0.0, out=Vn)

    # 就地写出
    if out_vn is not None:
        out_vn[band] = Vn[band]
    if out_vx is not None:
        out_vx[band] = Vx_pos[band]
    if out_vy is not None:
        out_vy[band] = Vy_pos[band]

    return Vn, Vx_pos, Vy_pos
