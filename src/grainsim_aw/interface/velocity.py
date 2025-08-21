from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from ..core.material import Dl_from_T, Ds_from_T


def compute_velocity(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict,
    *,
    normal: Tuple[np.ndarray, np.ndarray],
    eq: Tuple[np.ndarray, np.ndarray],  # (CLs, CSs) 仅在界面带有意义
    out_vn: np.ndarray | None = None,
    out_vx: np.ndarray | None = None,
    out_vy: np.ndarray | None = None,
):
    """
    用 Stefan 守恒 + “面闸门”离散计算界面法向速度。
    步骤：
      1) 计算四个面通量因子 N_face（固/液扩散 + 闸门）
      2) 轴向速度:
           Vx = (N_W + N_E) / ((1-k0) * CLs * dx)
           Vy = (N_S + N_N) / ((1-k0) * CLs * dy)
         —— 仅在界面带上做除法与赋值
      3) 半上风重组:
           Vn = max(Vx,0) * |nx| + max(Vy,0) * |ny|
      4) 仅向界面带写入 out_vn/out_vx/out_vy（若提供）
    说明：
      - 只依赖 masks['intf'] 作为界面带掩码；若缺失会抛 KeyError。
      - 为避免除零告警，分母只在带内做自适应极小值保护。
    """
    if "intf" not in masks:
        raise KeyError("masks['intf'] 缺失：请统一使用键名 'intf' 作为界面带掩码。")

    # 读入场与参数
    fs: np.ndarray = grid.fs
    CL: np.ndarray = grid.CL
    CS: np.ndarray = grid.CS
    T: np.ndarray = grid.T
    dx = float(grid.dx)
    dy = float(grid.dy)

    k0 = float(cfg.get("k0", 1.0))
    forbid_remelt = bool(cfg.get("forbid_remelt", True))

    band: np.ndarray = masks["intf"].astype(bool)
    nx, ny = normal  # (fields.nx, fields.ny)
    CLs, CSs = eq  # (fields.cls, fields.css)

    # 若当前没有界面点，直接清零返回
    if not np.any(band):
        if out_vn is not None:
            out_vn[...] = 0.0
        if out_vx is not None:
            out_vx[...] = 0.0
        if out_vy is not None:
            out_vy[...] = 0.0
        # 也返回全零，便于调试
        z = np.zeros_like(fs, dtype=float)
        return z, z, z

    # 物性（中心点）
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

    # 面通量因子 N_face（只在 band 上使用）
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * (1.0 - fs_W)
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * (1.0 - fs_E)
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * (1.0 - fs_S)
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * (1.0 - fs_N)

    # 分母（只在 band 内会被用到）
    den_x = (1.0 - k0) * CLs * dx
    den_y = (1.0 - k0) * CLs * dy

    # 自适应极小值保护（仅 band 内），避免 0/近 0 除法
    def _protect_den(den: np.ndarray, band_mask: np.ndarray) -> np.ndarray:
        den_safe = den.copy()
        max_abs = float(np.nanmax(np.abs(den[band_mask])))
        eps = max(1e-12, max_abs * 1e-12 + 1e-18)
        # 注意 sign(0)=0，这里用 where 保持正负号一致
        sign = np.where(den >= 0.0, 1.0, -1.0)
        den_safe[band_mask] = np.where(
            np.abs(den[band_mask]) < eps,
            sign[band_mask] * eps,
            den[band_mask],
        )
        return den_safe

    den_x_safe = _protect_den(den_x, band)
    den_y_safe = _protect_den(den_y, band)

    # 轴向速度：只在 band 上做除法并赋值，带外为 0，避免全场除法产生告警
    Vx = np.zeros_like(fs, dtype=float)
    Vy = np.zeros_like(fs, dtype=float)
    num_x = N_W + N_E
    num_y = N_S + N_N
    Vx[band] = num_x[band] / den_x_safe[band]
    Vy[band] = num_y[band] / den_y_safe[band]

    # 半上风重组（只取推进分量）
    Vx_pos = np.zeros_like(Vx)
    Vy_pos = np.zeros_like(Vy)
    Vx_pos[band] = np.maximum(Vx[band], 0.0)
    Vy_pos[band] = np.maximum(Vy[band], 0.0)

    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vx_pos[band] * np.abs(nx[band]) + Vy_pos[band] * np.abs(ny[band])

    if forbid_remelt:
        # 进一步保证非负
        Vn[band] = np.maximum(Vn[band], 0.0)

    # 写出（仅在界面带）
    if out_vn is not None:
        out_vn[band] = Vn[band]
    if out_vx is not None:
        out_vx[band] = Vx_pos[band]
    if out_vy is not None:
        out_vy[band] = Vy_pos[band]

    # 同时返回，便于调试/可视化
    return Vn, Vx_pos, Vy_pos
