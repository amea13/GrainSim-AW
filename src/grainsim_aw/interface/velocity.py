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
    eq: Tuple[np.ndarray, np.ndarray],  # (CLs, CSs)
    out_vn: Optional[np.ndarray] = None,
    out_vx: Optional[np.ndarray] = None,
    out_vy: Optional[np.ndarray] = None,
):
    fs, CL, CS, T = grid.fs, grid.CL, grid.CS, grid.T
    dx, dy = float(grid.dx), float(grid.dy)

    k0 = float(cfg.get("k0", 0.34))
    forbid_remelt = bool(cfg.get("forbid_remelt", True))

    band = np.asarray(masks["intf"], dtype=bool)
    if not np.any(band):
        z = np.zeros_like(fs, dtype=float)
        if out_vn is not None:
            out_vn[...] = 0.0
        if out_vx is not None:
            out_vx[...] = 0.0
        if out_vy is not None:
            out_vy[...] = 0.0
        return z, z, z

    nx, ny = normal

    # 规范化法向，避免法向幅值被误用为权重
    n2 = nx * nx + ny * ny
    invn = 1.0 / np.sqrt(np.maximum(n2, 1e-18))
    nxu = nx * invn
    nyu = ny * invn

    CLs, CSs = eq

    DL = Dl_from_T(T)
    DS = Ds_from_T(T)
    roll = np.roll

    # 面闸门（min 闸）
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))

    # 邻居中心
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # 四面等效通量项 N_face（与原思路一致）
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * (1.0 - fs_W)
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * (1.0 - fs_E)
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * (1.0 - fs_S)
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * (1.0 - fs_N)

    # Stefan 分母及稳健保护（按带内量级自适应）
    den_x = (1.0 - k0) * CLs * dx
    den_y = (1.0 - k0) * CLs * dy

    def _safe(den: np.ndarray) -> np.ndarray:
        out = den.copy()
        amp = float(np.nanmax(np.abs(den[band])))
        eps = max(1e-12, amp * 1e-12 + 1e-18)
        sgn = np.where(out >= 0.0, 1.0, -1.0)
        out = np.where(np.abs(out) < eps, sgn * eps, out)
        return out

    den_x = _safe(den_x)
    den_y = _safe(den_y)

    # 法向上风权重：在 x、y 两轴分别只取法向“下风侧”通量，并用法向分量大小加权
    wx_E = np.maximum(nxu, 0.0)  # nx>0 用东侧
    wx_W = np.maximum(-nxu, 0.0)  # nx<0 用西侧
    wy_N = np.maximum(nyu, 0.0)  # ny>0 用北侧
    wy_S = np.maximum(-nyu, 0.0)  # ny<0 用南侧

    # 轴向贡献（已是“沿法向上风”的贡献）
    Vx = wx_E * (N_E / den_x) + wx_W * (N_W / den_x)
    Vy = wy_N * (N_N / den_y) + wy_S * (N_S / den_y)

    # 合成法向速度。这里 Vx、Vy 已经按法向选择了上风面并乘以 |n_x|、|n_y|，直接相加即可
    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = (Vx + Vy)[band]

    # 禁止重熔
    if forbid_remelt:
        Vn[band] = np.maximum(Vn[band], 0.0)

    if out_vn is not None:
        out_vn[band] = Vn[band]
    if out_vx is not None:
        out_vx[band] = Vx[band]  # 输出已上风的轴向贡献，便于诊断
    if out_vy is not None:
        out_vy[band] = Vy[band]

    return Vn, Vx, Vy


def compute_velocity1(
    grid,
    cfg: Dict,
    masks: Dict[str, np.ndarray],
    *,
    normal: Tuple[np.ndarray, np.ndarray],
    eq: Tuple[np.ndarray, np.ndarray],  # (CL_star, CS_star)
    out_vn: Optional[np.ndarray] = None,
    out_vx: Optional[np.ndarray] = None,
    out_vy: Optional[np.ndarray] = None,
):
    """
    先计算 Vx、Vy，再合成 Vn = Vx*nx + Vy*ny。
    注意：这里的“w/E/S/N”是西/东/南/北邻胞；fs 为固相体积分数。
    """
    nx, ny = normal
    n2 = nx * nx + ny * ny
    invn = 1.0 / np.sqrt(np.maximum(n2, 1e-18))
    nxu = nx * invn
    nyu = ny * invn

    CL_star, CS_star = eq

    fs = grid.fs
    CL = getattr(grid, "CL")
    CS = getattr(grid, "CS")
    T = grid.T

    # 扩散系数：优先用网格上的场，否则用 cfg 标量
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    k0 = float(cfg.get("k0", cfg.get("k", 0.0)))
    dx = float(getattr(grid, "dx", cfg.get("dx", 1.0)))
    dy = float(getattr(grid, "dy", cfg.get("dy", dx)))
    eps = 1e-12
    roll = np.roll

    mask_int = masks.get("intf")
    if mask_int is None:
        # 若没有提供接口掩码，就全域计算再在最后清零
        mask_int = np.ones_like(fs, dtype=bool)

    fs_W = roll(fs, 1, 1)
    fs_E = roll(fs, -1, 1)
    fs_S = roll(fs, 1, 0)
    fs_N = roll(fs, -1, 0)

    # 邻居中心
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # 方向分量（对应式(12)；y 方向同理）
    denom_x = dx * np.maximum((1.0 - k0) * CL_star, eps)
    denom_y = dy * np.maximum((1.0 - k0) * CL_star, eps)

    # 单侧速度（别先相加）
    Vx_E = (DS * (CS_star - CS_E) * fs_E + DL * (CL_star - CL_E) * (1 - fs_E)) / denom_x
    Vx_W = (DS * (CS_star - CS_W) * fs_W + DL * (CL_star - CL_W) * (1 - fs_W)) / denom_x
    Vy_N = (DS * (CS_star - CS_N) * fs_N + DL * (CL_star - CL_N) * (1 - fs_N)) / denom_y
    Vy_S = (DS * (CS_star - CS_S) * fs_S + DL * (CL_star - CL_S) * (1 - fs_S)) / denom_y

    # 按法向选择迎风一侧
    Vx_sel = np.where(nxu >= 0.0, Vx_E, Vx_W)
    Vy_sel = np.where(nyu >= 0.0, Vy_N, Vy_S)

    # 只允许生长可在这里截断
    Vx_sel = np.maximum(Vx_sel, 0.0)
    Vy_sel = np.maximum(Vy_sel, 0.0)

    # 合成法向速度：用 |n| 当权重
    Vn = np.where(mask_int, np.abs(nxu) * Vx_sel + np.abs(nyu) * Vy_sel, 0.0)

    if out_vx is not None:
        out_vx[...] = Vx_sel
    if out_vy is not None:
        out_vy[...] = Vy_sel
    if out_vn is not None:
        out_vn[...] = Vn
    else:
        return Vn, Vx_sel, Vy_sel
