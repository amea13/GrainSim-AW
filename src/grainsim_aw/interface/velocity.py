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

    # 未凸归一化
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

    # 未凸归一化
    # 合成法向速度。这里 Vx、Vy 已经按法向选择了上风面并乘以 |n_x|、|n_y|，直接相加即可
    # Vn = np.zeros_like(fs, dtype=float)
    # Vn[band] = (Vx + Vy)[band]

    # 合成法向速度（上风 + 凸归一化，削弱 45° 偏置）
    Vn_raw = Vx + Vy
    wsum = np.abs(nxu) + np.abs(nyu)  # = |nx|+|ny|，∈[1, √2]
    Vn_iso = Vn_raw / np.maximum(wsum, 1e-12)  # 归一化成凸权重

    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vn_iso[band]

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


# 8向上风
def compute_velocity7(
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

    # ---- 额外：对角上风权重（仍保持上风、非负）
    rt2 = np.sqrt(2.0)
    w_NE = np.maximum((nxu + nyu) / rt2, 0.0)
    w_NW = np.maximum((-nxu + nyu) / rt2, 0.0)
    w_SE = np.maximum((nxu - nyu) / rt2, 0.0)
    w_SW = np.maximum((-nxu - nyu) / rt2, 0.0)

    # 对角邻居中心与“对角面开口”（用对角两点的 min 作连通判据）
    CL_NE = np.roll(np.roll(CL, -1, 0), -1, 1)
    CL_NW = np.roll(np.roll(CL, -1, 0), 1, 1)
    CL_SE = np.roll(np.roll(CL, 1, 0), -1, 1)
    CL_SW = np.roll(np.roll(CL, 1, 0), 1, 1)

    CS_NE = np.roll(np.roll(CS, -1, 0), -1, 1)
    CS_NW = np.roll(np.roll(CS, -1, 0), 1, 1)
    CS_SE = np.roll(np.roll(CS, 1, 0), -1, 1)
    CS_SW = np.roll(np.roll(CS, 1, 0), 1, 1)

    fs_NE = np.minimum(fs, np.roll(np.roll(fs, -1, 0), -1, 1))
    fs_NW = np.minimum(fs, np.roll(np.roll(fs, -1, 0), 1, 1))
    fs_SE = np.minimum(fs, np.roll(np.roll(fs, 1, 0), -1, 1))
    fs_SW = np.minimum(fs, np.roll(np.roll(fs, 1, 0), 1, 1))
    fL_NE, fL_NW, fL_SE, fL_SW = 1.0 - fs_NE, 1.0 - fs_NW, 1.0 - fs_SE, 1.0 - fs_SW

    # 对角“等效供给”，对角路径长度为 Ld = hypot(dx, dy)
    Ld = float(np.hypot(dx, dy))
    den_d = np.maximum((1.0 - k0) * CLs * Ld, 1e-12)

    N_NE = DS * (CSs - CS_NE) * fs_NE + DL * (CLs - CL_NE) * fL_NE
    N_NW = DS * (CSs - CS_NW) * fs_NW + DL * (CLs - CL_NW) * fL_NW
    N_SE = DS * (CSs - CS_SE) * fs_SE + DL * (CLs - CL_SE) * fL_SE
    N_SW = DS * (CSs - CS_SW) * fs_SW + DL * (CLs - CL_SW) * fL_SW

    Vd = (
        w_NE * (N_NE / den_d)
        + w_NW * (N_NW / den_d)
        + w_SE * (N_SE / den_d)
        + w_SW * (N_SW / den_d)
    )

    # 原 4 向上风贡献
    Vax = wx_E * (N_E / den_x) + wx_W * (N_W / den_x)
    Vay = wy_N * (N_N / den_y) + wy_S * (N_S / den_y)

    # 合成 + 归一化
    Vn_raw = Vax + Vay + Vd
    wsum8 = np.abs(nxu) + np.abs(nyu) + w_NE + w_NW + w_SE + w_SW
    Vn_iso = Vn_raw / np.maximum(wsum8, 1e-12)

    Vn = np.zeros_like(fs)
    Vn[band] = Vn_iso[band]
    if forbid_remelt:
        Vn[band] = np.maximum(Vn[band], 0.0)

    if out_vn is not None:
        out_vn[band] = Vn[band]
    if out_vx is not None:
        out_vx[band] = Vax[band]  # 输出已上风的轴向贡献，便于诊断
    if out_vy is not None:
        out_vy[band] = Vay[band]

    return Vn, Vax, Vay


# 8 向迎风、度量感知归一化
def compute_velocity3(
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
    # 单位法向
    n2 = nx * nx + ny * ny
    invn = 1.0 / np.sqrt(np.maximum(n2, 1e-18))
    nxu = nx * invn
    nyu = ny * invn

    CLs, CSs = eq
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)
    roll = np.roll

    # —— 面开口（min-gate）
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))

    # 邻居中心
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # 四面供给
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * (1.0 - fs_W)
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * (1.0 - fs_E)
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * (1.0 - fs_S)
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * (1.0 - fs_N)

    # Stefan 分母（各向）+ 保护
    def _safe(den: np.ndarray) -> np.ndarray:
        out = den.copy()
        amp = float(np.nanmax(np.abs(out[band])))
        eps = max(1e-12, amp * 1e-12 + 1e-18)
        sgn = np.where(out >= 0.0, 1.0, -1.0)
        out = np.where(np.abs(out) < eps, sgn * eps, out)
        return out

    den_x = _safe((1.0 - k0) * CLs * dx)
    den_y = _safe((1.0 - k0) * CLs * dy)

    # 上风权重（4 向）
    wx_E = np.maximum(nxu, 0.0)
    wx_W = np.maximum(-nxu, 0.0)
    wy_N = np.maximum(nyu, 0.0)
    wy_S = np.maximum(-nyu, 0.0)

    # —— 对角上风权重（4 对角）
    rt2 = np.sqrt(2.0)
    w_NE = np.maximum((nxu + nyu) / rt2, 0.0)
    w_NW = np.maximum((-nxu + nyu) / rt2, 0.0)
    w_SE = np.maximum((nxu - nyu) / rt2, 0.0)
    w_SW = np.maximum((-nxu - nyu) / rt2, 0.0)

    # 对角邻居与面开口（对角两点 min 判连通）
    CL_NE = roll(roll(CL, -1, 0), -1, 1)
    CS_NE = roll(roll(CS, -1, 0), -1, 1)
    CL_NW = roll(roll(CL, -1, 0), 1, 1)
    CS_NW = roll(roll(CS, -1, 0), 1, 1)
    CL_SE = roll(roll(CL, 1, 0), -1, 1)
    CS_SE = roll(roll(CS, 1, 0), -1, 1)
    CL_SW = roll(roll(CL, 1, 0), 1, 1)
    CS_SW = roll(roll(CS, 1, 0), 1, 1)

    fs_NE = np.minimum(fs, roll(roll(fs, -1, 0), -1, 1))
    fs_NW = np.minimum(fs, roll(roll(fs, -1, 0), 1, 1))
    fs_SE = np.minimum(fs, roll(roll(fs, 1, 0), -1, 1))
    fs_SW = np.minimum(fs, roll(roll(fs, 1, 0), 1, 1))
    fL_NE, fL_NW, fL_SE, fL_SW = 1.0 - fs_NE, 1.0 - fs_NW, 1.0 - fs_SE, 1.0 - fs_SW

    # 对角供给与分母
    Ld = float(np.hypot(dx, dy))
    den_d = np.maximum((1.0 - k0) * CLs * Ld, 1e-12)
    N_NE = DS * (CSs - CS_NE) * fs_NE + DL * (CLs - CL_NE) * fL_NE
    N_NW = DS * (CSs - CS_NW) * fs_NW + DL * (CLs - CL_NW) * fL_NW
    N_SE = DS * (CSs - CS_SE) * fs_SE + DL * (CLs - CL_SE) * fL_SE
    N_SW = DS * (CSs - CS_SW) * fs_SW + DL * (CLs - CL_SW) * fL_SW

    # 通道“速度” v_k = N_k / ((1-k0) C*_L * L_k)
    v_E, v_W = (N_E / den_x), (N_W / den_x)
    v_N, v_S = (N_N / den_y), (N_S / den_y)
    v_NE, v_NW, v_SE, v_SW = (
        (N_NE / den_d),
        (N_NW / den_d),
        (N_SE / den_d),
        (N_SW / den_d),
    )

    # —— 度量感知的凸归一化合成（关键改动）
    L0 = dx  # 参考长度；dx=dy 时取 dx 最自然
    sx, sy, sd = (L0 / dx), (L0 / dy), (L0 / Ld)  # 轴向=1；对角≈1/√2

    num = (
        sx * (wx_E * v_E + wx_W * v_W)
        + sy * (wy_N * v_N + wy_S * v_S)
        + sd * (w_NE * v_NE + w_NW * v_NW + w_SE * v_SE + w_SW * v_SW)
    )

    den = sx * (wx_E + wx_W) + sy * (wy_N + wy_S) + sd * (w_NE + w_NW + w_SE + w_SW)

    Vn_iso = num / np.maximum(den, 1e-12)

    # 带内取值 & 可选禁止重熔
    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vn_iso[band]
    if forbid_remelt:
        Vn[band] = np.maximum(Vn[band], 0.0)

    # 轴向上风贡献（便于诊断，可视化）
    Vax = wx_E * v_E + wx_W * v_W
    Vay = wy_N * v_N + wy_S * v_S
    Vax = np.where(band, Vax, 0.0)
    Vay = np.where(band, Vay, 0.0)

    if out_vn is not None:
        out_vn[...] = Vn
    if out_vx is not None:
        out_vx[...] = Vax
    if out_vy is not None:
        out_vy[...] = Vay
    return Vn, Vax, Vay


# 8 向迎风、度量感知归一化、分母保护、禁止重熔等都不动；只改对角开口的定义
def compute_velocity4(
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
    eta_corner = float(cfg.get("diag_corner_eta", 0.8))  # 角点惩罚系数 0.5~0.8

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
    # 单位法向
    n2 = nx * nx + ny * ny
    invn = 1.0 / np.sqrt(np.maximum(n2, 1e-18))
    nxu, nyu = nx * invn, ny * invn

    CLs, CSs = eq
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)
    roll = np.roll

    # —— 轴向面开口（min-gate）
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))
    fL_W, fL_E = 1.0 - fs_W, 1.0 - fs_E
    fL_S, fL_N = 1.0 - fs_S, 1.0 - fs_N

    # 邻居中心
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # 轴向供给
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * fL_W
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * fL_E
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * fL_S
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * fL_N

    # Stefan 分母 + 保护
    def _safe(den: np.ndarray) -> np.ndarray:
        out = den.copy()
        amp = float(np.nanmax(np.abs(out[band])))
        eps = max(1e-12, amp * 1e-12 + 1e-18)
        sgn = np.where(out >= 0.0, 1.0, -1.0)
        out = np.where(np.abs(out) < eps, sgn * eps, out)
        return out

    den_x = _safe((1.0 - k0) * CLs * dx)
    den_y = _safe((1.0 - k0) * CLs * dy)

    # 上风权重（4 向）
    wx_E = np.maximum(nxu, 0.0)
    wx_W = np.maximum(-nxu, 0.0)
    wy_N = np.maximum(nyu, 0.0)
    wy_S = np.maximum(-nyu, 0.0)

    # —— 对角上风权重
    rt2 = np.sqrt(2.0)
    w_NE = np.maximum((nxu + nyu) / rt2, 0.0)
    w_NW = np.maximum((-nxu + nyu) / rt2, 0.0)
    w_SE = np.maximum((nxu - nyu) / rt2, 0.0)
    w_SW = np.maximum((-nxu - nyu) / rt2, 0.0)

    # —— 对角供给：用“轴向面开口的组合”替代“角点 min”
    #  NE 需要同时穿过 N 和 E 两条面 → 用它们的联合开口（min 或几何均值），再乘角点惩罚
    def _corner_gate(fs_face_a, fs_face_b, mode="geom"):
        if mode == "geom":
            base = np.sqrt(fs_face_a * fs_face_b)
        else:  # "min"
            base = np.minimum(fs_face_a, fs_face_b)
        return eta_corner * base

    fs_NE = _corner_gate(fs_N, fs_E)
    fL_NE = _corner_gate(fL_N, fL_E)
    fs_NW = _corner_gate(fs_N, fs_W)
    fL_NW = _corner_gate(fL_N, fL_W)
    fs_SE = _corner_gate(fs_S, fs_E)
    fL_SE = _corner_gate(fL_S, fL_E)
    fs_SW = _corner_gate(fs_S, fs_W)
    fL_SW = _corner_gate(fL_S, fL_W)

    # 对角邻居中心（用来取 (C*-C_neighbor)）
    CL_NE = roll(roll(CL, -1, 0), -1, 1)
    CS_NE = roll(roll(CS, -1, 0), -1, 1)
    CL_NW = roll(roll(CL, -1, 0), 1, 1)
    CS_NW = roll(roll(CS, -1, 0), 1, 1)
    CL_SE = roll(roll(CL, 1, 0), -1, 1)
    CS_SE = roll(roll(CS, 1, 0), -1, 1)
    CL_SW = roll(roll(CL, 1, 0), 1, 1)
    CS_SW = roll(roll(CS, 1, 0), 1, 1)

    # 对角分母/长度
    Ld = float(np.hypot(dx, dy))
    den_d = np.maximum((1.0 - k0) * CLs * Ld, 1e-12)

    # 对角供给
    N_NE = DS * (CSs - CS_NE) * fs_NE + DL * (CLs - CL_NE) * fL_NE
    N_NW = DS * (CSs - CS_NW) * fs_NW + DL * (CLs - CL_NW) * fL_NW
    N_SE = DS * (CSs - CS_SE) * fs_SE + DL * (CLs - CL_SE) * fL_SE
    N_SW = DS * (CSs - CS_SW) * fs_SW + DL * (CLs - CL_SW) * fL_SW

    # 通道速度
    v_E, v_W = (N_E / den_x), (N_W / den_x)
    v_N, v_S = (N_N / den_y), (N_S / den_y)
    v_NE, v_NW, v_SE, v_SW = (
        (N_NE / den_d),
        (N_NW / den_d),
        (N_SE / den_d),
        (N_SW / den_d),
    )

    # —— 度量感知的凸归一化（轴向长度 Lx, Ly；对角长度 Ld）
    L0 = dx
    sx, sy, sd = (L0 / dx), (L0 / dy), (L0 / Ld)

    num = (
        sx * (wx_E * v_E + wx_W * v_W)
        + sy * (wy_N * v_N + wy_S * v_S)
        + sd * (w_NE * v_NE + w_NW * v_NW + w_SE * v_SE + w_SW * v_SW)
    )

    den = sx * (wx_E + wx_W) + sy * (wy_N + wy_S) + sd * (w_NE + w_NW + w_SE + w_SW)

    Vn_iso = num / np.maximum(den, 1e-12)

    # 带内 & 禁止重熔
    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vn_iso[band]
    if forbid_remelt:
        Vn[band] = np.maximum(Vn[band], 0.0)

    # 轴向上风贡献（诊断）
    Vax = wx_E * v_E + wx_W * v_W
    Vay = wy_N * v_N + wy_S * v_S
    Vax = np.where(band, Vax, 0.0)
    Vay = np.where(band, Vay, 0.0)

    if out_vn is not None:
        out_vn[...] = Vn
    if out_vx is not None:
        out_vx[...] = Vax
    if out_vy is not None:
        out_vy[...] = Vay
    return Vn, Vax, Vay


# 8 向上风 + 度量归一化 两面串联导通
def compute_velocity5(
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
    eta_corner = float(cfg.get("diag_corner_eta", 0.8))  # 角点惩罚(0.7~1.0)

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
    # 单位法向
    n2 = nx * nx + ny * ny
    invn = 1.0 / np.sqrt(np.maximum(n2, 1e-18))
    nxu = nx * invn
    nyu = ny * invn

    CLs, CSs = eq
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)
    roll = np.roll

    # —— 轴向面开口（min-gate）
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))
    fL_W, fL_E = 1.0 - fs_W, 1.0 - fs_E
    fL_S, fL_N = 1.0 - fs_S, 1.0 - fs_N

    # 邻居中心
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # 轴向供给
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * fL_W
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * fL_E
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * fL_S
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * fL_N

    # Stefan 分母 + 保护
    def _safe(den: np.ndarray) -> np.ndarray:
        out = den.copy()
        amp = float(np.nanmax(np.abs(out[band])))
        eps = max(1e-12, amp * 1e-12 + 1e-18)
        sgn = np.where(out >= 0.0, 1.0, -1.0)
        out = np.where(np.abs(out) < eps, sgn * eps, out)
        return out

    den_x = _safe((1.0 - k0) * CLs * dx)
    den_y = _safe((1.0 - k0) * CLs * dy)

    # 上风权重（4 向）
    wx_E = np.maximum(nxu, 0.0)
    wx_W = np.maximum(-nxu, 0.0)
    wy_N = np.maximum(nyu, 0.0)
    wy_S = np.maximum(-nyu, 0.0)

    # —— 对角上风权重
    rt2 = np.sqrt(2.0)
    w_NE = np.maximum((nxu + nyu) / rt2, 0.0)
    w_NW = np.maximum((-nxu + nyu) / rt2, 0.0)
    w_SE = np.maximum((nxu - nyu) / rt2, 0.0)
    w_SW = np.maximum((-nxu - nyu) / rt2, 0.0)

    # —— 对角“串联导通”开口（关键改动）
    # g_series(a,b) = a*b / (a+b+eps)；a=b=1 时自动=0.5
    def g_series(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return (a * b) / np.maximum(a + b + eps, eps)

    fs_NE = eta_corner * g_series(fs_N, fs_E)
    fL_NE = eta_corner * g_series(fL_N, fL_E)
    fs_NW = eta_corner * g_series(fs_N, fs_W)
    fL_NW = eta_corner * g_series(fL_N, fL_W)
    fs_SE = eta_corner * g_series(fs_S, fs_E)
    fL_SE = eta_corner * g_series(fL_S, fL_E)
    fs_SW = eta_corner * g_series(fs_S, fs_W)
    fL_SW = eta_corner * g_series(fL_S, fL_W)

    # 对角邻居中心（用于 (C*-C_neighbor)）
    CL_NE = roll(roll(CL, -1, 0), -1, 1)
    CS_NE = roll(roll(CS, -1, 0), -1, 1)
    CL_NW = roll(roll(CL, -1, 0), 1, 1)
    CS_NW = roll(roll(CS, -1, 0), 1, 1)
    CL_SE = roll(roll(CL, 1, 0), -1, 1)
    CS_SE = roll(roll(CS, 1, 0), -1, 1)
    CL_SW = roll(roll(CL, 1, 0), 1, 1)
    CS_SW = roll(roll(CS, 1, 0), 1, 1)

    # 对角供给与分母
    Ld = float(np.hypot(dx, dy))
    den_d = np.maximum((1.0 - k0) * CLs * Ld, 1e-12)
    N_NE = DS * (CSs - CS_NE) * fs_NE + DL * (CLs - CL_NE) * fL_NE
    N_NW = DS * (CSs - CS_NW) * fs_NW + DL * (CLs - CL_NW) * fL_NW
    N_SE = DS * (CSs - CS_SE) * fs_SE + DL * (CLs - CL_SE) * fL_SE
    N_SW = DS * (CSs - CS_SW) * fs_SW + DL * (CLs - CL_SW) * fL_SW

    # 通道速度
    v_E, v_W = (N_E / den_x), (N_W / den_x)
    v_N, v_S = (N_N / den_y), (N_S / den_y)
    v_NE, v_NW, v_SE, v_SW = (
        (N_NE / den_d),
        (N_NW / den_d),
        (N_SE / den_d),
        (N_SW / den_d),
    )

    # —— 度量感知的凸归一化（轴向 Lx,Ly；对角 Ld）
    L0 = dx
    sx, sy, sd = (L0 / dx), (L0 / dy), (L0 / Ld)

    num = (
        sx * (wx_E * v_E + wx_W * v_W)
        + sy * (wy_N * v_N + wy_S * v_S)
        + sd * (w_NE * v_NE + w_NW * v_NW + w_SE * v_SE + w_SW * v_SW)
    )

    den = sx * (wx_E + wx_W) + sy * (wy_N + wy_S) + sd * (w_NE + w_NW + w_SE + w_SW)

    Vn_iso = num / np.maximum(den, 1e-12)

    # 带内 & 禁止重熔
    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vn_iso[band]
    if forbid_remelt:
        Vn[band] = np.maximum(Vn[band], 0.0)

    # 轴向上风贡献（诊断）
    Vax = wx_E * v_E + wx_W * v_W
    Vay = wy_N * v_N + wy_S * v_S
    Vax = np.where(band, Vax, 0.0)
    Vay = np.where(band, Vay, 0.0)

    if out_vn is not None:
        out_vn[...] = Vn
    if out_vx is not None:
        out_vx[...] = Vax
    if out_vy is not None:
        out_vy[...] = Vay
    return Vn, Vax, Vay


# 四 向上风 + 度量归一化 + 45° 减速
def compute_velocity6(
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

    # —— 仅近 45° 的轻微减速参数（你可以在 cfg 里覆盖）
    diag_tau = float(cfg.get("diag_tau", 0.20))  # 45° 带宽（0.15~0.25）
    diag_slow = float(cfg.get("diag_slow", 0.2))  # 45° 减速强度（0~0.3 建议）
    diag_pow = float(cfg.get("diag_pow", 2.0))  # 形状指数（≥1，越大越集中 45°）

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
    # 单位法向（用它做上风权重 & 角度惩罚）
    n2 = nx * nx + ny * ny
    invn = 1.0 / np.sqrt(np.maximum(n2, 1e-18))
    nxu, nyu = nx * invn, ny * invn

    CLs, CSs = eq
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)
    roll = np.roll

    # —— 面开口（min-gate）
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))
    fL_W, fL_E = 1.0 - fs_W, 1.0 - fs_E
    fL_S, fL_N = 1.0 - fs_S, 1.0 - fs_N

    # 邻居中心
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # —— 四面供给
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * fL_W
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * fL_E
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * fL_S
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * fL_N

    # —— Stefan 分母 + 保护
    def _safe(den: np.ndarray) -> np.ndarray:
        out = den.copy()
        amp = float(np.nanmax(np.abs(out[band])))
        eps = max(1e-12, amp * 1e-12 + 1e-18)
        sgn = np.where(out >= 0.0, 1.0, -1.0)
        out = np.where(np.abs(out) < eps, sgn * eps, out)
        return out

    den_x = _safe((1.0 - k0) * CLs * dx)
    den_y = _safe((1.0 - k0) * CLs * dy)

    # —— 四向上风权重
    wx_E = np.maximum(nxu, 0.0)
    wx_W = np.maximum(-nxu, 0.0)
    wy_N = np.maximum(nyu, 0.0)
    wy_S = np.maximum(-nyu, 0.0)

    # —— 轴向贡献（上风）
    vE = N_E / den_x
    vW = N_W / den_x
    vN = N_N / den_y
    vS = N_S / den_y
    Vx = wx_E * vE + wx_W * vW
    Vy = wy_N * vN + wy_S * vS

    # —— 合成（去 L1 偏置的凸归一化）
    Vn_raw = Vx + Vy
    wsum = np.abs(nxu) + np.abs(nyu)  # ∈[1, √2]
    Vn_iso = Vn_raw / np.maximum(wsum, 1e-12)  # 去除 45° 的 L1 放大

    # —— 仅在接近 45° 时轻微减速（分级惩罚，不影响 0°）
    d = np.abs(np.abs(nxu) - np.abs(nyu))  # 距离 45° 的偏离
    wdiag = np.clip(1.0 - d / np.maximum(diag_tau, 1e-6), 0.0, 1.0) ** diag_pow
    Vn_tuned = Vn_iso * (1.0 - diag_slow * wdiag)

    # —— 带内 & 禁止重熔
    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vn_tuned[band]
    if forbid_remelt:
        Vn[band] = np.maximum(Vn[band], 0.0)

    # 输出诊断（轴向上风分量）
    Vax = np.where(band, Vx, 0.0)
    Vay = np.where(band, Vy, 0.0)
    if out_vn is not None:
        out_vn[...] = Vn
    if out_vx is not None:
        out_vx[...] = Vax
    if out_vy is not None:
        out_vy[...] = Vay
    return Vn, Vax, Vay


# 8 向只参与归一化，不参与通量
def compute_velocity9(
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

    # —— 8向只用于归一化的强度（调小→更接近4向；调大→更“圆”）
    # 典型 0.6~1.0；dx==dy 时 sd≈1/√2，这里再乘一个 gain 控制权重
    norm_diag_gain = float(cfg.get("norm_diag_gain", 0.8))

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

    # —— 单位化法向（只用于权重）
    n2 = nx * nx + ny * ny
    invn = 1.0 / np.sqrt(np.maximum(n2, 1e-18))
    nxu = nx * invn
    nyu = ny * invn

    CLs, CSs = eq
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)
    roll = np.roll

    # —— 面闸门（min gate）
    fs_W = np.minimum(fs, roll(fs, 1, 1))
    fs_E = np.minimum(fs, roll(fs, -1, 1))
    fs_S = np.minimum(fs, roll(fs, 1, 0))
    fs_N = np.minimum(fs, roll(fs, -1, 0))

    # 邻居中心
    CL_W, CL_E = roll(CL, 1, 1), roll(CL, -1, 1)
    CL_S, CL_N = roll(CL, 1, 0), roll(CL, -1, 0)
    CS_W, CS_E = roll(CS, 1, 1), roll(CS, -1, 1)
    CS_S, CS_N = roll(CS, 1, 0), roll(CS, -1, 0)

    # —— 四向供给 N_face
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * (1.0 - fs_W)
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * (1.0 - fs_E)
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * (1.0 - fs_S)
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * (1.0 - fs_N)

    # —— Stefan 分母 + 稳健保护
    def _safe(den: np.ndarray) -> np.ndarray:
        out = den.copy()
        amp = float(np.nanmax(np.abs(out[band])))
        eps = max(1e-12, amp * 1e-12 + 1e-18)
        sgn = np.where(out >= 0.0, 1.0, -1.0)
        out = np.where(np.abs(out) < eps, sgn * eps, out)
        return out

    den_x = _safe((1.0 - k0) * CLs * dx)
    den_y = _safe((1.0 - k0) * CLs * dy)

    # —— 四向上风权重
    wx_E = np.maximum(nxu, 0.0)
    wx_W = np.maximum(-nxu, 0.0)
    wy_N = np.maximum(nyu, 0.0)
    wy_S = np.maximum(-nyu, 0.0)

    # —— 轴向贡献（只用四向做通量）
    Vx = wx_E * (N_E / den_x) + wx_W * (N_W / den_x)
    Vy = wy_N * (N_N / den_y) + wy_S * (N_S / den_y)
    Vraw = Vx + Vy  # 未归一化

    # —— 8向仅用于“分母归一化”的权重（不参与通量）
    rt2 = np.sqrt(2.0)
    w_NE = np.maximum((nxu + nyu) / rt2, 0.0)
    w_NW = np.maximum((-nxu + nyu) / rt2, 0.0)
    w_SE = np.maximum((nxu - nyu) / rt2, 0.0)
    w_SW = np.maximum((-nxu - nyu) / rt2, 0.0)

    # 度量修正（对角长度 Ld>轴向），仅作用在“分母权重”上
    Ld = float(np.hypot(dx, dy))
    L0 = dx
    sx, sy, sd = (L0 / dx), (L0 / dy), (L0 / Ld)  # 轴向=1；对角≈1/√2（若 dx=dy）

    # 分母：四向 +（diag_gain×度量修正×对角权重）
    wsum8 = (
        sx * (np.abs(nxu))
        + sy * (np.abs(nyu))
        + norm_diag_gain * sd * (w_NE + w_NW + w_SE + w_SW)
    )
    Vn_iso = Vraw / np.maximum(wsum8, 1e-12)

    # —— 带内 + 可选禁止重熔
    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = Vn_iso[band]
    if forbid_remelt:
        Vn[band] = np.maximum(Vn[band], 0.0)

    # 输出（诊断）
    if out_vn is not None:
        out_vn[band] = Vn[band]
    if out_vx is not None:
        out_vx[band] = Vx[band]
    if out_vy is not None:
        out_vy[band] = Vy[band]
    return Vn, Vx, Vy


# 8 向仅用于平滑法向/权重，通量仍用 4 向
def compute_velocity8(
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

    # —— 轻平滑参数（只用于 normal/权重）
    smooth_alpha = float(cfg.get("smooth_alpha", 1))  # 0~1，推荐 0.3~0.6
    smooth_iters = int(cfg.get("smooth_iters", 2))  # 盒式迭代次数：1 或 2
    smooth_band_iters = int(cfg.get("smooth_band_iters", 1))  # 膨胀圈数：1

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

    # ——(A) 仅在界面带附近对 normal 做 3×3 盒式轻平滑，然后线性混合；带外不动
    def _box3(a: np.ndarray) -> np.ndarray:
        r = np.roll
        return (
            a
            + r(a, 1, 0)
            + r(a, -1, 0)
            + r(a, 0, 1)
            + r(a, 0, -1)
            + r(r(a, 1, 0), 1, 1)
            + r(r(a, 1, 0), -1, 1)
            + r(r(a, -1, 0), 1, 1)
            + r(r(a, -1, 0), -1, 1)
        ) / 9.0

    def _dilate(b: np.ndarray, iters: int = 1) -> np.ndarray:
        r = np.roll
        out = b.copy()
        for _ in range(max(0, iters)):
            out = (
                out
                | r(out, 1, 0)
                | r(out, -1, 0)
                | r(out, 0, 1)
                | r(out, 0, -1)
                | r(r(out, 1, 0), 1, 1)
                | r(r(out, 1, 0), -1, 1)
                | r(r(out, -1, 0), 1, 1)
                | r(r(out, -1, 0), -1, 1)
            )
        return out

    if smooth_alpha > 0.0 and smooth_iters > 0:
        nband = _dilate(band, iters=smooth_band_iters)
        nx_s, ny_s = nx.copy(), ny.copy()
        for _ in range(smooth_iters):
            nx_s = _box3(nx_s)
            ny_s = _box3(ny_s)
        nx = np.where(nband, (1.0 - smooth_alpha) * nx + smooth_alpha * nx_s, nx)
        ny = np.where(nband, (1.0 - smooth_alpha) * ny + smooth_alpha * ny_s, ny)

    # ——(B) 单位化法向（只用于上风权重）
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

    # 四面等效通量项
    N_W = DS * (CSs - CS_W) * fs_W + DL * (CLs - CL_W) * (1.0 - fs_W)
    N_E = DS * (CSs - CS_E) * fs_E + DL * (CLs - CL_E) * (1.0 - fs_E)
    N_S = DS * (CSs - CS_S) * fs_S + DL * (CLs - CL_S) * (1.0 - fs_S)
    N_N = DS * (CSs - CS_N) * fs_N + DL * (CLs - CL_N) * (1.0 - fs_N)

    # Stefan 分母 + 稳健保护（仅依赖轴向尺度；对角不参与）
    def _safe(den: np.ndarray) -> np.ndarray:
        out = den.copy()
        amp = float(np.nanmax(np.abs(out[band])))
        eps = max(1e-12, amp * 1e-12 + 1e-18)
        sgn = np.where(out >= 0.0, 1.0, -1.0)
        out = np.where(np.abs(out) < eps, sgn * eps, out)
        return out

    den_x = _safe((1.0 - k0) * CLs * dx)
    den_y = _safe((1.0 - k0) * CLs * dy)

    # 四向上风权重（只用 nxu/nyu）
    wx_E = np.maximum(nxu, 0.0)  # nx>0 用东侧
    wx_W = np.maximum(-nxu, 0.0)  # nx<0 用西侧
    wy_N = np.maximum(nyu, 0.0)  # ny>0 用北侧
    wy_S = np.maximum(-nyu, 0.0)  # ny<0 用南侧

    # 轴向贡献（仍为四向；不引入对角通量）
    Vx = wx_E * (N_E / den_x) + wx_W * (N_W / den_x)
    Vy = wy_N * (N_N / den_y) + wy_S * (N_S / den_y)

    # 不做凸归一化：保持你原来“未归一化”的合成
    Vn = np.zeros_like(fs, dtype=float)
    Vn[band] = (Vx + Vy)[band]

    # 禁止重熔
    if forbid_remelt:
        Vn[band] = np.maximum(Vn[band], 0.0)

    # 输出
    if out_vn is not None:
        out_vn[band] = Vn[band]
    if out_vx is not None:
        out_vx[band] = Vx[band]
    if out_vy is not None:
        out_vy[band] = Vy[band]
    return Vn, Vx, Vy
