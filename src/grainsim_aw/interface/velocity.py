from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from ..core.material import Dl_from_T, Ds_from_T


def _viewer(a: np.ndarray, pad: int = 1):
    """非环绕平移视图：越界补 0。"""
    Ny, Nx = a.shape
    ap = np.pad(a, pad_width=pad, mode="constant", constant_values=0.0)
    base = pad

    def V(di: int, dj: int) -> np.ndarray:
        return ap[base + di : base + di + Ny, base + dj : base + dj + Nx]

    return V


def compute_velocity(
    grid,
    fields,
    cfg: Dict,
    masks: Dict[str, np.ndarray],
    *,
    normal: Tuple[np.ndarray, np.ndarray],
    eq: Tuple[np.ndarray, np.ndarray],  # (CL_star, CS_star) == (Cl8, Cs8)
    out_vn: Optional[np.ndarray] = None,
    out_vx: Optional[np.ndarray] = None,
    out_vy: Optional[np.ndarray] = None,
):
    nx_full, ny_full = normal
    CL_star, CS_star = eq

    fs = grid.fs.astype(np.float64, copy=False)
    fl = 1.0 - fs
    CL = grid.CL.astype(np.float64, copy=False)
    CS = grid.CS.astype(np.float64, copy=False)
    T = grid.T.astype(np.float64, copy=False)

    # 物性：按中心温度
    DL = Dl_from_T(T).astype(np.float64, copy=False)
    DS = Ds_from_T(T).astype(np.float64, copy=False)

    k0 = float(cfg.get("k0", cfg.get("k", 0.34)))
    dx = float(getattr(grid, "dx", cfg.get("dx", 1e-6)))

    m = masks["intf"]  # 仅在界面带计算

    # 输出
    if out_vx is None:
        out_vx = np.zeros_like(fs, dtype=np.float64)
    else:
        out_vx.fill(0.0)
    if out_vy is None:
        out_vy = np.zeros_like(fs, dtype=np.float64)
    else:
        out_vy.fill(0.0)
    if out_vn is None:
        out_vn = np.zeros_like(fs, dtype=np.float64)
    else:
        out_vn.fill(0.0)

    # 邻域视图（一次 pad）
    V_CL = _viewer(CL, pad=1)
    V_CS = _viewer(CS, pad=1)
    V_fs = _viewer(fs, pad=1)
    V_fl = _viewer(fl, pad=1)

    # 邻居（x 向）
    CL_jm, CL_jp = V_CL(0, -1), V_CL(0, +1)
    CS_jm, CS_jp = V_CS(0, -1), V_CS(0, +1)
    fs_jm, fs_jp = V_fs(0, -1), V_fs(0, +1)
    fl_jm, fl_jp = V_fl(0, -1), V_fl(0, +1)

    # 邻居（y 向）
    CL_im, CL_ip = V_CL(-1, 0), V_CL(+1, 0)
    CS_im, CS_ip = V_CS(-1, 0), V_CS(+1, 0)
    fs_im, fs_ip = V_fs(-1, 0), V_fs(+1, 0)
    fl_im, fl_ip = V_fl(-1, 0), V_fl(+1, 0)

    # 系数
    inv = 1.0 / (dx * (1.0 - k0))
    coefL = DL * inv
    coefS = (k0 * DS) * inv

    # 只在界面带做除法，避免无关位置的告警
    termLx = np.zeros_like(fs)
    termSx = np.zeros_like(fs)
    termLy = np.zeros_like(fs)
    termSy = np.zeros_like(fs)

    termLx[m] = (1.0 - CL_jm[m] / CL_star[m]) * fl_jm[m] + (
        1.0 - CL_jp[m] / CL_star[m]
    ) * fl_jp[m]
    termSx[m] = (1.0 - CS_jm[m] / CS_star[m]) * fs_jm[m] + (
        1.0 - CS_jp[m] / CS_star[m]
    ) * fs_jp[m]

    termLy[m] = (1.0 - CL_im[m] / CL_star[m]) * fl_im[m] + (
        1.0 - CL_ip[m] / CL_star[m]
    ) * fl_ip[m]
    termSy[m] = (1.0 - CS_im[m] / CS_star[m]) * fs_im[m] + (
        1.0 - CS_ip[m] / CS_star[m]
    ) * fs_ip[m]

    vx_full = coefL * termLx + coefS * termSx
    vy_full = coefL * termLy + coefS * termSy

    out_vx[m] = vx_full[m]
    out_vy[m] = vy_full[m]

    # 合成 vn：与原 if/elif/else 等价的象限分支
    vxm = vx_full[m]
    vym = vy_full[m]
    ax = np.abs(nx_full[m])
    ay = np.abs(ny_full[m])

    v = np.zeros_like(vxm)
    m_pp = (vxm >= 0.0) & (vym >= 0.0)
    m_pn = (vxm >= 0.0) & (vym < 0.0)
    m_np = (vxm < 0.0) & (vym >= 0.0)

    v[m_pp] = vxm[m_pp] * ax[m_pp] + vym[m_pp] * ay[m_pp]
    v[m_pn] = vxm[m_pn] * ax[m_pn]
    v[m_np] = vym[m_np] * ay[m_np]

    out_vn[m] = v
    return out_vn, out_vx, out_vy
