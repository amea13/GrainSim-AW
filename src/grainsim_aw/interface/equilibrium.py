from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from .geometry import compute_normal, compute_curvature


def anisotropy_factor(
    nx: np.ndarray, ny: np.ndarray, theta: np.ndarray, eps: float
) -> np.ndarray:
    """f(phi, theta) = 1 - 15*eps*cos(4*(phi - theta))"""
    if eps == 0.0:
        return np.ones_like(nx, dtype=float)
    phi = np.arctan2(ny, nx)
    return 1.0 - 15.0 * float(eps) * np.cos(4.0 * (phi - theta))


def compute_equilibrium(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict,
    normal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    kappa: Optional[np.ndarray] = None,
    out_cls: np.ndarray | None = None,
    out_css: np.ndarray | None = None,
    out_ani: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    依据局部平衡 T = T* 反解 C_L^* 与 C_S^*：
      T* = T_L_eq + (C_L^* - C0) * m_L - Gamma * kappa * f(phi, theta)
      => C_L^* = C0 + [T - T_L_eq + Gamma * kappa * f] / m_L
         C_S^* = k0 * C_L^*
    仅在界面带赋值。
    """
    fs = grid.fs
    T = grid.T
    theta = grid.theta

    intf: np.ndarray = masks["intf"] if "intf" in masks else masks["mask_int"]
    if intf.dtype != bool:
        intf = intf.astype(bool, copy=False)

    # 物性/模型参数（如未提供，给出温和默认）
    TL_eq = float(cfg.get("TL_eq", 1809.15))
    C0 = float(cfg.get("C0", getattr(grid, "C0", 0.0)))
    mL = float(cfg.get("mL", -7800.0))  # 不能为 0
    Gamma = float(cfg.get("Gamma", 1.9e-7))
    k0 = float(cfg.get("k0", 0.34))
    eps = float(cfg.get("eps_anis", 0.0))

    # 法向/曲率：若未传入，则内部计算一次（便于独立使用）
    if normal is None:
        nx_tmp = np.zeros_like(fs, dtype=float)
        ny_tmp = np.zeros_like(fs, dtype=float)
        compute_normal(grid, masks, {}, out_nx=nx_tmp, out_ny=ny_tmp)
        normal = (nx_tmp, ny_tmp)
    nx, ny = normal

    if kappa is None:
        kappa = np.zeros_like(fs, dtype=float)
        compute_curvature(grid, masks, {}, out=kappa)

    # 各向异性因子
    ani = anisotropy_factor(nx, ny, theta, eps)
    if out_ani is not None:
        out_ani[intf] = ani[intf]

    # 反解 C_L^* / C_S^*
    if abs(mL) < 1e-20:
        mL = -1e-20
    num = (T - TL_eq) + Gamma * kappa * ani

    CLS = out_cls if out_cls is not None else np.zeros_like(fs, dtype=float)
    CSS = out_css if out_css is not None else np.zeros_like(fs, dtype=float)

    CLS[intf] = C0 + num[intf] / mL
    CSS[intf] = k0 * CLS[intf]

    return CLS, CSS
