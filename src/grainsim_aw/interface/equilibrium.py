from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def anisotropy_factor(
    nx: np.ndarray, ny: np.ndarray, theta: np.ndarray, eps: float, m: int = 4
) -> np.ndarray:
    """
    if (ny >= 0) angn = acos(-nx)
    else         angn = 2*pi - acos(-nx)
    ani = 1 - 15 * eps * cos(m * (angn - theta))
    """
    angn = np.empty_like(nx, dtype=np.float64)
    mask = ny >= 0.0

    angn[mask] = np.arccos(-nx[mask])
    angn[~mask] = 2.0 * np.pi - np.arccos(-nx[~mask])

    ani = 1.0 - 15.0 * float(eps) * np.cos(float(m) * (angn - theta))
    return ani


def compute_equilibrium(
    grid,
    fields,
    masks: Dict[str, np.ndarray],
    cfg: Dict,
    domain_cfg: Dict,
    normal: Tuple[np.ndarray, np.ndarray],
    kappa: np.ndarray,
    out_cls: np.ndarray | None = None,
    out_css: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:

    fs = grid.fs
    T = grid.T
    theta = grid.theta

    intf: np.ndarray = masks["intf"]

    # 物性/模型参数
    TL_eq = float(cfg.get("TL_eq", 1809.15))
    C0 = float(domain_cfg.get("C0", 0.0))
    mL = float(cfg.get("mL", -7800.0))
    Gamma = float(cfg.get("Gamma", 1.9e-7))
    k0 = float(cfg.get("k0", 0.34))
    eps = float(cfg.get("eps_anis", 0.04))

    # 法向
    nx, ny = normal

    # 各向异性因子
    ani = anisotropy_factor(nx, ny, theta, eps)

    # 输出缓冲
    if out_cls is None:
        out_cls = np.zeros_like(fs, dtype=np.float64)
    if out_css is None:
        out_css = np.zeros_like(fs, dtype=np.float64)

    # 仅界面元胞赋值（不做任何保护/截断）
    Cl8 = C0 + (T[intf] - TL_eq + Gamma * kappa[intf] * ani[intf]) / mL
    out_cls[intf] = Cl8
    out_css[intf] = k0 * Cl8

    return out_cls, out_css
