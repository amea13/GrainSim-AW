from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

__all__ = ["anisotropy_factor", "compute_equilibrium"]


def anisotropy_factor(
    nx: np.ndarray, ny: np.ndarray, theta: np.ndarray, eps: float, m: int = 4
) -> np.ndarray:
    """
    if (ny >= 0) angn = acos(-nx)
    else         angn = 2*pi - acos(-nx)
    ani = 1 - 15 * eps * cos(m * (angn - theta))
    """
    angn = np.empty_like(nx, dtype=np.float64)  # 与输入形状一致
    mask = ny >= 0.0

    # 严格保持与原实现在界面点上的分支与顺序
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
    """
    仅在界面带上计算并写回；对每个元素，计算公式与顺序与原版一致。
    """
    fs = grid.fs  # 仅用于创建同形状缓冲
    T = grid.T
    theta = grid.theta
    intf = masks["intf"]

    # 物性/模型参数（不更改含义）
    TL_eq = float(cfg.get("TL_eq", 1809.15))
    C0 = float(domain_cfg.get("C0", 0.0))
    mL = float(cfg.get("mL", -7800.0))
    Gamma = float(cfg.get("Gamma", 1.9e-7))
    k0 = float(cfg.get("k0", 0.34))
    eps = float(cfg.get("eps_anis", 0.04))
    # m 的默认 4 与原逻辑一致；若你在 cfg 里放了别名，也可改成 cfg.get("m", 4)
    mfold = 4

    # 法向（仅切出界面带视图，避免全域计算）
    nx_full, ny_full = normal
    nx = nx_full[intf]
    ny = ny_full[intf]
    th = theta[intf]

    # 输出缓冲
    if out_cls is None:
        out_cls = np.zeros_like(fs, dtype=np.float64)
    if out_css is None:
        out_css = np.zeros_like(fs, dtype=np.float64)

    # 各向异性因子（只对界面带计算；对每个点的分支与原版相同）
    ani = anisotropy_factor(nx, ny, th, eps, m=mfold)

    # 平衡浓度（保持同一条表达式、同样的运算次序）
    Cl8 = C0 + (T[intf] - TL_eq + Gamma * kappa[intf] * ani) / mL

    # 回写
    out_cls[intf] = Cl8
    out_css[intf] = k0 * Cl8
    return out_cls, out_css
