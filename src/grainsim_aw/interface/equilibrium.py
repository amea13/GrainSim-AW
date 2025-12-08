from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np


def anisotropy_factor(
    nx: np.ndarray,
    ny: np.ndarray,
    theta: np.ndarray,
    eps_anis: float,
    *,
    masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    f = np.ones_like(nx, dtype=float)
    if eps_anis == 0.0:
        return f

    mask = None
    if masks is not None and "intf" in masks:
        mask = masks["intf"].astype(bool, copy=False)

    if mask is None:
        mask = np.ones_like(nx, dtype=bool)

    # 只在界面上取法向角 φ
    phi = np.arctan2(ny[mask], nx[mask])  # ∈ (-π, π]
    delta = phi - theta[mask]  # 弧度差
    # 可选的规范化，保证数值稳定，但不改变 cos 的值
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi

    f_intf = 1.0 - 15.0 * float(eps_anis) * np.cos(4.0 * delta)
    f[mask] = f_intf
    return f


def compute_equilibrium(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict,
    domain_cfg: Dict,
    fields,
    normal: Tuple[np.ndarray, np.ndarray],
    kappa: np.ndarray,
    out_cls: np.ndarray | None = None,
    out_css: np.ndarray | None = None,
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

    intf: np.ndarray = masks["intf"]
    if intf.dtype != bool:
        intf = intf.astype(bool, copy=False)

    # 物性/模型参数（如未提供，给出温和默认）
    TL_eq = float(cfg.get("TL_eq", 1809.15))
    C0 = float(domain_cfg.get("C0", 0.0082))  # 初始浓度
    mL = float(cfg.get("mL", -7800.0))  # 不能为 0
    Gamma = float(cfg.get("Gamma", 1.9e-7))
    k0 = float(cfg.get("k0", 0.34))
    eps_anis = float(cfg.get("eps_anis", 0.04))
    # 法向/曲率：若未传入，则内部计算一次（便于独立使用）
    nx, ny = normal

    # 各向异性因子
    ani = anisotropy_factor(nx, ny, theta, eps_anis, masks=masks)

    fields.ani[...] = ani  # 记录各向异性因子，便于诊断

    # 反解 C_L^* / C_S^*
    num = (T - TL_eq) + Gamma * kappa * ani

    CLS = out_cls if out_cls is not None else np.zeros_like(fs, dtype=float)
    CSS = out_css if out_css is not None else np.zeros_like(fs, dtype=float)

    CLS[intf] = C0 + num[intf] / mL
    CSS[intf] = k0 * CLS[intf]

    return CLS, CSS
