"""
溶质传输求解器（无对流）
======================

【控制方程】（单一网格双变量，五点格式）
  令 α = 1 - f_s （液相体积分数），在零法向通量边界（Neumann）下：
    d( α C_L )/dt = ∇·( α D_L ∇C_L ) + S_pair
    d( (1-α) C_S )/dt = ∇·( (1-α) D_S ∇C_S ) - S_pair
  其中成对源项 S_pair = (1 - k) * C_L^n * df_s/dt

【时间离散】
  - 扩散与积累项：后向欧拉（系数取 t^{n+1}，即推进 fs 后）
  - 源项：使用 t^n 的 C_L 与给定的 df_s/dt

【空间离散】
  - 正交网格五点格式；面导通系数使用复合扩散率的调和插值：
      Γ^L = α D_L，Γ^S = (1-α) D_S
  - 鬼点层（ghost）由调用方维护；这里通过 pad(edge) 等效零法向梯度

【接口】
  - solute_advance(grid, cfg, dt, masks, CL_star, fs_dot)  # 原地更新 grid.CL/CS
  - step_solute(...)                                       # 与 TransportProcess 对接的薄包装
  - total_solute_mass(grid)                                # 诊断用总溶质量

【配置示例】
  cfg = {
    "k": 0.34,                 # 分配系数（若不从界面过程传入，可在此做常数近似）
    "eps": 1e-12,              # 活跃区阈值（跳过纯相单元）
    "solver": {"max_iter": 200, "tol": 1e-8},
    "clip": {"min": 0.0},      # 可选：对 CL/CS 做非负截断
  }
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from ..core.material import Dl_from_T, Ds_from_T

__all__ = ["solute_advance", "step_solute"]


# -----------------------
# 工具函数
# -----------------------
def _k_const(cfg: Dict) -> float:
    """常数分配系数 k（若界面过程未提供更精准耦合时使用）"""
    return float(cfg.get("k", 0.34))


def _harmonic(a: np.ndarray, b: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """调和平均：用于面处的复合扩散率插值"""
    return 2.0 * a * b / (a + b + eps)


def _build_conductance(
    Gamma: np.ndarray, dx: float, dy: float, nghost: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造 core 区域上的面导通系数 (Ge, Gw, Gn, Gs)：
      Ge 乘 (φ_E - φ_P)，Gw 乘 (φ_W - φ_P) 等。
    """
    g = int(nghost)
    ys = slice(g, -g)
    xs = slice(g, -g)

    Gp = Gamma[ys, xs]
    Ge_nb = np.roll(Gamma, -1, axis=1)[ys, xs]
    Gw_nb = np.roll(Gamma, 1, axis=1)[ys, xs]
    Gn_nb = np.roll(Gamma, -1, axis=0)[ys, xs]
    Gs_nb = np.roll(Gamma, 1, axis=0)[ys, xs]

    # 面处 Γ 的调和平均
    GeG = _harmonic(Gp, Ge_nb)
    GwG = _harmonic(Gp, Gw_nb)
    GnG = _harmonic(Gp, Gn_nb)
    GsG = _harmonic(Gp, Gs_nb)

    # 正交网格面积/距离度量：G_face = Γ_f * A / d
    Ge = GeG * dy / dx
    Gw = GwG * dy / dx
    Gn = GnG * dx / dy
    Gs = GsG * dx / dy
    return Ge, Gw, Gn, Gs


def _jacobi_sweep(
    aP: np.ndarray,
    Ge: np.ndarray,
    Gw: np.ndarray,
    Gn: np.ndarray,
    Gs: np.ndarray,
    b: np.ndarray,
    phi: np.ndarray,
    nghost: int,
    active: np.ndarray,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    向量化 Jacobi 迭代（仅在 core 上），鬼点通过 pad(edge) 实现零法向梯度。
    aP、Ge...、b、phi、active 的形状均为 core（Ny-2g, Nx-2g）。
    """
    for _ in range(max_iter):
        phip = np.pad(phi, ((1, 1), (1, 1)), mode="edge")  # 等效镜像 ghost
        phi_E = phip[1:-1, 2:]
        phi_W = phip[1:-1, :-2]
        phi_N = phip[2:, 1:-1]
        phi_S = phip[:-2, 1:-1]

        num = b + Ge * phi_E + Gw * phi_W + Gn * phi_N + Gs * phi_S
        phi_new = num / (aP + 1e-300)

        # 非活跃区保持原值（纯相跳过）
        phi_new = np.where(active, phi_new, phi)

        # 收敛判据：活跃区无穷范数
        if np.any(active):
            diff = np.max(np.abs(phi_new[active] - phi[active]))
            phi = phi_new
            if diff < tol:
                break
        else:
            phi = phi_new
            break
    return phi


# -----------------------
# 主入口
# -----------------------
def solute_advance(
    grid,
    cfg: Dict,
    dt: float,
    masks: Dict[str, np.ndarray],  # 目前未显式使用，保留扩展
    CL_star: np.ndarray,  # 界面液相平衡浓度 C_L^*（来自 equilibrium）
    fs_dot: np.ndarray,  # 本步固相率时间导数（来自界面推进）
) -> None:
    """
    单步推进溶质场（原地更新 grid.CL / grid.CS）

    输入
    ----
    grid : 具有属性 fs, CL, CS, T, dx, dy, nghost 的对象
    cfg  : 见模块顶部“配置示例”
    dt   : 时间步长
    masks: 相位掩码字典（当前保留；将来可能用于选择性更新）
    CL_star : C_L^*（与 grid 形状一致，含 ghost）
    fs_dot  : df_s/dt（与 grid 形状一致，含 ghost）
    """
    g = int(grid.nghost)
    ys = slice(g, -g)
    xs = slice(g, -g)

    # 物性/几何
    k = _k_const(cfg)
    T = grid.T
    dx = float(grid.dx)
    dy = float(grid.dy)
    Vc = dx * dy

    # t^{n+1} 的相分数（已由界面推进得到）
    fs_np1 = grid.fs
    alpha_np1 = 1.0 - fs_np1  # 液相体积分数

    # 用 fs_dot 近似重建 t^n 的相分数（后向欧拉一致）
    alpha_n = np.clip(alpha_np1 + dt * fs_dot, 0.0, 1.0)
    fs_n = 1.0 - alpha_n

    # t^n 的场
    CL_old = grid.CL.copy()
    CS_old = grid.CS.copy()

    # 扩散率（按中心点温度 T）
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    # 复合扩散率 Γ = 相分数 * D
    GammaL = alpha_np1 * DL
    GammaS = fs_np1 * DS

    # 面导通系数（core 形状）
    GeL, GwL, GnL, GsL = _build_conductance(GammaL, dx, dy, g)
    GeS, GwS, GnS, GsS = _build_conductance(GammaS, dx, dy, g)

    # 成对源（core）
    fs_dot_c = fs_dot[ys, xs]
    CL_old_c = CL_old[ys, xs]
    CL_star_c = CL_star[ys, xs]
    S_pair = (1.0 - k) * CL_star_c * fs_dot_c  # 单位：1/时间

    # 液相线性系统：aP_L * CL^{n+1} = b_L + ∑ G * 邻居
    alpha_np1_c = alpha_np1[ys, xs]
    alpha_n_c = alpha_n[ys, xs]
    aP_L = alpha_np1_c * Vc / dt + (GeL + GwL + GnL + GsL)
    b_L = alpha_n_c * CL_old_c * Vc / dt + S_pair * Vc

    # 固相线性系统
    fs_np1_c = fs_np1[ys, xs]
    fs_n_c = fs_n[ys, xs]
    aP_S = fs_np1_c * Vc / dt + (GeS + GwS + GnS + GsS)
    b_S = fs_n_c * CS_old[ys, xs] * Vc / dt - S_pair * Vc

    # 仅在“活跃区”求解（跳过近似纯相单元）
    eps = float(cfg.get("eps", 1e-12))
    active_L = alpha_np1_c > eps
    active_S = fs_np1_c > eps

    # 迭代参数
    solver = cfg.get("solver", {})
    max_iter = int(solver.get("max_iter", 100))
    tol = float(solver.get("tol", 1e-8))

    # 初值（取 t^n）
    CL_c = CL_old_c.copy()
    CS_c = CS_old[ys, xs].copy()

    # Jacobi 迭代求解
    CL_c = _jacobi_sweep(
        aP_L, GeL, GwL, GnL, GsL, b_L, CL_c, g, active_L, max_iter, tol
    )
    CS_c = _jacobi_sweep(
        aP_S, GeS, GwS, GnS, GsS, b_S, CS_c, g, active_S, max_iter, tol
    )

    # 可选截断（避免负浓度）
    cmin = float(cfg.get("clip", {"min": 0.0}).get("min", 0.0))
    if cmin is not None:
        CL_c = np.maximum(CL_c, cmin)
        CS_c = np.maximum(CS_c, cmin)

    # 回写 core
    grid.CL[ys, xs] = CL_c
    grid.CS[ys, xs] = CS_c


def step_solute(
    *,
    grid,
    cfg: Dict,
    dt: float,
    masks: Dict[str, np.ndarray],
    fs_dot: np.ndarray,
    CL_star: np.ndarray,
):
    """
    与 TransportProcess.step_solute 对接的薄包装。
    仅转调 solute_advance；参数名与顺序与编排器保持一致。
    """
    return solute_advance(grid, cfg, dt, masks, CL_star, fs_dot)
