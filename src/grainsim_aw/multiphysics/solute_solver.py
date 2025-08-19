from __future__ import annotations

"""
Solute transport solver on a fixed Cartesian grid.

Two-equation formulation on a single mesh with phase-weighted coefficients
and a paired source at the interface to enforce local partitioning while
keeping the global solute strictly conserved under zero-flux BCs.

Equations (no advection):
  d( alpha * C_L )/dt = div( alpha * D_L * grad C_L ) + S_pair
  d( (1-alpha) * C_S )/dt = div( (1-alpha) * D_S * grad C_S ) - S_pair
where alpha = 1 - f_s, and S_pair = (1 - k) * C_L^n * df_s/dt.

Time discretization: backward Euler for the accumulation and diffusion terms,
with coefficients evaluated from the known state at t^{n+1} (after fs update),
source uses C_L^n and df_s/dt supplied by the caller.

Spatial discretization: finite volume on a five-point stencil.
Face conductance uses harmonic interpolation of the composite diffusivity
Gamma = alpha*D_L (liquid) and Gamma = (1-alpha)*D_S (solid).
Ghost layers are assumed already filled by the caller (Neumann BC by mirroring).

This module provides a single entry point: solute_advance(grid, cfg, dt, masks, fs_dot)
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple
from ..core import Dl_from_T, Ds_from_T


__all__ = ["solute_advance"]


def _k_const(cfg: Dict) -> float:
    # default constant k
    return float(cfg.get("k", 0.34))


def _harmonic(a: np.ndarray, b: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    return 2.0 * a * b / (a + b + eps)


def _build_conductance(
    Gamma: np.ndarray, dx: float, dy: float, nghost: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return face conductances (Ge, Gw, Gn, Gs) on core cells for a scalar Gamma.
    Ge multiplies (phi_E - phi_P), etc.
    Using harmonic interpolation for face Gamma and orthogonal mesh metrics.
    """
    g = nghost
    ys = slice(g, -g)
    xs = slice(g, -g)

    Gp = Gamma[ys, xs]
    # neighbor Gamma values via safe shifts that land in ghost for boundary-adjacent cores
    Ge_nb = np.roll(Gamma, -1, axis=1)[ys, xs]
    Gw_nb = np.roll(Gamma, 1, axis=1)[ys, xs]
    Gn_nb = np.roll(Gamma, -1, axis=0)[ys, xs]
    Gs_nb = np.roll(Gamma, 1, axis=0)[ys, xs]

    # harmonic face Gamma
    GeG = _harmonic(Gp, Ge_nb)
    GwG = _harmonic(Gp, Gw_nb)
    GnG = _harmonic(Gp, Gn_nb)
    GsG = _harmonic(Gp, Gs_nb)

    # orthogonal face conductance: Gface = Gamma_f * A / d
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
    """Vectorized Jacobi iterations on core cells only.
    Inputs are core-shaped arrays.
    """
    # To fetch neighbor phi values, we work on a padded copy
    g = nghost
    for _ in range(max_iter):
        # neighbor values from the current iterate in the full field with ghosts
        # We build a temp full array to read neighbors by roll without mixing core only
        # Construct a view onto core of phi to write next values
        # Prepare neighbor fields from the surrounding full grid
        # Here phi_full is not needed since the neighbors we read are in core or ghost,
        # but ghost already mirrors core, and we only read by rolling the full array built from phi in-place into a temporary.
        # Build a slim full array by embedding phi into a zero array to leverage roll
        # to avoid extra copies, we instead access neighbors via slices on a temporary with ghosts.
        # Construct a compact window that includes one ghost layer around core
        # but since we already have ghosts in the caller's grid, and phi here is only core,
        # we cannot directly roll. Simplest is to request neighbor values by padding with edge values.
        # Use numpy.pad with edge mode to emulate mirrored ghosts.
        phi_pad = np.pad(phi, ((1, 1), (1, 1)), mode="edge")
        # neighbors in the padded interior aligned to core
        phi_E = phi_pad[1:-1, 2:]
        phi_W = phi_pad[1:-1, :-2]
        phi_N = phi_pad[2:, 1:-1]
        phi_S = phi_pad[:-2, 1:-1]

        num = b + Ge * phi_E + Gw * phi_W + Gn * phi_N + Gs * phi_S
        phi_new = num / (aP + 1e-300)

        # keep inactive cells unchanged
        phi_new = np.where(active, phi_new, phi)

        # convergence check on active cells
        diff = np.max(np.abs(phi_new[active] - phi[active])) if np.any(active) else 0.0
        phi = phi_new
        if diff < tol:
            break
    return phi


def _jacobi_sweep1(aP, Ge, Gw, Gn, Gs, b, phi, nghost, active, max_iter, tol):
    """
    替换为红黑 Gauss–Seidel（等价于棋盘颜色交替更新），
    并用无穷范数残差 ||r||∞ 作为收敛判据：r = b - (A phi)。
    """
    # 为了不改调用签名，这里保持函数名不变
    ny, nx = phi.shape

    def residual_inf(phi_cur):
        # 计算 r = b - (aP*phi - sum G*phi_nb)
        # 邻居值（用 pad 取邻居，等效零法向梯度）
        phip = np.pad(phi_cur, ((1, 1), (1, 1)), mode="edge")
        phi_E = phip[1:-1, 2:]
        phi_W = phip[1:-1, :-2]
        phi_N = phip[2:, 1:-1]
        phi_S = phip[:-2, 1:-1]
        Ap_phi = aP * phi_cur - (Ge * phi_E + Gw * phi_W + Gn * phi_N + Gs * phi_S)
        r = b - Ap_phi
        if np.any(active):
            return np.max(np.abs(r[active]))
        else:
            return 0.0

    # 红黑掩码
    ii, jj = np.indices((ny, nx))
    red = ((ii + jj) & 1) == 0
    black = ~red

    for _ in range(max_iter):
        # 邻居（每半步都基于最新 phi 重取）
        phip = np.pad(phi, ((1, 1), (1, 1)), mode="edge")
        phi_E = phip[1:-1, 2:]
        phi_W = phip[1:-1, :-2]
        phi_N = phip[2:, 1:-1]
        phi_S = phip[:-2, 1:-1]

        # 红色点
        num = b + Ge * phi_E + Gw * phi_W + Gn * phi_N + Gs * phi_S
        phi_red = num / (aP + 1e-300)
        phi = np.where(active & red, phi_red, phi)

        # 重新取邻居（黑点用最新红点）
        phip = np.pad(phi, ((1, 1), (1, 1)), mode="edge")
        phi_E = phip[1:-1, 2:]
        phi_W = phip[1:-1, :-2]
        phi_N = phip[2:, 1:-1]
        phi_S = phip[:-2, 1:-1]

        # 黑色点
        num = b + Ge * phi_E + Gw * phi_W + Gn * phi_N + Gs * phi_S
        phi_black = num / (aP + 1e-300)
        phi = np.where(active & black, phi_black, phi)

        # 残差收敛
        if residual_inf(phi) < tol:
            break

    return phi


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


def solute_advance(
    grid, cfg: Dict, dt: float, masks: Dict[str, np.ndarray], fs_dot: np.ndarray
) -> None:
    """Advance solute fields by one time step.

    Parameters
    ----------
    grid : object with attributes
        fs, CL, CS, T: 2D arrays with ghosts; dx, dy: spacings; nghost: ghost layers
    cfg : dict
        physics.solute sub-config; supports keys: k, solver, eps
    dt : float
        time step
    masks : dict
        not strictly required here, kept for future use
    fs_dot : 2D array
        time derivative of fs on the same grid, including ghosts
    """
    g = grid.nghost
    ys = slice(g, -g)
    xs = slice(g, -g)

    # properties
    k = _k_const(cfg)
    T = grid.T

    # phase fractions at n+1 (after interface update)
    fs_np1 = grid.fs
    alpha_np1 = 1.0 - fs_np1

    # reconstruct alpha^n, fs^n from fs_dot
    alpha_n = np.clip(alpha_np1 + dt * fs_dot, 0.0, 1.0)
    fs_n = 1.0 - alpha_n

    # old fields (at t^n)
    CL_old = grid.CL.copy()
    CS_old = grid.CS.copy()

    # diffusivities
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    # composite diffusivities Gamma = phase_fraction * D
    GammaL = alpha_np1 * DL
    GammaS = fs_np1 * DS

    # build conductances on core
    GeL, GwL, GnL, GsL = _build_conductance(GammaL, grid.dx, grid.dy, g)
    GeS, GwS, GnS, GsS = _build_conductance(GammaS, grid.dx, grid.dy, g)

    # volume per cell
    Vc = grid.dx * grid.dy

    # paired source, core region
    fs_dot_core = fs_dot[ys, xs]
    CL_old_core = CL_old[ys, xs]
    S_pair = (1.0 - k) * CL_old_core * fs_dot_core  # [per time]

    # accumulation coefficients aP and RHS b for liquid
    alpha_np1_c = alpha_np1[ys, xs]
    alpha_n_c = alpha_n[ys, xs]

    aP_L = alpha_np1_c * Vc / dt + (GeL + GwL + GnL + GsL)
    b_L = alpha_n_c * CL_old_core * Vc / dt + S_pair * Vc

    # accumulation for solid
    fs_np1_c = fs_np1[ys, xs]
    fs_n_c = fs_n[ys, xs]

    aP_S = fs_np1_c * Vc / dt + (GeS + GwS + GnS + GsS)
    b_S = fs_n_c * CS_old[ys, xs] * Vc / dt - S_pair * Vc

    # active masks to skip pure-phase cells
    eps = float(cfg.get("eps", 1e-12))
    active_L = alpha_np1_c > eps
    active_S = fs_np1_c > eps

    # solver settings
    solver = cfg.get("solver", {})
    max_iter = int(solver.get("max_iter", 100))
    tol = float(solver.get("tol", 1e-8))

    # initial guesses
    CL_core = CL_old_core.copy()
    CS_core = CS_old[ys, xs].copy()

    # liquid solve
    CL_core = _jacobi_sweep(
        aP_L, GeL, GwL, GnL, GsL, b_L, CL_core, g, active_L, max_iter, tol
    )
    # solid solve
    CS_core = _jacobi_sweep(
        aP_S, GeS, GwS, GnS, GsS, b_S, CS_core, g, active_S, max_iter, tol
    )

    # non-negative clipping if requested
    clip_cfg = cfg.get("clip", {"min": 0.0})
    cmin = float(clip_cfg.get("min", 0.0))
    if cmin is not None:
        CL_core = np.maximum(CL_core, cmin)
        CS_core = np.maximum(CS_core, cmin)

    # write back to grid core
    grid.CL[ys, xs] = CL_core
    grid.CS[ys, xs] = CS_core

    # done; ghost update left to the caller's usual update_ghosts at the next step


# 计算整个网格中溶质的总质量，用于诊断是否守恒
def total_solute_mass(grid) -> float:
    """
    诊断用总溶质量：M = ∑ [ α CL + (1-α) CS ] dx dy，
    其中 α = 1 - fs 为液相体积分数。
    只在 core 统计，避免 ghost 干扰。
    """
    alpha = 1.0 - grid.fs
    cell = grid.dx * grid.dy
    g = grid.nghost
    core = (slice(g, -g), slice(g, -g))
    M = np.sum(alpha[core] * grid.CL[core] + (1.0 - alpha[core]) * grid.CS[core]) * cell
    return float(M)
