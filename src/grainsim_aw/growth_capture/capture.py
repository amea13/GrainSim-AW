# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np


# ---------------------
# 顶点几何：ESVC 顶点
# ---------------------
@dataclass
class Verts:
    px: np.ndarray  # (Ny,Nx,4) 顶点 x 的绝对坐标
    py: np.ndarray  # (Ny,Nx,4) 顶点 y 的绝对坐标


def _core_center_indices(grid) -> Tuple[float, float]:
    g = int(grid.nghost)
    return (g + grid.ny / 2.0, g + grid.nx / 2.0)


def _cell_center_abs(
    i: int, j: int, dx: float, dy: float, i0: float, j0: float
) -> Tuple[float, float]:
    x = ((j - j0) + 0.5) * dx
    y = ((i - i0) + 0.5) * dy
    return x, y


def compute_verts(grid, masks: Dict[str, np.ndarray]) -> Verts:
    """
    基于偏心中心 (ecc_x,ecc_y)、取向 theta 与半对角线 L_dia，计算四个顶点绝对坐标。
    仅在界面带上写顶点；界面外填 NaN，便于后续捕捉时快速跳过。
    """
    mask_int = masks.get("intf")
    fs = grid.fs
    theta = grid.theta
    L = grid.L_dia
    ecc_x, ecc_y = grid.ecc_x, grid.ecc_y

    Ny, Nx = fs.shape
    px = np.full((Ny, Nx, 4), np.nan, dtype=fs.dtype)
    py = np.full((Ny, Nx, 4), np.nan, dtype=fs.dtype)

    dx = float(grid.dx)
    dy = float(grid.dy)
    i0, j0 = _core_center_indices(grid)
    I, J = np.indices(fs.shape, dtype=float)
    xC = ((J - j0) + 0.5) * dx + ecc_x
    yC = ((I - i0) + 0.5) * dy + ecc_y

    # 四个顶点方向：theta + pi/4 + k*(pi/2)
    base = theta
    for k in range(4):
        ang = base + k * (np.pi / 2.0)
        dxk = L * np.cos(ang)
        dyk = L * np.sin(ang)
        # 仅界面带写入
        px[..., k][mask_int] = (xC + dxk)[mask_int]
        py[..., k][mask_int] = (yC + dyk)[mask_int]

    return Verts(px=px, py=py)


# ---------------------
# 捕捉主逻辑（MDCS）
# ---------------------
@dataclass
class _Candidate:
    parent_i: int
    parent_j: int
    child_i: int
    child_j: int
    vidx: int
    xv: float
    yv: float
    gid: int
    theta: float
    margin: float
    df_parent: float


def _in_core(i: int, j: int, g: int, Ny: int, Nx: int) -> bool:
    return (g <= i < Ny - g) and (g <= j < Nx - g)


def _chebyshev_dist(i1: int, j1: int, i2: int, j2: int) -> int:
    return max(abs(i1 - i2), abs(j1 - j2))


def _map_abs_point_to_index(
    x: float, y: float, dx: float, dy: float, i0: float, j0: float
) -> Tuple[int, int]:
    j = int(np.floor(x / dx + j0))
    i = int(np.floor(y / dy + i0))
    return i, j


def geometry_and_capture(grid, cfg: Dict[str, Any], masks) -> None:
    """
    计算界面父胞的 ESVC 顶点并执行一次捕捉。状态在 grid 上原地更新。
    - cfg: MDCS 相关参数（例如 {"capture_seed_fs": 0.005}）
    - fields: IfaceFieldsBuf，使用其中的 masks（"intf"/"liq" 等）
    """
    mask_int = masks.get("intf")
    mask_liq = masks.get("liq")
    if mask_int is None or mask_liq is None:
        return  # 极简：按你的要求不做额外防御

    fs = grid.fs
    gid = grid.grain_id
    theta = grid.theta
    ecc_x = grid.ecc_x
    ecc_y = grid.ecc_y
    L_dia = grid.L_dia

    dx = float(grid.dx)
    dy = float(grid.dy)
    g = int(grid.nghost)
    Ny, Nx = fs.shape
    i0, j0 = _core_center_indices(grid)

    # 1) 顶点（绝对坐标）
    verts = compute_verts(grid, masks)
    px = verts.px
    py = verts.py

    # 预先父元胞盒
    def _cell_center_abs_vec(i: int, j: int) -> Tuple[float, float]:
        return _cell_center_abs(i, j, dx, dy, i0, j0)

    # delta_fs：平局破用；当前步尚未推进，取 0 即可
    delta_fs = np.zeros_like(fs)

    candidates_by_child: Dict[Tuple[int, int], List[_Candidate]] = {}
    parents = np.argwhere(mask_int)
    for i, j in parents:
        xP, yP = _cell_center_abs_vec(i, j)
        xminP, xmaxP = xP - dx * 0.5, xP + dx * 0.5
        yminP, ymaxP = yP - dy * 0.5, yP + dy * 0.5

        gid_p = int(gid[i, j])
        theta_p = float(theta[i, j])
        df_p = float(delta_fs[i, j])

        for k in range(4):
            xv = float(px[i, j, k])
            yv = float(py[i, j, k])
            if not np.isfinite(xv) or not np.isfinite(yv):
                continue

            # 越界父盒才考虑捕捉
            if (xminP <= xv <= xmaxP) and (yminP <= yv <= ymaxP):
                continue

            ci, cj = _map_abs_point_to_index(xv, yv, dx, dy, i0, j0)
            if not _in_core(ci, cj, g, Ny, Nx):
                continue
            if not bool(mask_liq[ci, cj]):
                continue
            if _chebyshev_dist(i, j, ci, cj) > 1:
                continue

            xC, yC = _cell_center_abs_vec(ci, cj)
            xminC, xmaxC = xC - dx * 0.5, xC + dx * 0.5
            yminC, ymaxC = yC - dy * 0.5, yC + dy * 0.5
            margin = min(xv - xminC, xmaxC - xv, yv - yminC, ymaxC - yv)
            if margin <= 0.0:
                continue

            key = (ci, cj)
            cand = _Candidate(
                parent_i=i,
                parent_j=j,
                child_i=ci,
                child_j=cj,
                vidx=k,
                xv=xv,
                yv=yv,
                gid=gid_p,
                theta=theta_p,
                margin=margin,
                df_parent=df_p,
            )
            lst = candidates_by_child.get(key)
            if lst is None:
                candidates_by_child[key] = [cand]
            else:
                lst.append(cand)

    if not candidates_by_child:
        return

    eps = 1e-12
    for (ci, cj), lst in candidates_by_child.items():
        if not lst:
            continue
        lst.sort(key=lambda c: (c.margin, c.df_parent, -c.gid), reverse=True)
        w = lst[0]

        if not bool(mask_liq[w.child_i, w.child_j]):
            continue

        # 继承 grain_id / theta
        gid[w.child_i, w.child_j] = w.gid
        theta[w.child_i, w.child_j] = w.theta

        # 偏心中心用胜出顶点
        xC, yC = _cell_center_abs(w.child_i, w.child_j, dx, dy, i0, j0)
        ecc_x[w.child_i, w.child_j] = w.xv - xC
        ecc_y[w.child_i, w.child_j] = w.yv - yC

        # 初始化 fs
        fs0 = float(cfg.get("capture_seed_fs", 0.005))
        if fs0 > fs[w.child_i, w.child_j]:
            fs[w.child_i, w.child_j] = fs0

        # 初始化 L_dia（与 fs0 对应的几何上限比例）
        th = theta[w.child_i, w.child_j]
        s_abs = abs(np.sin(th))
        c_abs = abs(np.cos(th))
        denom = max(max(s_abs, c_abs), eps)
        Lmax = dx / denom
        L_dia[w.child_i, w.child_j] = max(
            float(L_dia[w.child_i, w.child_j]), fs0 * Lmax
        )
