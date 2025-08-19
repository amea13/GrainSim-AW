# -*- coding: utf-8 -*-
"""
capture_rules.py — MDCS 落子/捕获规则（可运行版）
------------------------------------------------
流程：
  1) 对每个界面父元胞的 4 个顶点，判断是否越过父元胞盒；若越界，则映射到目标子元胞索引
  2) 只允许“一跳”（Chebyshev 距离 ≤ 1）且目标必须是液相（mask_liq）
  3) 同一子元胞的多方竞争，用“深入程度 margin”仲裁；平局用父 delta_fs 再用 grain_id 破
  4) 胜者落子：继承 grain_id/theta；将顶点绝对坐标设为新元胞的偏心中心；初始化 fs 和 L_dia

依赖：
  - geometry.vertices 返回的 Verts(px,py,cx,cy)（绝对坐标）
  - Grid 中已存在 ecc_x/ecc_y （偏心中心相对几何中心的偏移）
  - masks 至少含 'mask_liq','mask_int'
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np


# --------- 内部工具与数据结构 --------- #


@dataclass
class _Candidate:
    parent_i: int
    parent_j: int
    child_i: int
    child_j: int
    vidx: int  # 哪个顶点触发 (0..3)
    xv: float  # 顶点绝对坐标
    yv: float
    gid: int  # 父 grain_id
    theta: float  # 父取向
    margin: float  # 深入程度评分（越大越优）
    df_parent: float  # 父元胞本步 Δfs（用于平局破）


def _core_center_indices(grid) -> Tuple[float, float]:
    """core 几何中心在全域索引中的位置 (i0,j0)（小数）"""
    g = int(grid.nghost)
    return (g + grid.ny / 2.0, g + grid.nx / 2.0)


def _cell_center_abs(
    i: int, j: int, dx: float, dy: float, i0: float, j0: float
) -> Tuple[float, float]:
    """(i,j) 单元几何中心的绝对坐标（以 core 几何中心为原点）"""
    x = ((j - j0) + 0.5) * dx
    y = ((i - i0) + 0.5) * dy
    return x, y


def _map_abs_point_to_index(
    x: float, y: float, dx: float, dy: float, i0: float, j0: float
) -> tuple[int, int]:
    # 反算到网格索引：边界位于 ((j-j0))*dx 与 ((j-j0)+1)*dx 之间
    j = int(np.floor(x / dx + j0))
    i = int(np.floor(y / dy + i0))
    return i, j


def _in_core(i: int, j: int, g: int, Ny: int, Nx: int) -> bool:
    """是否在 core 范围内（不含 ghost）"""
    return (g <= i < Ny - g) and (g <= j < Nx - g)


def _chebyshev_dist(i1: int, j1: int, i2: int, j2: int) -> int:
    """切比雪夫距离"""
    return max(abs(i1 - i2), abs(j1 - j2))


# --------- 核心入口 --------- #


def apply(
    grid,
    verts,  # geometry.Verts(px, py, cx, cy) 绝对坐标
    nx: np.ndarray,  # 几何法向（目前未用到，可留作扩展）
    ny: np.ndarray,
    delta_fs: np.ndarray,  # 本步父元胞 Δfs（用于平局破）
    masks: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> None:
    """
    执行 MDCS 捕获：从父界面胞的顶点出发感染相邻液元，写入新元胞状态
    （原地更新 grid.fs/grid.grain_id/grid.theta/grid.ecc_x/ecc_y/grid.L_dia）
    """
    mask_int = masks.get("mask_int", None)
    mask_liq = masks.get("mask_liq", None)
    if mask_int is None or mask_liq is None:
        raise KeyError("masks 需包含 'mask_int' 与 'mask_liq'")

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

    # 顶点与中心（绝对坐标）
    px = verts.px  # (Ny,Nx,4)
    py = verts.py

    # 预先准备父元胞的几何中心盒（用于“是否越界”判断）
    # 也可以运行时计算，这里为清晰直接用函数
    # 注意：父元胞盒以“几何中心”为基准，而顶点是“偏心中心 L_dia 外延”的结果
    candidates_by_child: Dict[Tuple[int, int], List[_Candidate]] = {}

    # 遍历所有界面元胞
    parents = np.argwhere(mask_int)
    for i, j in parents:
        # 父几何中心与盒
        xP, yP = _cell_center_abs(i, j, dx, dy, i0, j0)
        xminP, xmaxP = xP - dx * 0.5, xP + dx * 0.5
        yminP, ymaxP = yP - dy * 0.5, yP + dy * 0.5

        # 当前父元胞信息
        gid_p = int(gid[i, j])
        theta_p = float(theta[i, j])
        df_p = float(delta_fs[i, j])

        # 遍历四个顶点
        for k in range(4):
            xv = float(px[i, j, k])
            yv = float(py[i, j, k])

            # 顶点可能为 NaN（非界面带时 vertices 置 NaN），跳过
            if not np.isfinite(xv) or not np.isfinite(yv):
                continue

            # 1) 是否越出父元胞盒：若仍在盒内则不捕获
            inside_parent = (xminP <= xv <= xmaxP) and (yminP <= yv <= ymaxP)
            if inside_parent:
                continue

            # 2) 绝对坐标映射回目标索引
            ci, cj = _map_abs_point_to_index(xv, yv, dx, dy, i0, j0)

            # ghost/域外过滤
            if not _in_core(ci, cj, g, Ny, Nx):
                continue

            # 目标必须是液相
            if not bool(mask_liq[ci, cj]):
                continue

            # 只允许“一跳”
            if _chebyshev_dist(i, j, ci, cj) > 1:
                continue

            # 3) 计算深入程度 margin（相对 child 盒）
            xC, yC = _cell_center_abs(ci, cj, dx, dy, i0, j0)
            xminC, xmaxC = xC - dx * 0.5, xC + dx * 0.5
            yminC, ymaxC = yC - dy * 0.5, yC + dy * 0.5
            margin = min(xv - xminC, xmaxC - xv, yv - yminC, ymaxC - yv)
            # 若因数值误差 margin 为负，说明越过两格或贴边误差，直接跳过
            if margin <= 0.0:
                continue

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
            key = (ci, cj)
            lst = candidates_by_child.get(key)
            if lst is None:
                candidates_by_child[key] = [cand]
            else:
                lst.append(cand)

    # ---- 冲突仲裁并执行落子 ---- #
    if not candidates_by_child:
        return

    eps = 1e-12  # 给 L_dia^max 的保护
    for (ci, cj), cand_list in candidates_by_child.items():
        if not cand_list:
            continue

        # 按 (margin, df_parent, -gid) 三重排序，取最大者
        cand_list.sort(key=lambda c: (c.margin, c.df_parent, -c.gid), reverse=True)
        w = cand_list[0]

        # 目标仍需是液相（可能被前一个胜者已更新为界面了；这里保守再判一次）
        if not bool(mask_liq[w.child_i, w.child_j]):
            continue

        # ---- 继承 grain_id / theta ----
        gid[w.child_i, w.child_j] = w.gid
        theta[w.child_i, w.child_j] = w.theta

        # ---- 偏心中心 = 父顶点绝对坐标 ----
        xC, yC = _cell_center_abs(w.child_i, w.child_j, dx, dy, i0, j0)
        ecc_x[w.child_i, w.child_j] = w.xv - xC
        ecc_y[w.child_i, w.child_j] = w.yv - yC

        # ---- 固相分数初始化 ----
        fs0 = float(cfg.get("capture_seed_fs", 0.005))
        if fs0 > fs[w.child_i, w.child_j]:
            fs[w.child_i, w.child_j] = fs0
            # 注意：masks 是旧的；真正 mask_liq→mask_int 的转换应由上层在本步结束后统一重算

        # ---- 半对角线初始化（与 fs0 对应的几何上限比例）----
        th = theta[w.child_i, w.child_j]
        s_abs = abs(np.sin(th))
        c_abs = abs(np.cos(th))
        denom = max(max(s_abs, c_abs), eps)
        Lmax = dx / denom
        # 取更大的（若已有累积则不回退）
        L_dia[w.child_i, w.child_j] = max(
            float(L_dia[w.child_i, w.child_j]), fs0 * Lmax
        )

    # 函数无返回；状态均已原地更新
