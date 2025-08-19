# -*- coding: utf-8 -*-
"""
geometry.py —— 偏心正方形几何：L_n、GF、L_dia 更新、顶点坐标
-------------------------------------------------------------
仅提供函数签名与详细注释，便于后续逐步填充实现。

约定与背景
- 网格正方形：dx = dy（若将来dx≠dy，需要按注释调整公式）
- “法向”来自 kernels.centroid_normal（圆核质心法，单位向量）
- 本模块只做几何量：不参与 Vn 的物理计算
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

__all__ = ["L_n", "shape_factor_GF", "update_Ldia", "vertices", "Verts"]


@dataclass(frozen=True)
class Verts:
    """
    顶点集合（绝对坐标版）
    px, py : (Ny, Nx, 4) 四个顶点的绝对坐标（单位: m），顺序为
             p0: +u 方向（u = [cosθ, sinθ]）
             p1: +v 方向（v = [-sinθ, cosθ]）
             p2: -u 方向
             p3: -v 方向
    cx, cy : (Ny, Nx)     本元胞“偏心中心”的绝对坐标（= 几何中心 + 偏移 ecc）
    """

    px: np.ndarray
    py: np.ndarray
    cx: np.ndarray
    cy: np.ndarray


def L_n(nx: np.ndarray, ny: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    由几何法向计算“界面穿越长度” L_n（法线穿过单元中心在该单元内的等效长度）

    输入
    ----
    nx, ny : ndarray
        圆核质心法得到的单位法向分量（与 fs 同形状，含 ghost）。只在界面带会用到。
    dx, dy : float
        网格步长（当前项目 dx=dy；若将来 dx!=dy 需按注释修正公式）

    输出
    ----
    Ln : ndarray
        与 fs 同形状（含 ghost）的正数数组；未使用位置的值无关紧要。
    """
    # 第一象限折叠后的分量幅值，并加极小保护
    eps = 1e-12
    c = np.maximum(np.abs(nx), eps)  # 相当于 |cos φ|
    s = np.maximum(np.abs(ny), eps)  # 相当于 |sin φ|

    # 分段计算（dx=dy 时与文献式等价；这里保留 dx/dy 以便将来扩展）
    Ln_c_ge_s = dx * (1.0 / c + s - (s * s) / c)  # 当 c >= s
    Ln_s_gt_c = dy * (1.0 / s + c - (c * c) / s)  # 当 c <  s

    Ln = np.where(c >= s, Ln_c_ge_s, Ln_s_gt_c)
    return Ln


def shape_factor_GF(
    fs: np.ndarray, nx: np.ndarray, ny: np.ndarray, masks: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    依据 CA 的一/二级邻胞固相分布，计算形状因子 GF_n（降低栅格各向异性）
    规则：
      1) 若四个轴向邻胞（N/S/E/W）中存在 ≥1 个固相：GF = 1
      2) 否则，若四个对角邻胞（NE/NW/SE/SW）中“恰有 1 个”为固相：
           若该对角为 NE 或 SW：GF = 1 / ( √2 * max(|nx - ny|/√2, eps) )
           若该对角为 NW 或 SE：GF = 1 / ( √2 * max(|nx + ny|/√2, eps) )
         （等价地分别用 e_d = (±1,∓1)/√2 与 e_d = (±1,±1)/√2 计算 |n·e_d|）
      3) 否则：GF = 1
    仅在界面带写入（mask_int），其它位置保持 1。
    """
    eps = 1e-12
    Ny, Nx = fs.shape
    GF = np.ones((Ny, Nx), dtype=float)

    mask_sol = masks.get("mask_sol", None)
    if mask_sol is None:
        raise KeyError("masks 中缺少 'mask_sol'")
    mask_int = masks.get("mask_int", np.ones_like(fs, dtype=bool))

    # —— 轴向邻胞（布尔图），使用 roll 将邻胞状态对齐到当前单元 —— #
    solN = np.roll(mask_sol, +1, axis=0)  # 北：来自 (i-1, j)
    solS = np.roll(mask_sol, -1, axis=0)  # 南：来自 (i+1, j)
    solW = np.roll(mask_sol, +1, axis=1)  # 西：来自 (i, j-1)
    solE = np.roll(mask_sol, -1, axis=1)  # 东：来自 (i, j+1)

    has_primary = solN | solS | solW | solE  # 任一轴向邻胞为固

    # —— 对角邻胞（布尔图） —— #
    solNE = np.roll(np.roll(mask_sol, +1, axis=0), -1, axis=1)  # (i-1, j+1)
    solNW = np.roll(np.roll(mask_sol, +1, axis=0), +1, axis=1)  # (i-1, j-1)
    solSE = np.roll(np.roll(mask_sol, -1, axis=0), -1, axis=1)  # (i+1, j+1)
    solSW = np.roll(np.roll(mask_sol, -1, axis=0), +1, axis=1)  # (i+1, j-1)

    diag_count = (
        solNE.astype(np.int8)
        + solNW.astype(np.int8)
        + solSE.astype(np.int8)
        + solSW.astype(np.int8)
    )

    # 仅当“无轴向固相”且“恰有一个对角固相”时启用修正
    mask_diag_single = (~has_primary) & (diag_count == 1)

    # —— 预计算 |n·e_d| 的两种形式 —— #
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    cos_plus = np.abs(nx + ny) * inv_sqrt2  # 对应 NW / SE
    cos_minus = np.abs(nx - ny) * inv_sqrt2  # 对应 NE / SW

    # —— NE 或 SW 单独为固：用 |nx - ny| —— #
    mask_NE_SW = mask_diag_single & (solNE ^ solSW) & ~(solNW | solSE)
    # （上式等价于：mask_diag_single & (solNE | solSW)；因 diag_count==1 已保证互斥）
    denom_NE_SW = np.maximum(cos_minus, eps)
    GF_NE_SW = inv_sqrt2 / denom_NE_SW
    GF[mask_NE_SW & mask_int] = GF_NE_SW[mask_NE_SW & mask_int]

    # —— NW 或 SE 单独为固：用 |nx + ny| —— #
    mask_NW_SE = mask_diag_single & (solNW ^ solSE) & ~(solNE | solSW)
    denom_NW_SE = np.maximum(cos_plus, eps)
    GF_NW_SE = inv_sqrt2 / denom_NW_SE
    GF[mask_NW_SE & mask_int] = GF_NW_SE[mask_NW_SE & mask_int]

    # 其余位置（含 has_primary 或 diag_count!=1）：保持 GF=1
    # 同时确保非界面带不被修改（已默认 1）
    GF[~mask_int] = 1.0

    return GF


def update_Ldia(grid, delta_fs: np.ndarray, theta: np.ndarray) -> None:
    """
    由 Δf_s 更新偏心正方形的“半对角线长度” L_dia（原地写 grid.L_dia）

    ΔL_dia = Δf_s * L_dia^max(θ)
    L_dia^max(θ) = dx / max(|sinθ|, |cosθ|)
    """
    dx = float(grid.dx)
    eps = 1e-12  # 极小保护，避免 |sinθ|=|cosθ|=0 时除零

    # 分母：max(|sinθ|, |cosθ|) 并做极小保护
    s = np.abs(np.sin(theta))
    c = np.abs(np.cos(theta))
    denom = np.maximum(np.maximum(s, c), eps)

    # 几何上限 L_dia^max(θ)
    Ldia_max = dx / denom

    # 增量并原地更新
    grid.L_dia += delta_fs * Ldia_max

    # 可选但推荐：不超过几何上限（当 fs 累积到 1 时 L_dia → L_dia^max）
    np.minimum(grid.L_dia, Ldia_max, out=grid.L_dia)


def vertices(grid, mask_int: np.ndarray) -> Verts:
    """
    由更新后的 L_dia 与 θ 计算偏心正方形（正八面体）的四个顶点与中心绝对坐标。

    约定与单位
    --------
    - 绝对坐标原点与温度模块一致：把 core 几何中心当作 (0,0)
    - 单位：米（m）
    - 需要 grid.ecc_x / grid.ecc_y 作为“偏心中心相对几何中心”的偏移场
      * 成核时应设为 0
      * 捕获时由 capture_rules 用“父顶点绝对坐标 - 子几何中心绝对坐标”赋值

    参数
    ----
    grid     : Grid（读取 L_dia, theta, dx, dy, nghost, nx, ny, ecc_x, ecc_y）
    mask_int : ndarray(bool)，界面带（只在这些位置会被捕获模块使用）

    返回
    ----
    Verts(px, py, cx, cy) : 顶点与中心的绝对坐标
    """
    fs_shape = grid.fs.shape
    Ny, Nx = fs_shape

    dx = float(grid.dx)
    dy = float(grid.dy)
    g = int(grid.nghost)

    # --- 1) 计算各单元“几何中心”的绝对坐标（以 core 几何中心为原点） ---
    # 索引网格
    ii, jj = np.indices((Ny, Nx), dtype=float)  # ii: 行(i,y), jj: 列(j,x)

    # core 几何中心在全域索引中的位置（小数）
    i0 = g + grid.ny / 2.0
    j0 = g + grid.nx / 2.0

    # 单元几何中心：((idx - 参考中心) + 0.5) * 步长
    x_cell = ((jj - j0) + 0.5) * dx
    y_cell = ((ii - i0) + 0.5) * dy

    # --- 2) 偏心中心的绝对坐标 = 几何中心 + 偏移 ---
    cx = x_cell + grid.ecc_x
    cy = y_cell + grid.ecc_y

    # --- 3) 四个顶点方向单位向量 ---
    theta = grid.theta
    L = grid.L_dia

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # u = (cosθ, sinθ), v = (-sinθ, cosθ)
    ux, uy = cos_t, sin_t
    vx, vy = -sin_t, cos_t

    # --- 4) 四个顶点的绝对坐标（广播计算） ---
    # 准备输出容器
    px = np.empty((Ny, Nx, 4), dtype=float)
    py = np.empty((Ny, Nx, 4), dtype=float)

    # p0 = C + L*u
    px[..., 0] = cx + L * ux
    py[..., 0] = cy + L * uy

    # p1 = C + L*v
    px[..., 1] = cx + L * vx
    py[..., 1] = cy + L * vy

    # p2 = C - L*u
    px[..., 2] = cx - L * ux
    py[..., 2] = cy - L * uy

    # p3 = C - L*v
    px[..., 3] = cx - L * vx
    py[..., 3] = cy - L * vy

    # --- 5) 非界面带位置（可选）：置 NaN，避免误用 ---
    # 捕获模块一般只读界面带；设 NaN 便于调试可视化。
    not_int = ~mask_int
    px[not_int, :] = np.nan
    py[not_int, :] = np.nan

    return Verts(px=px, py=py, cx=cx, cy=cy)
