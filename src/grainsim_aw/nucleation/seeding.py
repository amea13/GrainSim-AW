# -*- coding: utf-8 -*-
"""
seeding.py — 可控初始种子（一次性初始化）
支持:
- single_center : 在 core 几何中心放 1 个种子
- random        : 在 core 内随机放 N 个种子
- edge_line     : 在指定边上均匀放 count 个种子

每个种子：fs=1, L_dia=Lmax(theta), ecc=(0,0)，并按取向用“四顶点”一次性感染邻元为界面元。
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

# 依赖 Grid 的工具
from grainsim_aw.core.grid import update_ghosts


# ---------- 基本几何工具（绝对坐标以 core 几何中心为原点） ----------


def _core_center_indices(grid) -> Tuple[float, float]:
    g = int(grid.nghost)
    return (g + grid.ny / 2.0, g + grid.nx / 2.0)  # (i0, j0)


def _cell_center_abs(
    i: int, j: int, dx: float, dy: float, i0: float, j0: float
) -> Tuple[float, float]:
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
    return (g <= i < Ny - g) and (g <= j < Nx - g)


def _Ldia_max(theta: float, dx: float) -> float:
    s = abs(np.sin(theta))
    c = abs(np.cos(theta))
    return dx / max(s, c, 1e-12)


# ---------- 某个种子的“一次性感染”实现（按取向四顶点） ----------


def _infect_ring_by_vertices(
    grid, i0: int, j0: int, theta0: float, eps_over: float, fs_child: float, k0: float
) -> int:
    """
    以 (i0,j0) 为父（已 fs=1），用 L_seed=(0.5+eps_over)*Lmax 的四个顶点感染邻元为界面。
    新界面元初始化：fs=fs_child，L_dia=fs*Lmax，ecc=顶点偏移，grain_id/theta 继承；
    溶质：CS = k0 * CL（不改 CL）。
    返回感染成功的邻元个数。
    """
    fs = grid.fs
    gid_a = grid.grain_id
    th = grid.theta
    ecc_x = grid.ecc_x
    ecc_y = grid.ecc_y
    Ldia = grid.L_dia
    CL = grid.CL
    CS = grid.CS

    dx = float(grid.dx)
    dy = float(grid.dy)
    g = int(grid.nghost)
    Ny, Nx = fs.shape
    i0c, j0c = _core_center_indices(grid)

    # 父几何中心（绝对坐标）
    xC, yC = _cell_center_abs(i0, j0, dx, dy, i0c, j0c)

    # 四顶点（绝对坐标）
    Lmax = _Ldia_max(theta0, dx)
    Lseed = (0.5 + eps_over) * Lmax
    ct, st = np.cos(theta0), np.sin(theta0)
    ux, uy = ct, st
    vx, vy = -st, ct
    verts = [
        (xC + Lseed * ux, yC + Lseed * uy),
        (xC + Lseed * vx, yC + Lseed * vy),
        (xC - Lseed * ux, yC - Lseed * uy),
        (xC - Lseed * vx, yC - Lseed * vy),
    ]

    placed = 0
    tau_liq = 1e-12
    for xv, yv in verts:
        ci, cj = _map_abs_point_to_index(xv, yv, dx, dy, i0c, j0c)
        if not _in_core(ci, cj, g, Ny, Nx):
            continue
        if fs[ci, cj] > tau_liq:  # 仅感染“当前液相”元
            continue

        # 继承晶粒属性
        gid_a[ci, cj] = gid_a[i0, j0]
        th[ci, cj] = th[i0, j0]

        # 偏心中心：设为顶点绝对坐标
        xc, yc = _cell_center_abs(ci, cj, dx, dy, i0c, j0c)
        ecc_x[ci, cj] = xv - xc
        ecc_y[ci, cj] = yv - yc

        # 几何初始化
        fs[ci, cj] = max(fs[ci, cj], fs_child)
        Ldia[ci, cj] = max(Ldia[ci, cj], fs[ci, cj] * _Ldia_max(th[ci, cj], dx))

        # 溶质初始化：界面平衡 CS = k0 * CL（不改 CL）
        CS[ci, cj] = k0 * CL[ci, cj]

        placed += 1

    return placed


# ---------- 主入口：seed_initialize ----------


def seed_initialize(grid, rng: np.random.Generator, cfg: Dict[str, Any]) -> int:
    """
    可控初始种子（一次性初始化）。
    配置示例:
    {
        "mode": "single_center" | "random" | "edge_line",
        "theta_deg": 0,               # 可选：固定角（所有种子共用）；若缺省且 random_theta=True 则随机取向
        "random_theta": false,        # 可选：是否随机取向（均匀 [0, 2π)）
        "N": 10,                      # mode=="random" 时的数量
        "edge": "north",              # mode=="edge_line" 的边: "north"/"south"/"west"/"east"
        "count": 8,                   # mode=="edge_line" 的种子个数
        "eps_over_edge": 0.02,        # 顶点越过比例 ε（用于一次性感染）
        "capture_seed_fs": 0.005      # 新界面元初始 fs
    }
    返回：成功放置的“核心”种子数（非感染邻元数量）。
    """
    mode = str(cfg.get("mode", "single_center")).lower()
    N = int(cfg.get("N", 1))
    edge = str(cfg.get("edge", "north")).lower()
    eps_over = float(cfg.get("eps_over_edge", 0.02))
    fs_child = float(cfg.get("capture_seed_fs", 0.005))
    k0 = float(cfg.get("k0", 0.34))

    # 取向：固定角或随机
    if cfg.get("random_theta", False):
        # 每个种子独立随机取向
        def sample_theta() -> float:
            return float(rng.uniform(0.0, 2.0 * np.pi))

    else:
        # 固定角（默认 0°）
        theta_rad = float(np.deg2rad(cfg.get("theta_deg", 0.0)))

        def sample_theta() -> float:
            return theta_rad

    fs = grid.fs
    gid_a = grid.grain_id
    th = grid.theta
    ecc_x = grid.ecc_x
    ecc_y = grid.ecc_y
    Ldia = grid.L_dia
    CL = grid.CL
    CS = grid.CS

    dx = float(grid.dx)
    dy = float(grid.dy)
    g = int(grid.nghost)
    Ny, Nx = fs.shape
    core_y = range(g, Ny - g)
    core_x = range(g, Nx - g)

    # 生成种子坐标列表
    seeds: List[Tuple[int, int]] = []

    if mode == "single_center":
        i0 = g + grid.ny // 2
        j0 = g + grid.nx // 2
        seeds.append((i0, j0))

    elif mode == "random":
        # 在 core 内均匀抽样 N 个不重复位置
        total = grid.ny * grid.nx
        if N > total:
            N = total
        # 将 core 区展平采样，再还原 (i,j)
        flat_idx = rng.choice(total, size=N, replace=False)
        for k in flat_idx:
            di = int(k // grid.nx)
            dj = int(k % grid.nx)
            seeds.append((g + di, g + dj))

    elif mode == "edge_line":
        # 在指定边上均匀布点（count 个），避开 ghost
        count = max(1, int(cfg.get("count", 1)))
        if edge == "north":
            i = g
            js = np.linspace(g, g + grid.nx - 1, count, dtype=int)
            seeds.extend((i, j) for j in js)
        elif edge == "south":
            i = g + grid.ny - 1
            js = np.linspace(g, g + grid.nx - 1, count, dtype=int)
            seeds.extend((i, j) for j in js)
        elif edge == "west":
            j = g
            is_ = np.linspace(g, g + grid.ny - 1, count, dtype=int)
            seeds.extend((i, j) for i in is_)
        elif edge == "east":
            j = g + grid.nx - 1
            is_ = np.linspace(g, g + grid.ny - 1, count, dtype=int)
            seeds.extend((i, j) for i in is_)
        else:
            raise ValueError(f"edge_line.edge 不支持: {edge}")
    else:
        raise ValueError(f"init.mode 不支持: {mode}")

    # 逐个落子
    placed = 0
    next_gid = int(gid_a.max()) + 1
    for i0, j0 in seeds:
        # 避免覆盖已有非液相（极少见于重复初始化）
        if fs[i0, j0] > 1e-12:
            continue

        theta0 = sample_theta()

        # 1) 核心元：直接固相
        gid_a[i0, j0] = next_gid
        th[i0, j0] = theta0
        fs[i0, j0] = 1.0
        ecc_x[i0, j0] = 0.0
        ecc_y[i0, j0] = 0.0
        Ldia[i0, j0] = _Ldia_max(theta0, dx)

        # —— 溶质初始化（核心）：CS = k0 * CL_old；CL = 0 —— #
        CL_old = CL[i0, j0]
        CS[i0, j0] = k0 * CL_old
        CL[i0, j0] = 0.0

        # 2) 一次性感染：按取向四顶点感染（一级或二级邻胞，取决于 θ）
        _infect_ring_by_vertices(grid, i0, j0, theta0, eps_over, fs_child, k0)

        placed += 1
        next_gid += 1

    # 更新 ghost（顶点感染已修改 fs/几何）
    update_ghosts(grid)
    return placed
