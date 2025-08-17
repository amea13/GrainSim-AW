"""
可控初始种子（一次性初始化）：
- single_center：在核心区中心放 1 个种子（可设取向、fs）
- random:       在核心区随机放 N 个种子（均匀抽样）
- edge_line:    在某条边上按间距放一排种子
与 Nucleation.apply(逐步形核)独立；只在开局调用一次。
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _core_center_indices(grid) -> tuple[int, int]:
    """返回 core 区中心的（绝对）索引，而不是相对索引。"""
    g = grid.nghost
    cy = g + grid.ny // 2
    cx = g + grid.nx // 2
    return cy, cx


def _place_seed(grid, iy: int, ix: int, theta_rad: float, seed_fs: float, gid: int):
    """在 core 内放置一个种子；坐标是绝对索引。"""
    g = grid.nghost
    # core 的合法范围：[g, g+ny) × [g, g+nx)
    assert g <= iy < g + grid.ny and g <= ix < g + grid.nx, "seed 不在 core 内"
    grid.fs[iy, ix] = max(grid.fs[iy, ix], seed_fs)
    grid.grain_id[iy, ix] = gid
    grid.theta[iy, ix] = np.mod(theta_rad, 2.0 * np.pi)


def initialize(grid, rng: np.random.Generator, cfg_init: Dict[str, Any]) -> int:
    """
    根据 cfg_init 放置初始种子；返回放置的数量。
    cfg_init:
      mode: "single_center" | "random" | "edge_line"
      seed_fs: float (默认 1e-3)
      theta_deg: float（单中心/整排统一取向；也可填 theta_rad）
      count: int（random 模式数量）
      edge: "north"|"south"|"west"|"east"（edge_line）
      spacing: int（edge_line 的间隔，>=1）
      overwrite: bool（若 True 允许覆盖已有 grain_id；默认 False）
    """
    mode = str(cfg_init.get("mode", "single_center")).lower()
    seed_fs = float(cfg_init.get("seed_fs", 1e-3))
    overwrite = bool(cfg_init.get("overwrite", False))

    # 取向：优先 theta_rad，其次 theta_deg
    if "theta_rad" in cfg_init:
        theta_rad = float(cfg_init["theta_rad"])
    else:
        theta_deg = float(cfg_init.get("theta_deg", 0.0))
        theta_rad = theta_deg * np.pi / 180.0

    ys, xs = grid.core
    core_h, core_w = ys.stop - ys.start, xs.stop - xs.start

    # 当前最大 grain_id，新的从此基础上递增
    current_max = (
        int(np.max(grid.grain_id[ys, xs])) if np.any(grid.grain_id[ys, xs]) else 0
    )
    next_gid = current_max + 1

    placed = 0
    tau_liq = 1e-12

    if mode == "single_center":
        cy, cx = _core_center_indices(grid)
        if (not overwrite) and (
            grid.grain_id[cy, cx] != 0 or grid.fs[cy, cx] >= tau_liq
        ):
            logger.info("Seeding(single_center): 中心已有占用，跳过。")
            return 0
        _place_seed(grid, cy, cx, theta_rad, seed_fs, next_gid)
        placed = 1

    elif mode == "random":
        count = int(cfg_init.get("count", 10))
        # 可候选：core ∩ 液相且未分配 id
        mask_liq = (grid.fs[ys, xs] < tau_liq) & (grid.grain_id[ys, xs] == 0)
        candidates = np.transpose(np.nonzero(mask_liq))
        if candidates.size == 0:
            logger.info("Seeding(random): 无可用液相候选。")
            return 0
        count = min(count, candidates.shape[0])
        idx = rng.choice(candidates.shape[0], size=count, replace=False)
        for k, (iy_rel, ix_rel) in enumerate(candidates[idx]):
            iy = ys.start + int(iy_rel)
            ix = xs.start + int(ix_rel)
            _place_seed(grid, iy, ix, theta_rad, seed_fs, next_gid + k)
        placed = count

    elif mode == "edge_line":
        edge = str(cfg_init.get("edge", "north")).lower()
        spacing = max(1, int(cfg_init.get("spacing", 4)))
        if edge in ("north", "south"):
            j_indices = np.arange(xs.start, xs.stop, spacing, dtype=int)
            i = ys.start if edge == "north" else ys.stop - 1
            for k, j in enumerate(j_indices):
                if overwrite or (grid.grain_id[i, j] == 0 and grid.fs[i, j] < tau_liq):
                    _place_seed(grid, i, j, theta_rad, seed_fs, next_gid + placed)
                    placed += 1
        else:  # "west" | "east"
            i_indices = np.arange(ys.start, ys.stop, spacing, dtype=int)
            j = xs.start if edge == "west" else xs.stop - 1
            for k, i in enumerate(i_indices):
                if overwrite or (grid.grain_id[i, j] == 0 and grid.fs[i, j] < tau_liq):
                    _place_seed(grid, i, j, theta_rad, seed_fs, next_gid + placed)
                    placed += 1
    else:
        logger.warning("Seeding: 未知 mode=%s，跳过。", mode)
        return 0

    logger.info(
        "Seeding(%s): placed=%d, seed_fs=%.3g, theta=%.3f rad",
        mode,
        placed,
        seed_fs,
        theta_rad,
    )
    return placed
