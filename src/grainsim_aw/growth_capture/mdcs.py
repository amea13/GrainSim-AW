from __future__ import annotations
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


def step(
    grid, fields, cfg_mdcs: Dict[str, Any], cfg_orient: Dict[str, Any], dt: float, masks
):
    """
    最小推进：在界面元胞上令 Δfs = (gf_over_ln) * Vn * dt，并做 4 邻域捕捉：
      - fs 跨过阈值的界面胞，会给其 4 邻的液相元胞一个很小的 seed_fs（并继承 grain_id/theta）。
    这不是正式 MDCS，只是可视化“开始长”的第一步。
    """
    ys, xs = grid.core
    fs = grid.fs
    gid = grid.grain_id
    theta = grid.theta

    Vn = fields.Vn
    mask_int = masks.get("mask_int", np.zeros_like(fs, bool))
    mask_liq = masks.get("mask_liq", np.zeros_like(fs, bool))

    # 参数
    gf_over_ln = float(cfg_mdcs.get("gf_over_ln", 1.0))  # 近似 GF / L_n
    capture_threshold = float(cfg_mdcs.get("fs_capture_threshold", 0.6))
    seed_fs = float(cfg_mdcs.get("capture_seed_fs", 1e-3))
    tau_liq = 1e-12

    # —— 1) 在界面上增加 fs —— #
    Δfs = gf_over_ln * Vn * dt
    Δfs = np.where(mask_int, Δfs, 0.0)

    before = fs[ys, xs].copy()
    fs[ys, xs] = np.clip(fs[ys, xs] + Δfs[ys, xs], 0.0, 1.0)
    sum_dfs = float(np.sum(fs[ys, xs] - before))

    # —— 2) 触发捕捉：跨过阈值的格点作为“源” —— #
    crossed = (before < capture_threshold) & (fs[ys, xs] >= capture_threshold)

    captures = 0
    # 上
    mask_up = (
        crossed[1:, :] & (mask_liq[ys, xs][:-1, :]) & (fs[ys, xs][:-1, :] < tau_liq)
    )
    fs[ys, xs][:-1, :][mask_up] = np.maximum(fs[ys, xs][:-1, :][mask_up], seed_fs)
    gid[ys, xs][:-1, :][mask_up] = gid[ys, xs][1:, :][mask_up]
    theta[ys, xs][:-1, :][mask_up] = theta[ys, xs][1:, :][mask_up]
    captures += int(np.count_nonzero(mask_up))
    # 下
    mask_dn = (
        crossed[:-1, :] & (mask_liq[ys, xs][1:, :]) & (fs[ys, xs][1:, :] < tau_liq)
    )
    fs[ys, xs][1:, :][mask_dn] = np.maximum(fs[ys, xs][1:, :][mask_dn], seed_fs)
    gid[ys, xs][1:, :][mask_dn] = gid[ys, xs][:-1, :][mask_dn]
    theta[ys, xs][1:, :][mask_dn] = theta[ys, xs][:-1, :][mask_dn]
    captures += int(np.count_nonzero(mask_dn))
    # 左
    mask_lt = (
        crossed[:, 1:] & (mask_liq[ys, xs][:, :-1]) & (fs[ys, xs][:, :-1] < tau_liq)
    )
    fs[ys, xs][:, :-1][mask_lt] = np.maximum(fs[ys, xs][:, :-1][mask_lt], seed_fs)
    gid[ys, xs][:, :-1][mask_lt] = gid[ys, xs][:, 1:][mask_lt]
    theta[ys, xs][:, :-1][mask_lt] = theta[ys, xs][:, 1:][mask_lt]
    captures += int(np.count_nonzero(mask_lt))
    # 右
    mask_rt = (
        crossed[:, :-1] & (mask_liq[ys, xs][:, 1:]) & (fs[ys, xs][:, 1:] < tau_liq)
    )
    fs[ys, xs][:, 1:][mask_rt] = np.maximum(fs[ys, xs][:, 1:][mask_rt], seed_fs)
    gid[ys, xs][:, 1:][mask_rt] = gid[ys, xs][:, :-1][mask_rt]
    theta[ys, xs][:, 1:][mask_rt] = theta[ys, xs][:, :-1][mask_rt]
    captures += int(np.count_nonzero(mask_rt))

    logger.info("MDCS(min): sum(Δfs)=%.3e, captures=%d", sum_dfs, captures)
