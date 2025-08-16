"""
Thévoz–Rappaz 非均质形核（接口骨架）

约定（现在先不实现算法，只保证被 Engine 安全调用）：
- 只在“液相掩码”内尝试形核；
- 本函数可以写回 grid 的持久字段：grain_id、theta(/theta_class)，
  在需要时也可对命中元胞给一个极小的 fs（作为种子）；
- 若 cfg 不完整，先安静返回（不抛错），后续再逐步补参数与校验。

后续实现要点（占位注释）：
- 过冷分布参数：N_max、ΔT_n、σ_n；
- 取向抽样：扇区中心抽样 θ_local = ((c+0.5)/Nθ - 0.5) * Δ，Δ = 2π/fold；
- rng 只用 Engine 注入的随机源（不要新建全局随机）。
"""

from __future__ import annotations
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply(
    grid,
    Tbuf: np.ndarray,
    rng: np.random.Generator,
    cfg_nucl: Dict[str, Any],
    masks: Dict[str, np.ndarray],
) -> None:
    """
    形核入口（空实现占位版）。
    参数
    ----
    grid : Grid
        持久状态容器（fs/CL/CS/grain_id/theta/...）。
    Tbuf : np.ndarray
        当步温度缓冲（与 grid.fs 同形状，含 ghost）。
    rng : np.random.Generator
        统一注入的随机数源（保证可复现）。
    cfg_nucl : dict
        形核配置（此阶段允许为空 dict）。
    masks : dict
        相态掩码，至少包含 'mask_liq'、'mask_int'、'mask_sol'。
    返回
    ----
    None（就地更新 grid；当前占位版不做任何修改）
    """
    # —— 占位实现：现在什么都不做，只保证被安全调用 ——
    if masks is None or "mask_liq" not in masks:
        logger.debug("[nucleation] 未提供液相掩码，跳过。")
        return

    # 如果想确认被调用，可以打开这行日志（等你调通后可去掉）
    logger.info("[nucleation] thevoz.apply() 被调用（占位版，未执行形核）。")
    return
