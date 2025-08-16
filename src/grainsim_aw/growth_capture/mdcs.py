"""
growth_capture/mdcs.py
MDCS 主入口占位：只确定签名与日志，不做任何几何更新。
"""

from __future__ import annotations
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


def step(
    grid,
    fields,  # 来自 interface.compute_interface_fields 的 IfaceFields
    cfg_mdcs: Dict[str, Any],
    cfg_orient: Dict[str, Any],
    dt: float,
    masks: Dict[str, np.ndarray],
) -> None:
    """
    占位版：现在不修改 grid，只记录一次心跳日志。
    约定：真实实现时会写回 grid.fs 与 grid.L_dia，并可能设置新界面胞的 grain_id。
    """
    logger.info("MDCS.step() called (stub) — 未进行几何更新。")
    return
