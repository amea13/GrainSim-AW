"""
multiphysics/solute_solver.py
占位版：确定入口签名，不更新 CL/CS；只打一次日志。
"""

from __future__ import annotations
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


def advance(
    grid,
    cfg_sol: Dict[str, Any],
    dt: float,
    masks: Dict[str, np.ndarray],
) -> None:
    """
    占位实现：当前不修改 grid.CL / grid.CS。
    真实实现时会按照体积分数加权守恒式更新 CL/CS。
    """
    logger.info("SoluteSolver.advance() called (stub) — 未更新溶质场。")
    return
