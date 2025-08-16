"""
interface/equilibrium.py
对 Engine 暴露一个入口：compute_interface_fields(...)
现在是占位实现：返回与 grid.fs 形状一致的全 0 字段。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IfaceFields:
    Vn: np.ndarray  # 法向生长速度
    nx: np.ndarray  # 法向 x 分量
    ny: np.ndarray  # 法向 y 分量
    kappa: np.ndarray  # 曲率
    ani: np.ndarray  # 各向异性因子
    CLs: np.ndarray  # 界面液相浓度 C_L^*
    CSs: np.ndarray  # 界面固相浓度 C_S^*


def compute_interface_fields(
    grid,
    Tbuf: np.ndarray,
    cfg_if: Dict[str, Any],
    cfg_orient: Dict[str, Any],
    masks: Dict[str, np.ndarray],
) -> IfaceFields:
    """
    生成界面派生量（占位版：全 0）；约定所有数组形状与 grid.fs 完全一致（含 ghost）。
    非界面位置值为 0；真实实现时只在 masks['mask_int'] 上计算。
    """
    shape = grid.fs.shape
    zeros = lambda: np.zeros(shape, dtype=float)

    fields = IfaceFields(
        Vn=zeros(),
        nx=zeros(),
        ny=zeros(),
        kappa=zeros(),
        ani=zeros(),
        CLs=zeros(),
        CSs=zeros(),
    )
    logger.info("[interface] compute_interface_fields() 占位版返回全 0 字段。")
    return fields
