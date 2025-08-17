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
    cfg_if: Dict[str, Any],
    cfg_orient: Dict[str, Any],
    masks: Dict[str, np.ndarray],
):
    """
    占位加强版：在界面元胞上给一个常数 Vn，其它派生量先置零/一。
    如需温度信息请直接使用 grid.T。
    """
    shape = grid.fs.shape
    zeros = lambda: np.zeros(shape, dtype=float)

    # 占位派生量
    Vn = zeros()
    ani = np.ones(shape, dtype=float)  # 暂时各向同性
    nx = zeros()
    ny = zeros()
    kappa = zeros()
    CLs = zeros()
    CSs = zeros()

    # 示例：如将来有温度依赖，可直接读取
    # T = grid.T

    vconst = float(cfg_if.get("Vn_const", 0.0))
    if "mask_int" in masks and vconst > 0.0:
        Vn[masks["mask_int"]] = vconst

    fields = IfaceFields(Vn=Vn, nx=nx, ny=ny, kappa=kappa, ani=ani, CLs=CLs, CSs=CSs)
    logger.info(
        "Interface fields: Vn_const=%.3g, active=%d", vconst, int(np.count_nonzero(Vn))
    )
    return fields
