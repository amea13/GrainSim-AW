# src/grainsim_aw/interface/fields.py
# -*- coding: utf-8 -*-
"""
IfaceFieldsBuf — 单步“界面带场”的可写缓冲。
仅在 masks["intf"] 覆盖写入；支持复用与按上一帧界面带清零。
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class IfaceFieldsBuf:
    """
    【功能】承载单步界面带相关场，由各过程函数“就地写入”。
    【形状】所有数组与 grid.fs 一致；界面带外通常为 0。
    """

    # 几何/界面量
    kappa: np.ndarray  # 曲率
    nx: np.ndarray  # 法向 x
    ny: np.ndarray  # 法向 y
    cls: np.ndarray  # 界面液相浓度 C_L^*
    css: np.ndarray  # 界面固相浓度 C_S^*
    vn: np.ndarray  # 法向生长速率
    fs_dot: np.ndarray  # 固相率时间导数（溶质源项）
    vx: np.ndarray  # 法向 x 速度分量
    vy: np.ndarray  # 法向 y 速度分量
    ani: np.ndarray  # 各向异性诊断量

    # —— 工厂与维护 —— #
    @staticmethod
    def like(
        grid,
    ) -> "IfaceFieldsBuf":
        z = lambda: np.zeros_like(grid.fs)
        buf = IfaceFieldsBuf(
            kappa=z(),
            nx=z(),
            ny=z(),
            cls=z(),
            css=z(),
            vn=z(),
            fs_dot=z(),
            vx=z(),
            vy=z(),
            ani=z(),
        )
        return buf
