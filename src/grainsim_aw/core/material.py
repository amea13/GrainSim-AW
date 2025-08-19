# -*- coding: utf-8 -*-
"""
材料物性模型（底层通用）
- 扩散系数随温度的经验式
- 纯函数、无副作用；可被 interface / multiphysics 共同调用
"""
from __future__ import annotations
import numpy as np

__all__ = ["Dl_from_T", "Ds_from_T"]


def Dl_from_T(T: np.ndarray) -> np.ndarray:
    """液相扩散系数 [m^2/s]：7.67e-6 * exp(-12749.58 / T)"""
    Tc = np.clip(T, 1.0, None)
    return 7.67e-6 * np.exp(-12749.58 / Tc)


def Ds_from_T(T: np.ndarray) -> np.ndarray:
    """固相扩散系数 [m^2/s]：7.61e-6 * exp(-16185.23 / T)"""
    Tc = np.clip(T, 1.0, None)
    return 7.61e-6 * np.exp(-16185.23 / Tc)
