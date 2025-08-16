import numpy as np
from ..core.grid import Grid


def sample_constant(grid: Grid, T_const: float) -> np.ndarray:
    """返回一个与 Grid 同形状的温度缓冲。"""
    return np.full_like(grid.fs, fill_value=T_const, dtype=float)


def sample(grid: Grid, t: float, temperature_cfg: dict) -> np.ndarray:
    # 第一版只实现常数温度
    T0 = temperature_cfg.get("T_const", 1750.0)
    return sample_constant(grid, T0)
