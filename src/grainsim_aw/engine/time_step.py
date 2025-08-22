# -*- coding: utf-8 -*-
"""
自适应时间步（与 C++ fun_delta_t 对齐）
Δt = 0.25 * min{ ℓ / vmax, ℓ² / DLmax, ℓ² / DSmax }
其中 ℓ = min(dx, dy)，vmax 来自法向生长速度 |vn|max，
DL、DS 由温度场 T 通过经验式计算。
"""

from __future__ import annotations
from ..core.material import Dl_from_T as _Dl
from ..core.material import Ds_from_T as _Ds
import numpy as np


def compute_next_dt(grid, fields) -> float:
    """
    计算下一步 Δt。只依赖 grid 与 fields，不读 cfg。
    要求：
      - grid.dx, grid.dy 为标量
      - grid.T 为温度场 ndarray
      - fields.vn 为法向生长速度 ndarray
    """
    # 网格尺度
    ell = min(float(grid.dx), float(grid.dy))

    # 速度上界：|vn| 的最大值
    vmax = float(np.max(np.abs(fields.vn)))

    # 温度依赖的扩散系数上界
    T = grid.T
    DLmax = float(np.max(_Dl(T)))
    DSmax = float(np.max(_Ds(T)))

    # 三个约束时间尺度
    # 说明：当 vmax 或 D 为 0 时，按数学极限处理为 +inf，确保 min 取到其余两项
    t_vel = np.inf if vmax == 0.0 else (ell / vmax)
    t_dl = np.inf if DLmax == 0.0 else ((ell * ell) / DLmax)
    t_ds = np.inf if DSmax == 0.0 else ((ell * ell) / DSmax)

    # 安全系数固定为 0.25，与 C++ 对齐
    return 0.25 * min(t_vel, t_dl, t_ds)
