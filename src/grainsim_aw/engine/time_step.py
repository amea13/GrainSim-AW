from __future__ import annotations
from typing import Optional
import numpy as np

from ..core.material import Dl_from_T, Ds_from_T

__all__ = [
    "estimate_dt",
    "estimate_dt_from_grid",
    "vmax_from_fields",
    "adaptive_dt",
    "diagnose_dt",
]


def estimate_dt(
    dx: float,
    vmax: Optional[float],
    DL: Optional[float],
    DS: Optional[float] = None,
    safety: float = 0.2,
    cap: Optional[float] = None,
) -> float:
    """
    计算 Δt = safety * min(dx/vmax, dx^2/DL, dx^2/DS)。
    任一参数为 None 或非正时将忽略对应约束。
    """
    h = float(dx)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("dx 必须为正且有限。")

    candidates = []

    if vmax is not None and np.isfinite(vmax) and vmax > 0:
        candidates.append(h / float(vmax))
    if DL is not None and np.isfinite(DL) and DL > 0:
        candidates.append(h * h / float(DL))
    if DS is not None and np.isfinite(DS) and DS > 0:
        candidates.append(h * h / float(DS))

    if not candidates:
        raise ValueError("缺少稳定性约束：至少提供一个正的 v_max、D_L 或 D_S。")

    dt = float(safety) * min(candidates)
    if cap is not None and cap > 0:
        dt = min(dt, float(cap))
    return dt


def estimate_dt_from_grid(
    grid,
    vmax: Optional[float],
    DL: Optional[float],
    DS: Optional[float] = None,
    safety: float = 0.2,
    cap: Optional[float] = None,
) -> float:
    """
    从 grid 读取 dx, dy，取 h = min(dx, dy) 后调用 estimate_dt。
    """
    try:
        h = float(min(grid.dx, grid.dy))
    except Exception as e:
        raise AttributeError("grid 需包含浮点属性 dx 与 dy。") from e

    return estimate_dt(h, vmax=vmax, DL=DL, DS=DS, safety=safety, cap=cap)


def vmax_from_fields(
    vn: Optional[np.ndarray] = None,
    vx: Optional[np.ndarray] = None,
    vy: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    从速度或法向生长速率场估计 v_max。
    - 提供 vn 时返回 max(|vn|)。
    - 否则提供 vx 或 vy 时返回 max(sqrt(vx^2+vy^2))。
    若输入为空或无有限值，返回 None。
    """

    def _finite_max(a: np.ndarray) -> Optional[float]:
        if a is None:
            return None
        arr = np.asarray(a)
        mask = np.isfinite(arr)
        if not np.any(mask):
            return None
        return float(np.max(np.abs(arr[mask])))

    if vn is not None:
        return _finite_max(vn)

    if vx is not None or vy is not None:
        vx_ = 0.0 if vx is None else np.asarray(vx)
        vy_ = 0.0 if vy is None else np.asarray(vy)
        speed = np.sqrt(vx_ * vx_ + vy_ * vy_)
        return _finite_max(speed)

    return None


def _finite_max_scalar_from_field(field: Optional[np.ndarray]) -> Optional[float]:
    """
    返回场中有限值的最大值，若不存在则 None。
    用于把空间变系数转为最严格的全局约束。
    """
    if field is None:
        return None
    arr = np.asarray(field)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return None
    return float(np.max(arr[mask]))


def adaptive_dt(
    grid,
    fields,
    safety: float = 0.2,
    cap: Optional[float] = None,
    include_solid_diffusion: bool = True,
) -> float:
    """
    自适应时间步：
    - v_max 由 fields.Vn 的绝对值最大值得到
    - D_L = max(Dl_from_T(fields.T))
    - D_S = max(Ds_from_T(fields.T))，可通过 include_solid_diffusion 关闭

    要求：
    - grid 具有 dx, dy
    - fields 至少包含 Vn；若无 T，则只能使用速度约束
    """
    # 网格尺度
    try:
        h = float(min(grid.dx, grid.dy))
    except Exception as e:
        raise AttributeError("grid 需包含浮点属性 dx 与 dy。") from e

    # 速度约束
    v_max = vmax_from_fields(vn=getattr(fields, "Vn", None))

    # 扩散约束：对空间变系数取最大值，保证全域稳定
    T = getattr(fields, "T", None)
    DL = DS = None
    if T is not None:
        DL = _finite_max_scalar_from_field(Dl_from_T(T))
        if include_solid_diffusion:
            DS = _finite_max_scalar_from_field(Ds_from_T(T))

    return estimate_dt(h, vmax=v_max, DL=DL, DS=DS, safety=safety, cap=cap)


def diagnose_dt(
    grid,
    fields,
    safety: float = 0.2,
    include_solid_diffusion: bool = True,
) -> dict:
    """
    返回各约束项的数值，便于日志打印与调参。
    """
    h = float(min(grid.dx, grid.dy))
    v_max = vmax_from_fields(vn=getattr(fields, "Vn", None))
    T = getattr(fields, "T", None)

    DL = _finite_max_scalar_from_field(Dl_from_T(T)) if T is not None else None
    DS = (
        _finite_max_scalar_from_field(Ds_from_T(T))
        if (T is not None and include_solid_diffusion)
        else None
    )

    terms = {}
    if v_max is not None and v_max > 0:
        terms["dx_over_vmax"] = h / v_max
    if DL is not None and DL > 0:
        terms["dx2_over_DL"] = h * h / DL
    if DS is not None and DS > 0:
        terms["dx2_over_DS"] = h * h / DS

    if not terms:
        raise ValueError("无法诊断时间步：未找到有效的约束来源。")

    dt = safety * min(terms.values())
    return {"candidates": terms, "dt": dt, "safety": safety}
