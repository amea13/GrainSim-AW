from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np

__all__ = ["compute_curvature", "compute_normal"]


# =========================
# 曲率（中心差分法）
# κ = (2 f_x f_y f_xy -  f_xx f_y^2 - f_yy f_x^2) / (f_x^2 + f_y^2)^(3/2)
# 只在界面带写入 out
# =========================
def compute_curvature(
    grid,
    fields,
    masks: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    out: Optional[np.ndarray] = None,
) -> np.ndarray:

    fs = grid.fs
    dx = float(grid.dx)
    dy = float(grid.dy)
    nx = fields.nx
    ny = fields.ny

    intf: np.ndarray = masks["intf"]
    if intf is None:
        raise KeyError("masks 中缺少 'intf' 或 'mask_int'")
    intf = np.asarray(intf, dtype=bool)

    # 仅 core 区域参与写入，避免 ghost 受 roll 影响
    ys, xs = grid.core
    core_mask = np.zeros_like(intf, dtype=bool)
    core_mask[ys, xs] = True
    write_mask = intf & core_mask

    roll = np.roll

    # 一阶导
    fx = (roll(fs, -1, axis=1) - roll(fs, 1, axis=1)) / (2.0 * dx)
    fy = (roll(fs, -1, axis=0) - roll(fs, 1, axis=0)) / (2.0 * dy)

    # 二阶与混合导
    fxx = (roll(fs, -1, axis=1) + roll(fs, 1, axis=1) - 2.0 * fs) / (dx * dx)
    fyy = (roll(fs, -1, axis=0) + roll(fs, 1, axis=0) - 2.0 * fs) / (dy * dy)
    fxy = (
        roll(roll(fs, -1, axis=0), 1, axis=1)
        + roll(roll(fs, 1, axis=0), -1, axis=1)
        - roll(roll(fs, -1, axis=0), -1, axis=1)
        - roll(roll(fs, 1, axis=0), 1, axis=1)
    ) / (4.0 * dx * dy)

    g2 = fx * fx + fy * fy
    num = 2.0 * fx * fy * fxy - fxx * (fy * fy) - fyy * (fx * fx)
    den = np.power(g2, 1.5) + 1e-30

    kappa_full = num / den

    if out is None:
        out = np.zeros_like(fs, dtype=float)
    out[write_mask] = kappa_full[write_mask]
    return out


def compute_curvature1(
    grid,
    fields,
    masks: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    dx = float(grid.dx)
    dy = float(grid.dy)
    nx = np.asarray(fields.nx, dtype=float)
    ny = np.asarray(fields.ny, dtype=float)

    intf = masks["intf"].astype(bool)
    ys, xs = grid.core
    core_mask = np.zeros_like(intf, dtype=bool)
    core_mask[ys, xs] = True
    write_mask = intf & core_mask

    roll = np.roll

    # 可选：对法向做轻微平滑后再归一化，抑制噪声（只影响几何，不改守恒）
    if bool(cfg.get("curv_smooth_normals", True)):

        def box9(a):
            return (
                a
                + roll(a, 1, 0)
                + roll(a, -1, 0)
                + roll(a, 1, 1)
                + roll(a, -1, 1)
                + roll(roll(a, 1, 0), 1, 1)
                + roll(roll(a, 1, 0), -1, 1)
                + roll(roll(a, -1, 0), 1, 1)
                + roll(roll(a, -1, 0), -1, 1)
            ) / 9.0

        nx = box9(nx)
        ny = box9(ny)

    # 重新单位化，防止累计误差；纯相处设为 0（不写入也无所谓）
    g = np.sqrt(nx * nx + ny * ny)
    eps = 1e-12
    nz = g > eps
    nx = np.where(nz, nx / g, 0.0)
    ny = np.where(nz, ny / g, 0.0)

    # 散度（中心差分）
    dnx_dx = (roll(nx, -1, 1) - roll(nx, 1, 1)) / (2.0 * dx)
    dny_dy = (roll(ny, -1, 0) - roll(ny, 1, 0)) / (2.0 * dy)
    kappa_full = dnx_dx + dny_dy

    if out is None:
        out = np.zeros_like(nx, dtype=float)
    out[write_mask] = kappa_full[write_mask]

    # 可选：物理限幅（最小曲率半径 ~ c * min(dx,dy)）
    Rmin_cells = float(cfg.get("curv_Rmin_cells", 2.5))
    if Rmin_cells > 0:
        Rmin = Rmin_cells * min(dx, dy)
        kcap = 1.0 / max(Rmin, 1e-30)
        np.clip(out, -kcap, kcap, out=out)

    return out


# =========================
# 法向（圆核质心法，一阶矩权重）
# n = - (num_x, num_y) / |(num_x, num_y)|
# 只在界面带写入 out_nx/out_ny
# =========================
def compute_normal(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    out_nx: np.ndarray,
    out_ny: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """简化版本，使用预定义的偏移量列表但更Pythonic"""
    fs = grid.fs
    dx, dy = grid.dx, grid.dy
    intf_indices = np.where(masks["intf"])

    # 预定义偏移量和权重
    offsets_weights = _get_offsets_and_weights()

    # 向量化计算所有偏移
    di_array, dj_array, weights_array = map(np.array, zip(*offsets_weights))

    # 对每个界面点进行计算
    for i, j in zip(*intf_indices):
        # 计算所有邻域点的坐标
        ni_array = i + di_array
        nj_array = j + dj_array

        # 获取对应的fs值
        fs_values = fs[ni_array, nj_array]

        # 过滤掉fs=0的点
        valid_mask = fs_values != 0
        if not np.any(valid_mask):
            continue

        fs_valid = fs_values[valid_mask]
        weights_valid = weights_array[valid_mask]
        di_valid = di_array[valid_mask]
        dj_valid = dj_array[valid_mask]

        # 向量化计算
        weighted_fs = fs_valid * weights_valid
        xfz = np.sum(weighted_fs * dj_valid * dx)
        yfz = np.sum(weighted_fs * di_valid * dy)
        fm = np.sum(weighted_fs)

        # 计算法向量
        if fm > 0:
            xb, yb = xfz / fm, yfz / fm
            magnitude = np.sqrt(xb**2 + yb**2)
            if magnitude > 0:
                out_nx[i, j] = -xb / magnitude
                out_ny[i, j] = -yb / magnitude

    return out_nx, out_ny


def _get_offsets_and_weights():
    """生成偏移量和权重的更简洁方式"""
    offsets_weights = []

    # 核心5×5 (权重1.0)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            offsets_weights.append((di, dj, 1.0))

    # 环带权重规则
    ring3_patterns = [
        ([(0, 3), (0, -3), (3, 0), (-3, 0)], 1.0),  # 轴向
        (
            [(1, 3), (-1, 3), (1, -3), (-1, -3), (3, 1), (3, -1), (-3, 1), (-3, -1)],
            0.83,
        ),
        (
            [(2, 3), (-2, 3), (2, -3), (-2, -3), (3, 2), (3, -2), (-3, 2), (-3, -2)],
            0.65,
        ),
    ]

    for positions, weight in ring3_patterns:
        offsets_weights.extend([(di, dj, weight) for di, dj in positions])

    return offsets_weights
