from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

__all__ = ["compute_curvature", "compute_normal"]


def _viewer(a: np.ndarray, pad: int):
    """
    返回一个闭包 V(di, dj) -> 视图切片，等价于把 a 向下/右为正平移 (di,dj) 后
    超界部分补 0（常数 0 填充），但不产生新大数组；避免频繁分配。
    """
    Ny, Nx = a.shape
    ap = np.pad(a, pad_width=pad, mode="constant", constant_values=0.0)
    base_i = pad
    base_j = pad

    def V(di: int, dj: int) -> np.ndarray:
        i0 = base_i + di
        j0 = base_j + dj
        return ap[i0 : i0 + Ny, j0 : j0 + Nx]  # 纯视图

    return V


def compute_curvature(
    grid,
    masks: Dict[str, np.ndarray],
    out: Optional[np.ndarray] = None,
):
    """
    计算 fs 的曲率，使用二阶中心差分的标准公式。
    """
    fs = grid.fs.astype(np.float64, copy=False)
    mask = masks["intf"]

    if out is None:
        out = np.zeros_like(fs, dtype=np.float64)
    else:
        out.fill(0.0)

    dx = float(grid.dx)
    dx2 = dx * dx

    # 只需 pad=1 就能覆盖 ±1 及对角偏移
    V = _viewer(fs, pad=1)

    fs_ip = V(+1, 0)
    fs_im = V(-1, 0)
    fs_jp = V(0, +1)
    fs_jm = V(0, -1)

    dfsdx = (fs_jp - fs_jm) / (2.0 * dx)
    dfsdy = (fs_ip - fs_im) / (2.0 * dx)
    dfs2dx = (fs_jp + fs_jm - 2.0 * fs) / dx2
    dfs2dy = (fs_ip + fs_im - 2.0 * fs) / dx2

    fs_im_jp = V(-1, +1)
    fs_ip_jm = V(+1, -1)
    fs_im_jm = V(-1, -1)
    fs_ip_jp = V(+1, +1)
    dfsdxdy = (fs_im_jp + fs_ip_jm - fs_im_jm - fs_ip_jp) / (4.0 * dx2)

    num = (
        2.0 * dfsdx * dfsdy * dfsdxdy
        - dfs2dx * (dfsdy * dfsdy)
        - dfs2dy * (dfsdx * dfsdx)
    )
    den_base = dfsdx * dfsdx + dfsdy * dfsdy
    curv = num / np.sqrt(den_base * den_base * den_base)

    out[mask] = curv[mask]
    return out


def compute_normal(
    grid,
    masks: Dict[str, np.ndarray],
    out_nx: np.ndarray,
    out_ny: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    完全忠实原累加顺序（5x5，轴向±3，马步±(3,1)，马步±(3,2)），
    仅把逐元胞循环改成“偏移视图逐项累计”。
    """
    fs = grid.fs.astype(np.float64, copy=False)
    x = grid.x.astype(np.float64, copy=False)
    y = grid.y.astype(np.float64, copy=False)
    mask = masks["intf"]

    fx = fs * x
    fy = fs * y

    xfz = np.zeros_like(fs, dtype=np.float64)
    yfz = np.zeros_like(fs, dtype=np.float64)
    fm = np.zeros_like(fs, dtype=np.float64)

    # 需要用到 ±3 的偏移，pad=3 即可
    Vf = _viewer(fs, pad=3)
    Vfx = _viewer(fx, pad=3)
    Vfy = _viewer(fy, pad=3)

    def _acc(di: int, dj: int, w: float) -> None:
        # 逐项 add，保持与原实现相同的求和顺序（减少浮点尾差）
        np.add(xfz, w * Vfx(di, dj), out=xfz)
        np.add(yfz, w * Vfy(di, dj), out=yfz)
        np.add(fm, w * Vf(di, dj), out=fm)

    # 5x5 窗（权重 1.0）——严格按原先双循环的行优先顺序
    for di in (-2, -1, 0, 1, 2):
        for dj in (-2, -1, 0, 1, 2):
            _acc(di, dj, 1.0)

    # 半径 3，轴向（权重 1.0）
    _acc(-3, 0, 1.0)
    _acc(0, -3, 1.0)
    _acc(+3, 0, 1.0)
    _acc(0, +3, 1.0)

    # 半径 3，“马步”1格（权重 0.83）——保持原有累计顺序
    _acc(-3, +1, 0.83)
    _acc(-3, -1, 0.83)
    _acc(+3, +1, 0.83)
    _acc(+3, -1, 0.83)
    _acc(+1, -3, 0.83)
    _acc(-1, -3, 0.83)
    _acc(+1, +3, 0.83)
    _acc(-1, +3, 0.83)

    # 半径 3，“马步”2格（权重 0.65）
    _acc(-3, +2, 0.65)
    _acc(-3, -2, 0.65)
    _acc(+3, +2, 0.65)
    _acc(+3, -2, 0.65)
    _acc(+2, -3, 0.65)
    _acc(-2, -3, 0.65)
    _acc(+2, +3, 0.65)
    _acc(-2, +3, 0.65)

    # 只在界面元胞写回
    xb = np.empty_like(fs, dtype=np.float64)
    yb = np.empty_like(fs, dtype=np.float64)
    xb[mask] = xfz[mask] / fm[mask]
    yb[mask] = yfz[mask] / fm[mask]

    dxv = x[mask] - xb[mask]
    dyv = y[mask] - yb[mask]
    AB = np.sqrt(dxv * dxv + dyv * dyv)

    out_nx[mask] = dxv / AB
    out_ny[mask] = dyv / AB
    return out_nx, out_ny
