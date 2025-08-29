from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

__all__ = ["compute_curvature", "compute_normal"]


def compute_curvature(
    grid,
    masks: Dict[str, np.ndarray],
    out: Optional[np.ndarray] = None,
):
    """
    dfsdx  = (fs[i, j+1] - fs[i, j-1]) / (2*dx)
    dfsdy  = (fs[i+1, j] - fs[i-1, j]) / (2*dx)
    dfs2dx = (fs[i, j+1] + fs[i, j-1] - 2*fs[i, j]) / dx^2
    dfs2dy = (fs[i+1, j] + fs[i-1, j] - 2*fs[i, j]) / dx^2
    dfsdxdy= (fs[i-1, j+1] + fs[i+1, j-1] - fs[i-1, j-1] - fs[i+1, j+1]) / (4*dx^2)
    cur    = (2*dfsdx*dfsdy*dfsdxdy - dfs2dx*dfsdy^2 - dfs2dy*dfsdx^2) / sqrt((dfsdx^2+dfsdy^2)^3)
    """
    fs = grid.fs
    Ny, Nx = fs.shape

    if out is None:
        out = np.zeros_like(fs, dtype=np.float64)

    mask_intf = masks["intf"]

    dx = float(grid.dx)  # 对应 C++ 的 len
    dx2 = dx * dx

    for i in range(Ny):
        for j in range(Nx):
            if not mask_intf[i, j]:
                continue

            im, ip = i - 1, i + 1
            jm, jp = j - 1, j + 1

            dfsdx = (fs[i, jp] + (-fs[i, jm])) / (2.0 * dx)
            dfsdy = (fs[ip, j] + (-fs[im, j])) / (2.0 * dx)
            dfs2dx = (fs[i, jp] + fs[i, jm] - 2.0 * fs[i, j]) / dx2
            dfs2dy = (fs[ip, j] + fs[im, j] - 2.0 * fs[i, j]) / dx2
            dfsdxdy = (fs[im, jp] + fs[ip, jm] - fs[im, jm] - fs[ip, jp]) / (4.0 * dx2)

            num = (
                2.0 * dfsdx * dfsdy * dfsdxdy
                - dfs2dx * (dfsdy * dfsdy)
                - dfs2dy * (dfsdx * dfsdx)
            )
            den_base = dfsdx * dfsdx + dfsdy * dfsdy
            out[i, j] = num / np.sqrt(den_base * den_base * den_base)

    return out


def compute_normal(
    grid,
    masks: Dict[str, np.ndarray],
    out_nx: np.ndarray,
    out_ny: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    - 仅对 masks['intf'] 为 True 的元胞计算；
    - 使用 5x5 窗 + 半径3圈的三组权重(1.0, 0.83, 0.65)；
    - 直接写入 out_nx/out_ny
    依赖 grid.fs、grid.x、grid.y 三个二维数组。
    """

    fs = grid.fs
    x = grid.x  # 对应 C++: cell[i][j].x
    y = grid.y  # 对应 C++: cell[i][j].y

    mask_intf = masks["intf"]  # 按你的约束，不做兜底

    Ny, Nx = fs.shape

    # 遍历全域；是否计算由 mask 决定（不强加 i=2..ROWS+1 的显式范围）
    for i in range(Ny):
        for j in range(Nx):
            if not mask_intf[i, j]:
                continue

            # 对应 C++: if (cell[i][j].sta == 0) {...}
            # 下面完全照搬其加权累计写法
            xfz = 0.0
            yfz = 0.0
            fm = 0.0

            # 5x5 窗（权重 1.0）
            for n in range(5):  # n = 0..4
                for m in range(5):  # m = 0..4
                    ii = i + (n - 2)
                    jj = j + (m - 2)
                    fij = fs[ii, jj]
                    xfz += fij * 1.0 * x[ii, jj]
                    yfz += fij * 1.0 * y[ii, jj]
                    fm += fij * 1.0

            # 半径 3，轴向（权重 1.0）
            xfz += (
                fs[i - 3, j] * x[i - 3, j]
                + fs[i, j - 3] * x[i, j - 3]
                + fs[i + 3, j] * x[i + 3, j]
                + fs[i, j + 3] * x[i, j + 3]
            ) * 1.0
            yfz += (
                fs[i - 3, j] * y[i - 3, j]
                + fs[i, j - 3] * y[i, j - 3]
                + fs[i + 3, j] * y[i + 3, j]
                + fs[i, j + 3] * y[i, j + 3]
            ) * 1.0
            fm += (fs[i - 3, j] + fs[i, j - 3] + fs[i + 3, j] + fs[i, j + 3]) * 1.0

            # 半径 3，“马步”1格（权重 0.83）
            xfz += (
                fs[i - 3, j + 1] * x[i - 3, j + 1]
                + fs[i - 3, j - 1] * x[i - 3, j - 1]
                + fs[i + 3, j + 1] * x[i + 3, j + 1]
                + fs[i + 3, j - 1] * x[i + 3, j - 1]
                + fs[i + 1, j - 3] * x[i + 1, j - 3]
                + fs[i - 1, j - 3] * x[i - 1, j - 3]
                + fs[i + 1, j + 3] * x[i + 1, j + 3]
                + fs[i - 1, j + 3] * x[i - 1, j + 3]
            ) * 0.83
            yfz += (
                fs[i - 3, j + 1] * y[i - 3, j + 1]
                + fs[i - 3, j - 1] * y[i - 3, j - 1]
                + fs[i + 3, j + 1] * y[i + 3, j + 1]
                + fs[i + 3, j - 1] * y[i + 3, j - 1]
                + fs[i + 1, j - 3] * y[i + 1, j - 3]
                + fs[i - 1, j - 3] * y[i - 1, j - 3]
                + fs[i + 1, j + 3] * y[i + 1, j + 3]
                + fs[i - 1, j + 3] * y[i - 1, j + 3]
            ) * 0.83
            fm += (
                fs[i - 3, j + 1]
                + fs[i - 3, j - 1]
                + fs[i + 3, j + 1]
                + fs[i + 3, j - 1]
                + fs[i + 1, j - 3]
                + fs[i - 1, j - 3]
                + fs[i + 1, j + 3]
                + fs[i - 1, j + 3]
            ) * 0.83

            # 半径 3，“马步”2格（权重 0.65）
            xfz += (
                fs[i - 3, j + 2] * x[i - 3, j + 2]
                + fs[i - 3, j - 2] * x[i - 3, j - 2]
                + fs[i + 3, j + 2] * x[i + 3, j + 2]
                + fs[i + 3, j - 2] * x[i + 3, j - 2]
                + fs[i + 2, j - 3] * x[i + 2, j - 3]
                + fs[i - 2, j - 3] * x[i - 2, j - 3]
                + fs[i + 2, j + 3] * x[i + 2, j + 3]
                + fs[i - 2, j + 3] * x[i - 2, j + 3]
            ) * 0.65
            yfz += (
                fs[i - 3, j + 2] * y[i - 3, j + 2]
                + fs[i - 3, j - 2] * y[i - 3, j - 2]
                + fs[i + 3, j + 2] * y[i + 3, j + 2]
                + fs[i + 3, j - 2] * y[i + 3, j - 2]
                + fs[i + 2, j - 3] * y[i + 2, j - 3]
                + fs[i - 2, j - 3] * y[i - 2, j - 3]
                + fs[i + 2, j + 3] * y[i + 2, j + 3]
                + fs[i - 2, j + 3] * y[i - 2, j + 3]
            ) * 0.65
            fm += (
                fs[i - 3, j + 2]
                + fs[i - 3, j - 2]
                + fs[i + 3, j + 2]
                + fs[i + 3, j - 2]
                + fs[i + 2, j - 3]
                + fs[i - 2, j - 3]
                + fs[i + 2, j + 3]
                + fs[i - 2, j + 3]
            ) * 0.65

            xb = xfz / fm
            yb = yfz / fm

            AB = (
                (x[i, j] - xb) * (x[i, j] - xb) + (y[i, j] - yb) * (y[i, j] - yb)
            ) ** 0.5

            out_nx[i, j] = (x[i, j] - xb) / AB
            out_ny[i, j] = (y[i, j] - yb) / AB

    return out_nx, out_ny
