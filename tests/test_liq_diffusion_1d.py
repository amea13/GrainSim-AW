# -*- coding: utf-8 -*-
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 让解释器能找到 src/grainsim_aw
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainsim_aw.multiphysics import solute_advance  # 已在 __init__.py 导出


# ---- 极简 Grid 与工具 ----
class GridStub:
    def __init__(self, ny=100, nx=200, dx=1e-6, dy=1e-6, nghost=2, T0=1800.0):
        self.ny, self.nx, self.dx, self.dy, self.nghost = ny, nx, dx, dy, nghost
        Ny, Nx = ny + 2 * nghost, nx + 2 * nghost
        self.fs = np.zeros((Ny, Nx), dtype=np.float64)  # 全液相：fs=0
        self.CL = np.zeros((Ny, Nx), dtype=np.float64)
        self.CS = np.zeros((Ny, Nx), dtype=np.float64)
        self.T = np.full((Ny, Nx), T0, dtype=np.float64)

    @property
    def core(self):
        g = self.nghost
        return slice(g, -g), slice(g, -g)


def update_ghosts(grid: GridStub):
    g = grid.nghost
    for arr in (grid.fs, grid.CL, grid.CS, grid.T):
        arr[:g, :] = arr[g : 2 * g, :]
        arr[-g:, :] = arr[-2 * g : -g, :]
        arr[:, :g] = arr[:, g : 2 * g]
        arr[:, -g:] = arr[:, -2 * g : -g]


def total_mass(grid: GridStub) -> float:
    g = grid.nghost
    ys, xs = grid.core
    alpha = 1.0 - grid.fs
    cell = grid.dx * grid.dy
    return float(
        np.sum(
            alpha[ys, xs] * grid.CL[ys, xs] + (1.0 - alpha[ys, xs]) * grid.CS[ys, xs]
        )
        * cell
    )


# ---- 初始条件：纯液相，高斯包 ----
def build_case():
    ny, nx = 100, 200
    grid = GridStub(ny=ny, nx=nx, dx=1e-6, dy=1e-6, nghost=2, T0=1800.0)
    ys, xs = grid.core

    x = (np.arange(nx) + 0.5) * grid.dx
    x0 = x.mean()
    sigma0 = 6e-6  # 初始标准差 6 μm
    Cpeak = 1.0
    gauss = Cpeak * np.exp(-0.5 * ((x - x0) / sigma0) ** 2)

    grid.CL[ys, xs] = gauss[None, :]  # 每一行相同，便于看 1D 剖面
    grid.CS[ys, xs] = 0.0
    update_ghosts(grid)
    return grid


# ---- 可视化：左 CL 场，右 中心行剖面；下方总溶质量曲线 ----
def main():
    grid = build_case()
    dt = 1e-4  # s
    steps = 300  # 总时间 0.03 s
    fs_dot = np.zeros_like(grid.fs)  # 纯扩散，无源

    g = grid.nghost
    ys, xs = grid.core
    ny, nx = grid.ny, grid.nx
    x = (np.arange(nx) + 0.5) * grid.dx

    # 固定颜色范围，避免闪烁
    vmin_CL = float(grid.CL[ys, xs].min())
    vmax_CL = float(grid.CL[ys, xs].max())

    # 布局
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_im = fig.add_subplot(gs[0, 0])
    im = ax_im.imshow(
        grid.CL[ys, xs], origin="lower", vmin=vmin_CL, vmax=vmax_CL, aspect="auto"
    )
    ax_im.set_title("CL field (pure liquid diffusion)")
    cb = fig.colorbar(im, ax=ax_im, fraction=0.046)
    cb.set_label("CL")

    ax_line = fig.add_subplot(gs[0, 1])
    mid = ny // 2
    (line,) = ax_line.plot(x * 1e6, grid.CL[ys, xs][mid], lw=2)
    ax_line.set_title(f"Center-row profile (y={mid})")
    ax_line.set_xlabel("x (μm)")
    ax_line.set_ylabel("CL")
    ax_line.grid(True)
    ax_line.set_xlim(x[0] * 1e6, x[-1] * 1e6)
    ax_line.set_ylim(vmin_CL, vmax_CL * 1.05)

    ax_mass = fig.add_subplot(gs[1, :])
    (mass_line,) = ax_mass.plot([], [], lw=2)
    ax_mass.set_title("Total solute mass vs time")
    ax_mass.set_xlabel("time (s)")
    ax_mass.set_ylabel("mass")
    ax_mass.grid(True)

    times, masses = [0.0], [total_mass(grid)]
    ax_mass.set_xlim(0, steps * dt)
    ax_mass.set_ylim(masses[0] * 0.999, masses[0] * 1.001)

    def step(frame):
        # 推进一步
        solute_advance(
            grid,
            {
                "solver": {"max_iter": 120, "tol": 1e-8},
                "eps": 1e-12,
                "clip": {"min": 0.0},
            },
            dt,
            masks={},
            fs_dot=fs_dot,
        )
        update_ghosts(grid)

        # 更新影像与曲线
        im.set_data(grid.CL[ys, xs])
        line.set_data(x * 1e6, grid.CL[ys, xs][mid])

        t = times[-1] + dt
        times.append(t)
        masses.append(total_mass(grid))
        mass_line.set_data(times, masses)

        fig.suptitle(
            f"t = {t:.4f} s   mass rel. change = {(masses[-1]-masses[0])/masses[0]:.2e}"
        )
        return im, line, mass_line

    ani = FuncAnimation(fig, step, frames=steps, interval=30, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    # Windows PowerShell 运行：
    # $env:PYTHONPATH="$PWD/src"
    # python .\tests\vis_liq_diffusion.py
    main()
