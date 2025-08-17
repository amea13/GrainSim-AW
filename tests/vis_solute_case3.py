import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 让解释器能找到 src/grainsim_aw
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainsim_aw.multiphysics import solute_advance  # 已在 __init__.py 导出


# ---- 简易 Grid 与工具：和 test_solute_min 一样 ----
class GridStub:
    def __init__(self, ny=120, nx=160, dx=1.0, dy=1.0, nghost=2, T0=1800.0):
        self.ny, self.nx, self.dx, self.dy, self.nghost = ny, nx, dx, dy, nghost
        Ny, Nx = ny + 2 * nghost, nx + 2 * nghost
        self.fs = np.zeros((Ny, Nx), dtype=np.float64)
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


def total_solute_mass(grid: GridStub) -> float:
    g = grid.nghost
    core = (slice(g, -g), slice(g, -g))
    alpha = 1.0 - grid.fs
    cell = grid.dx * grid.dy
    return float(
        np.sum(alpha[core] * grid.CL[core] + (1.0 - alpha[core]) * grid.CS[core]) * cell
    )


# ---- 构造 Case3 初始场 ----
def build_case3():
    ny, nx = 120, 160
    grid = GridStub(ny=ny, nx=nx, dx=1e-6, dy=1e-6, nghost=2, T0=1800.0)
    ys, xs = grid.core
    j0, w = nx // 2, 3

    fs = np.zeros((grid.ny, grid.nx))
    fs[:, j0 - w : j0 + w + 1] = 0.5
    fs[:, j0 + w + 1 :] = 1.0
    grid.fs[ys, xs] = fs

    C0 = 0.05
    alpha = 1.0 - grid.fs
    grid.CL[...] = 0.0
    grid.CS[...] = 0.0
    grid.CL[ys, xs][alpha[ys, xs] > 1e-12] = C0  # 仅液相有 CL
    grid.CS[ys, xs][grid.fs[ys, xs] > 1e-12] = C0  # 仅固相有 CS
    update_ghosts(grid)

    fs_dot = np.zeros_like(grid.fs)
    band = np.zeros((ny, nx), dtype=bool)
    band[:, j0 - w : j0 + w + 1] = True
    fs_dot_band = np.zeros((ny, nx))
    fs_dot_band[band] = 0.5
    fs_dot[ys, xs] = fs_dot_band
    return grid, fs_dot


# ---- 可视化主程序 ----
def main():
    grid, fs_dot = build_case3()
    cfg = {
        "k": 0.34,
        "solver": {"max_iter": 800, "tol": 1e-8},
        "eps": 1e-12,
        "clip": {"min": 0.0},
    }
    dt = 0.01
    steps = 120

    g = grid.nghost
    ys, xs = grid.core
    alpha = 1.0 - grid.fs

    # 设定统一颜色范围，避免闪烁
    CL0 = grid.CL[ys, xs].copy()
    CS0 = grid.CS[ys, xs].copy()
    vmin_CL, vmax_CL = float(CL0.min()), float(CL0.max() + 0.005)
    vmin_CS, vmax_CS = float(CS0.min() - 0.002), float(CS0.max())
    vmin_fs, vmax_fs = 0.0, 1.0
    dens0 = alpha[ys, xs] * CL0 + (1.0 - alpha[ys, xs]) * CS0
    vmin_dens, vmax_dens = float(dens0.min()), float(dens0.max())

    # 布局：左上 fs，右上 CL，左下 CS，右下 总溶质量密度
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax_fs, ax_CL = axes[0]
    ax_CS, ax_dens = axes[1]

    im_fs = ax_fs.imshow(grid.fs[ys, xs], origin="lower", vmin=vmin_fs, vmax=vmax_fs)
    ax_fs.set_title("fs")
    fig.colorbar(im_fs, ax=ax_fs, fraction=0.046)

    im_CL = ax_CL.imshow(grid.CL[ys, xs], origin="lower", vmin=0.045, vmax=0.055)
    ax_CL.set_title("CL")
    fig.colorbar(im_CL, ax=ax_CL, fraction=0.046)

    im_CS = ax_CS.imshow(grid.CS[ys, xs], origin="lower", vmin=vmin_CS, vmax=vmax_CS)
    ax_CS.set_title("CS")
    fig.colorbar(im_CS, ax=ax_CS, fraction=0.046)

    dens = (1.0 - grid.fs[ys, xs]) * grid.CL[ys, xs] + grid.fs[ys, xs] * grid.CS[ys, xs]
    im_dens = ax_dens.imshow(dens, origin="lower", vmin=vmin_dens, vmax=vmax_dens)
    ax_dens.set_title("alpha*CL + (1-alpha)*CS")
    fig.colorbar(im_dens, ax=ax_dens, fraction=0.046)

    # 单独的守恒曲线
    fig2, axM = plt.subplots(figsize=(6, 3))
    (masses_t,) = axM.plot([], [], lw=2)
    axM.set_title("Total solute mass vs time")
    axM.set_xlabel("Time")
    axM.set_ylabel("Mass")
    axM.grid(True)
    times, masses = [], []
    M0 = total_solute_mass(grid)

    def init():
        masses_t.set_data([], [])
        return im_fs, im_CL, im_CS, im_dens, masses_t

    def step(_):
        nonlocal grid
        solute_advance(grid, cfg, dt, masks={}, fs_dot=fs_dot)
        update_ghosts(grid)

        # 更新影像
        im_fs.set_data(grid.fs[ys, xs])
        im_CL.set_data(grid.CL[ys, xs])
        im_CS.set_data(grid.CS[ys, xs])
        dens = (1.0 - grid.fs[ys, xs]) * grid.CL[ys, xs] + grid.fs[ys, xs] * grid.CS[
            ys, xs
        ]
        im_dens.set_data(dens)

        # 更新质量曲线
        t = (len(times) + 1) * dt
        times.append(t)
        masses.append(total_solute_mass(grid))
        masses_t.set_data(times, masses)
        axM.set_xlim(0, steps * dt)
        axM.set_ylim(M0 * 0.999, M0 * 1.001)

        fig.suptitle(
            f"Solute demo — t={t:.2f}, mass rel. change={(masses[-1]-M0)/M0:.2e}"
        )
        return im_fs, im_CL, im_CS, im_dens, masses_t

    ani = FuncAnimation(
        fig, step, init_func=init, frames=steps, interval=50, blit=False, repeat=False
    )
    plt.show()


if __name__ == "__main__":
    main()
