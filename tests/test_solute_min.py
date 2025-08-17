import numpy as np

# 改成你的真实导入路径
from grainsim_aw.multiphysics.solute_solver import solute_advance


class GridStub:
    """只保留 solute_advance 需要的字段"""

    def __init__(self, ny=64, nx=64, dx=1.0, dy=1.0, nghost=2, T0=1800.0):
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
    """镜像外推，实现零法向梯度边界"""
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
    M = np.sum(alpha[core] * grid.CL[core] + (1.0 - alpha[core]) * grid.CS[core]) * cell
    return float(M)


def case1_pure_diffusion_no_source():
    """全液相纯扩散，检验零源守恒"""
    grid = GridStub(ny=64, nx=64, dx=1e-6, dy=1e-6, T0=1800.0)
    g = grid.nghost
    ys, xs = grid.core

    # 全液相 fs = 0
    grid.fs.fill(0.0)
    # 初始 CL 做个二维高斯包，CS=0
    Y, X = np.mgrid[0 : grid.ny, 0 : grid.nx]
    X = (X - grid.nx / 2) / 10.0
    Y = (Y - grid.ny / 2) / 10.0
    CL0 = np.exp(-(X**2 + Y**2))
    grid.CL[ys, xs] = CL0
    grid.CS[ys, xs] = 0.0
    update_ghosts(grid)

    M0 = total_solute_mass(grid)
    dt = 1e-3
    fs_dot = np.zeros_like(grid.fs)

    cfg = {
        "k": 0.34,
        "solver": {"max_iter": 80, "tol": 1e-10},
        "eps": 1e-12,
        "clip": {"min": 0.0},
    }

    # 跑 50 步，纯扩散应该平滑而总量不变
    for _ in range(50):
        solute_advance(grid, cfg, dt, masks={}, fs_dot=fs_dot)
        update_ghosts(grid)
    M1 = total_solute_mass(grid)
    print("[Case1] Mass relative change:", (M1 - M0) / max(M0, 1e-300))


def case2_sharp_interface_no_leak():
    """左半为液相右半为固相，检验闸门抑制跨纯相泄漏"""
    grid = GridStub(ny=64, nx=64, dx=1.0, dy=1.0, T0=1800.0)
    ys, xs = grid.core
    g = grid.nghost

    # 左侧全液相 fs=0，右侧全固相 fs=1
    grid.fs[ys, xs] = 0.0
    grid.fs[ys, xs][:, grid.nx // 2 :] = 1.0

    # 只在液相侧给一块高浓度 CL，固相侧 CS=0
    grid.CL[ys, xs] = 0.0
    grid.CL[ys, xs][:, : grid.nx // 2] = 1.0
    grid.CS[ys, xs] = 0.0
    update_ghosts(grid)

    M0 = total_solute_mass(grid)
    dt = 1e-2
    fs_dot = np.zeros_like(grid.fs)

    cfg = {"solver": {"max_iter": 80, "tol": 1e-10}}

    # 跑 100 步，若闸门正确，右侧固相不会收到左侧液相扩散的泄漏
    for _ in range(100):
        solute_advance(grid, cfg, dt, masks={}, fs_dot=fs_dot)
        update_ghosts(grid)

    M1 = total_solute_mass(grid)
    # 右侧固相仍应保持接近零
    max_CS_right = grid.CS[ys, xs][:, grid.nx // 2 :].max()
    print(
        "[Case2] Mass rel. change:",
        (M1 - M0) / max(M0, 1e-300),
        " max CS right half:",
        max_CS_right,
    )


def case3_interface_pair_source_conservation():
    """给一个人工界面带和 fs_dot>0，检验成对源汇守恒与趋势"""
    grid = GridStub(ny=64, nx=64, dx=1.0, dy=1.0, T0=1800.0)
    ys, xs = grid.core

    # 构造一条竖直界面带：fs≈0.5，宽度 w=3 格
    fs = np.zeros((grid.ny, grid.nx))
    j0 = grid.nx // 2
    w = 3
    fs[:, j0 - w : j0 + w + 1] = 0.5
    fs[:, j0 + w + 1 :] = 1.0
    grid.fs[ys, xs] = fs

    # 初始 CL/CS 都是常数 C0
    C0 = 0.05
    grid.CL[ys, xs] = C0
    grid.CS[ys, xs] = C0
    update_ghosts(grid)

    # 在界面带内给一个正的 fs_dot，模拟固化推进
    fs_dot = np.zeros_like(grid.fs)
    mask_band = np.zeros((grid.ny, grid.nx), dtype=bool)
    mask_band[:, j0 - w : j0 + w + 1] = True
    fs_dot_band = np.zeros((grid.ny, grid.nx))
    fs_dot_band[mask_band] = 5e-3  # s^-1
    fs_dot[ys, xs] = fs_dot_band

    cfg = {
        "k": 0.34,
        "solver": {"max_iter": 80, "tol": 1e-10},
        "eps": 1e-12,
        "clip": {"min": 0.0},
    }
    dt = 1e-1  # 选小一点，确保 alpha_n 仍落在 [0,1]

    M0 = total_solute_mass(grid)
    solute_advance(grid, cfg, dt, masks={}, fs_dot=fs_dot)
    update_ghosts(grid)
    M1 = total_solute_mass(grid)

    # 期望：总量守恒，界面带附近液相 CL 上升，固相 CS 降低
    ys_c, xs_c = grid.core
    CL_band_mean = grid.CL[ys_c, xs_c][:, j0 - w : j0 + w + 1].mean()
    CS_band_mean = grid.CS[ys_c, xs_c][:, j0 - w : j0 + w + 1].mean()
    print(
        "[Case3] Mass rel. change:",
        (M1 - M0) / max(M0, 1e-300),
        " CL_band_mean:",
        CL_band_mean,
        " CS_band_mean:",
        CS_band_mean,
    )


if __name__ == "__main__":
    case1_pure_diffusion_no_source()
    case2_sharp_interface_no_leak()
    case3_interface_pair_source_conservation()
