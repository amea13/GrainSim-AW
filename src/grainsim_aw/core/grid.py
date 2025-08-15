from dataclasses import dataclass
import numpy as np


@dataclass
class Grid:
    fs: np.ndarray
    CL: np.ndarray
    CS: np.ndarray
    ny: int
    nx: int
    dx: float
    dy: float
    nghost: int

    @property
    def Ny(self):
        return self.ny + 2 * self.nghost

    @property
    def Nx(self):
        return self.nx + 2 * self.nghost

    @property
    def core(self):
        g = self.nghost
        return slice(g, -g), slice(g, -g)


def create_grid(domain_cfg: dict) -> Grid:
    ny, nx = domain_cfg["ny"], domain_cfg["nx"]
    dx, dy = domain_cfg["dx"], domain_cfg["dy"]
    g = domain_cfg.get("nghost", 3)
    fs = np.zeros((ny + 2 * g, nx + 2 * g), dtype=np.float64)
    CL = np.zeros_like(fs)
    CS = np.zeros_like(fs)
    return Grid(fs=fs, CL=CL, CS=CS, ny=ny, nx=nx, dx=dx, dy=dy, nghost=g)


def update_ghosts(grid: Grid, bc: str = "neumann0"):
    """极简版：只支持 neumann0(镜像)。"""
    g = grid.nghost
    if g == 0:
        return
    for arr in (grid.fs, grid.CL, grid.CS):
        # 上下
        arr[:g, :] = arr[g : 2 * g, :]
        arr[-g:, :] = arr[-2 * g : -g, :]
        # 左右
        arr[:, :g] = arr[:, g : 2 * g]
        arr[:, -g:] = arr[:, -2 * g : -g]


def classify_phases(fs: np.ndarray, nghost: int, tau_liq=1e-12, tau_sol=1 - 1e-12):
    """基于 fs 的三态掩码(液/界/固)，只返回包含 ghost 的布尔数组。"""
    g = nghost
    mask_liq = fs < tau_liq
    mask_sol = fs > tau_sol
    mask_int = ~(mask_liq | mask_sol)
    # 清掉 ghost 的统计副作用由调用方控制；这里按全域掩码返回
    return {"mask_liq": mask_liq, "mask_int": mask_int, "mask_sol": mask_sol}
