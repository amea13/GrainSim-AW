from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Grid:
    # ——持久字段（入快照/重启）——
    fs: np.ndarray  # 固相体积分数 [0,1]，float64
    CL: np.ndarray  # 液相体平均浓度，float64
    CS: np.ndarray  # 固相体平均浓度，float64
    grain_id: np.ndarray  # 晶粒 ID，int32
    theta: np.ndarray  # 晶粒取向角（弧度，规范到[0,2π)），float64
    L_dia: np.ndarray  # 偏心正方形“半对角线”长度，float64
    T: np.ndarray  # 温度场 [K]，float64（v0.2起持久字段）

    # —— 网格几何 ——
    ny: int
    nx: int
    dx: float
    dy: float
    nghost: int

    # —— 便捷属性 ——
    @property
    def Ny(self) -> int:
        return self.ny + 2 * self.nghost

    @property
    def Nx(self) -> int:
        return self.nx + 2 * self.nghost

    @property
    def core(self):
        """返回 core（不含 ghost）的二维切片 (ys, xs)。"""
        g = self.nghost
        return slice(g, -g), slice(g, -g)

    @property
    def shape(self):
        return self.fs.shape  # = (Ny, Nx)


# —— 工具：统一分配 (Ny,Nx) 数组 ——
def _alloc(ny: int, nx: int, nghost: int, *, dtype, fill=0.0):
    Ny = ny + 2 * nghost
    Nx = nx + 2 * nghost
    return np.full((Ny, Nx), fill_value=fill, dtype=dtype)


def create_grid(domain_cfg: dict) -> Grid:
    ny, nx = int(domain_cfg["ny"]), int(domain_cfg["nx"])
    dx, dy = float(domain_cfg["dx"]), float(domain_cfg["dy"])
    g = int(domain_cfg.get("nghost", 3))

    # 持久字段统一初始化
    fs = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    CL = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    CS = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    gid = _alloc(ny, nx, g, dtype=np.int32, fill=0)  # 0=未分配
    th = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)  # 取向角
    Ldia = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    T = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)  # 温度场

    return Grid(
        fs=fs,
        CL=CL,
        CS=CS,
        grain_id=gid,
        theta=th,
        L_dia=Ldia,
        T=T,
        ny=ny,
        nx=nx,
        dx=dx,
        dy=dy,
        nghost=g,
    )


def update_ghosts(grid: Grid, bc: str = "neumann0"):
    """极简版：只支持 neumann0(镜像)。"""
    g = grid.nghost
    if g == 0:
        return
    fields = (grid.fs, grid.CL, grid.CS, grid.grain_id, grid.theta, grid.L_dia, grid.T)
    for arr in fields:
        # 上下
        arr[:g, :] = arr[g : 2 * g, :]
        arr[-g:, :] = arr[-2 * g : -g, :]
        # 左右
        arr[:, :g] = arr[:, g : 2 * g]
        arr[:, -g:] = arr[:, -2 * g : -g]


def classify_phases(fs: np.ndarray, nghost: int, tau_liq=1e-12, tau_sol=1 - 1e-12):
    """
    基于 fs 的三态掩码（液/界/固），返回包含 ghost 的布尔数组。
    注意：统计或积分时请只在 grid.core 区域使用这些掩码。
    """
    mask_liq = fs < tau_liq
    mask_sol = fs > tau_sol
    mask_int = ~(mask_liq | mask_sol)
    return {"mask_liq": mask_liq, "mask_int": mask_int, "mask_sol": mask_sol}
