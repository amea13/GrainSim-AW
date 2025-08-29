from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Mapping, Union
import numpy as np


@dataclass(slots=True)
class Grid:
    # ——持久字段（入快照/重启）——
    fs: np.ndarray  # 固相体积分数 [0,1]，float64
    CL: np.ndarray  # 液相体平均浓度，float64
    CS: np.ndarray  # 固相体平均浓度，float64
    grain_id: np.ndarray  # 晶粒 ID，int32
    theta: np.ndarray  # 晶粒取向角，float64
    L_dia: np.ndarray  # 偏心正方形“半对角线”长度，float64
    T: np.ndarray  # 温度场 [K]，float64（v0.2起持久字段）
    ecc_x: np.ndarray  # 偏心正方形中心相对本元胞几何中心的 x 偏移 [m]
    ecc_y: np.ndarray  # 同上 y 偏移 [m]

    # —— 几何坐标（绝对坐标，包含 ghost）——
    x: np.ndarray  # 元胞中心 x 坐标 [m]
    y: np.ndarray  # 元胞中心 y 坐标 [m]

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
    gid = _alloc(ny, nx, g, dtype=np.int32, fill=0)
    th = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    Ldia = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    T = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    ecc_x = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    ecc_y = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)

    # 绝对坐标（含 ghost）
    Ny = ny + 2 * g
    Nx = nx + 2 * g
    x_coords = (np.arange(Nx) - g + 0.5) * dx  # core 从 0.5*dx 开始
    y_coords = (np.arange(Ny) - g + 0.5) * dy
    xx, yy = np.meshgrid(x_coords, y_coords)  # 形状 (Ny, Nx)

    return Grid(
        fs=fs,
        CL=CL,
        CS=CS,
        grain_id=gid,
        theta=th,
        L_dia=Ldia,
        T=T,
        ecc_x=ecc_x,
        ecc_y=ecc_y,
        x=xx,
        y=yy,
        ny=ny,
        nx=nx,
        dx=dx,
        dy=dy,
        nghost=g,
    )


def update_ghosts(grid: Grid, bc: Union[str, Mapping[str, str]] = "neumann0") -> None:
    """
    更新 ghost 带：
      - "neumann0"：零法向梯度（偶延拓）
      - "periodic"：周期
    说明：几何坐标 grid.x/grid.y 不更新（固定绝对坐标）。
    """
    g = grid.nghost
    if g == 0:
        return

    if isinstance(bc, str):
        bcx = bcy = bc
    else:
        bcx = bc.get("x", "neumann0")
        bcy = bc.get("y", "neumann0")

    # 需要更新 ghost 的“场”
    fields = (grid.fs, grid.CL, grid.CS, grid.grain_id, grid.theta, grid.L_dia, grid.T)

    for arr in fields:
        # 垂直方向（y）
        if bcy == "neumann0":
            arr[:g, :] = arr[g : 2 * g, :][::-1, :]
            arr[-g:, :] = arr[-2 * g : -g, :][::-1, :]
        elif bcy == "periodic":
            arr[:g, :] = arr[-2 * g : -g, :]
            arr[-g:, :] = arr[g : 2 * g, :]
        else:
            raise ValueError(f"不支持的 y 方向边界：{bcy!r}")

        # 水平方向（x）
        if bcx == "neumann0":
            arr[:, :g] = arr[:, g : 2 * g][:, ::-1]
            arr[:, -g:] = arr[:, -2 * g : -g][:, ::-1]
        elif bcx == "periodic":
            arr[:, :g] = arr[:, -2 * g : -g]
            arr[:, -g:] = arr[:, g : 2 * g]
        else:
            raise ValueError(f"不支持的 x 方向边界：{bcx!r}")


def classify_phases(grid) -> Dict[str, np.ndarray]:
    """
    三态掩码（包含 ghost）：
      - 液相：fs == 0
      - 固相：fs == 1
      - 界面：其它
    """
    fs = grid.fs
    mask_liq = fs == 0.0
    mask_sol = fs == 1.0
    mask_int = ~(mask_liq | mask_sol)

    # 保持布尔 dtype
    mask_liq = np.asarray(mask_liq, dtype=bool)
    mask_sol = np.asarray(mask_sol, dtype=bool)
    mask_int = np.asarray(mask_int, dtype=bool)

    return {
        "mask_liq": mask_liq,
        "liq": mask_liq,
        "mask_int": mask_int,
        "intf": mask_int,
        "mask_sol": mask_sol,
        "sol": mask_sol,
    }
