from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict


@dataclass
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
    ecc_x = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)
    ecc_y = _alloc(ny, nx, g, dtype=np.float64, fill=0.0)

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
        ecc_x=ecc_x,
        ecc_y=ecc_y,
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


def classify_phases(
    grid, tau_liq: float = 1e-12, tau_sol: float = 1.0 - 1e-12
) -> Dict[str, np.ndarray]:
    """
    【功能】基于 grid.fs 的三态掩码（液/界/固），返回包含 ghost 的布尔数组。
          统计或积分时请只在 core 区域使用这些掩码（例如 grid.core 或自行切片）。

    【输入】
    - grid: Grid
      需至少提供属性：
        - fs: np.ndarray   固相率场，shape=(ny, nx)
        - nghost: int      ghost 层厚度（仅用于调用方裁剪 core）
        - （可选）tau_liq/tau_sol: float 若 Grid 定义了，可覆盖默认阈值
    - tau_liq: float       视为“液相”的上阈（默认 1e-12）
    - tau_sol: float       视为“固相”的下阈（默认 1-1e-12）

    【输出】
    - masks: dict[str, np.ndarray]  （包含 ghost）
        必含键：
          - "mask_liq" | "mask_int" | "mask_sol"  # 保持你现有命名
        额外提供等价别名（便于新代码更简洁）：
          - "liq" | "intf" | "sol"

    【数值说明】
    - 阈值用于把 fs∈[0,1] 粗分为液/界/固；界面带 = ~(liq | sol)。
    - 若 grid 定义了 grid.tau_liq / grid.tau_sol，则优先使用之。
    """
    # 允许 Grid 覆盖默认阈值
    tl = getattr(grid, "tau_liq", tau_liq)
    ts = getattr(grid, "tau_sol", tau_sol)

    fs = grid.fs
    mask_liq = fs < tl
    mask_sol = fs > ts
    mask_int = ~(mask_liq | mask_sol)

    masks = {
        "mask_liq": mask_liq,
        "mask_int": mask_int,
        "mask_sol": mask_sol,
        # 别名（等价引用，不额外拷贝）
        "liq": mask_liq,
        "intf": mask_int,
        "sol": mask_sol,
    }
    return masks
