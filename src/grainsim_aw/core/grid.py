from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from typing import Dict
from typing import Mapping, Union
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
    nuc_x: np.ndarray
    nuc_y: np.ndarray

    # —— 网格几何 ——
    ny: int
    nx: int
    dx: float
    dy: float
    nghost: int

    # —— 相阈值（用于三态掩码）——
    tau_liq: float = 1e-12
    tau_sol: float = 1.0 - 1e-12

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
    tau_liq = float(domain_cfg.get("tau_liq", 1e-12))
    tau_sol = float(domain_cfg.get("tau_sol", 1.0 - 1e-12))

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
    nuc_x = _alloc(ny, nx, g, dtype=np.float64, fill=np.nan)
    nuc_y = _alloc(ny, nx, g, dtype=np.float64, fill=np.nan)

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
        ny=ny,
        nx=nx,
        dx=dx,
        dy=dy,
        nghost=g,
        tau_liq=tau_liq,
        tau_sol=tau_sol,
        nuc_x=nuc_x,
        nuc_y=nuc_y,
    )


def update_ghosts(grid: Grid, bc: Union[str, Mapping[str, str]] = "neumann0") -> None:
    """
    更新 ghost 带。
    支持：
      - "neumann0"：零法向梯度（偶延拓）
      - "periodic"：周期
    允许传入 dict：{"x": "...", "y": "..."} 分轴设置；未提供的轴沿用 "neumann0"。
    """
    g = grid.nghost
    if g == 0:
        return

    if isinstance(bc, str):
        bcx = bcy = bc
    else:
        bcx = bc.get("x", "neumann0")
        bcy = bc.get("y", "neumann0")

    # 需要处理的字段（若有不希望周期的字段，可在此排除或分开处理）
    fields = (
        grid.fs,
        grid.CL,
        grid.CS,
        grid.grain_id,
        grid.theta,
        grid.L_dia,
        grid.T,
        grid.nuc_x,
        grid.nuc_y,
    )

    for arr in fields:
        # 垂直方向（y）
        if bcy == "neumann0":
            # 顶部 ghost：取 [g:2g] 反向
            arr[:g, :] = arr[g : 2 * g, :][::-1, :]
            # 底部 ghost：取 [-2g:-g] 反向
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


def classify_phases(
    grid,
    tau_liq: Optional[float] = None,
    tau_sol: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """返回包含 ghost 的三态掩码：'liq' | 'intf' | 'sol'。"""
    tl = float(grid.tau_liq if tau_liq is None else tau_liq)
    ts = float(grid.tau_sol if tau_sol is None else tau_sol)

    fs = grid.fs
    mask_liq = fs < tl
    mask_sol = fs > ts
    mask_int = ~(mask_liq | mask_sol)

    # 保证布尔 dtype（以防上游传奇怪类型）
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
