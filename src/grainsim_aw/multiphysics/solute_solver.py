from typing import Dict, Tuple, Optional
import numpy as np
from ..core.material import Dl_from_T, Ds_from_T


def _viewer(a: np.ndarray, pad: int = 1):
    """非环绕平移视图：越界补 0（只用在取邻居时；核心区不受影响）。"""
    Ny, Nx = a.shape
    ap = np.pad(a, pad_width=pad, mode="constant", constant_values=0.0)
    base = pad

    def V(di: int, dj: int) -> np.ndarray:
        return ap[base + di : base + di + Ny, base + dj : base + dj + Nx]

    return V


def step_solute(
    grid,
    cfg: Dict,
    dt: float,
    masks: Dict[str, np.ndarray],
    fs_dot: np.ndarray,  # 本步 df_s/dt（与 grid 形状一致，含 ghost）
) -> None:
    """
    显式一次推进（与原 C++/Python 循环版逐元素等价）：
      - 计算 Cla/Csa 与成对源 Clbp
      - 第一圈 core 单元施加无扩散边界（清零对应面导通）
      - 重新计算 Clap0/Csap0
      - 用上一步浓度（precell）显式更新 CL/CS
    不做数值保护或裁剪；依赖 ghost 可用；要求 nghost ≥ 1。
    """
    fs = grid.fs
    CL = grid.CL
    CS = grid.CS
    T = grid.T
    dx = float(grid.dx)
    dy = float(grid.dy)
    g = int(grid.nghost)
    Ny, Nx = fs.shape

    # —— 上一步浓度（precell） ——
    CL_prev = CL.copy()
    CS_prev = CS.copy()

    # —— 物性（中心温度） ——
    DL = Dl_from_T(T).astype(np.float64, copy=False)
    DS = Ds_from_T(T).astype(np.float64, copy=False)

    # —— 常量系数 ——
    k = float(cfg.get("k", 0.34))
    Vc = dx * dy

    # —— 相位标记：-1 液 / 0 界面 / 1 固 ——
    sta = np.zeros_like(fs, dtype=np.int8)
    sta[fs == 0.0] = -1
    sta[fs == 1.0] = 1
    liq_or_int = sta <= 0  # -1 或 0
    sol_or_int = sta >= 0  #  0 或 1

    # —— 邻居视图（一次 pad） ——
    V_sta = _viewer(sta, pad=1)
    V_CLp = _viewer(CL_prev, pad=1)
    V_CSp = _viewer(CS_prev, pad=1)

    # 邻居状态布尔
    nb_sol_E = V_sta(0, +1) == 1
    nb_sol_W = V_sta(0, -1) == 1
    nb_sol_N = V_sta(-1, 0) == 1
    nb_sol_S = V_sta(+1, 0) == 1

    nb_liq_E = V_sta(0, +1) == -1
    nb_liq_W = V_sta(0, -1) == -1
    nb_liq_N = V_sta(-1, 0) == -1
    nb_liq_S = V_sta(+1, 0) == -1

    # —— 面导通系数（初始化为 0，再按相位填） ——
    shape = fs.shape
    Clae0 = np.zeros(shape, dtype=np.float64)  # 东
    Claw0 = np.zeros(shape, dtype=np.float64)  # 西
    Clan0 = np.zeros(shape, dtype=np.float64)  # 北
    Clas0 = np.zeros(shape, dtype=np.float64)  # 南

    Csae0 = np.zeros(shape, dtype=np.float64)
    Csaw0 = np.zeros(shape, dtype=np.float64)
    Csan0 = np.zeros(shape, dtype=np.float64)
    Csas0 = np.zeros(shape, dtype=np.float64)

    # 液/界面：初值
    base_ew_L = DL * dy / dx
    base_ns_L = DL * dx / dy
    Clae0[liq_or_int] = base_ew_L[liq_or_int]
    Claw0[liq_or_int] = base_ew_L[liq_or_int]
    Clan0[liq_or_int] = base_ns_L[liq_or_int]
    Clas0[liq_or_int] = base_ns_L[liq_or_int]
    # 邻居为固 → 清零对应面
    Clae0[nb_sol_E] = 0.0
    Claw0[nb_sol_W] = 0.0
    Clan0[nb_sol_N] = 0.0
    Clas0[nb_sol_S] = 0.0

    # 固/界面：初值
    base_ew_S = DS * dy / dx
    base_ns_S = DS * dx / dy
    Csae0[sol_or_int] = base_ew_S[sol_or_int]
    Csaw0[sol_or_int] = base_ew_S[sol_or_int]
    Csan0[sol_or_int] = base_ns_S[sol_or_int]
    Csas0[sol_or_int] = base_ns_S[sol_or_int]
    # 邻居为液 → 清零对应面
    Csae0[nb_liq_E] = 0.0
    Csaw0[nb_liq_W] = 0.0
    Csan0[nb_liq_N] = 0.0
    Csas0[nb_liq_S] = 0.0

    # —— 系数 p1 / p0（初始化） ——
    Clap1 = np.zeros(shape, dtype=np.float64)
    Csap1 = np.zeros(shape, dtype=np.float64)
    Clap1[liq_or_int] = Vc / dt
    Csap1[sol_or_int] = Vc / dt

    # 初值下的 p0（稍后边界清零后还会重算一遍，等价于原代码）
    Clap0 = Clap1 - (Clae0 + Claw0 + Clan0 + Clas0)
    Csap0 = Csap1 - (Csae0 + Csaw0 + Csan0 + Csas0)

    # —— 第一圈 core 单元的无扩散边界（等价 Cl0()） ——
    is_beg, is_end = g, Ny - g
    js_beg, js_end = g, Nx - g
    iT, iB = is_beg, is_end - 1
    jL, jR = js_beg, js_end - 1

    # 左/右边界：清零西/东面
    Claw0[is_beg:is_end, jL] = 0.0
    Clae0[is_beg:is_end, jR] = 0.0
    Csaw0[is_beg:is_end, jL] = 0.0
    Csae0[is_beg:is_end, jR] = 0.0

    # 上/下边界：清零北/南面
    Clan0[iT, js_beg:js_end] = 0.0
    Clas0[iB, js_beg:js_end] = 0.0
    Csan0[iT, js_beg:js_end] = 0.0
    Csas0[iB, js_beg:js_end] = 0.0

    # —— 边界处理后重新计算 p0（等价 aCl0()） ——
    Clap0 = Clap1 - (Clae0 + Claw0 + Clan0 + Clas0)
    Csap0 = Csap1 - (Csae0 + Csaw0 + Csan0 + Csas0)

    # —— 成对源：Clbp = CL * (1-k) * fs_dot * Vc（此时 CL 仍是上一步值） ——
    Clbp = CL_prev * (1.0 - k) * fs_dot * Vc

    # —— 邻居浓度（上一步） ——
    CLp_E, CLp_W = V_CLp(0, +1), V_CLp(0, -1)
    CLp_N, CLp_S = V_CLp(-1, 0), V_CLp(+1, 0)

    CSp_E, CSp_W = V_CSp(0, +1), V_CSp(0, -1)
    CSp_N, CSp_S = V_CSp(-1, 0), V_CSp(+1, 0)

    # —— 显式更新：仅 core 区（等价于 i=g..Ny-g-1, j=g..Nx-g-1） ——
    core = (slice(is_beg, is_end), slice(js_beg, js_end))

    # Cl：只在液/界面（liq_or_int）位置更新
    num_CL = (
        Clae0 * CLp_E
        + Claw0 * CLp_W
        + Clan0 * CLp_N
        + Clas0 * CLp_S
        + Clap0 * CL_prev
        + Clbp
    )
    m_CL = liq_or_int & False  # 占位，下面切 core 再筛
    # 写回（避免非 core 位置）
    c0, c1 = core
    m_liq_core = liq_or_int[c0, c1]
    CL[c0, c1][m_liq_core] = num_CL[c0, c1][m_liq_core] / Clap1[c0, c1][m_liq_core]

    # Cs：只在固/界面（sol_or_int）位置更新
    num_CS = (
        Csae0 * CSp_E + Csaw0 * CSp_W + Csan0 * CSp_N + Csas0 * CSp_S + Csap0 * CS_prev
    )
    m_sol_core = sol_or_int[c0, c1]
    CS[c0, c1][m_sol_core] = num_CS[c0, c1][m_sol_core] / Csap1[c0, c1][m_sol_core]
