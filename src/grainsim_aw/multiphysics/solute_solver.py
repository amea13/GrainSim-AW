from typing import Dict, Tuple, Optional
import numpy as np
from ..core.material import Dl_from_T, Ds_from_T  # 你已有


def step_solute(
    grid,
    cfg: Dict,
    dt: float,
    masks: Dict[str, np.ndarray],
    fs_dot: np.ndarray,  # 本步 df_s/dt（与 grid 形状一致，含 ghost）
) -> None:
    """
    纯 C++ 风格推进溶质场（显式一次迭代）：
      - 计算液相/固相导通系数（Cla/Csa）
      - 应用域边界无扩散（Cl0）
      - 计算 aCl0（Clap0/Csap0）
      - 用 precell（上一步浓度）显式更新 Cl/Cs
    不做任何数值保护或裁剪；依赖 ghost 可用；要求 nghost ≥ 1。
    """

    fs = grid.fs
    CL = grid.CL
    CS = grid.CS
    T = grid.T
    dx = float(grid.dx)
    dy = float(grid.dy)
    g = int(grid.nghost)

    Ny, Nx = fs.shape

    # ---- 显式所需：precell（上一时刻） ----
    CL_prev = CL.copy()
    CS_prev = CS.copy()

    # ---- Dl/Ds 取中心温度 ----
    DL = Dl_from_T(T)
    DS = Ds_from_T(T)

    # ---- k 与体元体积 ----
    k = float(cfg.get("k", 0.34))
    Vc = dx * dy

    # ---- sta 从 fs 判定：-1 液相 / 0 界面 / 1 固相 ----
    sta = np.zeros_like(fs, dtype=np.int8)
    sta[fs == 0.0] = -1
    sta[fs == 1.0] = 1
    # 其余保持 0

    # ---- 系数场（全域分配，逐格写）----
    Clae0 = np.zeros_like(fs, dtype=np.float64)
    Claw0 = np.zeros_like(fs, dtype=np.float64)
    Clan0 = np.zeros_like(fs, dtype=np.float64)
    Clas0 = np.zeros_like(fs, dtype=np.float64)
    Clap1 = np.zeros_like(fs, dtype=np.float64)
    Clap0 = np.zeros_like(fs, dtype=np.float64)
    Clbp = np.zeros_like(fs, dtype=np.float64)

    Csae0 = np.zeros_like(fs, dtype=np.float64)
    Csaw0 = np.zeros_like(fs, dtype=np.float64)
    Csan0 = np.zeros_like(fs, dtype=np.float64)
    Csas0 = np.zeros_like(fs, dtype=np.float64)
    Csap1 = np.zeros_like(fs, dtype=np.float64)
    Csap0 = np.zeros_like(fs, dtype=np.float64)

    # 方便：核心区索引范围（含边界第一圈）
    is_beg, is_end = g, Ny - g
    js_beg, js_end = g, Nx - g

    # ========================
    # 1) Cla：液相系数 + 源项 Clbp
    # ========================
    for i in range(is_beg, is_end):
        im, ip = i - 1, i + 1
        for j in range(js_beg, js_end):
            jm, jp = j - 1, j + 1

            if sta[i, j] == -1 or sta[i, j] == 0:
                # 初值
                Clae0[i, j] = DL[i, j] * dy / dx
                Claw0[i, j] = DL[i, j] * dy / dx
                Clan0[i, j] = DL[i, j] * dx / dy
                Clas0[i, j] = DL[i, j] * dx / dy

                # 邻居为固相 → 对应面导通置零
                if sta[ip, j] == 1:
                    Clas0[i, j] = 0.0
                if sta[im, j] == 1:
                    Clan0[i, j] = 0.0
                if sta[i, jp] == 1:
                    Clae0[i, j] = 0.0
                if sta[i, jm] == 1:
                    Claw0[i, j] = 0.0

                # 系数 p1 / p0
                Clap1[i, j] = Vc / dt
                Clap0[i, j] = (
                    Clap1[i, j] - Clae0[i, j] - Claw0[i, j] - Clan0[i, j] - Clas0[i, j]
                )

                # 成对源：Clbp = Cl * (1-k) * delta_fs * dx*dy / dt
                # 其中 delta_fs = fs_dot * dt
                Clbp[i, j] = CL[i, j] * (1.0 - k) * fs_dot[i, j] * Vc

    # ========================
    # 2) Csa：固相系数
    # ========================
    for i in range(is_beg, is_end):
        im, ip = i - 1, i + 1
        for j in range(js_beg, js_end):
            jm, jp = j - 1, j + 1

            if sta[i, j] == 1 or sta[i, j] == 0:
                # 初值
                Csae0[i, j] = DS[i, j] * dy / dx
                Csaw0[i, j] = DS[i, j] * dy / dx
                Csan0[i, j] = DS[i, j] * dx / dy
                Csas0[i, j] = DS[i, j] * dx / dy

                # 邻居为液相 → 对应面导通置零
                if sta[ip, j] == -1:
                    Csas0[i, j] = 0.0
                if sta[im, j] == -1:
                    Csan0[i, j] = 0.0
                if sta[i, jp] == -1:
                    Csae0[i, j] = 0.0
                if sta[i, jm] == -1:
                    Csaw0[i, j] = 0.0

                # 系数 p1 / p0
                Csap1[i, j] = Vc / dt
                Csap0[i, j] = (
                    Csap1[i, j] - Csae0[i, j] - Csaw0[i, j] - Csan0[i, j] - Csas0[i, j]
                )

    # ========================
    # 3) 域边界无扩散（等价于 C++ 的 Cl0()）
    #    在“第一圈 core 单元”把面导通清零
    # ========================
    # 左/右边界：j = g / j = Nx-g-1
    jL = js_beg
    jR = js_end - 1
    for i in range(is_beg, is_end):
        # 液相系数
        Claw0[i, jL] = 0.0
        Clae0[i, jR] = 0.0
        # 固相系数
        Csaw0[i, jL] = 0.0
        Csae0[i, jR] = 0.0

    # 上/下边界：i = g / i = Ny-g-1
    iT = is_beg
    iB = is_end - 1
    for j in range(js_beg, js_end):
        # 液相系数
        Clan0[iT, j] = 0.0
        Clas0[iB, j] = 0.0
        # 固相系数
        Csan0[iT, j] = 0.0
        Csas0[iB, j] = 0.0

    # ========================
    # 4) 重新计算 Clap0 / Csap0（等价 aCl0()）
    # ========================
    for i in range(is_beg, is_end):
        for j in range(js_beg, js_end):
            Clap0[i, j] = (
                Clap1[i, j] - Clae0[i, j] - Claw0[i, j] - Clan0[i, j] - Clas0[i, j]
            )
            Csap0[i, j] = (
                Csap1[i, j] - Csae0[i, j] - Csaw0[i, j] - Csan0[i, j] - Csas0[i, j]
            )

    # ========================
    # 5) 显式更新：Cl()
    # ========================
    for i in range(is_beg, is_end):
        im, ip = i - 1, i + 1
        for j in range(js_beg, js_end):
            jm, jp = j - 1, j + 1

            if sta[i, j] == -1 or sta[i, j] == 0:
                num = (
                    Clae0[i, j] * CL_prev[i, jp]
                    + Claw0[i, j] * CL_prev[i, jm]
                    + Clan0[i, j] * CL_prev[im, j]
                    + Clas0[i, j] * CL_prev[ip, j]
                    + Clap0[i, j] * CL_prev[i, j]
                    + Clbp[i, j]
                )
                CL[i, j] = num / Clap1[i, j]

    # ========================
    # 6) 显式更新：Cs()
    # ========================
    for i in range(is_beg, is_end):
        im, ip = i - 1, i + 1
        for j in range(js_beg, js_end):
            jm, jp = j - 1, j + 1

            if sta[i, j] == 1 or sta[i, j] == 0:
                num = (
                    Csae0[i, j] * CS_prev[i, jp]
                    + Csaw0[i, j] * CS_prev[i, jm]
                    + Csan0[i, j] * CS_prev[im, j]
                    + Csas0[i, j] * CS_prev[ip, j]
                    + Csap0[i, j] * CS_prev[i, j]
                )
                CS[i, j] = num / Csap1[i, j]
