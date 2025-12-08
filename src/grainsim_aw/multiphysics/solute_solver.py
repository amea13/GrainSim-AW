from typing import Dict, Optional
import numpy as np
from ..core.material import Dl_from_T, Ds_from_T


def _viewer(a: np.ndarray, pad: int = 1):
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
    fs = grid.fs
    CL = grid.CL
    CS = grid.CS
    T = grid.T
    dx = float(grid.dx)
    dy = float(grid.dy)
    g = int(grid.nghost)
    Ny, Nx = fs.shape

    # ------- 上一步场（precell；用于显式更新与步尾杠杆法） -------
    fs_prev = fs.copy()
    CL_prev = CL.copy()
    CS_prev = CS.copy()

    # ------- 物性 -------
    DL = Dl_from_T(T).astype(np.float64, copy=False)
    DS = Ds_from_T(T).astype(np.float64, copy=False)

    # ------- 常量/开关 -------
    k = float(cfg.get("k", 0.34))
    Vc = dx * dy
    fsdot_tol = float(cfg.get("fsdot_tol", 0.0))
    use_I_I = bool(cfg.get("use_I_I", True))  # 是否启用 I–I 衰减（式(5)），默认 False

    # ------- 相域判定（联合判定：几何 + 本步推进）-------
    fsdot_tol = float(cfg.get("fsdot_tol", 0.0))

    # 由推进速率判定的“本步界面”
    is_if_rate = np.abs(fs_dot) > fsdot_tol
    # 由几何判定的界面（步尾 fs 已更新到本步末）
    is_if_geom = (fs > 0.0) & (fs < 1.0)

    # 最终界面 = 两者并集
    is_interface = is_if_geom | is_if_rate

    # 纯液 / 纯固（纯固要排除“本步界面带”）
    is_liq = fs == 0.0
    sol_only = (fs == 1.0) & (~is_interface)

    # CL 的求解域 = 纯液 ∪ 界面
    liq_or_int = is_liq | is_interface

    # 三态标记仅用于“液-纯固硬闸门”
    sta = np.zeros_like(fs, dtype=np.int8)
    sta[is_liq] = -1  # 液
    sta[sol_only] = 1  # 纯固（非本步界面）
    # 剩下的是 0（界面）

    # ------- 邻居视图 -------
    V_sta = _viewer(sta, pad=1)
    V_CLp = _viewer(CL_prev, pad=1)
    V_CSp = _viewer(CS_prev, pad=1)
    V_fs = _viewer(fs, pad=1)
    V_int = _viewer(is_interface.astype(np.uint8), pad=1)
    V_sol = _viewer(sol_only.astype(np.uint8), pad=1)
    V_liq = _viewer(((fs == 0.0) & (~is_interface)).astype(np.uint8), pad=1)

    # 邻居状态（液/固用于硬闸门；纯固用于固相导通）
    nb_sol_E = V_sta(0, +1) == 1
    nb_sol_W = V_sta(0, -1) == 1
    nb_sol_N = V_sta(-1, 0) == 1
    nb_sol_S = V_sta(+1, 0) == 1

    nb_puresol_E = V_sol(0, +1) > 0
    nb_puresol_W = V_sol(0, -1) > 0
    nb_puresol_N = V_sol(-1, 0) > 0
    nb_puresol_S = V_sol(+1, 0) > 0

    # ------- 面导通系数 -------
    shape = fs.shape
    Clae0 = np.zeros(shape, dtype=np.float64)
    Claw0 = np.zeros(shape, dtype=np.float64)
    Clan0 = np.zeros(shape, dtype=np.float64)
    Clas0 = np.zeros(shape, dtype=np.float64)

    Csae0 = np.zeros(shape, dtype=np.float64)
    Csaw0 = np.zeros(shape, dtype=np.float64)
    Csan0 = np.zeros(shape, dtype=np.float64)
    Csas0 = np.zeros(shape, dtype=np.float64)

    # ==== 液/界面导通 ====
    base_ew_L = DL * dy / dx
    base_ns_L = DL * dx / dy

    fs_E, fs_W, fs_N, fs_S = V_fs(0, +1), V_fs(0, -1), V_fs(-1, 0), V_fs(+1, 0)
    int_E = V_int(0, +1) > 0
    int_W = V_int(0, -1) > 0
    int_N = V_int(-1, 0) > 0
    int_S = V_int(+1, 0) > 0
    liq_E = V_liq(0, +1) > 0
    liq_W = V_liq(0, -1) > 0
    liq_N = V_liq(-1, 0) > 0
    liq_S = V_liq(+1, 0) > 0

    is_I = is_interface
    is_L = (fs == 0.0) & (~is_interface)

    # 各方向衰减因子，先置 1
    fE = np.ones_like(fs, dtype=np.float64)
    fW = np.ones_like(fs, dtype=np.float64)
    fN = np.ones_like(fs, dtype=np.float64)
    fS = np.ones_like(fs, dtype=np.float64)

    # -- I–L：F_IL = 1/(1 + 0.5 * f_s^(界面格)) --
    mIL_E = (is_I & liq_E) | (is_L & int_E)
    mIL_W = (is_I & liq_W) | (is_L & int_W)
    mIL_N = (is_I & liq_N) | (is_L & int_N)
    mIL_S = (is_I & liq_S) | (is_L & int_S)

    fs_if_E = np.where(is_I & liq_E, fs, fs_E)
    fs_if_W = np.where(is_I & liq_W, fs, fs_W)
    fs_if_N = np.where(is_I & liq_N, fs, fs_N)
    fs_if_S = np.where(is_I & liq_S, fs, fs_S)

    fE[mIL_E] *= 1.0 / (1.0 + 0.5 * fs_if_E[mIL_E])
    fW[mIL_W] *= 1.0 / (1.0 + 0.5 * fs_if_W[mIL_W])
    fN[mIL_N] *= 1.0 / (1.0 + 0.5 * fs_if_N[mIL_N])
    fS[mIL_S] *= 1.0 / (1.0 + 0.5 * fs_if_S[mIL_S])

    # -- I–I：F_II = (1 - sigma^{IK}), sigma^{IK} = 0.5*(f_s^I + f_s^K)（可选） --
    if use_I_I:
        mII_E = is_I & int_E
        sigmaE = 0.5 * (fs + fs_E)
        mII_W = is_I & int_W
        sigmaW = 0.5 * (fs + fs_W)
        mII_N = is_I & int_N
        sigmaN = 0.5 * (fs + fs_N)
        mII_S = is_I & int_S
        sigmaS = 0.5 * (fs + fs_S)
        fE[mII_E] *= 1.0 - sigmaE[mII_E]
        fW[mII_W] *= 1.0 - sigmaW[mII_W]
        fN[mII_N] *= 1.0 - sigmaN[mII_N]
        fS[mII_S] *= 1.0 - sigmaS[mII_S]

    # 装配液/界面面导通（仅在液或界）
    Clae0[liq_or_int] = base_ew_L[liq_or_int] * fE[liq_or_int]
    Claw0[liq_or_int] = base_ew_L[liq_or_int] * fW[liq_or_int]
    Clan0[liq_or_int] = base_ns_L[liq_or_int] * fN[liq_or_int]
    Clas0[liq_or_int] = base_ns_L[liq_or_int] * fS[liq_or_int]

    # 液与纯固相邻 → 强制零通量
    Clae0[nb_sol_E] = 0.0
    Claw0[nb_sol_W] = 0.0
    Clan0[nb_sol_N] = 0.0
    Clas0[nb_sol_S] = 0.0

    # ==== 固相导通（仅纯固，且邻居也必须纯固） ====
    base_ew_S = DS * dy / dx
    base_ns_S = DS * dx / dy
    Csae0[sol_only] = base_ew_S[sol_only]
    Csaw0[sol_only] = base_ew_S[sol_only]
    Csan0[sol_only] = base_ns_S[sol_only]
    Csas0[sol_only] = base_ns_S[sol_only]
    Csae0[~nb_puresol_E] = 0.0
    Csaw0[~nb_puresol_W] = 0.0
    Csan0[~nb_puresol_N] = 0.0
    Csas0[~nb_puresol_S] = 0.0

    # ==== 外边界零通量（第一圈 core 单元清零导通） ====
    is_beg, is_end = g, Ny - g
    js_beg, js_end = g, Nx - g
    iT, iB = is_beg, is_end - 1
    jL, jR = js_beg, js_end - 1

    # 液相边界
    Claw0[is_beg:is_end, jL] = 0.0
    Clae0[is_beg:is_end, jR] = 0.0
    Clan0[iT, js_beg:js_end] = 0.0
    Clas0[iB, js_beg:js_end] = 0.0
    # 固相边界
    Csaw0[is_beg:is_end, jL] = 0.0
    Csae0[is_beg:is_end, jR] = 0.0
    Csan0[iT, js_beg:js_end] = 0.0
    Csas0[iB, js_beg:js_end] = 0.0

    # ------- 时间项与 p0 -------
    Clap1 = np.zeros(shape, dtype=np.float64)
    Csap1 = np.zeros(shape, dtype=np.float64)
    Clap1[liq_or_int] = Vc / dt
    Csap1[sol_only] = Vc / dt

    Clap0 = Clap1 - (Clae0 + Claw0 + Clan0 + Clas0)
    Csap0 = Csap1 - (Csae0 + Csaw0 + Csan0 + Csas0)

    # ------- 源（仅液/界）：(1-k) * C_L^n * fs_dot * Vc -------
    Clbp = CL_prev * (1.0 - k) * fs_dot * Vc

    # ------- 邻居浓度（上一步） -------
    CLp_E, CLp_W = V_CLp(0, +1), V_CLp(0, -1)
    CLp_N, CLp_S = V_CLp(-1, 0), V_CLp(+1, 0)
    CSp_E, CSp_W = V_CSp(0, +1), V_CSp(0, -1)
    CSp_N, CSp_S = V_CSp(-1, 0), V_CSp(+1, 0)

    c0, c1 = slice(is_beg, is_end), slice(js_beg, js_end)

    # —— CL ——
    num_CL = (
        Clae0 * CLp_E
        + Claw0 * CLp_W
        + Clan0 * CLp_N
        + Clas0 * CLp_S
        + Clap0 * CL_prev
        + Clbp
    )
    den_CL = Clap1 + 1e-300

    CL_core = CL[c0, c1].copy()
    mL_core = liq_or_int[c0, c1]
    tmp_CL = num_CL[c0, c1] / den_CL[c0, c1]
    CL_core[mL_core] = tmp_CL[mL_core]
    CL[c0, c1] = CL_core  # 回填

    # —— CS ——
    num_CS = (
        Csae0 * CSp_E + Csaw0 * CSp_W + Csan0 * CSp_N + Csas0 * CSp_S + Csap0 * CS_prev
    )
    den_CS = Csap1 + 1e-300

    CS_core = CS[c0, c1].copy()
    mS_core = sol_only[c0, c1]
    tmp_CS = num_CS[c0, c1] / den_CS[c0, c1]
    CS_core[mS_core] = tmp_CS[mS_core]
    CS[c0, c1] = CS_core  # 回填

    # ------- 步尾：杠杆法 & fs 写回 -------
    delta_fs = fs_dot * dt
    fs_new = np.clip(fs_prev + delta_fs, 0.0, 1.0)
    delta_eff = fs_new - fs_prev

    mask_int = is_interface

    grow = delta_eff > 0.0
    upd = mask_int & grow
    denom = fs_prev + delta_eff
    valid = upd & (denom > 0.0)

    CS[valid] = (
        CS_prev[valid] * fs_prev[valid] + k * CL_prev[valid] * delta_eff[valid]
    ) / denom[valid]

    fs[:] = fs_new
    CL[mask_int & (fs >= 1.0)] = 0.0
