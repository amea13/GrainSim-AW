from __future__ import annotations
from typing import Dict, Any
import numpy as np


def _normalize_angle(theta: np.ndarray) -> np.ndarray:
    """把角度规范到 [0, 2π)。"""
    twopi = 2.0 * np.pi
    return np.mod(theta, twopi)


def apply(
    grid,
    rng: np.random.Generator,
    cfg_nucl: Dict[str, Any],
    masks: Dict[str, np.ndarray],
) -> None:
    """
    形核入口
    参数
    ----
    grid : Grid
        持久状态容器（含 fs/CL/CS/grain_id/theta/L_dia/T 等）。
    rng : np.random.Generator
        统一注入的随机数源（保证可复现）。
    cfg_nucl : dict
        形核配置（此阶段允许为空 dict）。
    masks : dict
        相态掩码，至少包含 'mask_liq'、'mask_int'、'mask_sol'。

    最小可用形核：
    - 条件区域：液相且满足过冷条件；
    - 每个候选元胞以 p=prob_per_cell 的概率命中（独立伯努利）；
    - 命中元胞赋予递增 grain_id、取向 theta，并可选微增 fs。
    ----
    None（就地更新 grid；当前占位版不做任何修改）
    """

    gslice = grid.core  # (slice_y, slice_x)
    liq_core = masks["mask_liq"][gslice]

    gid = grid.grain_id[gslice]
    theta = grid.theta[gslice]
    fs = grid.fs[gslice]
    T = grid.T[gslice]  # ← 直接从持久字段读取温度

    # 1) 读取配置（给出合理默认）
    # 过冷判据：优先用 ΔT = T_L_eq - T ≥ dT_threshold；否则用温度阈值 T ≤ T_threshold；都缺则默认全候选
    dT_threshold = float(cfg_nucl.get("dT_threshold", np.nan))
    T_threshold = cfg_nucl.get("T_threshold", None)
    T_L_eq = cfg_nucl.get("T_L_eq", None)  # 也可来自 physics.interface
    prob_per_cell = float(
        cfg_nucl.get("prob_per_cell", cfg_nucl.get("rate_per_cell", 0.0))
    )
    prob_per_cell = max(0.0, min(1.0, prob_per_cell))  # 限幅到 [0,1]
    seed_fs = float(cfg_nucl.get("seed_fs", 0.0))
    fold = int(cfg_nucl.get("fold", 4))  # 取向相关参数兜底
    Ntheta = int(cfg_nucl.get("N_theta", 48))

    # 2) 过冷判定掩码
    if not np.isnan(dT_threshold) and (T_L_eq is not None):
        dT = float(T_L_eq) - T
        eligible = liq_core & (dT >= dT_threshold)
    elif T_threshold is not None:
        eligible = liq_core & (T <= float(T_threshold))
    else:
        eligible = liq_core.copy()

    # 3) 过滤掉已经成核（已有 grain_id）或 fs 已非液的格点
    tau_liq = 1e-12  # 与 classify_phases 的液阈值保持一致
    eligible &= (gid == 0) & (fs < tau_liq)

    num_eligible = int(np.count_nonzero(eligible))
    current_total = (
        int(np.max(grid.grain_id[gslice])) if np.any(grid.grain_id[gslice]) else 0
    )

    # 4) 伯努利抽样：每个候选以 p 命中
    r = rng.random(size=eligible.shape)
    hit = eligible & (r < prob_per_cell)

    n_new = int(np.count_nonzero(hit))

    # 5) 为命中元胞分配连续 grain_id
    current_max = (
        int(np.max(grid.grain_id[gslice])) if np.any(grid.grain_id[gslice]) else 0
    )
    new_ids = np.arange(current_max + 1, current_max + n_new + 1, dtype=np.int32)
    gid[hit] = new_ids  # 布尔掩码写入按内存序填充 new_ids

    # 6) 取向抽样（扇区中心抽样）
    delta = 2.0 * np.pi / float(fold)
    c = rng.integers(low=0, high=Ntheta, size=n_new, endpoint=False)
    theta_local = ((c + 0.5) / float(Ntheta) - 0.5) * delta
    drawn_theta = _normalize_angle(theta_local)
    theta[hit] = drawn_theta

    # 7) 可选：给一个很小的种子 fs，便于“可见”
    if seed_fs > 0.0:
        fs[hit] = np.maximum(fs[hit], seed_fs)

    total_now = (
        int(np.max(grid.grain_id[gslice])) if np.any(grid.grain_id[gslice]) else 0
    )
