from __future__ import annotations
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _normalize_angle(theta: np.ndarray) -> np.ndarray:
    """把角度规范到 [0, 2π)。"""
    twopi = 2.0 * np.pi
    return np.mod(theta, twopi)


def apply(
    grid,
    Tbuf: np.ndarray,
    rng: np.random.Generator,
    cfg_nucl: Dict[str, Any],
    masks: Dict[str, np.ndarray],
) -> None:
    """
    形核入口
    参数
    ----
    grid : Grid
        持久状态容器（fs/CL/CS/grain_id/theta/...）。
    Tbuf : np.ndarray
        当步温度缓冲（与 grid.fs 同形状，含 ghost）。
    rng : np.random.Generator
        统一注入的随机数源（保证可复现）。
    cfg_nucl : dict
        形核配置（此阶段允许为空 dict）。
    masks : dict
        相态掩码，至少包含 'mask_liq'、'mask_int'、'mask_sol'。
    返回

    最小可用形核：
    - 条件区域：液相且满足过冷条件；
    - 每个候选元胞以 p=prob_per_cell 的概率命中（独立伯努利）；
    - 命中元胞赋予递增 grain_id、取向 theta，并可选微增 fs。
    ----
    None（就地更新 grid；当前占位版不做任何修改）
    """

    # 检查液相掩码
    if masks is None or "mask_liq" not in masks:
        logger.debug("[nucleation] 未提供液相掩码，跳过。")
        return

    gslice = grid.core  # (slice_y, slice_x)
    liq_core = masks["mask_liq"][gslice]

    gid = grid.grain_id[gslice]
    theta = grid.theta[gslice]
    fs = grid.fs[gslice]
    T = Tbuf[gslice]

    # 1) 读取配置（给出合理默认）
    # 过冷判据：优先用 ΔT = T_L_eq - T ≥ dT_threshold；否则用温度阈值 T ≤ T_threshold；都缺则默认全候选
    dT_threshold = float(cfg_nucl.get("dT_threshold", np.nan))
    T_threshold = cfg_nucl.get("T_threshold", None)
    T_L_eq = cfg_nucl.get(
        "T_L_eq", None
    )  # 允许直接放在 nucleation 节；若你放在 physics.interface 也可在 Engine 侧透传
    prob_per_cell = float(
        cfg_nucl.get("prob_per_cell", cfg_nucl.get("rate_per_cell", 0.0))
    )
    prob_per_cell = max(0.0, min(1.0, prob_per_cell))  # 限幅到 [0,1]
    seed_fs = float(cfg_nucl.get("seed_fs", 0.0))  # 命中元胞的初始固相分数（可为 0）
    # 取向参数（各向异性阶数 fold、等分 N_theta）
    fold = int(
        cfg_nucl.get("fold", 4)
    )  # 也可以来自 physics.orientation，这里先本地兜底
    Ntheta = int(cfg_nucl.get("N_theta", 48))

    # 2) 过冷判定掩码
    if not np.isnan(dT_threshold) and (T_L_eq is not None):
        dT = float(T_L_eq) - T
        eligible = liq_core & (dT >= dT_threshold)
    elif T_threshold is not None:
        eligible = liq_core & (T <= float(T_threshold))
    else:
        # 没有阈值配置：仅以液相作为候选（便于先看到效果）
        eligible = liq_core.copy()

    # 3) 过滤掉已经成核（已有 grain_id）或 fs 已非液的格点
    tau_liq = 1e-12  # 与 classify_phases 的液阈值保持一致
    eligible &= (gid == 0) & (fs < tau_liq)

    num_eligible = int(np.count_nonzero(eligible))
    if num_eligible == 0 or prob_per_cell <= 0.0:
        logger.info(
            "Nucleation: new=0, total=%d (eligible=%d, p=%.3g)",
            int(np.max(grid.grain_id[gslice], initial=0)),
            num_eligible,
            prob_per_cell,
        )
        return

    # 4) 伯努利抽样：每个候选以 p 命中
    hit = np.zeros_like(eligible, dtype=bool)
    # 生成与候选同数目的随机数，仅在 eligible 上比较
    r = rng.random(size=eligible.shape)
    hit = eligible & (r < prob_per_cell)

    n_new = int(np.count_nonzero(hit))
    if n_new == 0:
        logger.info(
            "Nucleation: new=0, total=%d (eligible=%d, p=%.3g)",
            int(np.max(grid.grain_id[gslice], initial=0)),
            num_eligible,
            prob_per_cell,
        )
        return

    # 5) 为命中元胞分配连续 grain_id
    current_max = (
        int(np.max(grid.grain_id[gslice])) if np.any(grid.grain_id[gslice]) else 0
    )
    new_ids = np.arange(current_max + 1, current_max + 1 + n_new, dtype=np.int32)
    gid[hit] = new_ids  # 注意：按布尔掩码写入会按内存顺序填充 new_ids

    # 6) 取向抽样（扇区中心抽样）
    #   Δ = 2π / fold
    #   c ∈ {0,1,…,Ntheta-1} 均匀整数
    #   θ_local = ((c+0.5)/Ntheta - 0.5) * Δ
    delta = 2.0 * np.pi / float(fold)
    c = rng.integers(low=0, high=Ntheta, size=n_new, endpoint=False)
    theta_local = ((c + 0.5) / float(Ntheta) - 0.5) * delta
    # 这里没有全局锚角，直接用局部取向；若你以后有晶粒取向锚，可在此相加
    drawn_theta = _normalize_angle(theta_local)
    theta[hit] = drawn_theta

    # 7) 可选：给一个很小的种子 fs，便于“可见”
    if seed_fs > 0.0:
        fs[hit] = np.maximum(fs[hit], seed_fs)

    total_now = int(np.max(grid.grain_id[gslice]))
    logger.info(
        "Nucleation: new=%d, total=%d (eligible=%d, p=%.3g)",
        n_new,
        total_now,
        num_eligible,
        prob_per_cell,
    )

    # 可选：DEBUG 输出取向桶计数（验证均匀性）
    if logger.isEnabledFor(logging.DEBUG):
        # 以 Ntheta 桶统计 theta 的分布（只看新核）
        counts = np.bincount(c, minlength=Ntheta)
        logger.debug(
            "Nucleation θ_class histogram (Ntheta=%d): %s", Ntheta, counts.tolist()
        )
