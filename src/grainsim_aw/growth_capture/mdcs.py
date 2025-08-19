# src/grainsim_aw/growth_capture/mdcs.py

from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .kernels import centroid_normal
from .geometry import L_n, shape_factor_GF, update_Ldia, vertices
from .capture_rules import apply as apply_capture_rules


def capture_pass(
    grid,
    masks: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    parent_mask: np.ndarray | None = None,
) -> None:
    """
    仅执行一次“偏心正方形捕捉”。不计算 Δfs，不更新 L_dia。
    - parent_mask: 可选 “父界面胞集合”。若为 None，默认用 masks['mask_int']。
      典型用法：parent_mask = masks['mask_int'] | seeds_mask  （把本步新核也当父胞）
    - 本函数内部不会修改 masks；调用方应在捕捉后统一重算 masks。
    """
    if parent_mask is None:
        parent_mask = masks["mask_int"]

    # 只为父胞生成顶点。非父胞位置将被 vertices 置为 NaN，apply 会自动跳过
    verts = vertices(grid, parent_mask)

    # 构造“捕捉用掩码”：强制用 parent_mask 作为父界面集合
    masks_cap = dict(masks)
    masks_cap["mask_int"] = parent_mask

    # 这三个阵当前 apply 不依赖（df_parent 仅用于平局破），给零阵即可
    Z = np.zeros_like(grid.fs)
    apply_capture_rules(grid, verts, Z, Z, Z, masks_cap, cfg)
    # 注意：此处已原地修改 grid.fs / grain_id / theta / ecc / L_dia（仅初始化新界面元）


def advance_no_capture(
    grid,
    fields,  # IfaceFields：至少含 Vn
    cfg: Dict[str, Any],
    dt: float,
    masks: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    仅推进 Δfs 与 L_dia（不执行捕捉）。
    返回 fs_dot（供溶质源项用）。
    需要调用方在进入本函数前已 update_ghosts(grid)，因为圆核质心法依赖 ghost。
    """
    fs = grid.fs
    Vn = fields.Vn
    dx, dy = float(grid.dx), float(grid.dy)
    mask_int = masks["mask_int"]

    # 1) 圆核质心法：几何法向（用于几何长度与 GF）
    nx_c, ny_c = centroid_normal(fs)

    # 2) 法向 -> 界面穿越长度 L_n
    Ln = L_n(nx_c, ny_c, dx, dy)

    # 3) 一/二级邻胞规则 -> 形状因子 GF
    GF = shape_factor_GF(fs, nx_c, ny_c, masks)

    # 4) Δf_s（仅界面带，单向增长，不超过剩余空间）
    eps = 1e-30
    delta_fs = np.zeros_like(fs, dtype=float)
    delta_fs[mask_int] = (
        GF[mask_int] * Vn[mask_int] * dt / np.maximum(Ln[mask_int], eps)
    )
    np.maximum(delta_fs, 0.0, out=delta_fs)
    delta_fs = np.minimum(delta_fs, 1.0 - fs)
    fs += delta_fs  # 原地更新

    # 5) 由 Δf_s 更新 L_dia（仅几何，不做捕捉）
    update_Ldia(grid, delta_fs, grid.theta)

    # 输出给溶质源项
    fs_dot = delta_fs / dt
    return fs_dot
