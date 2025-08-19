# -*- coding: utf-8 -*-
"""
界面局部平衡与派生场的编排
- 计算界面法向、曲率、各向异性因子
- 基于 T = T* 的局部平衡，反解 C_L* 与 C_S*
- 速度仍为占位，由 velocity.compute_velocity 提供
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from .anisotropy import compute_normals_and_curvature, anisotropy_factor
from .velocity import compute_velocity


@dataclass(frozen=True)
class IfaceFields:
    Vn: np.ndarray  # 法向生长速度
    Vx: np.ndarray  # 法向速度 x 分量
    Vy: np.ndarray  # 法向速度 y 分量
    nx: np.ndarray  # 界面法向 x 分量
    ny: np.ndarray  # 界面法向 y 分量
    kappa: np.ndarray  # 曲率
    ani: np.ndarray  # 各向异性因子 f(phi, theta)
    CLs: np.ndarray  # 界面液相浓度 C_L^*
    CSs: np.ndarray  # 界面固相浓度 C_S^*


def compute_interface_fields(
    grid,
    cfg_if: Dict[str, Any],
    cfg_orient: Dict[str, Any],  # 预留，将来可切换取向来源。目前直接用 grid.theta
    masks: Dict[str, np.ndarray],
) -> IfaceFields:
    """
    计算固液界面的几何量与局部平衡闭式解。
    仅在界面带 mask_int 内赋物理值，其它位置置零或合理常数。

    物理模型
    --------
    T* = T_L_eq + (C_L* - C0) m_L - Gamma * kappa * f(phi, theta)
    C_S* = k0 * C_L*
    在局部平衡假设下，界面处 T = T*，于是可反解：
    C_L* = C0 + [ T - T_L_eq + Gamma * kappa * f ] / m_L

    配置键（cfg_if）
    ----------------
    TL_eq : float      理论液相线温度
    C0    : float      初始合金含量
    mL    : float      液相线斜率（可能为负）
    Gamma : float      Gibbs Thomson 系数
    k0    : float      分配系数
    eps_anis : float   各向异性幅值 epsilon
    curv_smooth : int  曲率计算前对 fs 的盒式平滑次数，默认 0
    Vn_const : float   可选常数法向速度，用于占位

    返回
    ----
    IfaceFields：字段形状与 grid.fs 相同，含 ghost
    """
    shape = grid.fs.shape
    zeros = lambda: np.zeros(
        shape, dtype=float
    )  # lambda匿名函数赋值给变量 zeros，每次调用生成新的全零数组

    # 读取界面带掩码
    mask_int = masks.get("mask_int", None)
    if mask_int is None:
        # 若未提供，按 0<fs<1 自动判定
        mask_int = (grid.fs > 1e-12) & (grid.fs < 1.0 - 1e-12)

    # 物性参数，提供合理默认，实际使用请在 config 中指定
    TL_eq = float(cfg_if.get("TL_eq", 933.47))
    C0 = float(cfg_if.get("C0", 0.0))
    mL = float(cfg_if.get("mL", -1.0))  # 注意不可为 0
    Gamma = float(cfg_if.get("Gamma", 0.0))
    k0 = float(cfg_if.get("k0", 1.0))
    eps = float(cfg_if.get("eps_anis", 0.0))
    curv_smooth = int(cfg_if.get("curv_smooth", 0))

    # 1) 几何量：法向与曲率
    nx = zeros()
    ny = zeros()
    kappa = zeros()
    # 中间变量存取
    nx_, ny_, kappa_ = compute_normals_and_curvature(grid.fs, grid.dx, grid.dy)
    # 仅赋值中间变量
    nx[mask_int] = nx_[mask_int]
    ny[mask_int] = ny_[mask_int]
    kappa[mask_int] = kappa_[mask_int]

    nx, ny, kappa = -nx, -ny, -kappa

    # 2) 各向异性因子 f(phi, theta)
    ani = np.ones(shape, dtype=float)
    if eps != 0.0:
        ani_, _phi = anisotropy_factor(nx_, ny_, grid.theta, eps)
        ani[mask_int] = ani_[mask_int]  # 非界面处保持 1

    # 3) 局部平衡：反解 C_L*, C_S*
    CLs = zeros()
    CSs = zeros()

    # 避免 mL = 0 数值问题
    if abs(mL) < 1e-20:
        mL = -1e-20

    T = grid.T
    # 只在界面带计算
    num = T[mask_int] - TL_eq + Gamma * kappa[mask_int] * ani[mask_int]
    CLs[mask_int] = C0 + num / mL
    CSs[mask_int] = k0 * CLs[mask_int]

    # 4) 计算界面移动速度 Stefan守恒
    Vn, Vx, Vy = compute_velocity(
        cfg_if,
        masks["mask_int"],
        nx,
        ny,
        grid=grid,
        CLs=CLs,
        CSs=CSs,
    )

    fields = IfaceFields(
        Vn=Vn, Vx=Vx, Vy=Vy, nx=nx, ny=ny, kappa=kappa, ani=ani, CLs=CLs, CSs=CSs
    )

    return fields
