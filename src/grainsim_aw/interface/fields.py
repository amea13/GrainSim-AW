# src/grainsim_aw/interface/fields.py
# -*- coding: utf-8 -*-
"""
IfaceFieldsBuf — 单步“界面带场”的可写缓冲。
仅在 masks["intf"] 覆盖写入；支持复用与按上一帧界面带清零。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class IfaceFieldsBuf:
    """
    【功能】承载单步界面带相关场，由各过程函数“就地写入”。
    【形状】所有数组与 grid.fs 一致；界面带外通常为 0。
    """

    masks: Dict[str, np.ndarray]  # 至少包含 "intf"

    # 几何/界面量
    kappa: np.ndarray  # 曲率
    nx: np.ndarray  # 法向 x
    ny: np.ndarray  # 法向 y
    cls: np.ndarray  # 界面液相浓度 C_L^*
    css: np.ndarray  # 界面固相浓度 C_S^*
    vn: np.ndarray  # 法向生长速率
    fs_dot: np.ndarray  # 固相率时间导数（溶质源项）

    # 可选：速度分量与各向异性诊断量
    vx: Optional[np.ndarray] = None
    vy: Optional[np.ndarray] = None
    ani: Optional[np.ndarray] = None

    # —— 工厂与维护 —— #
    @staticmethod
    def like(
        grid,
        masks,
        with_vec: bool = False,
        with_ani: bool = False,
        *,
        need_clear: bool = True,
    ) -> "IfaceFieldsBuf":
        z = lambda: np.zeros_like(grid.fs)
        buf = IfaceFieldsBuf(
            masks=masks,
            kappa=z(),
            nx=z(),
            ny=z(),
            cls=z(),
            css=z(),
            vn=z(),
            fs_dot=z(),
            vx=z() if with_vec else None,
            vy=z() if with_vec else None,
            ani=z() if with_ani else None,
        )
        if need_clear:
            buf._clear_all()
        return buf

    def reset(self, masks, *, need_clear: bool = True) -> None:
        """更新掩码；按需清零。"""
        self.masks = masks
        if need_clear:
            self._clear_all()

    def clear_on(self, mask: np.ndarray) -> None:
        """仅对给定 mask 清零，避免全场 fill(0) 的 O(N) 开销。"""
        self.kappa[mask] = 0
        self.nx[mask] = 0
        self.ny[mask] = 0
        self.cls[mask] = 0
        self.css[mask] = 0
        self.vn[mask] = 0
        self.fs_dot[mask] = 0
        if self.vx is not None:
            self.vx[mask] = 0
        if self.vy is not None:
            self.vy[mask] = 0
        if self.ani is not None:
            self.ani[mask] = 0

    def _clear_all(self) -> None:
        self.kappa.fill(0)
        self.nx.fill(0)
        self.ny.fill(0)
        self.cls.fill(0)
        self.css.fill(0)
        self.vn.fill(0)
        self.fs_dot.fill(0)
        if self.vx is not None:
            self.vx.fill(0)
        if self.vy is not None:
            self.vy.fill(0)
        if self.ani is not None:
            self.ani.fill(0)
