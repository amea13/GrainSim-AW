from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np

FloatArr = np.ndarray  # 仅做可读性标签


@dataclass(slots=True)
class IfaceFieldsBuf:
    kappa: FloatArr
    nx: FloatArr
    ny: FloatArr
    cls: FloatArr
    css: FloatArr
    vn: FloatArr
    fs_dot: FloatArr
    vx: FloatArr
    vy: FloatArr
    ani: FloatArr
    test: FloatArr

    # —— 工厂 —— #
    @staticmethod
    def like(grid) -> "IfaceFieldsBuf":
        z = lambda: np.zeros_like(grid.fs, dtype=np.float64, order="C")
        return IfaceFieldsBuf(
            kappa=z(),
            nx=z(),
            ny=z(),
            cls=z(),
            css=z(),
            vn=z(),
            fs_dot=z(),
            vx=z(),
            vy=z(),
            ani=z(),
            test=z(),
        )

    # —— 校验与维护 —— #
    def __post_init__(self) -> None:
        shape = self.kappa.shape
        # 所有数组形状一致、连续、float64
        for name, a in self._named_arrays():
            if a.shape != shape:
                raise ValueError(f"{name} shape {a.shape} != {shape}")
            if a.dtype != np.float64:
                raise TypeError(f"{name} dtype {a.dtype} != float64")
            # 确保 C 连续，避免后续 ravel 索引慢
            if not a.flags.c_contiguous:
                setattr(self, name, np.ascontiguousarray(a, dtype=np.float64))

    def _arrays(self) -> Iterable[FloatArr]:
        return (
            self.kappa,
            self.nx,
            self.ny,
            self.cls,
            self.css,
            self.vn,
            self.fs_dot,
            self.vx,
            self.vy,
            self.ani,
        )

    def _named_arrays(self) -> Iterable[tuple[str, FloatArr]]:
        return (
            ("kappa", self.kappa),
            ("nx", self.nx),
            ("ny", self.ny),
            ("cls", self.cls),
            ("css", self.css),
            ("vn", self.vn),
            ("fs_dot", self.fs_dot),
            ("vx", self.vx),
            ("vy", self.vy),
            ("ani", self.ani),
            ("test", self.test),
        )

    # —— 清零 —— #
    def clear_all(self) -> None:
        for a in self._arrays():
            a.fill(0.0)

    def reset(
        self, mask: Optional[np.ndarray], *, sparse_threshold: float = 0.25
    ) -> None:
        """
        按上一帧界面带清零；mask 为 None 时整场清零。
        对稀疏 mask 采用“索引表 + ravel 赋值”，减少布尔筛选开销；
        对稠密 mask 走布尔索引更快。
        """
        if mask is None:
            self.clear_all()
            return

        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or mask.shape != self.kappa.shape:
            raise ValueError(f"mask shape {mask.shape} != {self.kappa.shape}")

        density = float(mask.mean())  # True 占比
        if density <= sparse_threshold:
            idx = np.flatnonzero(mask.ravel())
            if idx.size == 0:
                return
            for a in self._arrays():
                a.ravel()[idx] = 0.0
        else:
            for a in self._arrays():
                a[mask] = 0.0
