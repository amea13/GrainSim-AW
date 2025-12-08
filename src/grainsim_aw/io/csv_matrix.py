import os
import numpy as np
from typing import Iterable, Tuple, Optional


def _core_view(a: np.ndarray, nghost: int, include_ghost: bool) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("仅支持二维数组")
    if include_ghost or nghost <= 0:
        return a
    g = int(nghost)
    return a[g:-g, g:-g]


def _fmt_for_dtype(a: np.ndarray, float_fmt: str) -> str:
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_):
        return "%d"
    return float_fmt


def _write_block(fh, title: Optional[str], a: np.ndarray, fmt: str, delimiter: str):
    if title:
        fh.write(f"### {title}\n")
    # 用 numpy 写矩阵一行一行
    np.savetxt(fh, a, fmt=fmt, delimiter=delimiter)
    fh.write("\n")  # 空行分隔


def dump_grids_to_csv(
    grid,
    fields,
    out_csv: str,
    *,
    include_ghost: bool = False,
    float_fmt: str = "%.11f",
    delimiter: str = ",",
    # 想自定义导出顺序可传入形如 [("fs","grid.fs"), ("vn","fields.vn"), ...]
    which: Optional[Iterable[Tuple[str, str]]] = None,
) -> None:
    """
    将 grid 与 fields 中的二维场按“矩阵块”顺序写入一个 CSV。
    每个块的形状与网格一致（默认去掉 ghost）。
    - grid: 你的 Grid 实例（需有属性 nghost）
    - fields: 你的 IfaceFieldsBuf 实例
    - which: 自定义要导出的字段列表，元素为 (块名, 路径字符串)
             路径以 "grid." 或 "fields." 开头，例如 "grid.fs"、"fields.vn"
             若为 None 使用默认集合
    """
    # 默认导出清单（可按需增删）
    if which is None:
        which = [
            ("fs", "grid.fs"),
            ("T", "grid.T"),
            ("CL", "grid.CL"),
            ("CS", "grid.CS"),
            ("vel", "fields.vn"),  # 如需速度模长改成：lambda g,f: np.hypot(f.vx, f.vy)
            ("nx", "fields.nx"),
            ("ny", "fields.ny"),
            ("cur", "fields.kappa"),  # 曲率
            (
                "ani",
                "fields.ani",
            ),  # 暂以各向异性因子代替 gf；若你有 gf 场，改成 "fields.gf"
            ("Ldia", "grid.L_dia"),  # 名称保持 Ldia，取 grid.L_dia
            # ——其余没有在 C++ 顺序里的，排在后面（可选保留/删除）——
            ("grain_id", "grid.grain_id"),
            ("theta", "grid.theta"),
            ("ecc_x", "grid.ecc_x"),
            ("ecc_y", "grid.ecc_y"),
            ("cls", "fields.cls"),
            ("css", "fields.css"),
            ("fs_dot", "fields.fs_dot"),
            ("vx", "fields.vx"),
            ("vy", "fields.vy"),
            ("test", "fields.test"),
        ]

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    nghost = int(getattr(grid, "nghost", 0))

    with open(out_csv, "w", newline="") as fh:
        for title, path in which:
            # 解析路径，拿到 ndarray
            root, attr = path.split(".", 1)
            obj = grid if root == "grid" else fields
            if obj is None:
                continue
            if not hasattr(obj, attr):
                continue
            arr = getattr(obj, attr)
            if arr is None:
                continue

            a = _core_view(np.asarray(arr), nghost, include_ghost)
            if a.ndim != 2:
                # 只写二维场，其他维度跳过
                continue

            fmt = _fmt_for_dtype(a, float_fmt)
            _write_block(fh, title, a, fmt, delimiter)
