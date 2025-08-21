from __future__ import annotations
import os
import numpy as np

__all__ = ["dump_matrix"]


def dump_matrix(
    field: np.ndarray, out_csv: str, *, float_fmt: str = "%.8e", delimiter: str = ","
) -> None:
    """
    将二维场按网格矩阵格式导出为 CSV
    行列一一对应网格
    不写坐标 不写表头

    参数
    field : 形状为 (rows, cols) 的 ndarray
    out_csv : 输出文件路径
    float_fmt : 浮点数格式 默认 %.8e
    delimiter : 分隔符 默认逗号
    """
    a = np.asarray(field)
    if a.ndim != 2:
        raise ValueError("field 必须是二维数组")

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    # 根据数据类型选择合适的格式
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_):
        fmt = "%d"
    else:
        fmt = float_fmt

    np.savetxt(out_csv, a, fmt=fmt, delimiter=delimiter)
