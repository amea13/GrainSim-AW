from pathlib import Path
import json
import numpy as np
from ..core.grid import Grid


# 准备输出目录
def prepare_out(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# 写入配置元数据
def write_meta(cfg: dict, out: Path):
    (out / "meta_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# 快照保存函数
def snapshot(grid: Grid, t: float, step: int, out: Path):
    np.savez_compressed(
        out / f"step_{step:06d}.npz",
        fs=grid.fs,
        CL=grid.CL,
        CS=grid.CS,
        grain_id=grid.grain_id,
        theta=grid.theta,
        L_dia=grid.L_dia,
        t=t,
        step=step,
    )
