from pathlib import Path
import json
import numpy as np
from ..core.grid import Grid


def prepare_out(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_meta(cfg: dict, out: Path):
    (out / "meta_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def snapshot(grid: Grid, t: float, step: int, out: Path):
    np.savez_compressed(
        out / f"step_{step:06d}.npz",
        fs=grid.fs,
        CL=grid.CL,
        CS=grid.CS,
        grain_id=grid.grain_id,  # ← 新增
        theta=grid.theta,  # ← 新增
        L_dia=grid.L_dia,  # ← 新增
        t=t,
        step=step,
    )
