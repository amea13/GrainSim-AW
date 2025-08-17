from pathlib import Path
import json
import numpy as np
from ..core.grid import Grid
from typing import Optional, Dict


def prepare_out(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_meta(cfg: dict, out: Path):
    (out / "meta_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def snapshot(
    grid: Grid,
    t: float,
    step: int,
    out: Path,
    extras: Optional[Dict[str, np.ndarray]] = None,
):
    payload = {
        "fs": grid.fs,
        "CL": grid.CL,
        "CS": grid.CS,
        "grain_id": grid.grain_id,
        "theta": grid.theta,
        "L_dia": grid.L_dia,
        "T": grid.T,  # ← 新增：T 作为持久字段直接写入
        "t": t,
        "step": step,
    }
    if extras:
        # 仍支持写入其他派生量或调试量
        payload.update(extras)
    np.savez_compressed(out / f"step_{step:06d}.npz", **payload)
