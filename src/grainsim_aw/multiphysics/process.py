from typing import Dict, Any

from .solute_solver import step_solute as _step_solute
from .temperature_adapter import update as _update_temperature


class TransportProcess:
    def step_solute(self, grid, cfg: Dict[str, Any], dt: float, fields):
        return _step_solute(
            grid=grid,
            cfg=cfg,
            dt=dt,
            masks=fields.masks,
            fs_dot=fields.fs_dot,
            CL_star=fields.cls,
        )

    def update_temperature(self, grid, cfg: Dict[str, Any], t: float):
        return _update_temperature(grid=grid, cfg=cfg, t=t)
