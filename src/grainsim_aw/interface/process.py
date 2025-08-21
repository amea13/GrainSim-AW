# src/grainsim_aw/interface/process.py
from typing import Dict, Any
from .fields import IfaceFieldsBuf as Fields
from .geometry import compute_curvature, compute_normal
from .equilibrium import compute_equilibrium
from .velocity import compute_velocity


class InterfaceProcess:
    def curvature(self, grid, cfg: Dict[str, Any], fields: Fields) -> None:
        compute_curvature(grid, fields.masks, cfg, out=fields.kappa)

    def normal(self, grid, cfg: Dict[str, Any], fields: Fields) -> None:
        compute_normal(grid, fields.masks, cfg, out_nx=fields.nx, out_ny=fields.ny)

    def equilibrium(self, grid, cfg: Dict[str, Any], fields: Fields) -> None:
        compute_equilibrium(
            grid, fields.masks, cfg, out_cls=fields.cls, out_css=fields.css
        )

    def velocity(self, grid, cfg: Dict[str, Any], fields: Fields) -> None:
        compute_velocity(
            grid,
            fields.masks,
            cfg,
            normal=(fields.nx, fields.ny),
            eq=(fields.cls, fields.css),
            out_vn=fields.vn,
            out_vx=fields.vx,
            out_vy=fields.vy,
        )
