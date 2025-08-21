from typing import Dict, Any
import numpy as np

from .capture import geometry_and_capture as _geometry_and_capture
from .advance import advance_interface as _advance_interface


class GrowthProcess:
    def geometry_and_capture(self, grid, cfg: Dict[str, Any], fields) -> None:
        return _geometry_and_capture(grid, cfg, fields)

    def advance_solid(
        self, grid, vn: np.ndarray, dt: float, cfg: Dict[str, Any], fields
    ):
        return _advance_interface(grid, vn, dt, cfg, fields)
