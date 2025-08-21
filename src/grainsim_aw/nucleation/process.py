from typing import Dict, Any
import numpy as np
from .seeding import seed_initialize
from .thevoz import apply as thevoz_apply


class NucleationProcess:
    def seed_manual(self, grid, rng: np.random.Generator, cfg: Dict[str, Any]):
        return seed_initialize(grid, rng, cfg)

    def nucleate(
        self,
        grid,
        rng: np.random.Generator,
        cfg: Dict[str, Any],
        masks: Dict[str, np.ndarray],
    ):
        return thevoz_apply(grid, rng, cfg, masks)
