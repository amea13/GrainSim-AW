import numpy as np
from ..core.grid import create_grid, update_ghosts, classify_phases
from ..io.writer import prepare_out, write_meta, snapshot
from ..temperature.adapter import sample as sample_temperature

class Simulator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.grid = create_grid(cfg["domain"])
        self.rng = np.random.default_rng(cfg["run"]["seed"])
        self.out = prepare_out(cfg["run"]["output_dir"])
        write_meta(cfg, self.out)
        update_ghosts(self.grid, cfg["domain"]["bc"])

    def run(self):
        dt = self.cfg["time"]["dt"]
        t_end = self.cfg["time"]["t_end"]
        save_every = self.cfg["time"]["save_every"]

        t = 0.0
        step = 0
        while t < t_end:
            step += 1
            t += dt

            # Pre-step
            update_ghosts(self.grid, self.cfg["domain"]["bc"])
            Tbuf  = sample_temperature(self.grid, t, self.cfg.get("temperature", {"T_const": 1750.0}))
            masks = classify_phases(self.grid.fs, self.grid.nghost)

            # Core physics （第一批为空实现，留出接口）
            # Nucleation.apply(...)
            # Interface.equilibrium(...)
            # GrowthRule.MDCS.step(...)
            # SoluteSolver.advance(...)

            # Post-step
            if step % save_every == 0:
                snapshot(self.grid, t, step, self.out)

        snapshot(self.grid, t, step, self.out)
