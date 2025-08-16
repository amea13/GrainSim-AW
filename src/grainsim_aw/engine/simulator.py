import numpy as np
import logging
from ..core.grid import create_grid, update_ghosts, classify_phases
from ..nucleation import apply as nucl_apply
from ..interface import compute_interface_fields
from ..growth_capture import step as mdcs_step
from ..multiphysics import solute_advance
from ..io.writer import prepare_out, write_meta, snapshot
from ..multiphysics.temperature_adapter import sample as sample_temperature

logger = logging.getLogger(__name__)


def _total_solute_mass(grid) -> float:
    """
    诊断用总溶质量：M = ∑ [ α CL + (1-α) CS ] dx dy，
    其中 α = 1 - fs  是液相体积分数（标准守恒写法）。
    """
    alpha = 1.0 - grid.fs
    cell = grid.dx * grid.dy
    # 只统计 core（避免 ghost 干扰）
    g = grid.nghost
    core = (slice(g, -g), slice(g, -g))
    M = np.sum(alpha[core] * grid.CL[core] + (1.0 - alpha[core]) * grid.CS[core]) * cell
    return float(M)


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
            Tbuf = sample_temperature(
                self.grid, t, self.cfg.get("temperature", {"T_const": 1750.0})
            )
            masks = classify_phases(self.grid.fs, self.grid.nghost)

            # Core physics （第一批为空实现，留出接口）
            nucl_apply(self.grid, Tbuf, self.rng, self.cfg.get("nucleation", {}), masks)
            logger.info("Nucleation done (stub)")  # 这行是 INFO，肯定能看到
            # Interface.equilibrium(...)
            fields = compute_interface_fields(
                self.grid,
                Tbuf,
                self.cfg.get("physics", {}).get("interface", {}),
                self.cfg.get("physics", {}).get("orientation", {}),
                masks,
            )
            logger.info("Interface fields computed (stub)")
            # GrowthRule.MDCS.step(...)
            mdcs_step(
                self.grid,
                fields,
                self.cfg.get("physics", {}).get("mdcs", {}),
                self.cfg.get("physics", {}).get("orientation", {}),
                self.cfg["time"]["dt"],
                masks,
            )
            logger.info("MDCS done (stub)")
            # SoluteSolver.advance(...)
            # 调用溶质占位解算
            solute_advance(
                self.grid,
                self.cfg.get("physics", {}).get("solute", {}),
                self.cfg["time"]["dt"],
                masks,
            )
            logger.info("solute_advance done (stub)")

            # 诊断：打印一次总溶质量（以后 E1 验收就看这个是否随时间守恒）
            M = _total_solute_mass(self.grid)
            logger.info(f"Total solute mass (diag): {M:.6e}")

            # Post-step
            if step % save_every == 0:
                snapshot(self.grid, t, step, self.out)

        snapshot(self.grid, t, step, self.out)
