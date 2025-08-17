import numpy as np
from typing import Optional
import logging
from ..core.grid import create_grid, update_ghosts, classify_phases
from ..nucleation import apply as nucl_apply
from ..nucleation import initialize as seed_initialize
from ..interface import compute_interface_fields
from ..growth_capture import step as mdcs_step
from ..multiphysics import solute_advance
from ..multiphysics import sample_T
from ..viz.liveplot import LivePlotter
from ..io.writer import prepare_out, write_meta, snapshot

logger = logging.getLogger(__name__)


def _total_solute_mass(grid) -> float:
    """
    诊断用总溶质量：M = ∑ [ α CL + (1-α) CS ] dx dy，
    其中 α = 1 - fs 为液相体积分数。
    只在 core 统计，避免 ghost 干扰。
    """
    alpha = 1.0 - grid.fs
    cell = grid.dx * grid.dy
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

        init_cfg = self.cfg.get("init", {})
        if init_cfg:
            placed = seed_initialize(self.grid, self.rng, init_cfg)
            logger.info("Init seeds placed: %d", placed)

        self.live: Optional[LivePlotter] = None
        live_cfg = self.cfg.get("viz", {}).get("live", {})
        if live_cfg.get("enabled", False):
            try:
                self.live = LivePlotter(
                    field=live_cfg.get("field", "fs"),
                    autoscale=bool(live_cfg.get("autoscale", False)),
                    show_ghosts=bool(live_cfg.get("show_ghosts", False)),
                )
                logger.info(
                    "LivePlotter enabled: field=%s", live_cfg.get("field", "fs")
                )
            except Exception:
                logger.exception(
                    "Failed to init LivePlotter; continue without live window."
                )

    def run(self):
        dt = self.cfg["time"]["dt"]
        t_end = self.cfg["time"]["t_end"]
        save_every = self.cfg["time"]["save_every"]

        t = 0.0
        step = 0

        if self.live:
            self.live.start(self.grid)

        while t < t_end:
            step += 1
            t += dt

            # Pre-step
            update_ghosts(self.grid, self.cfg["domain"]["bc"])
            # 温度为持久字段：每步采样并写入 grid.T
            self.grid.T[:] = sample_T(self.grid, t, self.cfg.get("temperature", {}))
            masks = classify_phases(self.grid.fs, self.grid.nghost)

            # Core physics（各模块内部直接读取 grid.T）
            nucl_apply(self.grid, self.rng, self.cfg.get("nucleation", {}), masks)
            logger.info("Nucleation done (stub)")

            fields = compute_interface_fields(
                self.grid,
                self.cfg.get("physics", {}).get("interface", {}),
                self.cfg.get("physics", {}).get("orientation", {}),
                masks,
            )
            logger.info("Interface fields computed (stub)")

            mdcs_step(
                self.grid,
                fields,
                self.cfg.get("physics", {}).get("mdcs", {}),
                self.cfg.get("physics", {}).get("orientation", {}),
                dt,
                masks,
            )
            logger.info("MDCS done (stub)")

            solute_advance(
                self.grid,
                self.cfg.get("physics", {}).get("solute", {}),
                dt,
                masks,
            )
            logger.info("solute_advance done (stub)")

            # 诊断
            M = _total_solute_mass(self.grid)
            logger.info(f"Total solute mass (diag): {M:.6e}")

            # Post-step
            if step % save_every == 0:
                snapshot(self.grid, t, step, self.out)

            # 直播更新
            live_cfg = self.cfg.get("viz", {}).get("live", {})
            every: int = int(live_cfg.get("every", 1))
            if self.live and self.live.alive:
                if step % max(1, every) == 0:
                    self.live.update(self.grid, t, step)
            elif self.live and not self.live.alive:
                logger.info("Live window closed by user; disabling live updates.")
                self.live = None

        snapshot(self.grid, t, step, self.out)

        # 是否保持窗口：默认 True，可用配置关闭
        live_cfg = self.cfg.get("viz", {}).get("live", {})
        keep_open = bool(live_cfg.get("keep_open", True))

        if keep_open and self.live is not None and self.live.alive:
            logger.info(
                "Simulation finished — live window will stay open. Close it to exit."
            )
            self.live.block()
