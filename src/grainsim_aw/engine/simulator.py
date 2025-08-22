"""
求解器（Simulator）
===================

【功能】
本模块实现“编排器”角色，严格按照既定流程调用各物理过程的 Process 类，
自身不复写具体数值细节，尽量减少跨模块耦合与重复。

【输入】
- cfg: dict
  运行配置字典。推荐结构如下，字段名与默认值仅作参考，具体以各子模块实现为准。

配置示例（JSON 语义）
--------------------
{
  "domain": {
    "nx": 256, "ny": 256,
    "dx": 1.0e-6, "dy": 1.0e-6,
    "bc": { "x": "periodic", "y": "wall" },
    "C0": 0.02
  },
  "time": {
    "dt": 2.0e-4,
    "t_end": 16.0,
    "save_every": 50
  },
  "run": {
    "seed": 42,
    "output_dir": "data/output/run-minimal"
  },
  "viz": {
    "live": { "enable": true, "interval": 1 }
  },
  "init": {
    "mode": "random",
    "count": 20,
    "k0": 0.34
  },
  "nucleation": { "rate": 1.0, "sigma": 0.1 },
  "physics": {
    "interface": { "k0": 0.34 },
    "mdcs": { "capture_radius": 1.0 },
    "solute": { "scheme": "jacobi", "max_iter": 200, "tol": 1e-8 }
  },
  "temperature": {
    "mode": "table",
    "T0": 1800.0
  }
}

【输出】
- run() 期间：
  1) 按 save_every 写出快照到输出目录
  2) 若启用实时显示，则刷新 LivePlotter
  3) 日志记录关键统计量，例如总溶质量

【流程】
初始化（__init__）四步：
  1. 配置字典、创建网格、随机数生成器、输出目录与元数据
  2. 初始化基础物理场并更新 ghost
  3. 可选：手动形核初始种子
  4. 实时可视化工具初始化

主循环（run）十二步：
  1. 更新 ghosts 与相掩码
  2. Thevoz 形核
  3. ESVC 几何与捕捉
  4. 计算曲率
  5. 计算法向（圆质心法）
  6. 界面平衡固、液相浓度
  7. 界面法向生长速率
  8. 推进固相率与 ESVC 半对角线，得到 fs_dot
  9. 溶质场一步
  10. 温度更新
  11. 保存快照
  12. 刷新可视化

注：如需自适应时间步，可在标注处添加策略并控制稳定性。
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np

# --- 核心数据结构与基础操作 ---
from ..core.grid import create_grid, update_ghosts, classify_phases, Grid
from ..interface.fields import IfaceFieldsBuf as Fields

# --- 过程门面类（各包的 process.py 提供） ---
from ..nucleation.process import NucleationProcess
from ..growth_capture.process import GrowthProcess
from ..interface.process import InterfaceProcess
from ..multiphysics.process import TransportProcess

# --- 可视化与输出 ---
from ..viz.liveplot import LivePlotter
from ..io.writer import prepare_out, write_meta, snapshot
from ..io.csv_matrix import dump_matrix
from ..engine.time_step import compute_next_dt

logger = logging.getLogger(__name__)


class Simulator:
    """
    高层求解器编排器

    【输入】
    - cfg: dict
      运行配置字典，结构见模块顶部示例。

    【成员】
    - cfg: dict      原始配置
    - grid: Grid     计算域网格与场
    - rng: np.random.Generator 随机数生成器
    - out: str       输出目录
    - live: Optional[LivePlotter] 实时可视化
    - nuc: NucleationProcess
    - gro: GrowthProcess
    - itf: InterfaceProcess
    - trn: TransportProcess
    """

    # =====================
    # 初始化阶段（四步）
    # =====================
    def __init__(self, cfg: Dict[str, Any]):
        """
        【功能】初始化求解器，完成基础构建、场初始化、可选初始形核以及可视化准备。

        【输入】
        - cfg: dict
          运行配置。必须包含 time、domain、run 三段。

        【输出】
        - 内部状态被创建与就绪，不返回值。
        """
        self.cfg = cfg

        # 1) 配置与基础构建：网格、随机数、输出、元数据
        self.grid: Grid = create_grid(cfg["domain"])
        self.rng = np.random.default_rng(cfg["run"]["seed"])
        self.out = prepare_out(cfg["run"]["output_dir"])
        write_meta(cfg, self.out)

        # 2) 初始化基础场与 ghost
        C0 = float(cfg["domain"].get("C0", 0.0))
        self.grid.fs[:] = 0.0
        self.grid.CL[:] = C0
        # 初始温度，采用温度适配器在 t=0 的采样
        # 这里不直接导入适配函数，而是复用 TransportProcess 的统一入口
        self.trn = TransportProcess()
        self.trn.update_temperature(self.grid, cfg.get("temperature", {}), t=0.0)
        update_ghosts(self.grid, cfg["domain"]["bc"])

        # 3) 可选：手动初始形核
        self.nuc = NucleationProcess()
        init_cfg = dict(cfg.get("init", {}))
        init_cfg.setdefault(
            "k0",
            float(cfg.get("physics", {}).get("interface", {}).get("k0", 0.34)),
        )
        if init_cfg:
            self.nuc.seed_manual(self.grid, self.rng, init_cfg)

        # 4) 实时可视化初始化
        self.live: Optional[LivePlotter] = LivePlotter(
            cfg.get("viz", {}).get("live", {})
        )

        # 其他过程对象
        self.gro = GrowthProcess()
        self.itf = InterfaceProcess()

    # =====================
    # 运行阶段（十二步）
    # =====================
    def run(self) -> None:
        """
        【功能】启动主循环，按既定顺序推进物理过程。

        【输入】无（使用 __init__ 已就绪的内部成员）

        【输出】无（产生快照文件并可视化；必要统计写日志）
        """
        # 1) 时间参数读取与初始化
        dt = float(self.cfg["time"]["dt"])
        t_end = float(self.cfg["time"]["t_end"])
        save_every = int(self.cfg["time"]["save_every"])

        t = 0.0
        step = 0
        dt_next = dt  # 下一步使用的 Δt

        masks = classify_phases(self.grid)  # 约定键：liq | intf | sol
        fields = Fields.like(self.grid)

        # 2) 可视化启动
        if self.live:
            self.live.start(self.grid)

        try:
            while t < t_end:
                step += 1
                t += dt

                # 3-1) 更新 ghosts 与相掩码
                update_ghosts(self.grid, self.cfg["domain"]["bc"])

                # 3-2) Thevoz 形核
                self.nuc.nucleate(
                    self.grid, self.rng, self.cfg.get("nucleation", {}), masks
                )

                # dump_matrix(self.grid.fs, f"debug/fs0{step:06d}.csv")
                # dump_matrix(self.grid.L_dia, f"debug/L_dia{step:06d}.csv")

                # 3-3) ESVC 几何与捕捉
                self.gro.geometry_and_capture(
                    self.grid, self.cfg.get("physics", {}).get("mdcs", {}), masks
                )

                masks = classify_phases(self.grid)
                fields = Fields.like(self.grid)

                # 3-4) 计算曲率
                self.itf.curvature(
                    self.grid,
                    self.cfg.get("physics", {}).get("interface", {}),
                    fields,
                    masks,
                )

                # 3-5) 计算法向（圆质心法）
                self.itf.normal(
                    self.grid,
                    self.cfg.get("physics", {}).get("interface", {}),
                    fields,
                    masks,
                )

                # dump_matrix(fields.kappa, f"debug/Kappa{step:06d}.csv")
                # dump_matrix(fields.nx, f"debug/Nx{step:06d}.csv")
                # dump_matrix(fields.ny, f"debug/Ny{step:06d}.csv")

                # 3-6) 界面平衡固、液相浓度
                self.itf.equilibrium(
                    self.grid,
                    self.cfg.get("physics", {}).get("interface", {}),
                    fields,
                    masks,
                )

                # dump_matrix(fields.cls, f"debug/Cls{step:06d}.csv")

                # 3-7) 界面法向生长速率
                self.itf.velocity(
                    self.grid,
                    self.cfg.get("physics", {}).get("interface", {}),
                    fields,
                    masks,
                )

                # 3-8) 推进固相，更新 fs 与 ESVC 半对角线，得到 fs_dot
                self.gro.advance_solid(
                    self.grid,
                    fields.vn,
                    dt,
                    self.cfg.get("physics", {}).get("mdcs", {}),
                    fields,
                    masks,
                )

                # 3-9) 溶质场一步
                self.trn.step_solute(
                    self.grid,
                    self.cfg.get("physics", {}).get("solute", {}),
                    dt,
                    fields,
                )

                # 3-10) 温度更新
                self.trn.update_temperature(
                    self.grid,
                    self.cfg.get("temperature", {}),
                    t,
                )

                # 评估下一步 dt（与 C++ fun_delta_t 时序一致，在步末计算）
                # dt = compute_next_dt(self.grid, fields)

                # 3-11) 保存快照
                if step % save_every == 0:
                    snapshot(self.grid, t, step, self.out)

                # 3-12) 刷新可视化
                if self.live:
                    self.live.update(self.grid, t, step)

            # 循环结束后保存一次
            snapshot(self.grid, t, step, self.out)

        finally:
            if self.live:
                self.live.close()
