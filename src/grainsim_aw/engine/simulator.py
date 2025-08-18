import numpy as np  # numpy做数值计算
from typing import Optional  # typing 确保类型注解
import logging  # logging 记录日志

from ..core.grid import (
    create_grid,
    update_ghosts,
    classify_phases,
)  # 管理计算网格，包括创建和更新边界ghost cells
from ..nucleation import apply as nucl_apply  # Thevoz方法异质形核
from ..nucleation import (
    initialize as seed_initialize,
)  # 手动初始化晶核
from ..interface import compute_interface_fields  # 计算界面相关的场
from ..growth_capture import step as mdcs_step  # 处理MDCS捕获
from ..multiphysics import solute_advance, total_solute_mass  # 溶质场偏微分方程求解
from ..multiphysics import sample_T  # 温度场加载
from ..io.writer import prepare_out, write_meta, snapshot  # 数据输出
from ..viz.liveplot import LivePlotter  # 实时可视化

# 创建一个日志记录器
logger = logging.getLogger(__name__)


# 模拟器类：负责初始化、运行和实时可视化
class Simulator:
    # 初始化方法 初始化模拟器，设置网格、随机数生成器、输出目录、可视化工具等
    def __init__(self, cfg: dict):
        self.cfg = cfg  # 配置字典
        self.grid = create_grid(cfg["domain"])  # 创建计算网格
        self.rng = np.random.default_rng(cfg["run"]["seed"])  # 随机数生成器
        self.out = prepare_out(cfg["run"]["output_dir"])  # 输出目录
        write_meta(cfg, self.out)  # 写入元数据
        update_ghosts(self.grid, cfg["domain"]["bc"])  # 更新ghost层
        self.live = LivePlotter(
            self.cfg.get("viz", {}).get("live", {})
        )  # 实时可视化工具
        init_cfg = self.cfg.get("init", {})  # 手动晶粒初始化配置
        if init_cfg:
            placed = seed_initialize(self.grid, self.rng, init_cfg)
            logger.info("Init seeds placed: %d", placed)

    # 运行方法
    def run(self):
        # 时间参数 存储参数
        dt = self.cfg["time"]["dt"]
        t_end = self.cfg["time"]["t_end"]
        save_every = self.cfg["time"]["save_every"]

        t = 0.0
        step = 0
        # 启动实时显示
        self.live.start(self.grid)
        try:
            # 时间循环
            while t < t_end:
                step += 1
                t += dt

                # 预处理：更新网格边界条件
                update_ghosts(self.grid, self.cfg["domain"]["bc"])

                # 温度场更新
                self.grid.T[:] = sample_T(self.grid, t, self.cfg.get("temperature", {}))

                # 确定固相、液相、界面区域 生成掩膜
                masks = classify_phases(self.grid.fs, self.grid.nghost)

                # 处理异质形核
                nucl_apply(self.grid, self.rng, self.cfg.get("nucleation", {}), masks)
                logger.info("Nucleation done (stub)")

                # 计算界面相关的物理量 主要是CL^*
                fields = compute_interface_fields(
                    self.grid,
                    self.cfg.get("physics", {}).get("interface", {}),
                    self.cfg.get("physics", {}).get("orientation", {}),
                    masks,
                )
                logger.info("Interface fields computed (stub)")

                # 保存旧固相分数场以计算变化率
                fs_old = self.grid.fs.copy()

                # 处理MDCS捕获 更新固相分数
                mdcs_step(
                    self.grid,
                    fields,
                    self.cfg.get("physics", {}).get("mdcs", {}),
                    self.cfg.get("physics", {}).get("orientation", {}),
                    dt,
                    masks,
                )
                logger.info("MDCS done (stub)")

                # 计算固相分数场变化率
                fs_dot = (self.grid.fs - fs_old) / dt

                # 处理溶质场
                solute_advance(
                    self.grid,
                    self.cfg.get("physics", {}).get("solute", {}),
                    dt,
                    masks,
                    fs_dot=fs_dot,
                )
                logger.info("solute_advance done (stub)")

                # 计算总溶质质量
                M = total_solute_mass(self.grid)
                logger.info(f"Total solute mass (diag): {M:.6e}")

                # 后处理
                # 保存快照
                if step % save_every == 0:
                    snapshot(self.grid, t, step, self.out)

                # 更新实时显示
                self.live.update(self.grid, t, step)

            # 保存最后快照
            snapshot(self.grid, t, step, self.out)
        finally:
            # 关闭窗口
            self.live.close()
