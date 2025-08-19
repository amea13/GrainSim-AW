import numpy as np  # numpy做数值计算
from typing import Optional  # typing 确保类型注解
import logging  # logging 记录日志

from ..core.grid import (
    create_grid,
    update_ghosts,
    classify_phases,
)  # 管理计算网格，包括创建和更新边界ghost cells
from ..nucleation import apply as step_nucleation  # Thevoz方法异质形核
from ..nucleation import (
    initialize as seed_initialize,
)  # 手动初始化晶核
from ..interface import compute_interface_fields  # 计算界面相关的场
from ..growth_capture import capture_pass, advance_no_capture  # 处理MDCS捕获
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

                # A) 先更新 ghosts & 掩码
                update_ghosts(self.grid, self.cfg["domain"]["bc"])
                masks = classify_phases(self.grid.fs, self.grid.nghost)

                # B) Thevoz 形核：返回本步新核掩码（seeds_mask），并在 grid 上把核元设置好：
                #    fs=1, L_dia=Lmax(theta), ecc=0, grain_id/theta 赋值
                seeds_mask = step_nucleation(
                    self.grid, self.rng, self.cfg.get("nucleation", {}), masks
                )

                # C) 全局“捕捉优先”——把“旧界面带 ∪ 新核”统一作为父胞执行一次捕捉
                seeds_mask = seeds_mask  # 你的 thevoz 返回值
                if seeds_mask is None:
                    parent_mask = masks["mask_int"]
                else:
                    parent_mask = (
                        masks["mask_int"] | seeds_mask
                    )  # 两者都是 bool ndarray
                capture_pass(
                    self.grid,
                    masks,
                    self.cfg["physics"]["mdcs"],
                    parent_mask=parent_mask,
                )

                # D) 捕捉后，ghosts 与 masks 已过期；立刻重算（供 Vn 与推进使用）
                update_ghosts(self.grid, self.cfg["domain"]["bc"])
                masks = classify_phases(self.grid.fs, self.grid.nghost)

                # E) 计算界面热力学平衡（Vn、nx,ny、κ、各向异性、C* 等）
                fields = compute_interface_fields(
                    self.grid,
                    self.cfg.get("physics", {}).get("interface", {}),
                    self.cfg.get("physics", {}).get("orientation", {}),
                    masks,
                )

                # F) 仅推进 Δfs / L_dia
                fs_dot = advance_no_capture(
                    self.grid,
                    fields,
                    self.cfg.get("physics", {}).get("mdcs", {}),
                    dt,
                    masks,
                )

                # G) 溶质/温度更新（使用最新 fs 与 fs_dot）
                solute_advance(
                    self.grid,
                    self.cfg.get("physics", {}).get("solute", {}),
                    dt,
                    masks,
                    fs_dot=fs_dot,
                )
                self.grid.T[:] = sample_T(self.grid, t, self.cfg.get("temperature", {}))
                # 计算总溶质质量
                M = total_solute_mass(self.grid)

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
