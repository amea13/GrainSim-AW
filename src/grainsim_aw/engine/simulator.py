import numpy as np  # numpy做数值计算
from typing import Optional  # typing 确保类型注解
import logging  # logging 记录日志

from ..core.grid import (
    create_grid,
    update_ghosts,
    classify_phases,
)  # 管理计算网格，包括创建和更新边界ghost cells
from ..nucleation import apply as step_nucleation  # Thevoz方法异质形核
from ..nucleation import seed_initialize
from ..interface import compute_interface_fields  # 计算界面相关的场
from ..growth_capture import capture_pass, advance_no_capture  # 处理MDCS捕获
from ..multiphysics import solute_advance, total_solute_mass  # 溶质场偏微分方程求解
from ..multiphysics import sample_T  # 温度场加载
from ..io.writer import prepare_out, write_meta, snapshot  # 数据输出
from ..viz.liveplot import LivePlotter  # 实时可视化

# 创建一个日志记录器
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


# 模拟器类：负责初始化、运行和实时可视化
class Simulator:
    # 初始化方法 初始化模拟器，设置网格、随机数生成器、输出目录、可视化工具等
    def __init__(self, cfg: dict):
        self.cfg = cfg  # 配置字典
        self.grid = create_grid(cfg["domain"])  # 创建计算网格
        C0 = float(self.cfg.get("physics", {}).get("interface", {}).get("C0", 0.0))
        self.grid.T[:] = sample_T(self.grid, 0.0, self.cfg.get("temperature", {}))
        self.grid.CL[:] = C0
        self.grid.fs[:] = 0.0
        self.rng = np.random.default_rng(cfg["run"]["seed"])  # 随机数生成器
        self.out = prepare_out(cfg["run"]["output_dir"])  # 输出目录
        write_meta(cfg, self.out)  # 写入元数据
        update_ghosts(self.grid, cfg["domain"]["bc"])  # 更新ghost层
        init_cfg = dict(self.cfg.get("init", {}))
        init_cfg.setdefault(
            "k0",
            float(self.cfg.get("physics", {}).get("interface", {}).get("k0", 0.34)),
        )
        if init_cfg:
            placed = seed_initialize(self.grid, self.rng, init_cfg)
            logger.info("Init seeds placed: %d", placed)
        self.live = LivePlotter(
            self.cfg.get("viz", {}).get("live", {})
        )  # 实时可视化工具

    def _pick_seed_and_ring(self, grid, masks):
        """返回 (i0,j0), ring[<=4个(i,j)] —— 同一 grain_id 的核心种子与其最近的4个界面元。"""
        g = grid.nghost
        Ny, Nx = grid.fs.shape
        # 1) 找核心种子：fs≈1 且在 core
        seeds = np.argwhere((grid.fs > 1 - 1e-12) & masks["mask_sol"])
        if seeds.size == 0:
            return None, []
        # 选“最靠近 core 几何中心”的那个（single_center 场景就是它）
        ic = g + grid.ny / 2.0
        jc = g + grid.nx / 2.0
        k = np.argmin(np.hypot(seeds[:, 0] - ic, seeds[:, 1] - jc))
        i0, j0 = map(int, seeds[k])

        gid = grid.grain_id[i0, j0]
        # 2) 该 grain 的界面元
        ring_all = np.argwhere((grid.grain_id == gid) & masks["mask_int"])
        if ring_all.size == 0:
            return (i0, j0), []
        # 取最近的4个（理论上就是一次性感染的那4个）
        d = np.hypot(ring_all[:, 0] - i0, ring_all[:, 1] - j0)
        idx = np.argsort(d)[:4]
        ring = [tuple(map(int, ij)) for ij in ring_all[idx]]
        return (i0, j0), ring

    def _debug_check_seed_init(self, grid, seed, ring):
        """打印核心与四界面胞在“初始化后”的一致性（fs/L_dia/取向/偏心/成分）"""
        if seed is None or len(ring) == 0:
            logger.info("CHK init: 未找到界面元")
            return
        i0, j0 = seed
        th = float(grid.theta[i0, j0])
        Lmax = grid.dx / max(abs(np.sin(th)), abs(np.cos(th)), 1e-12)

        vals = []
        for i, j in ring:
            vals.append(
                dict(
                    ij=(i, j),
                    fs=float(grid.fs[i, j]),
                    Ldia=float(grid.L_dia[i, j]),
                    Ldia_over_fsLmax=float(
                        grid.L_dia[i, j] / max(grid.fs[i, j] * Lmax, 1e-30)
                    ),
                    gid=int(grid.grain_id[i, j]),
                    th=float(grid.theta[i, j]),
                    ecc=(float(grid.ecc_x[i, j]), float(grid.ecc_y[i, j])),
                    CL=float(grid.CL[i, j]),
                    CS=float(grid.CS[i, j]),
                )
            )
        logger.info(
            "CHK init: seed=(%d,%d) gid=%d th=%.3f",
            i0,
            j0,
            int(grid.grain_id[i0, j0]),
            th,
        )
        for v in vals:
            logger.info(
                "  ring %s: fs=%.4g, Ldia=%.3g, Ldia/(fs·Lmax)=%.3f, "
                "gid=%d, th=%.3f, ecc=(%.3g,%.3g), CL=%.4g, CS=%.4g",
                v["ij"],
                v["fs"],
                v["Ldia"],
                v["Ldia_over_fsLmax"],
                v["gid"],
                v["th"],
                v["ecc"][0],
                v["ecc"][1],
                v["CL"],
                v["CS"],
            )

    def _debug_check_iface_step1(self, grid, masks, seed, ring, fields):
        """打印四界面胞在“算完 CL*、各向异性、Vn 等”后的关键量是否一致"""
        if seed is None or len(ring) == 0:
            logger.info("CHK iface: 无 ring")
            return
        for i, j in ring:
            logger.info(
                "CHK iface @(%d,%d): nx=%.3g ny=%.3g kappa=%.3g ani=%.3f | "
                "CLs=%.5g CSs=%.5g | Vx=%.3g Vy=%.3g Vn=%.3g",
                i,
                j,
                float(fields.nx[i, j]),
                float(fields.ny[i, j]),
                float(fields.kappa[i, j]),
                float(fields.ani[i, j]),
                float(fields.CLs[i, j]),
                float(fields.CSs[i, j]),
                float(fields.Vx[i, j]),
                float(fields.Vy[i, j]),
                float(fields.Vn[i, j]),
            )

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
        masks = classify_phases(self.grid.fs, self.grid.nghost)
        try:
            # 时间循环
            while t < t_end:
                step += 1
                t += dt

                # A) 先更新 ghosts & 掩码
                update_ghosts(self.grid, self.cfg["domain"]["bc"])

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

                if step <= 100:
                    seed, ring = self._pick_seed_and_ring(self.grid, masks)
                    self._debug_check_seed_init(self.grid, seed, ring)

                # D) 捕捉后，ghosts 与 masks 已过期；立刻重算（供 Vn 与推进使用）
                update_ghosts(self.grid, self.cfg["domain"]["bc"])
                masks = classify_phases(self.grid.fs, self.grid.nghost)

                logger.info(
                    "AFTER-CAPTURE masks: int=%d sol=%d liq=%d",
                    int(masks["mask_int"].sum()),
                    int(masks["mask_sol"].sum()),
                    int(masks["mask_liq"].sum()),
                )

                # E) 计算界面热力学平衡（Vn、nx,ny、κ、各向异性、C* 等）
                fields = compute_interface_fields(
                    self.grid,
                    self.cfg.get("physics", {}).get("interface", {}),
                    self.cfg.get("physics", {}).get("orientation", {}),
                    masks,
                )

                if step == 1:
                    seed, ring = self._pick_seed_and_ring(self.grid, masks)
                    self._debug_check_iface_step1(self.grid, masks, seed, ring, fields)
                # —— 在 fields = compute_interface_fields(...) 之后，加这一块 —— #
                if step <= 3:  # 只在前几步打印，防刷屏
                    from grainsim_aw.growth_capture.kernels import centroid_normal
                    from grainsim_aw.growth_capture.geometry import L_n, shape_factor_GF

                    mm = masks["mask_int"]
                    int_cnt = int(mm.sum())

                    # 法向、Ln、GF
                    nx_c, ny_c = centroid_normal(self.grid.fs)
                    Ln = L_n(nx_c, ny_c, self.grid.dx, self.grid.dy)
                    GF = shape_factor_GF(self.grid.fs, nx_c, ny_c, masks)

                    # 统计（只看界面带）
                    def stats(arr):
                        if int_cnt == 0:
                            return (np.nan, np.nan, np.nan)
                        v = arr[mm]
                        return (
                            float(np.nanmin(v)),
                            float(np.nanmean(v)),
                            float(np.nanmax(v)),
                        )

                    Vn = fields.Vn
                    vmin, vavg, vmax = stats(Vn)
                    lmin, lavg, lmax = stats(Ln)
                    gmin, gavg, gmax = stats(GF)

                    # 估算 Δfs（不写回，仅打印）
                    dt = self.cfg["time"]["dt"]
                    eps = 1e-30
                    df_est = np.zeros_like(self.grid.fs)
                    df_est[mm] = GF[mm] * Vn[mm] * dt / np.maximum(Ln[mm], eps)
                    dmin, davg, dmax = stats(df_est)

                    logger.info(
                        "DBG step=%d: int_cnt=%d | Vn[min/avg/max]=%.3g/%.3g/%.3g | "
                        "Ln[min/avg/max]=%.3g/%.3g/%.3g | GF[min/avg/max]=%.3g/%.3g/%.3g | "
                        "dfs_est[min/avg/max]=%.3g/%.3g/%.3g",
                        step,
                        int_cnt,
                        vmin,
                        vavg,
                        vmax,
                        lmin,
                        lavg,
                        lmax,
                        gmin,
                        gavg,
                        gmax,
                        dmin,
                        davg,
                        dmax,
                    )

                    # 选一个界面元，打印关键量（定位到具体数值）
                    if int_cnt > 0:
                        iy, ix = np.argwhere(mm)[0]
                        logger.info(
                            "DBG probe @(%d,%d): fs=%.4g, Vn=%.4g, nx=%.4g, ny=%.4g, Ln=%.4g, GF=%.4g, est_dfs=%.4g",
                            int(iy),
                            int(ix),
                            float(self.grid.fs[iy, ix]),
                            float(Vn[iy, ix]),
                            float(nx_c[iy, ix]),
                            float(ny_c[iy, ix]),
                            float(Ln[iy, ix]),
                            float(GF[iy, ix]),
                            float(df_est[iy, ix]),
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
                    CL_star=fields.CLs,
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
