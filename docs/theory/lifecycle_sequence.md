# 时间推进序列（Lifecycle） — v0.1（2025-08-14）

> 本页是**流程契约**：定义一次时间步里“先做什么、谁能改什么、何时写文件”。它面向人阅读与实现对照，非可执行代码。

## 1. 文档目的

* 固定先后顺序，避免数值/逻辑歧义。
* 将副作用（I/O、日志、随机）集中到引擎统一管理。
* 为实现提供“填空题”骨架；任何模块替换只要遵守本契约即可。

## 2. 参与者（角色）

* **Engine（引擎/调度）**：时间循环、ghost 同步、日志、快照与检查点。
* **TemperatureAdapter（温度适配器）**：按 (x,y,t) 提供温度 *T*（派生量，不入快照）。
* **Nucleation（形核）**：基于欠冷的 Thévoz 统计，决定新核（写 `grain_id, theta(/theta_class)`）。
* **Interface（界面平衡）**：计算界面处 `C_L^*, C_S^*`、各向异性因子、曲率等（派生量）。
* **GrowthRule/MDCS（几何生长）**：由 `V_n` 与法向更新 `Δfs, L_dia`，捕获新胞（写 `fs, L_dia`）。
* **SoluteSolver（溶质守恒）**：体积分数加权 + 成对源/汇项，更新 `CL, CS`。
* **Writer/Logger**：仅由 Engine 调用；模块本身不得触碰文件系统。

## 3. 状态分层（只读/持久/临时）

* **Config（只读）**：来自配置文件的常量与开关（例：`dt, k, D_L, fold, N_theta`）。
* **Grid（持久）**：`fs, CL, CS, grain_id, theta, L_dia, (lock/state)`，含 ghost 层；快照/重启所需。
* **Derived（派生量，临时）**：`T, ΔT, nx, ny, kappa, ani, V_n, C_L^*, C_S^* ...`，当步生成，用完丢弃。
* **Workspace（模块工作区）**：系数矩阵、残差、MDCS 局部几何、前沿列表等，仅模块内部复用。
* **External data**：外部温度文件等；配置只给路径与插值方式。

## 4. 不变量与约束（每步在 core 上检查）

* 值域：`0 ≤ fs ≤ 1`；`CL, CS` 为有限数（非 NaN/Inf）。
* 守恒：零通量边界下，总溶质量变化 `|ΔM| ≤ tol`（阈值 `tol` 来自配置）。
* 取向：`theta` 规范到 `[0, 2π)`；若启用 `theta_class`，其取值为 `{哨兵} ∪ [0, Nθ]`。
* 边界：任何计算仅读写 core；ghost 区域由边界更新器统一写入。

## 5. 副作用与随机性纪律

* I/O（快照、检查点、meta、日志）**只允许** Engine → Writer/Logger；其他模块禁止文件操作。
* 随机性统一由 Engine 注入 `rng`；禁止模块内创建全局随机源。
* 前沿列表的更新与遍历顺序固定（如行优先），保证可复现。

## 6. 单步时间推进顺序（自然语言）

1. **Pre-step**：同步 ghost；采样温度 `T`；由 `fs` 得相掩码（液/界/固）。
2. **Nucleation**：在液相依据欠冷与 Thévoz 统计产生新核；写入 `grain_id, theta(/class)`，必要时微增 `fs`。
3. **Interface（派生）**：在界面元胞计算 `C_L^*, C_S^*`、各向异性因子与曲率；不改持久场。
4. **GrowthRule/MDCS**：按 `V_n` 与法向推进几何，写回 `fs, L_dia`（并据需要扩展 `grain_id`）。
5. **SoluteSolver**：按体积分数加权方程更新 `CL, CS`，加入 `(1-k)·CL·∂fs/∂t` 的成对源/汇；可选回扩散与限幅。
6. **Post-step**：检查不变量；到保存周期写快照；到检查点周期写检查点（含 `rng` 状态）；记日志；满足早停条件则提前结束。

## 7. 伪代码骨架（钉死顺序与 I/O 位置）

```text
init(cfg):
  grid  ← create_grid(cfg.domain, seed=cfg.run.seed)
  rng   ← make_rng(cfg.run.seed)
  t, step ← 0.0, 0
  Writer.write_meta(cfg)
  update_ghosts(grid, cfg.domain.bc)

while t < cfg.time.t_end:
  step += 1
  t    += cfg.time.dt

  # Pre-step
  update_ghosts(grid, cfg.domain.bc)
  Tbuf   ← TemperatureAdapter.sample(grid, t, cfg.temperature)
  masks  ← classify_phases(grid.fs)

  # Core physics
  Nucleation.apply(grid, Tbuf, rng, cfg.nucleation, masks)                               # 写 grain_id/theta
  Interface.equilibrium(grid, Tbuf, cfg.physics.interface, cfg.physics.orientation, masks) # 仅派生量
  GrowthRule.MDCS.step(grid, cfg.physics.mdcs, cfg.physics.orientation, dt=cfg.time.dt, masks=masks) # 写 fs/L_dia
  SoluteSolver.advance(grid, cfg.physics.solute, dt=cfg.time.dt, masks=masks)            # 写 CL/CS

  # Post-step
  assert_invariants(grid, cfg.debug)
  if step % cfg.time.save_every == 0: Writer.snapshot(grid, t, step, cfg.run)
  if cfg.time.checkpoint_every>0 and step % cfg.time.checkpoint_every == 0:
      Writer.checkpoint(grid, t, step, rng_state=rng.state)
  Logger.tick(step, t, grid)
  if early_stop(grid): break

Writer.snapshot(grid, t, step, cfg.run)  # 结束补一次
```

## 8. 阶段 I/O 与副作用一览

| 阶段           | 读                 | 写(持久)                              | 产出(派生/临时)                                | 允许的副作用                                    |
| ------------ | ----------------- | ---------------------------------- | ---------------------------------------- | ----------------------------------------- |
| Pre-step     | `grid`            | —                                  | `Tbuf`、相掩码                               | 无                                         |
| Nucleation   | `grid, Tbuf, rng` | `grain_id, theta(/class), (少量 fs)` | —                                        | 无                                         |
| Interface    | `grid, Tbuf`      | —                                  | `C_L^*, C_S^*`、`nx, ny, kappa, ani, V_n` | 无                                         |
| GrowthRule   | `grid, 派生量`       | `fs, L_dia`（必要时扩展 `grain_id`）      | 前沿列表/几何临时量                               | 无                                         |
| SoluteSolver | `grid, ∂fs/∂t`    | `CL, CS`                           | 系数/残差                                    | 无                                         |
| Post-step    | `grid, t, step`   | —                                  | —                                        | **Writer.snapshot/checkpoint**、**Logger** |

## 9. 早停条件与结束

* **全固化**：core 区 `fs` 接近 1（阈值可由配置给定）。
* **无前沿**：界面元胞集合为空。
* **达到终止条件**：`t ≥ t_end` 或外部中断信号。
* 结束时**仍需**补写一次快照，保证结果闭环。

## 10. 变更记录

* v0.1（2025-08-14）：首版，约定 Pre → Nucleation → Interface → MDCS → Solute → Post 的固定顺序；I/O 与随机性集中在 Engine。
