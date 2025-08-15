# 模块契约卡片（Contracts）— v0.1（2025-08-15）

> 本页给出关键模块的职责、输入/输出、不变量、依赖与测试要点。用于实现与评审的“对照清单”。

---

## 0. 通用约定

* 本页术语与数据结构以《data\_model.md》与《lifecycle\_sequence.md》为准。
* **只读 vs. 可写**：模块默认仅读 `Grid.core`；只有明确声明“写回持久字段”的步骤可以改写。
* **随机性**：统一由 Engine 注入 `rng`；模块不得创建全局随机源。
* **I/O**：除 Writer 外，任何模块不得触碰文件系统。

---

## 1) Core / Grid

**功能定位**：网格与字段的容器；提供 ghost 同步与常见掩码工具。

**关键接口（签名示意）**

* `Grid.create(nx, ny, dx, dy, nghost, seed) -> Grid`
* `update_ghosts(grid, bc) -> None`
* `classify_phases(fs, thresh=(f_liq, f_sol)) -> {mask_liq, mask_int, mask_sol}`

**输入与输出**

* 入：尺寸、步长、nghost、边界类型。
* 出：持久字段数组（`fs, CL, CS, grain_id, theta, L_dia, (lock/state)`），以及常用掩码（派生）。

**不变量**

* 形状 `(Ny, Nx)` 与 dtype（float64 / int32 / uint8）恒定。
* `.core` 内 `0 ≤ fs ≤ 1`；`CL, CS` 有限。
* ghost 仅由 `update_ghosts` 写入；统计只在 `.core` 上进行。

**依赖**：无（基础层）。

**错误与边界**

* 输入尺寸/步长非正 → 抛错。
* `nghost < 需求核半径` → 抛错并提示 MDCS/离散模板所需最小值。

**性能注意**：字段 C-order，避免 per-cell Python 循环。

**测试要点**

* 创建后三大场形状一致；`fs` 值域合法；ghost 同步后边界复制/周期行为正确。

---

## 2) Engine / Simulator

**功能定位**：调度时间推进；集中管理 ghost、随机源、日志、快照。

**关键接口**

* `Simulator.from_config(cfg) -> Simulator`
* `Simulator.run() -> None`
* `Simulator.step(dt) -> None`（内部依次调用各模块接口，顺序见 lifecycle）

**输入与输出**

* 入：`cfg`、模块句柄（或函数指针）。
* 出：副作用（快照、检查点、日志）；持久状态更新由各模块完成。

**不变量**

* 调用顺序固定：Pre → Nucleation → Interface → MDCS → Solute → Post。
* 只有 Engine 允许 I/O；模块只能通过返回值或写回 Grid 传递信息。

**依赖**：Core/Grid、Nucleation、Interface、GrowthRule/MDCS、SoluteSolver、TemperatureAdapter、Writer、Logger。

**错误与边界**

* 模块抛出的异常向上冒泡；Engine 负责加步号/时间上下文并中止运行。

**测试要点**

* 在“空物理”实现下能按 `save_every` 产生快照；早停条件触发正确。

---

## 3) TemperatureAdapter

**功能定位**：提供 `(x,y,t)` 上的温度场采样与插值（派生量）。

**关键接口**

* `sample(grid, t, cfg_temperature) -> Tbuf`（形状 `(Ny, Nx)`；或懒加载/分块）

**输入与输出**

* 入：Grid（用于坐标）、时间 `t`、适配器配置（constant/file/function、插值方式）。
* 出：`Tbuf`（临时）；不得修改持久字段。

**不变量**：`Tbuf` 与 Grid 形状匹配；不写回 Grid。

**依赖**：Core/Grid、外部数据读取器（若 mode=file）。

**错误与边界**

* 缺少文件或数据集 → 抛出带路径信息的异常。
* 插值越界 → 使用边界外推策略或抛错（由配置决定）。

**测试要点**

* 常数模式返回常量数组；文件模式在规则网格上与解析函数基准一致。

---

## 4) Nucleation / Thévoz

**功能定位**：在液相按 Thévoz–Rappaz 正态分布统计激活形核位点，写入 `grain_id` 与取向 `theta(/class)`；必要时微增 `fs`。

**关键接口**

* `apply(grid, Tbuf, rng, cfg_nucl, masks) -> None`

**输入与输出**

* 入：液相掩码、温度/过冷、`N_max, ΔT_n, σ_n` 等配置、随机源。
* 出：更新 `grain_id`、`theta(/theta_class)`；可对命中胞 `fs += ε`（可选）。

**不变量**

* 仅在液相单元尝试形核；同一步内同一单元最多被赋一次 `grain_id`。
* 取向：采用“扇区中心抽样”：`θ_local=((c+0.5)/Nθ-0.5)Δ`，`Δ=2π/fold`。

**依赖**：TemperatureAdapter（用于过冷）、Orientation 参数（`fold, Nθ`）。

**错误与边界**

* `N_max ≤ 0` 或 `σ_n ≤ 0` → 抛错；随机序列必须来自注入的 `rng`。

**测试要点**

* 大样本下激活率随过冷符合正态累积分布；`seed` 固定时结果可复现。

---

## 5) Interface / Equilibrium（热力学平衡与几何派生）

**功能定位**：在界面元胞求 `C_L^*、C_S^*`，并计算各向异性因子与曲率等供生长与溶质使用（派生量，不改持久）。

**关键接口**

* `equilibrium(grid, Tbuf, cfg_if, cfg_orient, masks) -> Derived`
  返回需要的派生量视图（如 `C_L_star, C_S_star, Vn, nx, ny, kappa, ani`）。

**输入与输出**

* 入：界面掩码、温度、物性（`m_L, T_L_eq, Γ, k`）、取向参数。
* 出：派生量；不写回 Grid。

**不变量**

* 局部平衡：`T* = T_L_eq + (C_L^*-C_0)m_L − Γ κ f(φ,θ)`；`C_S^* = k C_L^*`。
* 任何派生量 NaN/Inf 需回退/限幅，并记录诊断计数。

**依赖**：Core/Grid；可能复用 GrowthCapture 的法向/曲率工具。

**测试要点**

* 一维稳态基准下 `C_L^*` 与解析一致；各向异性函数周期性与峰位正确。

---

## 6) GrowthRule / MDCS（几何更新与捕获）

**功能定位**：以 MDCS 规则推进界面几何，更新 `fs` 与 `L_dia`，并捕获符合条件的液相元胞归入晶粒。

**关键接口**

* `step(grid, cfg_mdcs, cfg_orient, dt, masks) -> None`

**内部子流程（固定顺序）**

1. 前沿遍历：从 `mask_int` 得到界面元胞列表 `front`（行优先，保证可复现）。
2. 圆核质心：以直径 `7Δx` 的核求远重心 `B` → 法向 `n=(nx,ny)`；计算 `L_n`。
3. 更新量：`Δfs = GF * V_n * dt / L_n`；剪裁到 `[0,1]`；`ΔL_dia = Δfs * L_dia^max`。
4. 写回：`fs += Δfs`，`L_dia += ΔL_dia`；计算临时顶点 `P1..P4` 并做 **capture**，设置新胞 `grain_id`。
5. 刷新前沿：更新 `mask_int`/`front`。

**输入与输出**

* 入：派生量（`Vn, nx, ny, ani, kappa` 等）、MDCS 配置（`GF, kernel_radius_dx, fs_capture_threshold`）。
* 出：写回 `fs, L_dia`；必要时写入新 `grain_id`。

**不变量**

* `0 ≤ fs ≤ 1`；单步内 `fs` 非递减（若模型假设如此）。
* 仅修改 `.core`；ghost 在步首/必要时由 Engine 统一更新。

**依赖**：Core/Grid、Interface 派生量。

**性能注意**：使用稀疏 `front`；避免全域 dense 计算；向量化或小循环。

**测试要点**

* 在无各向异性且匀速 `Vn` 下，圆形/方形模板基准增长速率正确；capture 只在阈值之上发生。

---

## 7) SoluteSolver（体积分数加权守恒）

**功能定位**：在整域上求解无对流/（可选）对流的溶质守恒，写回 `CL, CS`，并保证守恒。

**关键接口**

* `advance(grid, cfg_sol, dt, masks) -> None`

**输入与输出**

* 入：`fs` 的时间导数或增量（由 Engine 或本模块内部差分得到）、`D_L, D_S, k`、离散方案与限幅器。
* 出：写回 `CL, CS`。

**离散原则**（无对流）

* `(L)`：`∂(αCL)/∂t = ∇·(α D_L ∇CL) + (1−k)CL ∂f_s/∂t`
* `(S)`：`∂((1−α)CS)/∂t = ∇·((1−α) D_S ∇CS) − (1−k)CL ∂f_s/∂t`
* 面通量按相开口系数闸门（如 `α_f = min(α_P, α_N)`），避免固/液之间虚假泄漏。

**不变量**

* 零通量边界下总溶质量守恒在阈值内。
* `CL, CS` 有限；若越界按配置限幅。

**依赖**：Core/Grid（掩码与 ghost），MDCS 提供的 `∂f_s/∂t` 或 `Δf_s`。

**测试要点**

* 一维纯扩散解析解对比；守恒误差随时间累积受控。

---

## 8) Writer（快照 / 检查点 / 元数据）

**功能定位**：写出持久状态快照、检查点与 meta；保证可复现与安全写入。

**关键接口**

* `write_meta(cfg) -> None`
* `snapshot(grid, t, step, run_cfg) -> Path`
* `checkpoint(grid, t, step, rng_state) -> Path`

**输入与输出**

* 入：Grid 持久字段、时间/步号、运行配置（路径、格式、字段列表）。
* 出：磁盘文件（npz/hdf5/zarr 等）。

**不变量**

* 仅写《data\_model.md》中列为“持久字段”的数组；调试派生量需在配置白名单中。
* 原子化写入：先写临时文件再重命名，避免中途崩溃导致半文件。

**依赖**：文件系统；可选压缩库。

**错误与边界**

* 目录不可写或磁盘满 → 抛错并附带路径；Writer 不做重试。

**测试要点**

* 快照能被成功读回；字段集合与形状匹配；meta 中包含完整配置与种子。

---

### 备注：模块间最小依赖图

```
TemperatureAdapter ─┐
                     ├─> Interface/Equilibrium ─┐
Nucleation ─────────────────────────────────────┤
                                               ├─> GrowthRule/MDCS ──> SoluteSolver
Core/Grid  ─────────────────────────────────────┘
Engine ──(调度/I-O/随机)────────────────────────────────────────> Writer/Logger
```

> 任何新增模块/字段请先按本页格式补齐“契约卡片”，并在 data\_model.md 与 lifecycle\_sequence.md 中登记或更新顺序。
