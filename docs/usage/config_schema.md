# 配置键表与校验规则（config\_schema）—— v0.2（2025-08-17）

本文件定义可读写配置的键、类型、默认值与校验规则。目标是让配置成为可验证的外部契约。

## 0. 约定

* 格式：TOML
* 数值类型：浮点默认使用 SI 单位，浓度单位需显式声明
* 角度单位：弧度
* 数组形状：逻辑域为 `(ny, nx)`，存储域包含 ghost 层

---

## 1. \[project]

| 键       | 类型  | 默认            | 约束与说明            |
| ------- | --- | ------------- | ---------------- |
| name    | str | "dendrite-ca" | 仅用于元数据标识         |
| version | str | 自动注入          | 由程序写入 git 提交或包版本 |

## 2. \[domain]

| 键          | 类型    | 默认         | 约束与说明                                                |
| ---------- | ----- | ---------- | ---------------------------------------------------- |
| nx         | int   | 必填         | nx > 0                                               |
| ny         | int   | 必填         | ny > 0                                               |
| dx         | float | 必填         | dx > 0 \[m]                                          |
| dy         | float | 必填         | dy > 0 \[m]                                          |
| nghost     | int   | 3          | ≥ 0。若留空可由程序按所用模板自动取最大需求                              |
| bc         | str   | "neumann0" | 边界条件。可选 "neumann0"、"dirichlet"、"periodic"。也可用分向量写法见下 |
| bc\_x      | str   | 继承 bc      | 若提供则覆盖 x 方向边界类型                                      |
| bc\_y      | str   | 继承 bc      | 若提供则覆盖 y 方向边界类型                                      |
| bc\_values | table | 无          | 当使用 dirichlet 时需给出 fs, CL, CS, T 的边界值表               |

示例：

```toml
[domain.bc_values]
fs = 0.0
CL = 0.02
CS = 0.02
T = 1800.0
```

## 3. \[time]

| 键                 | 类型    | 默认 | 约束与说明                        |
| ----------------- | ----- | -- | ---------------------------- |
| dt                | float | 必填 | dt > 0 \[s]                  |
| t\_end            | float | 必填 | t\_end ≥ dt                  |
| save\_every       | int   | 50 | ≥ 1，按步保存快照                   |
| checkpoint\_every | int   | 0  | 0 表示关闭检查点。大于 0 表示每隔该步数写一次检查点 |

## 4. \[run]

| 键                | 类型          | 默认                                                        | 约束与说明                               |
| ---------------- | ----------- | --------------------------------------------------------- | ----------------------------------- |
| seed             | int         | 0                                                         | 任意整数。作为统一随机源种子                      |
| output\_dir      | str         | "data/output/run"                                         | 可写目录。若存在会追加时间戳子目录                   |
| snapshot\_format | str         | "npz"                                                     | 可选 "npz"、"hdf5"、"zarr"              |
| snapshot\_fields | array\[str] | \["fs","CL","CS","grain\_id","theta","L\_dia","lock","T"] | 写入的持久字段清单                           |
| log\_level       | str         | "INFO"                                                    | 可选 "DEBUG"、"INFO"、"WARNING"、"ERROR" |

## 5. \[physics]

### 5.1 通用

| 键           | 类型  | 默认     | 约束与说明                              |
| ----------- | --- | ------ | ---------------------------------- |
| unit\_of\_C | str | "mass" | 可选 "mass" 或 "atomic"，统一 CL 与 CS 单位 |

### 5.2 取向与各向异性 \[physics.orientation]

| 键                    | 类型    | 默认          | 约束与说明                          |
| -------------------- | ----- | ----------- | ------------------------------ |
| fold                 | int   | 4           | 晶体平面内对称数，四折对称取 4               |
| N\_theta             | int   | 48          | 每个主扇区内的离散类数                    |
| class\_liq\_sentinel | int   | -10         | 液相取向类的哨兵值                      |
| anisotropy\_epsilon  | float | 0.02        | 0 ≤ epsilon < 1，用于张力各向异性因子     |
| anisotropy\_mode     | str   | "cos\_fold" | 常用形式为 `1 + ε cos(fold·θ)`，预留扩展 |

### 5.3 界面热平衡与几何 \[physics.interface]

| 键              | 类型    | 默认        | 约束与说明                             |
| -------------- | ----- | --------- | --------------------------------- |
| m\_L           | float | 必填        | 液相线斜率 \[K per unit C]             |
| T\_L\_eq       | float | 必填        | 基准液相线温度 \[K]                      |
| Gamma          | float | 必填        | Gibbs Thomson 系数 \[K·m]           |
| k              | float | 必填        | 0 ≤ k ≤ 1，分配系数                    |
| anisotropy\_fn | str   | "default" | 各向异性因子函数名，默认与 anisotropy\_mode 一致 |

### 5.4 生长规则 MDCS \[physics.mdcs]

| 键                  | 类型  | 默认 | 约束与说明                      |
| ------------------ | --- | -- | -------------------------- |
| kernel\_radius\_dx | int | 3  | 圆核半径，单位为格点数。典型为 3 对应直径 7Δx |

### 5.5 溶质场 \[physics.solute]

| 键       | 类型    | 默认               | 约束与说明                                     |
| ------- | ----- | ---------------- | ----------------------------------------- |
| D\_L    | float | 必填               | 液相扩散系数 \[m^2 s^-1]，≥ 0                    |
| D\_S    | float | 必填               | 固相扩散系数 \[m^2 s^-1]，≥ 0                    |
| scheme  | str   | "semi\_implicit" | 可选 "explicit"、"semi\_implicit"、"implicit" |
| limiter | str   | "none"           | 选项 "none"、"clip\_0\_1"，用于限制数值越界           |

## 6. 温度场适配器 \[temperature]

| 键             | 类型    | 默认         | 约束与说明                                  |
| ------------- | ----- | ---------- | -------------------------------------- |
| mode          | str   | "constant" | 可选 "constant"、"file"、"function"        |
| T\_const      | float | 1600.0     | 当 mode=constant 时使用 \[K]               |
| file          | str   | 无          | 当 mode=file 时给出数据文件路径                  |
| dataset       | str   | 无          | 文件中的数据集名称或列名                           |
| interp\_space | str   | "bilinear" | 空间插值。可选 "nearest"、"bilinear"、"bicubic" |
| interp\_time  | str   | "linear"   | 时间插值。可选 "step"、"linear"                |

## 7. 形核统计 \[nucleation]

| 键         | 类型    | 默认   | 约束与说明                      |
| --------- | ----- | ---- | -------------------------- |
| enable    | bool  | true | 是否启用非均质形核                  |
| N\_max    | float | 1e6  | 最大可用形核点密度 \[m^-3] 或按元胞体积折算 |
| DeltaT\_n | float | 5.0  | 形核过冷均值 \[K]                |
| sigma\_n  | float | 1.0  | 过冷标准差 \[K]                 |

## 8. I O 与可视化

### 8.1 \[io]

| 键           | 类型   | 默认     | 约束与说明                |
| ----------- | ---- | ------ | -------------------- |
| write\_meta | bool | true   | 是否写 meta.json 或同等元数据 |
| compression | str  | "auto" | 对应各格式的压缩策略           |

### 8.2 \[viz]

| 键      | 类型          | 默认        | 约束与说明        |
| ------ | ----------- | --------- | ------------ |
| enable | bool        | false     | 是否在运行中生成基础图像 |
| fields | array\[str] | \["fs"]   | 绘制字段列表       |
| cmap   | str         | "viridis" | 调色方案名        |

## 9. 调试与诊断 \[debug]

| 键                  | 类型          | 默认   | 约束与说明                           |
| ------------------ | ----------- | ---- | ------------------------------- |
| dump\_derived      | array\[str] | \[]  | 选择性导出派生量，如 \["nx","ny","kappa"] |
| assert\_invariants | bool        | true | 是否在每步检查不变量与守恒                   |
| tol\_mass          | float       | 1e-8 | 守恒阈值系数，定义在数据模型中                 |

---

## 10. 组合约束与校验逻辑

* `t_end ≥ dt` 必须成立
* `0 ≤ k ≤ 1`
* `anisotropy_epsilon ∈ [0,1)`
* `N_theta ≥ 1` 且为整数
* `kernel_radius_dx ≥ 0` 且 `nghost ≥ kernel_radius_dx`
* 当 `domain.bc` 为 `dirichlet` 时必须提供 `domain.bc_values`
* 当 `temperature.mode = file` 时必须提供 `temperature.file` 和 `temperature.dataset`

---

## 11. 最小示例

```toml
[domain]
nx = 64
ny = 64
dx = 1e-6
dy = 1e-6
nghost = 3
bc = "neumann0"

[time]
dt = 1e-3
t_end = 1e-1
save_every = 20

[run]
seed = 42
output_dir = "data/output/run-minimal"
snapshot_format = "npz"
snapshot_fields = ["fs","CL","CS","grain_id","theta","L_dia","lock","T"]

[physics]
unit_of_C = "mass"

[physics.orientation]
fold = 4
N_theta = 48
anisotropy_epsilon = 0.02
class_liq_sentinel = -10

[physics.interface]
m_L = -300.0
T_L_eq = 1800.0
Gamma = 2.0e-7
k = 0.3

[physics.mdcs]
kernel_radius_dx = 3

[physics.solute]
D_L = 3.0e-9
D_S = 1.0e-12
scheme = "semi_implicit"
limiter = "clip_0_1"

[temperature]
mode = "constant"
T_const = 1750.0
```

## 12. 进阶示例（文件驱动温度场）

```toml
[domain]
nx = 128
ny = 96
dx = 5e-7
dy = 5e-7
nghost = 3
bc_x = "periodic"
bc_y = "neumann0"

[time]
dt = 5e-4
t_end = 0.2
save_every = 50
checkpoint_every = 200

[run]
seed = 2025
output_dir = "data/output/run-adv"
snapshot_format = "hdf5"
snapshot_fields = ["fs","CL","CS","grain_id","theta","L_dia","lock","T"]
log_level = "DEBUG"

[physics]
unit_of_C = "atomic"

[physics.orientation]
fold = 4
N_theta = 48
anisotropy_epsilon = 0.03
class_liq_sentinel = -10

[physics.interface]
m_L = -250.0
T_L_eq = 1750.0
Gamma = 1.8e-7
k = 0.25

[physics.mdcs]
kernel_radius_dx = 3

[physics.solute]
D_L = 2.5e-9
D_S = 8.0e-13
scheme = "semi_implicit"
limiter = "none"

[temperature]
mode = "file"
file = "data/input/T_field.h5"
dataset = "T"
interp_space = "bilinear"
interp_time = "linear"

[debug]
dump_derived = ["nx","ny","kappa"]
assert_invariants = true
```

以上键与规则建议保持稳定。新增物理时应先在本文件登记键名、类型与约束，再在加载器中实现校验与回退策略。
