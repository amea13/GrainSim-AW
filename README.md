# GrainSim-AW

本项目实现二维金属凝固时**多取向枝晶生长**的元胞自动机模拟，基于界面热力学平衡和 Stefan 溶质守恒，耦合宏观温度场数据，输出枝晶生长过程的数值结果。

## 快速开始

### 环境要求
- Python 3.8+
- 依赖包：numpy, scipy 等（详见项目配置）

### 基本运行
```bash
python -m grainsim_aw.main --config configs/run-minimal.toml
```

## 项目结构

```
src/grainsim_aw/
├── core/           # 核心数据结构（网格、材料属性）
├── engine/         # 模拟引擎（时间步进、模拟器主循环）
├── nucleation/     # 形核模型（Thévoz统计、晶核生成）
├── growth_capture/ # 晶粒生长规则（MDCS、界面捕捉）
├── interface/      # 界面处理（热力学平衡、溶质扩散、速度计算）
├── multiphysics/   # 多物理耦合（温度、溶质求解）
├── io/             # 输入输出（数据写入、配置加载）
└── viz/            # 可视化（绘图、动画）
```

## 主要特性

- **元胞自动机方法**：离散化网格上的晶粒生长模拟
- **多取向形核**：基于物理的异质形核模型
- **界面热平衡**：考虑曲率效应和溶质偏析的界面温度
- **溶质守恒**：体积分数加权的 Stefan 问题求解
- **温度耦合**：读取外部温度场进行宏微插值
- **CET 转变**：模拟柱状晶向等轴晶的转变过程

## 配置与运行

详见 `docs/usage/config_schema.md`，配置示例在 `configs/run-minimal.toml`

## 输出文件

- `meta_config.json`：计算配置和晶粒元数据
- `step_*.npz`：每个时间步的快照（晶粒ID、固相分数等）

## 文档

- [一页纸总纲](docs/usage/onepager.md) - 项目核心目标与范围
- [理论基础](docs/theory/lifecycle_sequence.md) - 时间推进序列

## 许可证

详见 LICENSE 文件

