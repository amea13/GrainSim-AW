# GrainSim-AW
本项目实现二维金属凝固时枝晶生长的元胞自动机模拟，基于界面热力学平衡和stfen溶质守恒，耦合宏观温度场数据，输出枝晶生长过程的数值结果。


```
GrainSim-AW
├─ config.json
├─ data
│  └─ output
│     └─ run-minimal
│        ├─ meta_config.json
│        └─ step_003001.npz
├─ docs
│  ├─ api
│  │  ├─ data_model.md
│  │  └─ modules
│  │     └─ contracts.md
│  ├─ theory
│  │  └─ lifecycle_sequence.md
│  └─ usage
│     ├─ acceptance_criteria.md
│     ├─ config_schema.md
│     └─ onepager.md
├─ LICENSE
├─ README.md
├─ src
│  └─ grainsim_aw
│     ├─ config_loader.py
│     ├─ core
│     │  ├─ grid.py
│     │  ├─ material.py
│     │  └─ __init__.py
│     ├─ engine
│     │  ├─ simulator.py
│     │  ├─ time_step.py
│     │  └─ __init__.py
│     ├─ errors.py
│     ├─ growth_capture
│     │  ├─ advance.py
│     │  ├─ capture.py
│     │  ├─ process.py
│     │  └─ __init__.py
│     ├─ interface
│     │  ├─ equilibrium.py
│     │  ├─ fields.py
│     │  ├─ geometry.py
│     │  ├─ process.py
│     │  ├─ velocity.py
│     │  └─ __init__.py
│     ├─ io
│     │  ├─ csv_matrix.py
│     │  ├─ writer.py
│     │  └─ __init__.py
│     ├─ logging_config.py
│     ├─ main.py
│     ├─ multiphysics
│     │  ├─ process.py
│     │  ├─ solute_solver.py
│     │  ├─ temperature_adapter.py
│     │  └─ __init__.py
│     ├─ nucleation
│     │  ├─ process.py
│     │  ├─ seeding.py
│     │  ├─ thevoz.py
│     │  └─ __init__.py
│     ├─ viz
│     │  ├─ animation.py
│     │  ├─ liveplot.py
│     │  ├─ plot.py
│     │  └─ __init__.py
│     └─ __init__.py
└─ tests

```
























