from pathlib import Path
import json


def load_cfg(
    path="config.json",
) -> dict:  # 定义名为load_cfg  的函数，通常表示 "load configuration"（加载配置）。

    # 读取并解析 JSON 配置文件
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # 基本结构校验
    for sec in ("domain", "time", "run"):
        assert sec in data and isinstance(data[sec], dict), f"缺少配置节 [{sec}]"
    
    d, t, r = data["domain"], data["time"], data["run"]

    # 最小校验 + 安全默认
    assert d["nx"] > 0 and d["ny"] > 0 and d["dx"] > 0 and d["dy"] > 0
    assert t["dt"] > 0 and t["t_end"] >= t["dt"]
    t.setdefault("save_every", 50)
    r.setdefault("seed", 0)
    r.setdefault("output_dir", "data/output/run-minimal")
    d.setdefault("nghost", 3)
    d.setdefault("bc", "neumann0")  # 统一先用零通量
    return data
