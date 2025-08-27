from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import tomllib


def load_cfg(path: str | Path = "configs/run-minimal.toml") -> dict:
    path = Path(path)
    if path.suffix.lower() not in {".toml", ".tml"}:
        raise ValueError(f"只支持 TOML 配置，收到文件：{path.name}")

    with path.open("rb") as f:
        cfg: Dict[str, Any] = tomllib.load(f)

    # 必要配置节
    for sec in ("domain", "time", "run"):
        if sec not in cfg or not isinstance(cfg[sec], dict):
            raise ValueError(f"缺少配置节 [{sec}] 或类型错误")

    d, t, r = cfg["domain"], cfg["time"], cfg["run"]

    # 默认值与规范化
    d.setdefault("nghost", 3)
    d.setdefault("bc", {"x": "neumann0", "y": "neumann0"})
    if isinstance(d["bc"], str):  # 允许用户写成单值，自动展开
        d["bc"] = {"x": d["bc"], "y": d["bc"]}
    else:
        d["bc"].setdefault("x", "neumann0")
        d["bc"].setdefault("y", "neumann0")

    t.setdefault("save_every", 50)
    r.setdefault("seed", 0)
    r.setdefault("output_dir", "data/output/run-minimal")

    # 数值校验
    for k in ("nx", "ny"):
        v = int(d.get(k, 0))
        if v <= 0:
            raise ValueError(f"[domain].{k} 必须 > 0")
        d[k] = v
    for k in ("dx", "dy"):
        v = float(d.get(k, 0.0))
        if v <= 0.0:
            raise ValueError(f"[domain].{k} 必须 > 0")
        d[k] = v

    dt = float(t.get("dt", 0.0))
    t_end = float(t.get("t_end", 0.0))
    if dt <= 0.0:
        raise ValueError("[time].dt 必须 > 0")
    if t_end < dt:
        raise ValueError("[time].t_end 必须 ≥ dt")

    # C0 重复定义冲突检查（可选，但很有用）
    C0_domain = d.get("C0", None)
    C0_iface = cfg.get("physics", {}).get("interface", {}).get("C0", None)
    if (
        C0_domain is not None
        and C0_iface is not None
        and abs(float(C0_domain) - float(C0_iface)) > 1e-12
    ):
        raise ValueError(
            "检测到 C0 在 [domain] 与 [physics.interface] 两处定义且数值不一致，请合并为一处。"
        )

    # 确保输出目录存在
    Path(r["output_dir"]).mkdir(parents=True, exist_ok=True)

    return cfg
