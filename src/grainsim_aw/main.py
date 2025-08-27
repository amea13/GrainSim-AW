from __future__ import annotations
import sys
import logging
from pathlib import Path
from .config_loader import load_cfg
from .engine.simulator import Simulator


def main(cfg_path: str | Path = "configs/run-minimal.toml") -> None:
    # 入口里，第一行就配置日志（只配一次）
    logging.basicConfig(
        level=logging.INFO,  # 如需更详细日志：改为 logging.DEBUG
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.debug("DEBUG OK: main() 日志初始化完成")

    cfg_path = Path(cfg_path)

    if cfg_path.suffix.lower() not in {".toml", ".tml"}:
        raise ValueError(f"配置文件必须是 TOML，收到：{cfg_path.name}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path.resolve()}")

    cfg = load_cfg(cfg_path)
    logging.info("使用配置：%s", cfg_path)

    sim = Simulator(cfg)
    sim.run()
    logging.info("INFO OK: main() 运行完成，输出目录：%s", cfg["run"]["output_dir"])


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/run-minimal.toml"
    main(cfg_path)
