import sys
import logging
from .config_loader import load_cfg
from .engine.simulator import Simulator


def main(cfg_path="config.json"):
    # 入口里，第一行就配置日志（只配一次）
    logging.basicConfig(
        level=logging.DEBUG,  # 想看 debug 就用 DEBUG；想清爽点用 INFO
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.debug("DEBUG OK: main() 已配置日志")
    cfg = load_cfg(cfg_path)
    sim = Simulator(cfg)
    sim.run()
    logging.info("INFO OK: main() 运行完成，输出目录：%s", cfg["run"]["output_dir"])


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    main(cfg_path)
