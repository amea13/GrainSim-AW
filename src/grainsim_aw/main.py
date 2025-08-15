import sys
from .config_loader import load_cfg
from .engine.simulator import Simulator

def main(cfg_path="config.json"):
    cfg = load_cfg(cfg_path)
    sim = Simulator(cfg)
    sim.run()
    print("运行完成，输出目录：", cfg["run"]["output_dir"])

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    main(cfg_path)
