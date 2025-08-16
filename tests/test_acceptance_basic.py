# tests/test_acceptance_basic.py
from pathlib import Path
import subprocess, sys, json, os, time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.json"


def run_cmd(args):
    # 确保能找到 src/grainsim_aw
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    print(">", " ".join(map(str, args)))
    return subprocess.run(args, check=True, cwd=ROOT, env=env)


def main():
    assert CONFIG.exists(), "找不到 config.json"

    t0 = time.time()
    # 以“包模块”方式运行 main（相对导入才成立）
    run_cmd([sys.executable, "-m", "grainsim_aw.main", str(CONFIG)])
    dt = time.time() - t0
    print(f"运行用时：{dt:.3f}s")

    # 检查输出
    out_dir = json.loads(CONFIG.read_text(encoding="utf-8"))["run"]["output_dir"]
    out_path = ROOT / out_dir
    assert out_path.exists(), "输出目录未创建"
    snaps = list(out_path.glob("step_*.npz"))
    assert len(snaps) >= 1, "没有找到快照文件"
    print("✅ 骨架心跳正常。")


if __name__ == "__main__":
    main()
