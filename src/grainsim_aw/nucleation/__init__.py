"""
nucleation 包的对外 API。

对 Engine 只暴露一个入口：apply()
后续即使我们在包内增加更多文件（比如 orientation.py），
Engine 的导入路径也保持不变：grainsim_aw.nucleation.apply
"""

from .thevoz import apply

__all__ = ["apply"]
