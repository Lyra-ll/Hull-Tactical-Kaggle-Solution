# features/__init__.py
"""
Feature engineering subpackage.

- 把与特征构造相关的模块放在这里（weather、pools 等）。
- 显式成为包，避免与外部同名模块冲突。
"""
from . import weather  # 可选：对外暴露 `features.weather`
__all__ = ["weather"]
