"""ae_lgbm_wide_deep package

This file turns the folder into a Python package so you can run:
    python -m ae_lgbm_wide_deep.main --mode validate

It also exposes a tiny helper to add the *project root* to sys.path in case
you insist on running the script as a file (python ae_lgbm_wide_deep/main.py).
You usually don't need it when using module-mode (-m).
"""
from __future__ import annotations
import os, sys

__all__ = ["add_project_root_to_sys_path", "__version__"]
__version__ = "0.0.1"


def add_project_root_to_sys_path() -> str:
    """Ensure the parent directory (project root) is on sys.path.
    Returns the path that was inserted (or already present).
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
