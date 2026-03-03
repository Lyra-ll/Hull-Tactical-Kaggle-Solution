#!/usr/bin/env python3
"""
Export a Kaggle-ready model bundle by running your existing validation/training pipeline
(with the export hooks enabled inside pipeline.py / encoder.py).

Usage (Windows cmd / PowerShell):
  python export_bundle.py --out hull_model_bundle --mode validate

What it does:
- Sets cfg.EXPORT_BUNDLE_DIR to --out
- Runs main.py in the requested mode (default: validate)
- pipeline.fit_evaluate will automatically write:
    foldXX_fp.pkl
    foldXX_weather.pkl (if enabled)
    foldXX_wd_cols.json
    foldXX_wd_lgbm.txt
    foldXX_seedYY_ae_state.pth + foldXX_seedYY_ae_norm.pkl
    bundle_meta.json
- Also writes decision_config.json into the bundle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config as cfg


def _dump_decision_config(out_dir: Path) -> None:
    keys = [
        "DECISION_MODE",
        "DECISION_COEF",
        "DECISION_CENTER_SHIFT",
        "DECISION_THRESHOLD",
        "DECISION_DECAY",
        "DECISION_VOL_WINDOW",
        "DECISION_EMA_ENABLE",
        "DECISION_EMA_ALPHA",
        "MIN_INVESTMENT",
        "MAX_INVESTMENT",
        "DECISION_CLIP_LOW",
        "DECISION_CLIP_HIGH",
        "DECISION_USE_WEATHER_REGIME",
    ]
    d = {}
    for k in keys:
        if hasattr(cfg, k):
            d[k] = getattr(cfg, k)
    (out_dir / "decision_config.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"[bundle] wrote decision_config.json (keys={len(d)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="hull_model_bundle")
    ap.add_argument("--mode", type=str, default="validate", choices=["validate", "train"])
    ap.add_argument("--folds", type=str, default="last_k", choices=["last_k", "all"])
    ap.add_argument("--no_copy_code", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enable export hooks
    cfg.EXPORT_BUNDLE_DIR = str(out_dir)
    cfg.EXPORT_BUNDLE_FOLDS = args.folds
    cfg.EXPORT_BUNDLE_COPY_CODE = not args.no_copy_code
    cfg.EXPORT_BUNDLE_OVERWRITE = True

    # Run your existing entrypoint
    import main as main_mod

    # emulate CLI call so your artifacts go under artifacts/run-*
    sys.argv = ["main.py", "--mode", args.mode]
    main_mod.main()

    # dump decision params (so Kaggle side doesn't need your full config.py)
    _dump_decision_config(out_dir)

    print("[bundle] DONE. Now zip this folder and upload as a Kaggle Dataset.")


if __name__ == "__main__":
    main()
