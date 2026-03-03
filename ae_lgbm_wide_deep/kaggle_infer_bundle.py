#!/usr/bin/env python3
"""
Kaggle notebook entrypoint: load a bundle exported by export_bundle.py and run the Evaluation API.

You will paste this file's content into a Kaggle Notebook cell, OR upload it with your bundle
(and run it as the notebook's main script).

Expected bundle layout (written by pipeline export hooks):
  bundle_meta.json
  decision_config.json
  fold01_fp.pkl
  fold01_weather.pkl                (optional)
  fold01_wd_cols.json
  fold01_wd_lgbm.txt
  fold01_seed01_ae_state.pth
  fold01_seed01_ae_norm.pkl
  ...

This script:
- Loads per-fold preprocessors + weather models + AE encoders + LGBM boosters
- Ensembling: mean across folds; within each fold, mean across AE seeds for Z
- Applies decision layer (long_only / tilt1), with optional EMA + vol^decay scaling

NOTE:
- You may need to adjust BUNDLE_DIR to your Kaggle Dataset mount path.
- You may need to adjust the column name for the API output (usually "prediction").
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# optional polars
try:
    import polars as pl
except Exception:
    pl = None

import pandas as pd
import joblib
import torch
import lightgbm as lgb

# Ensure we can import preprocess/weather/encoder if they are copied into the bundle
# (export_bundle.py defaults to copying them)
import sys


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class DecisionParams:
    mode: str = "long_only"          # "long_only" or "tilt1"
    coef: float = 1.0
    center_shift: float = 0.0
    threshold: float = 0.0
    decay: float = 0.0
    vol_window: int = 20
    ema_enable: bool = False
    ema_alpha: float = 0.10
    clip_low: float = 0.0
    clip_high: float = 2.0
    min_inv: float = 0.0
    max_inv: float = 2.0


class OnlineDecisionState:
    """
    Maintains rolling volatility estimate and optional EMA smoothing.
    Assumes a single time-step per predict call (common for evaluation API),
    but will also handle batches by iterating row-wise.
    """
    def __init__(self, p: DecisionParams):
        self.p = p
        self.ret_hist = deque(maxlen=max(2, int(p.vol_window)))
        self.sig_hist = deque(maxlen=max(2, int(p.vol_window)))
        self._ema_last: Optional[float] = None

    def update_vol(self, signal: float, lagged_ret: Optional[float] = None) -> float:
        # Prefer realized lagged returns if present; otherwise fall back to signal history.
        if lagged_ret is not None and np.isfinite(lagged_ret):
            self.ret_hist.append(float(lagged_ret))
            arr = np.asarray(self.ret_hist, dtype=float)
            vol = float(np.std(arr)) if len(arr) >= 2 else 1.0
        else:
            self.sig_hist.append(float(signal))
            arr = np.asarray(self.sig_hist, dtype=float)
            vol = float(np.std(arr)) if len(arr) >= 2 else 1.0
        return max(vol, 1e-6)

    def map_alloc(self, pred: float, lagged_ret: Optional[float] = None) -> float:
        p = self.p
        center = 0.5 + float(p.center_shift)
        signal = float(pred) - center

        # deadzone
        if p.threshold and abs(signal) < float(p.threshold):
            alloc = 0.0 if p.mode == "long_only" else 1.0
        else:
            if p.mode == "tilt1":
                alloc = 1.0 + float(p.coef) * signal
            else:
                alloc = float(p.coef) * signal  # long_only timing (can be negative; we clip)
        # volatility scaling
        if p.decay and float(p.decay) > 0:
            vol = self.update_vol(signal, lagged_ret=lagged_ret)
            alloc = alloc / (vol ** float(p.decay))

        # clip ranges
        lo = float(p.clip_low) if p.clip_low is not None else float(p.min_inv)
        hi = float(p.clip_high) if p.clip_high is not None else float(p.max_inv)
        lo = min(lo, float(p.min_inv))
        hi = max(hi, float(p.max_inv))
        alloc = float(np.clip(alloc, lo, hi))

        # EMA smoothing (scalar)
        if p.ema_enable:
            a = float(p.ema_alpha)
            if self._ema_last is None:
                self._ema_last = alloc
            else:
                self._ema_last = a * alloc + (1.0 - a) * self._ema_last
            alloc = self._ema_last

        return alloc


class FoldBundle:
    def __init__(self, fold: int, bundle_dir: Path):
        self.fold = fold
        self.bundle_dir = bundle_dir
        self.fp = joblib.load(bundle_dir / f"fold{fold:02d}_fp.pkl")

        # weather is optional
        w_path = bundle_dir / f"fold{fold:02d}_weather.pkl"
        self.weather = joblib.load(w_path) if w_path.exists() else None

        self.wd_cols = json.loads((bundle_dir / f"fold{fold:02d}_wd_cols.json").read_text(encoding="utf-8"))
        self.booster = lgb.Booster(model_file=str(bundle_dir / f"fold{fold:02d}_wd_lgbm.txt"))

        # load AE seeds for this fold
        self.ae_seeds: List[Tuple[torch.nn.Module, Dict]] = []
        for norm_path in sorted(bundle_dir.glob(f"fold{fold:02d}_seed*_ae_norm.pkl")):
            # derive weight path
            prefix = norm_path.name.replace("_norm.pkl", "")
            w_path = bundle_dir / f"{prefix}_state.pth"
            meta = joblib.load(norm_path)

            # import EncoderNet from bundle (encoder.py copied there)
            from encoder import EncoderNet  # type: ignore

            net = EncoderNet(
                in_dim=int(meta["in_dim"]),
                enc_dim=int(meta["enc_dim"]),
                hidden=int(meta["hidden"]),
                dropout=float(meta["dropout"]),
                n_targets=int(meta.get("n_targets", 1)),
            )
            net.load_state_dict(torch.load(w_path, map_location="cpu"))
            net.eval()
            self.ae_seeds.append((net, meta))

        if not self.ae_seeds:
            raise RuntimeError(f"No AE seeds found for fold {fold} in {bundle_dir}")

    def _compute_Z_mean(self, X_fp_np: np.ndarray) -> np.ndarray:
        Zs = []
        x = np.asarray(X_fp_np, dtype=np.float32)
        for net, meta in self.ae_seeds:
            mu = np.asarray(meta["mu"], dtype=np.float32)
            sigma = np.asarray(meta["sigma"], dtype=np.float32)
            x_std = (x - mu) / sigma
            with torch.no_grad():
                z, _, _ = net(torch.from_numpy(x_std))
                Zs.append(z.numpy())
        return np.mean(np.stack(Zs, axis=0), axis=0)

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        # 1) preprocessor
        X_fp_df = self.fp.transform_df(X_raw).reset_index(drop=True)
        X_fp_np = X_fp_df.to_numpy(dtype=np.float32, copy=False)

        # 2) AE -> Z
        Z = self._compute_Z_mean(X_fp_np)

        # 3) weather features (optional, from raw X to match training)
        parts = [X_fp_df]
        if self.weather is not None:
            from weather import transform as weather_transform  # type: ignore
            W = weather_transform(self.weather, X_raw).reset_index(drop=True)
            parts.append(W)

        # Z columns must match training naming
        z_cols = [c for c in self.wd_cols if c.startswith("ae_z")]
        if len(z_cols) == 0:
            # fallback: create generic names in the same count
            z_cols = [f"ae_z{i}" for i in range(Z.shape[1])]
        Z_df = pd.DataFrame(Z[:, :len(z_cols)], columns=z_cols)
        parts.append(Z_df)

        X_wd = pd.concat(parts, axis=1)
        X_wd = X_wd.reindex(columns=self.wd_cols, fill_value=0.0)

        p = self.booster.predict(X_wd.to_numpy(dtype=np.float32, copy=False))
        return np.asarray(p, dtype=float)


class BundleEnsemble:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir
        sys.path.insert(0, str(bundle_dir))  # so preprocess/weather/encoder modules resolve for joblib + imports

        meta_path = bundle_dir / "bundle_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        # which folds to load
        folds = meta.get("export_folds", [])
        if not folds:
            # fallback: detect from fp files
            folds = []
            for p in sorted(bundle_dir.glob("fold*_fp.pkl")):
                s = p.stem  # fold01_fp
                fold = int(s[4:6])
                folds.append(fold)
        self.folds = sorted(set(int(x) for x in folds))

        self.fold_models = [FoldBundle(f, bundle_dir) for f in self.folds]

        # decision params
        dcfg_path = bundle_dir / "decision_config.json"
        dcfg = json.loads(dcfg_path.read_text(encoding="utf-8")) if dcfg_path.exists() else {}
        self.decision = OnlineDecisionState(DecisionParams(
            mode=str(dcfg.get("DECISION_MODE", "long_only")),
            coef=_safe_float(dcfg.get("DECISION_COEF", 1.0), 1.0),
            center_shift=_safe_float(dcfg.get("DECISION_CENTER_SHIFT", 0.0), 0.0),
            threshold=_safe_float(dcfg.get("DECISION_THRESHOLD", 0.0), 0.0),
            decay=_safe_float(dcfg.get("DECISION_DECAY", 0.0), 0.0),
            vol_window=int(dcfg.get("DECISION_VOL_WINDOW", 20)),
            ema_enable=bool(dcfg.get("DECISION_EMA_ENABLE", False)),
            ema_alpha=_safe_float(dcfg.get("DECISION_EMA_ALPHA", 0.10), 0.10),
            clip_low=_safe_float(dcfg.get("DECISION_CLIP_LOW", 0.0), 0.0),
            clip_high=_safe_float(dcfg.get("DECISION_CLIP_HIGH", 2.0), 2.0),
            min_inv=_safe_float(dcfg.get("MIN_INVESTMENT", 0.0), 0.0),
            max_inv=_safe_float(dcfg.get("MAX_INVESTMENT", 2.0), 2.0),
        ))

    def predict_alloc(self, X_raw: pd.DataFrame) -> np.ndarray:
        ps = [fb.predict_proba(X_raw) for fb in self.fold_models]
        p = np.mean(np.stack(ps, axis=0), axis=0)

        # if API provides lagged returns, use them for volatility scaling
        lag_col = None
        for c in ["lagged_forward_returns", "lagged_return", "lag_return", "lagged_ret"]:
            if c in X_raw.columns:
                lag_col = c
                break

        allocs = []
        for idx in range(len(p)):
            lagged = None
            if lag_col is not None:
                lagged = _safe_float(X_raw.iloc[idx][lag_col], None)
            allocs.append(self.decision.map_alloc(float(p[idx]), lagged_ret=lagged))
        return np.asarray(allocs, dtype=float)


# ---------------- Kaggle Evaluation API hook ----------------

BUNDLE_DIR = os.getenv("HULL_BUNDLE_DIR", "/kaggle/input/hull-model-bundle")  # change this in notebook if needed
_bundle = None

def predict(test):
    global _bundle
    if _bundle is None:
        _bundle = BundleEnsemble(Path(BUNDLE_DIR))

    if pl is not None and isinstance(test, pl.DataFrame):
        df = test.to_pandas()
    else:
        # assume pandas
        df = test.copy()

    alloc = _bundle.predict_alloc(df)

    if pl is not None and isinstance(test, pl.DataFrame):
        return test.with_columns(pl.Series("prediction", alloc))
    else:
        out = df[["row_id"]].copy() if "row_id" in df.columns else pd.DataFrame(index=df.index)
        out["prediction"] = alloc
        return out


def _main():
    # Kaggle provides an inference server; if it's not available, we just run a local smoke test.
    try:
        from kaggle_evaluation.default_inference_server import DefaultInferenceServer
        server = DefaultInferenceServer(predict)
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            server.serve()
        else:
            # local debug path; adjust to your dataset mount if you use it
            server.run_local_gateway(("/kaggle/input/hull-tactical-market-prediction/",))
    except Exception as e:
        print("[WARN] kaggle_evaluation not available here or failed to start server:", e)


if __name__ == "__main__":
    _main()
