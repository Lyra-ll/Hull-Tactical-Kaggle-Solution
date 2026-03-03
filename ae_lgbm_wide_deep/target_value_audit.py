#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Target Value Audit (MFER vs FR)
--------------------------------
Purpose
=======
A reproducible, leak-safe audit to evaluate whether
`market_forward_excess_returns` (MFER) contains predictive value vs
`forward_returns` (FR), and whether MFER is useful as an auxiliary
signal (multi-task proxy) or as a component in sample weighting.

What it does
============
0) Basic profiling: correlation(FR, MFER), distribution stability (rolling stats)
1) Quick predictability baselines (simple models):
   - Regression R^2 for MFER (Ridge) vs Always-Mean baseline
   - Classification AUC for sign(MFER>0) vs sign(FR>0) with Logistic/LGBM (tiny)
2) Auxiliary-task emulation (no deep net):
   - Train OOF FR-classifier (LightGBM) baseline
   - Train OOF MFER-regressor (Ridge) and stack its OOF-pred as an extra feature
     into FR-classifier (LightGBM). Compare OOF AUC (FR main metric).
3) Weight fusion experiment:
   - Try w = α|FR| + (1-α)|MFER|, α∈{1.0, 0.75, 0.5, 0.25, 0.0}
   - Compare OOF AUC (FR target) under each weight scheme
4) Optional regime-conditional value:
   - If a `weather_state` column exists in the input (int states 0..K-1),
     report per-state AUC for (baseline vs stacked) FR-classifier.
5) Block-bootstrap confidence intervals for OOF AUC (time-aware CI)

Outputs (under --artifacts)
===========================
- audit_summary.json : All key metrics and decisions
- oof_fr_baseline.csv : date_id, y_fr, pred, weight_alpha
- oof_fr_stacked.csv  : date_id, y_fr, pred, weight_alpha
- per_state_auc.csv   : (optional) if weather_state present

Assumptions / Safety
====================
- Input is the *train* features CSV (e.g., train_final_features.csv)
- We exclude any column whose name suggests leakage: contains any of
  ["forward", "dls_target", "resp_", "action_", "_lead", "shift_", "label", "target"]
- Features are numeric; missing values imputed by training-fold median
- CV is chronological by date_id (n_splits configurable)

Run
===
python -m ae_lgbm_wide_deep.target_value_audit \
  --train_csv train_final_features.csv \
  --date_col date_id \
  --artifacts artifacts/audit-MFER \
  --n_splits 5

Tip: You can also place this script at project root and run directly.
"""
from __future__ import annotations
import os
import json
import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# LightGBM (sklearn API)
try:
    from lightgbm import LGBMClassifier
except Exception as e:  # graceful message if missing
    LGBMClassifier = None

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

RANDOM_SEED = 42


# ----------------------------- utils: safety & IO -----------------------------

def is_safe_feature(name: str) -> bool:
    s = str(name).lower()
    banned = [
        "forward", "dls_target", "resp_", "action_", "_lead", "shift_", "label", "target"
    ]
    return not any(b in s for b in banned)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------- utils: CV splits -------------------------------

# -------------------------- utils: robust fold imputation ---------------------

def fold_impute_pair(Xtr: pd.DataFrame, Xva: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Robust per-fold imputation:
    1) 将 ±inf 先置为 NaN
    2) 用训练折的列中位数填充
    3) 若仍有列在训练折中是全 NaN（中位数也是 NaN），则把这些列从 Xtr/Xva 同步删除
    4) 训练折里为常量的列（无信息）也顺手删掉（可防止部分线性模型数值问题）
    5) 兜底：若仍残留 NaN，填 0.0
    """
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xva = Xva.replace([np.inf, -np.inf], np.nan)

    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xva = Xva.fillna(med)

    # 仍然有 NaN 的列（例如训练折全空），统一删除
    bad_nan = [c for c in Xtr.columns if Xtr[c].isna().any()]
    # 训练折常量列（无方差），也删除
    bad_const = [c for c in Xtr.columns if Xtr[c].nunique(dropna=True) <= 1]
    bad_cols = sorted(set(bad_nan) | set(bad_const))
    if bad_cols:
        Xtr = Xtr.drop(columns=bad_cols)
        Xva = Xva.drop(columns=bad_cols, errors="ignore")

    # 兜底：把任何残留 NaN 补 0
    Xtr = Xtr.fillna(0.0)
    Xva = Xva.fillna(0.0)
    return Xtr, Xva



def chronological_splits(date_series: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return time-ordered CV splits like TimeSeriesSplit, using indices.
    Assumes date_series is sortable; we keep original order.
    """
    order = np.argsort(date_series.values)
    n = len(order)
    # Use sklearn TSS for robustness, but map back to original indices
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for tr_idx, va_idx in tss.split(np.arange(n)):
        splits.append((order[tr_idx], order[va_idx]))
    return splits


# -------------------------- utils: block bootstrap CI -------------------------

def block_bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, dates: np.ndarray,
                           B: int = 200, block: int = 20, seed: int = RANDOM_SEED) -> Tuple[float, Tuple[float, float]]:
    """Time-aware bootstrap for ROC AUC with non-overlapping random date blocks.
    dates: array of date_id aligned with y_true/y_score
    Returns: (auc_mean, (auc_lo, auc_hi)) at 95% level.
    """
    rng = np.random.default_rng(seed)
    # Unique sorted dates -> build contiguous blocks
    udates = np.unique(dates)
    if len(udates) < block:
        try:
            auc = roc_auc_score(y_true, y_score)
            return float(auc), (float(auc), float(auc))
        except Exception:
            return float('nan'), (float('nan'), float('nan'))
    # create block start indices
    starts = np.arange(0, len(udates), block)
    blocks = [udates[s:s+block] for s in starts]
    aucs = []
    for _ in range(B):
        # sample blocks with replacement until we cover ~len(udates)
        sampled = []
        while len(sampled) < len(udates):
            sampled.extend(blocks[int(rng.integers(0, len(blocks)))] )
        sampled = np.array(sampled[:len(udates)])
        mask = np.isin(dates, sampled)
        yb = y_true[mask]
        pb = y_score[mask]
        if len(np.unique(yb)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yb, pb))
        except Exception:
            continue
    if not aucs:
        return float('nan'), (float('nan'), float('nan'))
    aucs = np.array(aucs)
    return float(aucs.mean()), (float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975)))


# ---------------------------- data preparation --------------------------------

def prepare_frame(train_csv: str, date_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(train_csv)
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not in CSV")
    # Targets
    if "forward_returns" not in df.columns or "market_forward_excess_returns" not in df.columns:
        raise ValueError("CSV must contain 'forward_returns' and 'market_forward_excess_returns'")
    # Basic cleanup: keep only numeric + date + optional weather_state
    keep_cols = [c for c in df.columns if (pd.api.types.is_numeric_dtype(df[c]) or c == date_col)]
    if "weather_state" in df.columns:  # allow optional state
        keep_cols.append("weather_state")
    df = df[keep_cols].copy()
    # Feature list (safe)
    feats = [c for c in df.columns if c not in (date_col, "forward_returns", "market_forward_excess_returns", "weather_state") and is_safe_feature(c)]
    X = df[feats].copy()
    y_fr = df["forward_returns"].astype(float)
    y_mfer = df["market_forward_excess_returns"].astype(float)
    meta = df[[date_col]].copy()
    if "weather_state" in df.columns:
        meta["weather_state"] = df["weather_state"].astype(int)
    return X, pd.DataFrame({"FR": y_fr, "MFER": y_mfer}), meta


# ------------------------- step 0: basic profiling ----------------------------

def basic_profile(y: pd.DataFrame, date_series: pd.Series, artifacts: str) -> Dict[str, float]:
    fr = y["FR"].values
    mfer = y["MFER"].values
    out: Dict[str, float] = {}
    # correlation
    pear = float(np.corrcoef(fr, mfer)[0,1])
    spear = float(pd.Series(fr).corr(pd.Series(mfer), method="spearman"))
    out["corr_pearson"] = pear
    out["corr_spearman"] = spear
    # rolling stability (mean, std) over 252d
    df = pd.DataFrame({"date": date_series.values, "FR": fr, "MFER": mfer}).sort_values("date")
    for col in ("FR", "MFER"):
        rmean = df[col].rolling(252, min_periods=64).mean().abs().mean()  # |mean| stability proxy
        rstd  = df[col].rolling(252, min_periods=64).std().mean()
        out[f"{col}_roll_mean_absmean_252"] = float(rmean)
        out[f"{col}_roll_std_mean_252"] = float(rstd)
    with open(os.path.join(artifacts, "00_basic_profile.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


# ---------------- step 1: quick predictability baselines (simple) -------------

def quick_baselines(X: pd.DataFrame, y: pd.DataFrame, date_series: pd.Series, n_splits: int, artifacts: str) -> Dict[str, float]:
    splits = chronological_splits(date_series, n_splits=n_splits)
    # Prepare containers
    oof_logit_fr = np.full(len(X), np.nan)
    oof_logit_mfer = np.full(len(X), np.nan)
    oof_ridge_mfer = np.full(len(X), np.nan)

    for fold, (tr, va) in enumerate(splits):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        # median impute (train fit, val transform)
        Xtr, Xva = fold_impute_pair(Xtr, Xva)

        # Logistic for FR sign
        y_fr_cls_tr = (y["FR"].iloc[tr].values > 0).astype(int)
        y_fr_cls_va = (y["FR"].iloc[va].values > 0).astype(int)
        try:
            clf_fr = LogisticRegression(max_iter=1000, n_jobs=1, solver="lbfgs")
            clf_fr.fit(Xtr, y_fr_cls_tr)
            oof_logit_fr[va] = clf_fr.predict_proba(Xva)[:,1]
        except Exception:
            pass

        # Logistic for MFER sign
        y_mf_cls_tr = (y["MFER"].iloc[tr].values > 0).astype(int)
        y_mf_cls_va = (y["MFER"].iloc[va].values > 0).astype(int)
        try:
            clf_mf = LogisticRegression(max_iter=1000, n_jobs=1, solver="lbfgs")
            clf_mf.fit(Xtr, y_mf_cls_tr)
            oof_logit_mfer[va] = clf_mf.predict_proba(Xva)[:,1]
        except Exception:
            pass

        # Ridge for MFER value
        try:
            rg = Ridge(alpha=1.0, random_state=RANDOM_SEED)
            rg.fit(Xtr, y["MFER"].iloc[tr].values)
            oof_ridge_mfer[va] = rg.predict(Xva)
        except Exception:
            pass

    # Scores
    out: Dict[str, float] = {}
    # FR AUC (logit)
    try:
        mask = ~np.isnan(oof_logit_fr)
        out["quick_FR_logit_auc"] = float(roc_auc_score((y["FR"].values>0).astype(int)[mask], oof_logit_fr[mask]))
    except Exception:
        out["quick_FR_logit_auc"] = float("nan")
    # MFER sign AUC
    try:
        mask = ~np.isnan(oof_logit_mfer)
        out["quick_MFER_logit_auc"] = float(roc_auc_score((y["MFER"].values>0).astype(int)[mask], oof_logit_mfer[mask]))
    except Exception:
        out["quick_MFER_logit_auc"] = float("nan")
    # MFER R^2 (ridge)
    try:
        mask = ~np.isnan(oof_ridge_mfer)
        out["quick_MFER_ridge_r2"] = float(r2_score(y["MFER"].values[mask], oof_ridge_mfer[mask]))
    except Exception:
        out["quick_MFER_ridge_r2"] = float("nan")

    with open(os.path.join(artifacts, "01_quick_baselines.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Also store OOFs for diagnostics
    diag = pd.DataFrame({
        "date_id": date_series.values,
        "y_fr": (y["FR"].values>0).astype(int),
        "y_mfer": (y["MFER"].values>0).astype(int),
        "oof_logit_fr": oof_logit_fr,
        "oof_logit_mfer": oof_logit_mfer,
        "oof_ridge_mfer": oof_ridge_mfer,
    })
    diag.to_csv(os.path.join(artifacts, "01_quick_oof.csv"), index=False)
    return out


# ------------- step 2: FR-classifier baseline vs stacked-aux ------------------

def lgbm_params_default() -> Dict:
    return dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=4,
    )


def oof_fr_auc_with_weights(
    X: pd.DataFrame,
    y_fr: np.ndarray,
    date_series: pd.Series,
    weights: Optional[np.ndarray],
    stacked_aux: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> Tuple[float, np.ndarray]:
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is required for this audit. Please install lightgbm.")
    splits = chronological_splits(date_series, n_splits=n_splits)
    oof = np.full(len(X), np.nan)
    for fold, (tr, va) in enumerate(splits):
        Xtr = X.iloc[tr].copy()
        Xva = X.iloc[va].copy()
        # median impute
        # add stacked aux feature first
        if stacked_aux is not None:
            Xtr = Xtr.assign(MFER_hat=stacked_aux[tr])
            Xva = Xva.assign(MFER_hat=stacked_aux[va])

        # robust per-fold imputation (handles NaN/inf/all-NaN/constant)
        Xtr, Xva = fold_impute_pair(Xtr, Xva)
        ytr = y_fr[tr]
        yva = y_fr[va]
        wtr = None if weights is None else weights[tr]
        # tiny, safe defaults
        clf = LGBMClassifier(**lgbm_params_default())
        try:
            clf.fit(Xtr, ytr, sample_weight=wtr)
            oof[va] = clf.predict_proba(Xva)[:, 1]
        except Exception:
            # fall back to unweighted if weighting explodes
            clf.fit(Xtr, ytr)
            oof[va] = clf.predict_proba(Xva)[:, 1]
    
    # --- 新增：遮掉未覆盖的样本（最早那段） ---
    mask = ~np.isnan(oof)
    if mask.sum() == 0:
        raise RuntimeError("OOF predictions are all NaN; check splits.")
    auc = float(roc_auc_score(y_fr[mask], oof[mask]))

    # --- 新增：为后续下游（保存/作图）做安全填充 ---
    fill_val = float(np.nanmean(oof[mask]))  # 用已观察到的均值
    oof = np.where(mask, oof, fill_val)

    return auc, oof



def build_stacked_aux_oof(
    X: pd.DataFrame,
    y_mfer: np.ndarray,
    date_series: pd.Series,
    n_splits: int = 5,
) -> np.ndarray:
    """OOF predictions for MFER using Ridge (trained fold-wise).
    These OOF preds will be added as an extra feature to FR classifier
    to emulate multi-task synergy.
    """
    splits = chronological_splits(date_series, n_splits=n_splits)
    oof = np.full(len(X), np.nan)
    for fold, (tr, va) in enumerate(splits):
        Xtr = X.iloc[tr].copy(); Xva = X.iloc[va].copy()
        Xtr, Xva = fold_impute_pair(Xtr, Xva)
        rg = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        rg.fit(Xtr, y_mfer[tr])
        oof[va] = rg.predict(Xva)
    mask = ~np.isnan(oof)
    if mask.any():
        fill_val = float(np.nanmean(oof[mask]))
    else:
        fill_val = 0.0  # 极端兜底（理论上不会发生）
    oof = np.where(mask, oof, fill_val)
    return oof


# ---------------------- step 3: weight fusion experiment ----------------------

def run_weight_grid(
    X: pd.DataFrame,
    y_fr_cls: np.ndarray,
    y_fr_abs: np.ndarray,
    y_mfer_abs: np.ndarray,
    date_series: pd.Series,
    stacked_aux: Optional[np.ndarray],
    n_splits: int,
    artifacts: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    grid = [1.0, 0.75, 0.5, 0.25, 0.0]
    rows = []
    oof_snapshots = {}
    for a in grid:
        w = a * y_fr_abs + (1.0 - a) * y_mfer_abs
        auc_base, oof_base = oof_fr_auc_with_weights(X, y_fr_cls, date_series, weights=w, stacked_aux=None, n_splits=n_splits)
        auc_stk, oof_stk  = oof_fr_auc_with_weights(X, y_fr_cls, date_series, weights=w, stacked_aux=stacked_aux, n_splits=n_splits)
        rows.append({"alpha": a, "auc_baseline": auc_base, "auc_stacked": auc_stk, "delta": auc_stk - auc_base})
        oof_snapshots[f"oof_base_alpha_{a}"] = oof_base
        oof_snapshots[f"oof_stacked_alpha_{a}"] = oof_stk
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(artifacts, "03_weight_grid_auc.csv"), index=False)
    # Persist the best alpha's oofs for quick plotting later
    best_row = tab.iloc[tab["auc_stacked"].idxmax()]
    best_a = float(best_row["alpha"])
    pd.DataFrame({
        "date_id": date_series.values,
        "y_fr": y_fr_cls,
        "pred_baseline": oof_snapshots[f"oof_base_alpha_{best_a}"] ,
        "pred_stacked":  oof_snapshots[f"oof_stacked_alpha_{best_a}"] ,
        "alpha": best_a,
    }).to_csv(os.path.join(artifacts, "oof_fr_baseline_vs_stacked_best_alpha.csv"), index=False)
    return tab, {"best_alpha": best_a}


# --------------- optional: per-state AUC if weather_state present -------------

def per_state_auc(oof: np.ndarray, y_cls: np.ndarray, states: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in np.unique(states):
        m = states == s
        if m.sum() < 30 or len(np.unique(y_cls[m])) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_cls[m], oof[m])
        rows.append({"state": int(s), "n": int(m.sum()), "auc": float(auc) if not math.isnan(auc) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="train_final_features.csv")
    ap.add_argument("--date_col", type=str, default="date_id")
    ap.add_argument("--artifacts", type=str, default=os.path.join("artifacts", f"audit-MFER-{int(time.time())}"))
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--block", type=int, default=20, help="bootstrap block size (days)")
    args = ap.parse_args()

    ensure_dir(args.artifacts)
    # 0) load & prep
    X, y, meta = prepare_frame(args.train_csv, args.date_col)
    date = meta[args.date_col]
    has_state = "weather_state" in meta.columns

    # 0) basic profile
    prof = basic_profile(y, date, args.artifacts)

    # 1) quick baselines
    qb = quick_baselines(X, y, date, n_splits=args.n_splits, artifacts=args.artifacts)

    # 2) stacked aux (OOF for MFER via ridge)
    stacked_aux = build_stacked_aux_oof(X, y["MFER"].values, date, n_splits=args.n_splits)

    # 3) weight grid + FR OOF AUCs (baseline vs stacked)
    y_fr_cls = (y["FR"].values > 0).astype(int)
    tab, extra = run_weight_grid(
        X=X,
        y_fr_cls=y_fr_cls,
        y_fr_abs=np.abs(y["FR"].values),
        y_mfer_abs=np.abs(y["MFER"].values),
        date_series=date,
        stacked_aux=stacked_aux,
        n_splits=args.n_splits,
        artifacts=args.artifacts,
    )

    # 4) best alpha CI via block bootstrap
    #    (compare baseline vs stacked under best alpha)
    best_alpha = extra["best_alpha"]
    # load snapshot we saved
    snap = pd.read_csv(os.path.join(args.artifacts, "oof_fr_baseline_vs_stacked_best_alpha.csv"))
    # CI
    auc_b, (lo_b, hi_b) = block_bootstrap_auc_ci(snap["y_fr"].values, snap["pred_baseline"].values, snap["date_id"].values, block=args.block)
    auc_s, (lo_s, hi_s) = block_bootstrap_auc_ci(snap["y_fr"].values, snap["pred_stacked"].values,  snap["date_id"].values, block=args.block)

    # 5) optional per-state AUC report
    per_state = None
    if has_state:
        per_state = {
            "baseline": per_state_auc(snap["pred_baseline"].values, snap["y_fr"].values, meta["weather_state"].values),
            "stacked":  per_state_auc(snap["pred_stacked"].values,  snap["y_fr"].values, meta["weather_state"].values),
        }
        df_state = per_state["baseline"].merge(per_state["stacked"], on="state", suffixes=("_base", "_stack"))
        df_state["delta"] = df_state["auc_stack"] - df_state["auc_base"]
        df_state.to_csv(os.path.join(args.artifacts, "04_per_state_auc.csv"), index=False)

    # 6) final summary & decision hints
    summary = {
        "basic_profile": prof,
        "quick_baselines": qb,
        "weight_grid": tab.to_dict(orient="records"),
        "best_alpha": best_alpha,
        "oof_auc_ci_baseline": {"mean": auc_b, "lo": lo_b, "hi": hi_b},
        "oof_auc_ci_stacked":  {"mean": auc_s, "lo": lo_s, "hi": hi_s},
        "recommendation": (
            "keep_as_auxiliary" if (auc_s - auc_b) >= 0.002 and auc_s > auc_b and (lo_s - hi_b) > 0.0 else "no_clear_gain"
        ),
    }
    with open(os.path.join(args.artifacts, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # write minimal console report
    print("\n===== Target Value Audit (MFER vs FR) =====")
    print(f"corr(FR, MFER) pearson={summary['basic_profile']['corr_pearson']:.4f}, spearman={summary['basic_profile']['corr_spearman']:.4f}")
    print(f"quick FR(logit) AUC={summary['quick_baselines'].get('quick_FR_logit_auc', float('nan')):.4f}")
    print(f"quick MFER(logit) AUC={summary['quick_baselines'].get('quick_MFER_logit_auc', float('nan')):.4f}")
    print(f"quick MFER(ridge) R2={summary['quick_baselines'].get('quick_MFER_ridge_r2', float('nan')):.4f}")
    print("-- weight grid (see 03_weight_grid_auc.csv) --")
    print(tab)
    print(f"best alpha={best_alpha:.2f}")
    print(f"OOF AUC baseline (CI): {auc_b:.4f} [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"OOF AUC stacked  (CI): {auc_s:.4f} [{lo_s:.4f}, {hi_s:.4f}]")
    print(f"Recommendation: {summary['recommendation']}")
    if has_state:
        print("Per-state AUC saved -> 04_per_state_auc.csv")


if __name__ == "__main__":
    main()
