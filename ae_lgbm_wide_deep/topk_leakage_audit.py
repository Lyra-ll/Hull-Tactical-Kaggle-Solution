#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topk_leakage_audit.py

Purpose
-------
Diagnose whether your "global TopK" list is leaking (look-ahead) versus your
train-only AUC ranking being too noisy.

It runs multiple *feature-selection modes* under the SAME CV splitter (cv.time_series_folds),
then reports:
- OOF AUC (weighted, action_1d)
- OOF "official-style" score (Volatility-Adjusted Sharpe with penalties), using OOF predictions

Modes
-----
1) global_file
   - Use your existing ranking file(s) (e.g. kf_out/feature_ranking_clean.csv) as-is.
   - This may be leaky if that ranking used future/validation segments.

2) global_trainonly_auc
   - Build a single TopK list using ONLY rows strictly before the first validation window used.
   - Then use this list for all folds.

3) fold_auc
   - For each fold, rank candidate features by univariate (directionless) AUC on TRAIN ONLY,
     then pick TopK for that fold.

Notes
-----
- This audit is meant to be *fast* and *controlled*.
- It does NOT attempt to replicate your whole pipeline (AE/weather/decision layer).
  The goal is to isolate feature-selection leakage / instability.

Usage (recommended)
-------------------
python topk_leakage_audit.py --k 80 --candidate 300 --modes global_file global_trainonly_auc fold_auc
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("lightgbm is required for this audit script. Please install lightgbm.") from e


# =============================
# Safety: banned columns
# =============================
BANNED_EXACT = {
    "forward_returns",
    "market_forward_excess_returns",
    "risk_free_rate",
}
BANNED_SUBSTR = (
    "dls_target",
    "resp_",
    "action_",
    "_lead",
    "shift_",
    "label",
    "target",
    "sample_weight",
    "weight",
)

def is_safe_feature(name: str) -> bool:
    s = str(name).lower()
    if s in BANNED_EXACT:
        return False
    for b in BANNED_SUBSTR:
        if b in s:
            return False
    return True


# =============================
# Official-style score (from your pasted "Hull starter" snippet)
# =============================
def official_score(
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    position: np.ndarray,
) -> float:
    fr = np.asarray(forward_returns, dtype=np.float64)
    rf = np.asarray(risk_free_rate, dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)

    # Strategy = rf*(1-pos) + pos*fr
    strat_ret = rf * (1.0 - pos) + pos * fr

    strat_excess = strat_ret - rf
    # geometric mean of (1 + excess) - 1
    try:
        strat_mean = float(np.prod(1.0 + strat_excess) ** (1.0 / len(strat_excess)) - 1.0)
    except FloatingPointError:
        return 0.0

    strat_std = float(np.std(strat_ret, ddof=1))
    if not np.isfinite(strat_std) or strat_std == 0.0:
        return 0.0

    sharpe = strat_mean / strat_std * np.sqrt(252.0)
    s_vol = strat_std * np.sqrt(252.0) * 100.0

    # Market
    mkt_excess = fr - rf
    mkt_mean = float(np.prod(1.0 + mkt_excess) ** (1.0 / len(mkt_excess)) - 1.0)
    mkt_std = float(np.std(fr, ddof=1))
    mkt_vol = mkt_std * np.sqrt(252.0) * 100.0
    if not np.isfinite(mkt_vol) or mkt_vol == 0.0:
        return 0.0

    # Penalties
    vol_penalty = 1.0 + max(0.0, s_vol / mkt_vol - 1.2)

    return_gap = max(0.0, (mkt_mean - strat_mean) * 100.0 * 252.0)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    return float(sharpe / (vol_penalty * return_penalty))


def proba_to_position(proba: np.ndarray) -> np.ndarray:
    # Simple, monotonic mapping: [0,1] -> [0,2]
    p = np.asarray(proba, dtype=np.float64)
    return np.clip(2.0 * p, 0.0, 2.0)


# =============================
# Data loading / prep
# =============================
def detect_date_col(df: pd.DataFrame, cfg_module=None) -> str:
    if cfg_module is not None and hasattr(cfg_module, "DATE_COL"):
        dc = getattr(cfg_module, "DATE_COL")
        if dc in df.columns:
            return dc
    for cand in ("date_id", "Date", "DATE"):
        if cand in df.columns:
            return cand

    # Common case: saved index column "Unnamed: 0"
    if df.columns.size > 0 and str(df.columns[0]).lower().startswith("unnamed"):
        # If it looks like a date_id column, accept it.
        s0 = df.iloc[:, 0]
        if np.issubdtype(s0.dtype, np.number) and s0.nunique() > 100:
            df.rename(columns={df.columns[0]: "date_id"}, inplace=True)
            return "date_id"

    raise ValueError("Cannot find date column (expected: date_id / Date / DATE or config.DATE_COL).")


def build_xywr(
    df: pd.DataFrame,
    date_col: str,
    max_nan_ratio: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray]:
    """
    Returns:
      X (numeric, filtered, safe),
      y (action_1d),
      w (abs(forward_returns)),
      dates (pd.Series),
      forward_returns,
      risk_free_rate
    """
    if "action_1d" in df.columns:
        y = df["action_1d"].astype(np.uint8).to_numpy()
    elif "forward_returns" in df.columns:
        y = (df["forward_returns"] > 0).astype(np.uint8).to_numpy()
    else:
        raise ValueError("Need action_1d or forward_returns in training CSV.")

    if "forward_returns" not in df.columns or "risk_free_rate" not in df.columns:
        raise ValueError("Training CSV must contain forward_returns and risk_free_rate for official score audit.")

    fr = df["forward_returns"].astype(np.float64).to_numpy()
    rf = df["risk_free_rate"].astype(np.float64).to_numpy()

    w = np.abs(fr).astype(np.float32)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    # candidate numeric feature table
    num = df.select_dtypes(include=[np.number]).copy()
    # drop exact banned + date col
    drop = {date_col, *BANNED_EXACT}
    # drop target-like prefixes
    for c in list(num.columns):
        cl = str(c).lower()
        if (c in drop) or (not is_safe_feature(c)):
            drop.add(c)
            continue
        if cl.startswith(("action_", "resp_", "dls_target_")):
            drop.add(c)

    feat_cols = [c for c in num.columns if c not in drop and is_safe_feature(c)]
    X = num[feat_cols].copy()

    # global missing filter: match pipeline behavior (<= threshold AND <1.0)
    nan_ratio = X.isna().mean(0)
    keep_cols = nan_ratio.index[(nan_ratio <= max_nan_ratio) & (nan_ratio < 1.0)].tolist()
    X = X[keep_cols].copy()

    # sanity: no banned exacts
    for bad in BANNED_EXACT:
        if bad in X.columns:
            raise RuntimeError(f"[LEAK] {bad} is in X; it must never be used as a feature.")

    dates = df[date_col]
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    return X, y, w, dates, fr, rf


def read_rank_files(paths: List[str], avail_cols: List[str]) -> List[str]:
    """
    Read ranked feature names from CSVs.
    Accepts:
      - column named 'feature'
      - otherwise first column
    Returns a de-duplicated ordered list filtered to avail_cols.
    """
    avail = set(avail_cols)
    out: List[str] = []
    seen = set()
    for p in paths:
        if not p:
            continue
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            col = "feature" if "feature" in df.columns else df.columns[0]
            names = [str(x) for x in df[col].tolist()]
        except Exception:
            # tolerate weird files
            try:
                names = [str(x).strip() for x in open(p, "r", encoding="utf-8").read().splitlines() if str(x).strip()]
            except Exception:
                continue
        for n in names:
            if n in avail and n not in seen and is_safe_feature(n):
                out.append(n)
                seen.add(n)
    return out


# =============================
# Ranking helpers
# =============================
def univariate_auc_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    w: Optional[np.ndarray],
    cols: List[str],
) -> Dict[str, float]:
    """
    Directionless univariate AUC per feature:
      score = max(auc, 1-auc) in [0.5, 1.0]
    NaNs are filled with median computed on *this* sample.
    """
    out: Dict[str, float] = {}
    yy = y
    ww = w
    for c in cols:
        s = X[c]
        if s.isna().all():
            out[c] = 0.5
            continue
        med = float(s.median(skipna=True))
        x = s.fillna(med).to_numpy(dtype=np.float64)
        try:
            auc = roc_auc_score(yy, x, sample_weight=ww)
        except Exception:
            out[c] = 0.5
            continue
        if not np.isfinite(auc):
            out[c] = 0.5
            continue
        out[c] = float(max(auc, 1.0 - auc))
    return out


def select_topk_by_scores(scores: Dict[str, float], k: int) -> List[str]:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [c for c, _ in items[:k]]


# =============================
# Model (fixed)
# =============================
def lgbm_gain_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    w: Optional[np.ndarray],
    cols: List[str],
    seed: int = 42,
) -> Dict[str, float]:
    """Fold-internal feature ranking using a single LightGBM fit.

    IMPORTANT: This ranks using ONLY the given (training-fold) rows.
    It does *not* touch validation fold labels, so it is CV-safe.

    We use a small model + early stopping on a time-ordered internal split
    inside the training fold to avoid overfitting and keep it fast.
    """
    cols = [c for c in cols if c in X.columns]
    if not cols:
        return {}

    n = len(X)
    # time-ordered split: last 20% as internal valid
    split = int(n * 0.8)
    split = max(100, min(split, n - 100)) if n >= 250 else max(1, int(n * 0.8))

    X_tr = X.iloc[:split][cols]
    y_tr = y[:split]
    X_va = X.iloc[split:][cols]
    y_va = y[split:]

    w_tr = w[:split] if w is not None else None
    w_va = w[split:] if w is not None else None

    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=True)
    dva = lgb.Dataset(X_va, label=y_va, weight=w_va, free_raw_data=True)

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        min_gain_to_split=0.0,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        verbosity=-1,
        seed=seed,
        feature_fraction_seed=seed,
        bagging_seed=seed,
        data_random_seed=seed,
        deterministic=True,
    )

    try:
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=300,
            valid_sets=[dva],
            valid_names=["va"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    except Exception:
        # fall back: train without early stopping
        booster = lgb.train(params, dtr, num_boost_round=100)

    gains = booster.feature_importance(importance_type="gain")
    out = {c: float(g) for c, g in zip(cols, gains)}
    return out

def fit_predict_lgbm(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    w_tr: Optional[np.ndarray],
    w_va: Optional[np.ndarray],
    seed: int,
) -> Tuple[np.ndarray, float, Optional[int]]:
    params = dict(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.05,
        n_estimators=1200,
        random_state=seed,
        n_jobs=-1,
    )
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    proba = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, proba, sample_weight=w_va)
    best_iter = getattr(clf, "best_iteration_", None)
    return proba, float(auc), int(best_iter) if best_iter else None


@dataclass
class RunResult:
    mode: str
    embargo: int
    oof_auc: float
    oof_score: float
    folds: int
    notes: str


# =============================
# Main
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=80, help="TopK size.")
    ap.add_argument("--candidate", type=int, default=300, help="Candidate pool size (from ranking files or fallback all).")
    ap.add_argument("--modes", nargs="+", default=["global_file", "global_trainonly_auc"],
                    choices=["global_file", "global_trainonly_auc", "fold_auc", "fold_lgbm"],
                    help="Selection modes to compare.")
    ap.add_argument("--embargo", nargs="*", type=int, default=None, help="Embargo values to sweep. Default uses config.EMBARGO_SIZE.")
    ap.add_argument("--purge", type=int, default=None, help="Purge size. Default uses config.PURGE_SIZE.")
    ap.add_argument("--n_splits", type=int, default=None, help="Number of date blocks (default uses config.N_SPLITS).")
    ap.add_argument("--n_folds", type=int, default=None, help="DEPRECATED alias for --n_splits (kept for compatibility).")
    ap.add_argument("--warmup_blocks", type=int, default=None, help="Warmup blocks (default uses config.WARMUP_BLOCKS).")
    ap.add_argument("--last_k", type=int, default=None, help="Use last_k folds (default uses config.N_LAST_FOLDS_TO_USE_INFERENCE). Use 0 for all.")
    ap.add_argument("--rank_files", nargs="*", default=None, help="CSV ranking files (default uses config.TOPK_FILES).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (default uses config.GLOBAL_SEED).")
    ap.add_argument("--max_nan_ratio", type=float, default=None, help="Global missing threshold (default uses config.MISSING_THRESHOLD or MAX_NAN_RATIO).")
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module("config")
    cvmod = importlib.import_module("cv")

    raw_path = getattr(cfg, "RAW_DATA_FILE", "train_final_features.csv")
    df = pd.read_csv(raw_path)

    date_col = detect_date_col(df, cfg)
    min_id = int(getattr(cfg, "ANALYSIS_START_DATE_ID", 1055))
    if date_col in df.columns:
        df = df[df[date_col] >= min_id].reset_index(drop=True)

    max_nan_ratio = args.max_nan_ratio
    if max_nan_ratio is None:
        max_nan_ratio = getattr(cfg, "MAX_NAN_RATIO", None)
        if max_nan_ratio is None:
            max_nan_ratio = getattr(cfg, "MISSING_THRESHOLD", 0.30)
    max_nan_ratio = float(max(0.0, min(1.0, max_nan_ratio)))

    X, y, w, dates, fr, rf = build_xywr(df, date_col, max_nan_ratio=max_nan_ratio)

    # Candidate pool
    rank_files = args.rank_files if args.rank_files is not None else list(getattr(cfg, "TOPK_FILES", []))
    ranked_cols = read_rank_files(rank_files, list(X.columns)) if rank_files else []
    if not ranked_cols:
        ranked_cols = list(X.columns)

    candidate_pool = ranked_cols[: max(args.candidate, args.k)]
    candidate_pool = [c for c in candidate_pool if c in X.columns]

    purge = int(args.purge) if args.purge is not None else int(getattr(cfg, "PURGE_SIZE", 0))
    embargo_list = args.embargo if args.embargo is not None else [int(getattr(cfg, "EMBARGO_SIZE", 40))]

    n_splits = args.n_splits if args.n_splits is not None else (args.n_folds if args.n_folds is not None else int(getattr(cfg, "N_SPLITS", 5)))
    warmup_blocks = args.warmup_blocks if args.warmup_blocks is not None else int(getattr(cfg, "WARMUP_BLOCKS", 1))
    use_k = args.last_k if args.last_k is not None else int(getattr(cfg, "N_LAST_FOLDS_TO_USE_INFERENCE", 3))
    seed = args.seed if args.seed is not None else int(getattr(cfg, "GLOBAL_SEED", 42))

    print(f"[audit] raw={raw_path} rows={len(df)} | features={X.shape[1]} | k={args.k} | candidate={len(candidate_pool)}")
    print(f"[audit] cv: n_splits={n_splits} warmup_blocks={warmup_blocks} purge={purge} embargo_list={embargo_list} last_k={use_k}")
    print(f"[audit] modes={args.modes}")

    results: List[RunResult] = []

    for embargo in embargo_list:
        # IMPORTANT: cv.time_series_folds expects pd.Series (uses .values / .isin)
        dates_ser = dates if isinstance(dates, pd.Series) else pd.Series(dates)
        folds = cvmod.time_series_folds(
            dates=dates_ser,
            n_splits=int(n_splits),
            embargo=int(embargo),
            purge=int(purge),
            warmup_blocks=int(warmup_blocks),
        )
        if use_k and use_k > 0:
            folds = cvmod.last_k_folds(folds, k=int(use_k))
        if not folds:
            raise RuntimeError("No folds produced. Check your date column and cv settings.")

        # train-only global window: strictly before the earliest validation index used
        min_val_start = min(int(va.min()) for _, va in folds)
        trainonly_idx = np.arange(0, min_val_start, dtype=int)

        # Precompute global_file topk (just slice)
        global_file_topk = candidate_pool[: args.k]

        # Precompute global_trainonly_auc topk (one-time per embargo)
        global_trainonly_topk: Optional[List[str]] = None
        if "global_trainonly_auc" in args.modes:
            scores = univariate_auc_scores(
                X=X.iloc[trainonly_idx][candidate_pool],
                y=y[trainonly_idx],
                w=w[trainonly_idx],
                cols=candidate_pool,
            )
            global_trainonly_topk = select_topk_by_scores(scores, args.k)

        for mode in args.modes:
            oof_pred = np.full(len(X), np.nan, dtype=np.float64)
            fold_aucs: List[float] = []
            notes: List[str] = []

            for fold_i, (tr_idx, va_idx) in enumerate(folds):
                if mode == "global_file":
                    topk_cols = global_file_topk
                elif mode == "global_trainonly_auc":
                    assert global_trainonly_topk is not None
                    topk_cols = global_trainonly_topk
                elif mode == "fold_auc":
                    scores = univariate_auc_scores(
                        X=X.iloc[tr_idx][candidate_pool],
                        y=y[tr_idx],
                        w=w[tr_idx],
                        cols=candidate_pool,
                    )
                    topk_cols = select_topk_by_scores(scores, args.k)
                elif mode == "fold_lgbm":
                    scores = lgbm_gain_scores(
                        X=X.iloc[tr_idx],
                        y=y[tr_idx],
                        w=w[tr_idx],
                        cols=candidate_pool,
                        seed=seed + 1000 * int(embargo) + 10 * fold_i,
                    )
                    topk_cols = select_topk_by_scores(scores, args.k) if scores else list(global_file_topk)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Fit model
                X_tr = X.iloc[tr_idx][topk_cols]
                X_va = X.iloc[va_idx][topk_cols]
                proba, auc, best_iter = fit_predict_lgbm(
                    X_tr=X_tr,
                    X_va=X_va,
                    y_tr=y[tr_idx],
                    y_va=y[va_idx],
                    w_tr=w[tr_idx],
                    w_va=w[va_idx],
                    seed=seed + 1000 * int(embargo) + 10 * fold_i,
                )
                oof_pred[va_idx] = proba
                fold_aucs.append(auc)
                if best_iter is not None and fold_i == 0:
                    notes.append(f"best_iter(first_fold)={best_iter}")

            mask = np.isfinite(oof_pred)
            oof_auc = float(roc_auc_score(y[mask], oof_pred[mask], sample_weight=w[mask]))

            # Official score on OOF rows (same rows used for AUC)
            pos = proba_to_position(oof_pred[mask])
            oof_score = float(official_score(fr[mask], rf[mask], pos))

            results.append(RunResult(
                mode=mode,
                embargo=int(embargo),
                oof_auc=oof_auc,
                oof_score=oof_score,
                folds=len(folds),
                notes="; ".join(notes) if notes else "",
            ))

            print(f"[audit] mode={mode:20s} embargo={embargo:3d} | OOF AUC={oof_auc:.6f} | OOF score={oof_score:.6f} | folds={len(folds)}")
            if fold_aucs:
                print(f"        per-fold AUC={['%.4f'%x for x in fold_aucs]}")

    # Summary table
    print("\n[audit] SUMMARY")
    rows = []
    for r in results:
        rows.append(dict(mode=r.mode, embargo=r.embargo, oof_auc=r.oof_auc, oof_score=r.oof_score, folds=r.folds, notes=r.notes))
    summ = pd.DataFrame(rows).sort_values(["embargo", "oof_score"], ascending=[True, False])
    with pd.option_context("display.max_rows", 200, "display.max_colwidth", 120):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()