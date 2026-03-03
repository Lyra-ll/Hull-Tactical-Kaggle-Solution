# decision_tune_v2_TILT1_FULL.py
# ============================================================
# Decision layer tuner (TILT1 + EMA + FULL SEARCH)
#
# Upgrades:
#   1) Search space includes: Shape (long/tilt), EMA (alpha), Center Shift, Vol Window.
#   2) Dynamic Volatility Proxy: Re-calculates vol proxy per trial based on window size.
#   3) Causal EMA: Implements the date-aware smoothing logic from pipeline_TILT1.
#
# Usage:
#   python decision_tune_v2_TILT1_FULL.py --run_dir artifacts/run-XXXX --pred pred_blend --n_trials 500
#   python decision_tune_v2_TILT1_FULL.py --run_dir artifacts/run-XXXX --pred pred_blend --n_trials 500 --apply_config
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import optuna
except Exception:
    optuna = None

# ---------------------------------------------------------
# 1. CORE LOGIC PORTS (From pipeline_TILT1.py)
# We embed these to ensure tuning matches new pipeline logic perfectly
# even if local pipeline.py is outdated.
# ---------------------------------------------------------

def _ema_by_date(values: np.ndarray, date_id: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA smoothing by date_id."""
    x = np.asarray(values, dtype=float)
    d = np.asarray(date_id)
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    # unique dates (not necessarily sorted in input, but we sort unique)
    uniq, inv = np.unique(d, return_inverse=True)
    
    # Calculate daily means first (robust to multiple rows per day)
    sums = np.zeros(len(uniq), dtype=float)
    cnts = np.zeros(len(uniq), dtype=float)
    np.add.at(sums, inv, x)
    np.add.at(cnts, inv, 1.0)
    per = sums / np.maximum(cnts, 1.0)

    # EMA on daily means
    order = np.argsort(uniq, kind="mergesort")
    per_s = per[order]
    ema_s = np.empty_like(per_s)
    
    prev = per_s[0]
    ema_s[0] = prev
    a = float(alpha)
    a = min(max(a, 0.0), 1.0)
    
    for i in range(1, len(per_s)):
        prev = a * per_s[i] + (1.0 - a) * prev
        ema_s[i] = prev

    # Map back to original rows
    # We computed EMA on sorted unique dates, so we map back via inverse
    # But inv maps to unsorted uniq. We need a mapping from unsorted_uniq_idx -> sorted_uniq_idx
    # Actually simpler:
    # 1. ema_s corresponds to uniq[order]
    # 2. We need to map back to 'inv' which corresponds to 'uniq'
    # Let's re-order ema_s back to the order of 'uniq'
    
    # map: sorted_idx -> value
    # we want: unsorted_idx -> value
    # order maps: sorted_i -> unsorted_i
    # so: ema_s[k] is value for uniq[order[k]]
    
    ema_unsorted = np.empty_like(ema_s)
    ema_unsorted[order] = ema_s
    
    out = ema_unsorted[inv]
    return out.astype(np.float32, copy=False)

def _vol_proxy_dynamic(
    values: np.ndarray,
    date_id: Optional[np.ndarray],
    window: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Dynamic vol proxy calculation for tuning loop."""
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[0]
    
    if date_id is None:
        order = np.arange(n, dtype=int)
    else:
        d = np.asarray(date_id, dtype=np.float64)
        order = np.lexsort((np.arange(n, dtype=int), d))

    v_sorted = v[order]
    s = pd.Series(v_sorted)
    # shift(1) to be causal
    vol = s.shift(1).rolling(window=window, min_periods=max(5, window // 2)).std()
    vol = vol.ffill()
    
    base_std = float(np.nanstd(v_sorted) or 1.0)
    vol = vol.fillna(base_std)
    out_sorted = np.maximum(vol.to_numpy(dtype=np.float64), eps)

    out = np.empty(n, dtype=np.float64)
    # Inverse map
    # out[order[i]] = out_sorted[i]
    out[order] = out_sorted
    return out

def _decision_allocation_logic(
    pred: np.ndarray,
    vol: Optional[np.ndarray],
    cfg: Any,
    regime_scale: Optional[np.ndarray] = None,
    date_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Standalone implementation of the decision logic matching pipeline_TILT1.py.
    """
    # Parse Config
    coef = float(getattr(cfg, "DECISION_COEF", 5.0))
    decay = float(getattr(cfg, "DECISION_DECAY", 0.0))
    thr = float(getattr(cfg, "DECISION_THRESHOLD", 0.0))
    
    shape = str(getattr(cfg, "DECISION_SHAPE", "long_only")).lower()
    
    center_shift = float(getattr(cfg, "DECISION_CENTER_SHIFT", 0.0))
    
    ema_enable = bool(getattr(cfg, "DECISION_EMA_ENABLE", False))
    ema_alpha = float(getattr(cfg, "DECISION_EMA_ALPHA", 0.1))
    
    clip_low = float(getattr(cfg, "DECISION_CLIP_LOW", 0.0))
    clip_high = float(getattr(cfg, "DECISION_CLIP_HIGH", 2.0))
    min_inv = float(getattr(cfg, "MIN_INVESTMENT", 0.0))
    max_inv = float(getattr(cfg, "MAX_INVESTMENT", 2.0))

    pred = np.asarray(pred, dtype=float)

    # 1. Signal (Center Shift)
    center = 0.5 + center_shift
    signal = pred - center

    # 2. Deadzone
    if thr > 0:
        signal = np.where(np.abs(signal) < thr, 0.0, signal)

    # 3. Shape Mapping
    if shape == "tilt1":
        # Tilt around 1.0
        alloc = 1.0 + coef * signal
    else:
        # Standard long-only
        alloc = coef * signal

    # 4. Vol Scaling
    if (vol is None) or (decay <= 0):
        pass # denom = 1.0
    else:
        v = np.asarray(vol, dtype=float)
        denom = np.power(np.maximum(v, 1e-6), decay) + 1e-6
        alloc = alloc / denom

    # 5. Regime Overlay
    if regime_scale is not None:
        rs = np.asarray(regime_scale, dtype=float)
        rs = np.where(np.isfinite(rs), rs, 1.0)
        alloc = alloc * rs

    # 6. EMA Smoothing
    if ema_enable and (date_id is not None):
        if ema_alpha > 0:
            alloc = _ema_by_date(alloc, date_id, ema_alpha)

    # 7. Final Clip
    alloc = np.clip(alloc, min_inv, max_inv).astype(np.float32, copy=False)
    
    # Meta stats
    if shape == "tilt1":
        coverage = float(np.mean(np.abs(alloc - 1.0) > 0.01)) if alloc.size else 0.0
    else:
        coverage = float(np.mean(alloc > 0.01)) if alloc.size else 0.0
        
    meta = {
        "coverage": coverage
    }
    return alloc, meta

# ---------------------------------------------------------
# 2. SCORE FUNCTION (Official)
# ---------------------------------------------------------

def _score_official_fallback(
    solution: pd.DataFrame,
    position: np.ndarray,
    min_invest: float = 0.0,
    max_invest: float = 2.0,
) -> float:
    if solution is None or len(solution) == 0:
        return 0.0
    
    fr = solution["forward_returns"].to_numpy(dtype=np.float64, copy=False)
    rf = solution["risk_free_rate"].to_numpy(dtype=np.float64, copy=False)
    p = np.asarray(position, dtype=np.float64).reshape(-1)
    
    p = np.clip(p, min_invest, max_invest)

    strat_ret = rf * (1.0 - p) + p * fr
    strat_excess = strat_ret - rf
    mkt_excess = fr - rf

    if strat_ret.size < 2: return 0.0

    one_plus = 1.0 + strat_excess
    if np.any(one_plus <= 0): return 0.0
    strat_geo = float(np.exp(np.mean(np.log(one_plus))) - 1.0)

    mkt_one_plus = 1.0 + mkt_excess
    if np.any(mkt_one_plus <= 0): return 0.0
    mkt_geo = float(np.exp(np.mean(np.log(mkt_one_plus))) - 1.0)

    strat_std = float(np.std(strat_excess) + 1e-12)
    mkt_std = float(np.std(mkt_excess) + 1e-12)

    sharpe = strat_geo / strat_std * np.sqrt(252.0)

    s_vol = strat_std * np.sqrt(252.0) * 100.0
    mkt_vol = mkt_std * np.sqrt(252.0) * 100.0
    if mkt_vol == 0.0: return 0.0

    vol_penalty = 1.0 + max(0.0, s_vol / mkt_vol - 1.2)
    return_gap = max(0.0, (mkt_geo - strat_geo) * 100.0 * 252.0)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    out = sharpe / (vol_penalty * return_penalty)
    return float(out) if np.isfinite(out) else 0.0

# ---------------------------------------------------------
# 3. TUNING INFRASTRUCTURE
# ---------------------------------------------------------

@dataclass
class BestResult:
    pred_col: str
    score: float
    coverage: float
    params: Dict[str, Any]
    created_at: str

def _coerce_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _detect_pred_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.lower().startswith("pred")]
    pref = ["pred_blend", "pred_ae_lgbm", "pred_lgbm"]
    ordered: List[str] = []
    for p in pref:
        if p in df.columns: ordered.append(p)
    for c in cols:
        if c not in ordered: ordered.append(c)
    return ordered

def tune_one_pred(
    df: pd.DataFrame,
    pred_col: str,
    n_trials: int,
    seed: int,
    clip_low: float,
    clip_high: float,
    min_invest: float,
    max_invest: float,
    use_regime: bool,
    regime_col: Optional[str],
    min_valid: int,
    tune_regime: bool,
    tune_decay: bool,
    vol_source: str,
) -> Tuple[BestResult, pd.DataFrame]:
    
    if optuna is None: raise RuntimeError("optuna not installed.")
    
    # 1. Prepare Data
    fr = _coerce_numeric(df["forward_returns"])
    rf = _coerce_numeric(df["risk_free_rate"])
    pred = _coerce_numeric(df[pred_col])
    date_id = _coerce_numeric(df["date_id"]) if "date_id" in df.columns else np.arange(len(df), dtype=float)

    mask = np.isfinite(fr) & np.isfinite(rf) & np.isfinite(pred)
    
    rs_full = None
    if regime_col and (regime_col in df.columns):
        rs_full = _coerce_numeric(df[regime_col])
    elif "regime_scale" in df.columns:
        rs_full = _coerce_numeric(df["regime_scale"])
        
    idx = np.flatnonzero(mask)
    if idx.size < min_valid:
        raise ValueError(f"Too few valid rows: {idx.size}")

    # Sliced Data
    sol = pd.DataFrame({"forward_returns": fr[idx], "risk_free_rate": rf[idx]})
    pred_v = pred[idx]
    date_v = date_id[idx]
    rs_v = rs_full[idx] if (rs_full is not None) else None
    
    # For Vol Source, we need the raw series to re-compute window dynamically
    # If vol_source='pred', we use pred_v
    # If vol_source='resp', we use resp_1d
    raw_vol_series = None
    vol_src_str = str(vol_source).lower()
    
    if vol_src_str in ("pred", "prediction"):
        raw_vol_series = pred_v
    elif vol_src_str in ("resp", "ret"):
        if "resp_1d" in df.columns:
            raw_vol_series = _coerce_numeric(df["resp_1d"])[idx]
        else:
            print("[warn] resp_1d missing for vol. Using pred instead.")
            raw_vol_series = pred_v
            
    # Cache for vol proxies to avoid recomputing same window repeatedly
    vol_cache: Dict[int, np.ndarray] = {}

    def get_vol_for_window(w: int) -> Optional[np.ndarray]:
        if raw_vol_series is None: return None
        if w not in vol_cache:
            vol_cache[w] = _vol_proxy_dynamic(raw_vol_series, date_v, w)
        return vol_cache[w]

    # 2. Objective Function
    def objective(trial: optuna.Trial) -> float:
        # === Search Space (FULL) ===
        
        # 1. Shape & Coef
        shape = trial.suggest_categorical("shape", ["long_only", "tilt1"])
        
        if shape == "tilt1":
            # Tilt usually needs smaller coef as it adds to 1.0
            coef = trial.suggest_float("coef", 0.01, 10.0, log=True)
            # Center shift only makes sense for tilt/long_only
            center_shift = trial.suggest_float("center_shift", -0.05, 0.05)
        else:
            coef = trial.suggest_float("coef", 0.1, 50.0, log=True)
            center_shift = trial.suggest_float("center_shift", -0.05, 0.05)
            
        threshold = trial.suggest_float("threshold", 0.0, 0.05)
        
        # 2. Volatility
        if tune_decay and (raw_vol_series is not None):
            decay = trial.suggest_float("decay", 0.0, 1.5)
            vol_win = trial.suggest_categorical("vol_window", [5, 10, 20, 40, 60])
            vol_trial = get_vol_for_window(vol_win)
        else:
            decay = 0.0
            vol_win = 20
            vol_trial = None

        # 3. Regime
        if tune_regime and (rs_v is not None):
            use_reg = trial.suggest_categorical("use_regime", [True, False])
        else:
            use_reg = use_regime

        # 4. EMA (New!)
        ema_en = trial.suggest_categorical("ema_enable", [True, False])
        if ema_en:
            ema_a = trial.suggest_float("ema_alpha", 0.05, 0.5)
        else:
            ema_a = 0.1

        # Construct Config
        cfg_local = SimpleNamespace(
            DECISION_COEF=coef,
            DECISION_SHAPE=shape,
            DECISION_CENTER_SHIFT=center_shift,
            DECISION_THRESHOLD=threshold,
            
            DECISION_DECAY=decay,
            DECISION_VOL_WINDOW=vol_win,
            
            DECISION_EMA_ENABLE=ema_en,
            DECISION_EMA_ALPHA=ema_a,
            
            DECISION_CLIP_LOW=clip_low,
            DECISION_CLIP_HIGH=clip_high,
            MIN_INVESTMENT=min_invest,
            MAX_INVESTMENT=max_invest,
        )

        # Allocation
        alloc, _ = _decision_allocation_logic(
            pred_v, vol_trial, cfg_local,
            regime_scale=rs_v if use_reg else None,
            date_id=date_v
        )

        # Score
        try:
            sc = float(_score_official_fallback(sol, alloc, min_invest=min_invest, max_invest=max_invest))
        except Exception:
            sc = 0.0
            
        return sc if np.isfinite(sc) else -1e9

    # 3. Run Optimization
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 4. Process Best Result
    best = study.best_trial
    bp = best.params
    
    # Reconstruct best config
    best_shape = bp.get("shape", "long_only")
    best_cfg = SimpleNamespace(
        DECISION_COEF=bp["coef"],
        DECISION_SHAPE=best_shape,
        DECISION_CENTER_SHIFT=bp.get("center_shift", 0.0),
        DECISION_THRESHOLD=bp["threshold"],
        DECISION_DECAY=bp.get("decay", 0.0),
        DECISION_EMA_ENABLE=bp.get("ema_enable", False),
        DECISION_EMA_ALPHA=bp.get("ema_alpha", 0.1),
        DECISION_CLIP_LOW=clip_low,
        DECISION_CLIP_HIGH=clip_high,
        MIN_INVESTMENT=min_invest,
        MAX_INVESTMENT=max_invest,
    )
    
    best_vol_win = bp.get("vol_window", 20)
    best_vol = get_vol_for_window(best_vol_win) if (bp.get("decay", 0.0) > 0) else None
    
    best_use_reg = bool(bp.get("use_regime", use_regime))
    
    alloc, _ = _decision_allocation_logic(
        pred_v, best_vol, best_cfg,
        regime_scale=rs_v if best_use_reg else None,
        date_id=date_v
    )
    
    final_score = _score_official_fallback(sol, alloc, min_invest, max_invest)
    if best_shape == "tilt1":
        coverage = float(np.mean(np.abs(alloc - 1.0) > 0.01))
    else:
        coverage = float(np.mean(alloc > 0.01))

    # Pack params for JSON
    out_params = {
        "coef": bp["coef"],
        "shape": best_shape,
        "center_shift": bp.get("center_shift", 0.0),
        "threshold": bp["threshold"],
        "decay": bp.get("decay", 0.0),
        "vol_window": best_vol_win,
        "ema_enable": bp.get("ema_enable", False),
        "ema_alpha": bp.get("ema_alpha", 0.1),
        "use_regime": best_use_reg,
        "vol_source": str(vol_source),
        
        "clip_low": clip_low, "clip_high": clip_high,
        "min_invest": min_invest, "max_invest": max_invest
    }

    best_res = BestResult(
        pred_col=pred_col,
        score=float(final_score),
        coverage=coverage,
        params=out_params,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    
    # Save trials
    rows = []
    for t in study.trials:
        if t.value is None: continue
        r = dict(t.params)
        r["value"] = float(t.value)
        rows.append(r)
    df_trials = pd.DataFrame(rows).sort_values("value", ascending=False)
    
    return best_res, df_trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pred", default="pred_blend")
    ap.add_argument("--n_trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vol_source", default="pred", help="Must be 'pred' for online consistency")
    
    ap.add_argument("--use_regime", action="store_true")
    ap.add_argument("--tune_regime", action="store_true")
    ap.add_argument("--tune_decay", action="store_true", default=True) # Default True for full search
    ap.add_argument("--regime_col", default="regime_scale")
    
    ap.add_argument("--apply_config", action="store_true")
    ap.add_argument("--config_path", default="config.py")

    args = ap.parse_args()
    
    run_dir = args.run_dir
    oof_path = os.path.join(run_dir, "oof_pack.csv")
    
    if not os.path.exists(oof_path):
        print(f"ERROR: {oof_path} not found.")
        return 1
        
    df = pd.read_csv(oof_path)
    
    # Resolve Pred Column
    tune_cols = [args.pred]
    if args.pred == "all":
        tune_cols = _detect_pred_cols(df)
    
    overall_best = None
    
    for col in tune_cols:
        print(f"\n[TUNE] >>> {col} (Full Search: Shape/EMA/Vol/Shift) <<<")
        best_res, df_trials = tune_one_pred(
            df=df, pred_col=col,
            n_trials=args.n_trials, seed=args.seed,
            clip_low=0.0, clip_high=2.0, min_invest=0.0, max_invest=2.0,
            use_regime=args.use_regime, regime_col=args.regime_col,
            min_valid=1000,
            tune_regime=args.tune_regime,
            tune_decay=args.tune_decay,
            vol_source=args.vol_source
        )
        
        out_csv = os.path.join(run_dir, f"tune_tilt1_{col}.csv")
        df_trials.to_csv(out_csv, index=False)
        
        print(f"[RESULT] Score={best_res.score:.4f} | Shape={best_res.params['shape']} | EMA={best_res.params['ema_enable']}")
        
        if (overall_best is None) or (best_res.score > overall_best.score):
            overall_best = best_res
            
    # Save & Apply
    if overall_best:
        best_json = os.path.join(run_dir, "best_decision_params_v2.json")
        with open(best_json, "w") as f:
            json.dump({
                "pred_col": overall_best.pred_col,
                "score": overall_best.score,
                "params": overall_best.params,
                "created_at": overall_best.created_at
            }, f, indent=2)
            
        print(f"\n[DONE] Best Overall: {overall_best.pred_col} -> Score {overall_best.score:.4f}")
        
        # Hints
        p = overall_best.params
        print("\n[CONFIG HINTS]")
        print(f"DECISION_MODE = \"{p['shape']}\"")
        print(f"DECISION_COEF = {p['coef']:.6f}")
        print(f"DECISION_CENTER_SHIFT = {p['center_shift']:.6f}")
        print(f"DECISION_DECAY = {p['decay']:.6f}")
        print(f"DECISION_VOL_WINDOW = {p['vol_window']}")
        print(f"DECISION_EMA_ENABLE = {p['ema_enable']}")
        print(f"DECISION_EMA_ALPHA = {p['ema_alpha']:.4f}")
        
        if args.apply_config:
            try:
                sys.path.insert(0, os.getcwd())
                from tools.apply_best_decision_to_config import apply_best_json_to_config
                apply_best_json_to_config(best_json_path=best_json, config_path=args.config_path, pred_col=overall_best.pred_col)
                print("[APPLY] Config updated.")
            except Exception as e:
                print(f"[WARN] Apply config failed: {e}")

    return 0

if __name__ == "__main__":
    main()