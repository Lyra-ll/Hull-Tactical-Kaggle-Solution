#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalman-enhanced feature generator + comprehensive ranking (no leakage).

Fixes vs v1
-----------
- ✅ No more DataFrame fragmentation warnings: build KF columns in memory then `pd.concat` once.
- ✅ Parquet optional: will try pyarrow/fastparquet; otherwise automatically fall back to CSV.
- ✅ New flags: `--save-format {auto,parquet,csv}` and `--engine {auto,pyarrow,fastparquet}`.

What it does
------------
1) Loads dev (train_final_features.csv) and/or holdout (test_final_features.csv) by reading config.py.
2) Builds labels (resp_{h}d, action_{h}d) from forward_returns if missing, with proper horizon NaN handling.
3) Generates causal 1D Kalman Filter features for a set of numeric base features (and forward_returns):
      <feat>_kf_trend, <feat>_kf_innov, <feat>_kf_prec
   * q/r are auto-scaled per feature unless you override via CLI.
4) Computes per-feature usefulness metrics against each horizon:
      - AUC(action_{h}d, score=feature)
      - Pearson IC(feature, resp_{h}d)
      - Spearman IC(feature, resp_{h}d)
   and aggregates to an overall rank.
5) Saves:
      ./kf_out/dev_kf_features.(parquet|csv)
      ./kf_out/holdout_kf_features.(parquet|csv)
      ./kf_out/feature_ranking.csv

Run
---
# dev only (default):
python kf_feature_rank.py

# include holdout processing and ranking based on dev labels only (to avoid peeking):
python kf_feature_rank.py --do-holdout

# customize KF hyperparams scaling (q = q_scale * var(feature), r = r_scale * var(feature))
python kf_feature_rank.py --q-scale 1e-4 --r-scale 1e-3

# force CSV (no pyarrow installed):
python kf_feature_rank.py --save-format csv

Notes
-----
- No smoother is used; the filter is strictly causal.
- Ranking is computed on the dev split ONLY to avoid using holdout info.
- If "ANALYSIS_START_DATE_ID" is present in config.py, we respect it on dev.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

# ======== Global switches for pruning ========
TOPK = 60                # <<< 这里改你想要的 K（例如 60）
PRUNE_POOL = 200         # 在做去冗余前，先从综合排名里取前多少做候选池
SPEARMAN_THRES = 0.92    # 去冗余时的 Spearman 相关阈值
# ============================================

# ---------- Utilities ----------

def is_safe_feature_name(name: str) -> bool:
    """Return False for any feature name that hints at leakage or labels.
    Filters out forward/target/label/lead/etc. Applied to *all* feature sources.
    """
    s = name.lower()
    banned_substrings = [
        "forward",        # any forward-looking field name
        "dls_target",     # soft labels
        "resp_", "action_",  # labels
        "_lead", "shift-",   # future shifts
        "label", "target",   # generic
    ]
    return not any(b in s for b in banned_substrings)

def load_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("_cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)  # type: ignore
    return cfg


def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    have = all(c in out.columns for c in ["action_1d","action_3d","action_5d","resp_1d","resp_3d","resp_5d"])
    if have:
        return out
    if "forward_returns" not in out.columns:
        raise ValueError("Missing labels and forward_returns. Need at least 'forward_returns' to build labels.")
    fr = out["forward_returns"]
    for h in (1,3,5):
        stacked = [fr.shift(-i) for i in range(h)]
        multi = pd.concat(stacked, axis=1).sum(axis=1, min_count=h)
        out[f"resp_{h}d"] = multi
        act = (multi > 0).astype(float)
        act[multi.isna()] = np.nan
        out[f"action_{h}d"] = act
    return out


# Causal 1D Kalman filter producing trend, innovation and predictive variance

def kf_1d(obs: np.ndarray, q: float, r: float, x0: float = 0.0, p0: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(obs)
    x = np.zeros(n); p = np.zeros(n)
    innov = np.full(n, np.nan); s_var = np.full(n, np.nan)
    x_prev, p_prev = x0, p0
    for t in range(n):
        # predict
        x_pred = x_prev
        p_pred = p_prev + q
        ot = obs[t]
        if np.isnan(ot):
            # no update if missing observation
            x[t], p[t] = x_pred, p_pred
        else:
            s = p_pred + r
            k = p_pred / s
            innov[t] = ot - x_pred
            x[t] = x_pred + k * innov[t]
            p[t] = (1.0 - k) * p_pred
            s_var[t] = s
        x_prev, p_prev = x[t], p[t]
    return x, innov, s_var


def auto_qr(vec: np.ndarray, q_scale: float, r_scale: float) -> Tuple[float,float]:
    v = np.nanvar(vec)
    if not np.isfinite(v) or v <= 0:
        v = 1e-6
    return float(max(1e-12, q_scale * v)), float(max(1e-12, r_scale * v))


def find_base_features(df: pd.DataFrame, prefix_hint: str = "EIPVSD") -> List[str]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # hard bans: anything forward/target/label-related is excluded
    banned = {
        "forward_returns",
        "sample_weight","sw","weight","weights",
        # labels & variants
        "resp_1d","resp_3d","resp_5d","action_1d","action_3d","action_5d",
        # soft-label (targets) should NEVER be used as features
        "dls_target_1d","dls_target_3d","dls_target_5d",
    }
    def ok(c: str) -> bool:
        if c in banned:
            return False
        if not is_safe_feature_name(c):
            return False
        c_low = c.lower()
        # prefer your repo’s typical prefixes and rolling/diff hints
        return (c and c[0] in set(prefix_hint)) or ("_rol_" in c_low) or ("_diff" in c_low)
    return [c for c in num_cols if ok(c)]


def build_kf_block(df: pd.DataFrame, cols: List[str], q_scale: float, r_scale: float, add_forward_returns: bool=True) -> pd.DataFrame:
    """Build KF features without column-by-column assignment to avoid fragmentation."""
    if add_forward_returns and ("forward_returns" in df.columns):
        cols = cols + ["forward_returns"]
    cols = list(dict.fromkeys(cols))  # de-dup, keep order

    blocks: Dict[str, np.ndarray] = {}
    for c in cols:
        arr = df[c].astype(float).values
        q, r = auto_qr(arr, q_scale=q_scale, r_scale=r_scale)
        trend, innov, s_var = kf_1d(arr, q=q, r=r)
        blocks[f"{c}_kf_trend"] = trend
        blocks[f"{c}_kf_innov"] = innov

        # 避免极小方差导致的“巨大但有限”的精度值
        safe_s = np.where(np.isfinite(s_var) & (s_var > 1e-6), s_var, np.nan)
        prec = 1.0 / safe_s

        # winsorize 上限到 99.9 分位，进一步加 log 压缩一个版本
        if np.isfinite(prec).sum() > 50:
            cap = np.nanpercentile(prec, 99.9)
            prec = np.clip(prec, 0, cap)

        blocks[f"{c}_kf_prec"] = prec
        blocks[f"{c}_kf_logprec"] = np.log1p(prec)  # 新增一个稳定版
        
    out = pd.DataFrame(blocks, index=df.index)
    # defragment
    out = out.copy()
    return out


def auc_binary(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(y) | np.isnan(score))
    y = y[mask].astype(float); score = score[mask].astype(float)
    if y.size < 3: return None
    pos = (y > 0.5)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1==0 or n0==0: return None
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score)+1)
    s = pd.Series(score[order])
    rank_series = s.groupby(s).cumcount() + 1
    counts = s.value_counts()
    cum = counts.cumsum()
    starts = cum - counts + 1
    avg_ranks = (starts + cum) / 2.0
    avg_map = avg_ranks.reindex(s.values).values
    ranks[order] = avg_map
    sum_pos = ranks[pos]
    auc = (sum_pos.sum() - n1*(n1+1)/2.0) / (n1*n0)
    return float(auc)


def corr_spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]; y = y[mask]
    if x.size < 3: return None
    return float(pd.Series(x).corr(pd.Series(y), method='spearman'))


def corr_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]; y = y[mask]
    if x.size < 3: return None
    vx, vy = np.var(x), np.var(y)
    if vx<=0 or vy<=0: return None
    return float(pd.Series(x).corr(pd.Series(y), method='pearson'))


def feature_ranking(dev: pd.DataFrame, all_feature_cols: List[str]) -> pd.DataFrame:
    dev = ensure_labels(dev)
    rows = []
    for f in all_feature_cols:
        x = dev[f].astype(float).values
        metrics = {}
        for h in (1,3,5):
            y_bin = dev[f"action_{h}d"].values
            y_cont = dev[f"resp_{h}d"].values
            metrics[f"auc_{h}d"] = auc_binary(y_bin, x)
            metrics[f"ic_{h}d"] = corr_pearson(x, y_cont)
            metrics[f"rankic_{h}d"] = corr_spearman(x, y_cont)
        rows.append({"feature": f, **metrics})
    rank_df = pd.DataFrame(rows)

    # build an overall score: average of ranks across available metrics
    metric_cols = [c for c in rank_df.columns if c != "feature"]
    for c in metric_cols:
        rank_df[f"rank_{c}"] = (-rank_df[c]).rank(method='average', na_option='bottom')  # negative for descending
    rank_cols = [c for c in rank_df.columns if c.startswith("rank_")]
    rank_df["rank_overall"] = rank_df[rank_cols].mean(axis=1)
    rank_df.sort_values("rank_overall", inplace=True)
    return rank_df


def try_save_table(df: pd.DataFrame, path_base: Path, save_format: str = "auto", engine: str = "auto") -> Path:
    """Save df as parquet if possible, else CSV. Returns final path."""
    out_path = path_base
    if save_format not in {"auto","parquet","csv"}:
        save_format = "auto"
    if engine not in {"auto","pyarrow","fastparquet"}:
        engine = "auto"

    if save_format in {"auto","parquet"}:
        try:
            df.to_parquet(str(path_base.with_suffix('.parquet')), index=False, engine=None if engine=="auto" else engine)
            return path_base.with_suffix('.parquet')
        except Exception as e:
            if save_format == "parquet":
                raise
            # fall back
    # CSV fallback
    df.to_csv(str(path_base.with_suffix('.csv')), index=False)
    return path_base.with_suffix('.csv')


def main():
    root = Path.cwd()
    cfg_path = root / "config.py"
    if not cfg_path.exists():
        print("❌ 找不到 config.py，请在项目根目录运行。")
        sys.exit(1)
    cfg = load_config(cfg_path)

    out_dir = root / "kf_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-holdout", action='store_true', help="Whether to also build KF features for holdout file")
    ap.add_argument("--q-scale", type=float, default=1e-5, help="Q scale: q = q_scale * var(feature)")
    ap.add_argument("--r-scale", type=float, default=1e-4, help="R scale: r = r_scale * var(feature)")
    ap.add_argument("--max-base", type=int, default=200, help="Cap number of base features to run KF on (top N by variance)")
    ap.add_argument("--save-format", choices=["auto","parquet","csv"], default="auto")
    ap.add_argument("--engine", choices=["auto","pyarrow","fastparquet"], default="auto")
    args = ap.parse_args()

    # Load dev
    dev_path = root / getattr(cfg, "RAW_DATA_FILE", "train_final_features.csv")
    if not dev_path.exists():
        print(f"❌ 找不到训练特征文件: {dev_path}")
        sys.exit(1)
    dev = pd.read_csv(dev_path)

    # Respect analysis start
    start_id = getattr(cfg, "ANALYSIS_START_DATE_ID", None)
    if start_id is not None and "date_id" in dev.columns:
        dev = dev[dev["date_id"] >= start_id].reset_index(drop=True)

    # Choose base features from dev
    base_feats = find_base_features(dev)
    if len(base_feats) > args.max_base:
        var_rank = pd.Series({c: np.nanvar(dev[c].values.astype(float)) for c in base_feats}).sort_values(ascending=False)
        base_feats = list(var_rank.head(args.max_base).index)
    print(f"Base features selected for KF: {len(base_feats)}")

    # Build KF block on dev
    dev_kf = build_kf_block(dev, base_feats, q_scale=args.q_scale, r_scale=args.r_scale, add_forward_returns=False)
    dev_kf_path_base = out_dir / "dev_kf_features"
    final_dev_kf_path = try_save_table(dev_kf, dev_kf_path_base, save_format=args.save_format, engine=args.engine)
    print(f"Saved dev KF features -> {final_dev_kf_path}")

    # Merge for ranking
    dev_aug = pd.concat([dev, dev_kf], axis=1)

    # Build full feature list (original + manual + KF) with GLOBAL safety filter
    orig_manual = [c for c in dev.columns if pd.api.types.is_numeric_dtype(dev[c])]
    banned = {"forward_returns","sample_weight","resp_1d","resp_3d","resp_5d","action_1d","action_3d","action_5d",
              "dls_target_1d","dls_target_3d","dls_target_5d"}
    orig_manual = [c for c in orig_manual if (c not in banned) and is_safe_feature_name(c)]
    all_feats = sorted(set([c for c in (orig_manual + list(dev_kf.columns)) if is_safe_feature_name(c)]))

    # Ranking on dev only (to avoid holdout peeking)
    rank_df = feature_ranking(dev_aug, all_feats)
    # save clean ranking
    rank_path = out_dir / "feature_ranking.csv"
    rank_df.to_csv(rank_path, index=False)
    print(f"Saved feature ranking -> {rank_path}")

    # also save a leakage-filtered version explicitly (same as rank_df now), for clarity
    clean_rank_path = out_dir / "feature_ranking_clean.csv"
    rank_df.to_csv(clean_rank_path, index=False)

    # Redundancy pruning: greedy by overall rank using Spearman corr threshold
    def prune_redundant(df_feat: pd.DataFrame, rank_df: pd.DataFrame, max_corr: float = 0.92, top_n: int = 200) -> list[str]:
        pool = rank_df.sort_values("rank_overall").head(top_n)["feature"].tolist()
        kept: list[str] = []
        for f in pool:
            ok = True
            x = df_feat[f].astype(float)
            for g in kept:
                y = df_feat[g].astype(float)
                # robust correlation with NaN handling
                m = ~(x.isna() | y.isna())
                if m.sum() < 20:
                    continue
                rho = pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman")
                if pd.notna(rho) and abs(rho) >= max_corr:
                    ok = False
                    break
            if ok:
                kept.append(f)
        return kept

    pruned_top = prune_redundant(dev_aug, rank_df, max_corr=SPEARMAN_THRES, top_n=PRUNE_POOL)[:TOPK]
    # --- robust save of pruned list with Windows-friendly newlines + read-back sanity check ---
    pruned_path = out_dir / "top_features_pruned.txt"
    sep = "\r\n"  # 用 CRLF，避免在某些Windows编辑器/工具里被当成一行
    pruned_text = sep.join(pruned_top) + sep
    pruned_path.write_text(pruned_text, encoding="utf-8")

    # 立刻回读校验：每行一个特征名、且这些列都在 dev_aug 里存在
    _read_back = [l.strip() for l in pruned_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    _missing = [f for f in _read_back if f not in dev_aug.columns]
    if _missing:
        # 若仍有异常，给出前几项示例，方便你定位
        raise RuntimeError(f"[KF] pruned文件回读校验失败：有 {len(_missing)} 列不在dev列名中，例如：{_missing[:5]}\n"
                        f"请检查KF生成是否与排名同步、或是否手动编辑破坏了换行。")

    print(f"Saved pruned top-{TOPK} ->", pruned_path)
    print(f"Saved pruned top-{TOPK} ->", out_dir / "top_features_pruned.txt")

    # Optionally process holdout KF block (no ranking computed on holdout)
    if args.do_holdout:
        hold_path = root / getattr(cfg, "HOLDOUT_DATA_FILE", "test_final_features.csv")
        if hold_path.exists():
            hold = pd.read_csv(hold_path)
            hold_kf = build_kf_block(hold, base_feats, q_scale=args.q_scale, r_scale=args.r_scale, add_forward_returns=False)
            hold_kf_path_base = out_dir / "holdout_kf_features"
            final_hold_kf_path = try_save_table(hold_kf, hold_kf_path_base, save_format=args.save_format, engine=args.engine)
            print(f"Saved holdout KF features -> {final_hold_kf_path}")
        else:
            print(f"⚠️ 找不到持有集特征文件: {hold_path}")

    # Quick preview of top features
    head_cols = ["feature","rank_overall","rank_auc_1d","rank_auc_3d","rank_auc_5d","rank_ic_1d","rank_ic_3d","rank_ic_5d","rank_rankic_1d","rank_rankic_3d","rank_rankic_5d"]
    preview = rank_df[[c for c in head_cols if c in rank_df.columns]].head(30)
    print(f"Top-{TOPK} features by overall rank (clean):", preview.to_string(index=False))
    print(f"Pruned Top-{TOPK} (<={SPEARMAN_THRES} Spearman pairwise):", "".join(pruned_top))


if __name__ == "__main__":
    main()
