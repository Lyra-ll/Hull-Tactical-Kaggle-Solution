"""
- 负责读取训练/holdout 数据；构造 y（目标）与 w（样本权重）。
- 统一剥离“危险列”（resp_*/action_*/dls_target_* 等）避免泄露。
- 不做折内抽样与标准化；只做列对齐与基础类型修复。
Core functions:
- load_train(cfg) -> Bundle(X, y, w, index)
- load_holdout(cfg) -> Bundle
- build_y(train_df, cfg): 硬/软标签的分支；与 cfg.TARGET_KIND 对齐
- build_w(train_df, cfg): w = |forward_returns| × time_decay（可关）

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import re
import os


# =============== 基本数据容器 ===============
@dataclass
class DataBundle:
    """训练/评估所需的数据打包

    X: pd.DataFrame
    y: Dict[str, np.ndarray]          # 可能含 action_1d/3d/5d 或 fr
    date: pd.Series                  # date_id 列
    w: np.ndarray                    # 样本权重
    cols: List[str]                  # X 的列名列表
    sol: Optional[pd.DataFrame]      # (forward_returns, risk_free_rate) 仅用于离线评分/分析；不参与训练
    """
    X: pd.DataFrame
    y: Dict[str, np.ndarray]
    date: pd.Series
    w: np.ndarray
    cols: List[str]
    sol: pd.DataFrame | None = None

# =============== 列发现 & 规则口径 ===============
def _detect_date_col(df: pd.DataFrame, cfg: Any | None = None) -> str:
    '''寻找日期列'''
    if cfg is not None and hasattr(cfg, "DATE_COL"):
        dc = getattr(cfg, "DATE_COL")
        if dc in df.columns:
            return dc
    for cand in ("date_id", "Date", "DATE"):
        if cand in df.columns:
            return cand
    raise ValueError("未找到日期列（期望: date_id / Date / DATE 或 config.DATE_COL）")

# 统一黑名单（唯一真源）：任何包含这些片段的列都不可进入 X/模型
# 注意：不包含 'forward'，这样 holdout 可保留 forward_returns 供离线分析，
# 但我们在构建 X 时会始终排除它。
_BANNED = ("dls_target", "resp_", "action_", "_lead", "shift_", "label", "target",
           "sample_weight", "weight") 

def _is_safe_feature(name: str) -> bool:
    s = name.lower()
    return not any(b in s for b in _BANNED)

def _assert_holdout_purity(df: pd.DataFrame) -> None:
    """holdout严格过滤"""
    bad = [c for c in df.columns if any(b in c.lower() for b in _BANNED)]
    if bad:
        raise AssertionError(f"[LeakGuard] Holdout 含禁止列: {bad}")
# =============== y / w / X 构建 ===============
def _build_y_train(df:pd.DataFrame) -> Dict[str, np.ndarray]:
    y:Dict[str, np.ndarray] = {}
    
    # 1. 加载 Action (硬标签，用于 LGBM 训练)
    for key in ("action_1d", "action_3d", "action_5d"):
        if key in df.columns:
            y[key] = df[key].astype(np.uint8).values
            
    # 2. [新增] 加载 Resp (真实收益，用于计算 Sharpe 和 波动率)
    for key in ("resp_1d", "resp_3d", "resp_5d"):
        if key in df.columns:
            # 保持 float32 精度
            y[key] = df[key].astype(np.float32).values
            
    # 3. 兜底逻辑
    if not y and "forward_returns" in df.columns:
        y["fr"] = (df["forward_returns"]>0).astype(np.uint8).values
        
    if not y:
        raise ValueError("未找到目标列（期望: action_* 或 forward_returns）")
    return y

def _make_weight_train(df: pd.DataFrame) -> np.ndarray:
    """样本权重赋值"""
    if "forward_returns" in df.columns:
        w = df["forward_returns"].to_numpy(dtype=np.float32)
        w = np.abs(w)
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ones(len(df), dtype=np.float32)

def _build_X(df: pd.DataFrame, date_col:str) -> tuple[pd.DataFrame, List[str]]:
    """从 df 中构建纯净的数值特征矩阵 X（排除 date/目标/权重/泄露列）"""
    num = df.select_dtypes(include=[np.number])
    drop = {date_col, "forward_returns", "market_forward_excess_returns"}
    drop |= {c for c in num.columns if c.startswith(("action_", "resp_", "dls_target_"))}
    safe_cols = [c for c in num.columns if c not in drop and _is_safe_feature(c)]
    X = num[safe_cols].copy()
    return X, safe_cols

def _merge_kf_features(df: pd.DataFrame, cfg: Any, mode: str = "dev") -> pd.DataFrame:
    """
    Merge Kalman-filter-derived features into the main dataframe **safely**.

    Why this exists:
    - 旧实现是“长度相等就直接按 index 拼接”，在任何 reorder / filter / concat 发生后都可能错位，
      这种错位会把 KF 特征随机打乱 → AUC/Sharpe 断崖式下跌（典型症状：加入 KF 就狂掉）。

    Strategy:
    1) Prefer key-based merge (row_id/date_id) when possible.
    2) Fallback to index-based concat ONLY when length matches AND (optional) date_id spot-check passes.
    3) Emit diagnostics and optionally refuse to merge when alignment looks suspicious.
    """
    if not getattr(cfg, "KF_ENABLE", False):
        return df

    kf_dir = getattr(cfg, "KF_DIR", getattr(cfg, "kF_DIR", "kf_out"))
    fname = getattr(cfg, "KF_DEV_FILE", "dev_kf_features") if mode == "dev" else getattr(cfg, "KF_HOLDOUT_FILE", "holdout_kf_features")

    path_pq = os.path.join(kf_dir, fname + ".parquet")
    path_csv = os.path.join(kf_dir, fname + ".csv")

    kf_df: Optional[pd.DataFrame] = None
    if os.path.exists(path_pq):
        print(f"[data] Loading KF ({mode}) from {path_pq}")
        try:
            kf_df = pd.read_parquet(path_pq)
        except Exception as e:
            print(f"[data] ❌ Failed to read parquet KF file: {e}")
            kf_df = None
    elif os.path.exists(path_csv):
        print(f"[data] Loading KF ({mode}) from {path_csv}")
        try:
            kf_df = pd.read_csv(path_csv)
        except Exception as e:
            print(f"[data] ❌ Failed to read csv KF file: {e}")
            kf_df = None
    else:
        print(f"[data] ⚠️ KF enabled but file missing: {fname} in {kf_dir}")
        return df

    if kf_df is None or len(kf_df) == 0:
        print("[data] ⚠️ KF dataframe is empty. Skip merge.")
        return df

    # ---- helpers ----
    def _infer_join_keys(df_main: pd.DataFrame, df_kf: pd.DataFrame) -> List[str]:
        # user-configurable
        keys = getattr(cfg, "KF_JOIN_KEYS", None)
        if isinstance(keys, (list, tuple)) and len(keys) > 0:
            keys = [str(k) for k in keys]
            keys = [k for k in keys if (k in df_main.columns) and (k in df_kf.columns)]
            if keys:
                return keys

        # heuristics: prefer (date_id, row_id) if both exist; else row_id only.
        if ("date_id" in df_main.columns) and ("date_id" in df_kf.columns) and ("row_id" in df_main.columns) and ("row_id" in df_kf.columns):
            return ["date_id", "row_id"]
        if ("row_id" in df_main.columns) and ("row_id" in df_kf.columns):
            return ["row_id"]

        # Do NOT auto-merge on date_id only (almost always many-to-many).
        return []

    def _spotcheck_date_alignment(df_main: pd.DataFrame, df_kf: pd.DataFrame) -> bool:
        # If both have a date column, make sure a few positions match before allowing index-concat.
        try:
            dc_main = _detect_date_col(df_main, cfg)
            dc_kf = _detect_date_col(df_kf, cfg)
        except Exception:
            return True  # no date col -> cannot check, allow (but this is rare)

        if (dc_main not in df_main.columns) or (dc_kf not in df_kf.columns):
            return True

        n = len(df_main)
        if n == 0:
            return True
        # deterministic sample positions
        idxs = np.unique(np.clip(np.array([0, 1, 2, n // 3, n // 2, (2 * n) // 3, n - 3, n - 2, n - 1]), 0, n - 1))
        a = df_main.iloc[idxs][dc_main].to_numpy()
        b = df_kf.iloc[idxs][dc_kf].to_numpy()
        ok = bool(np.all(a == b))
        if not ok:
            print(f"[data][kf] ❌ index-fallback date spot-check failed on {len(idxs)} positions. "
                  f"main[{dc_main}] != kf[{dc_kf}] at sampled rows.")
        return ok

    def _rename_collisions(df_main: pd.DataFrame, df_kf: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        feat_cols = [c for c in df_kf.columns if c not in keys]
        collisions = sorted(set(feat_cols) & set(df_main.columns))
        if not collisions:
            return df_kf
        prefix = str(getattr(cfg, "KF_COL_PREFIX", "kf_"))
        ren = {}
        for c in collisions:
            if c.startswith(prefix):
                ren[c] = prefix + "dup_" + c[len(prefix):]
            else:
                ren[c] = prefix + c
        print(f"[data][kf] ⚠️ column collisions detected: n={len(collisions)} | sample={collisions[:5]} -> renamed with prefix")
        return df_kf.rename(columns=ren)

    strict = bool(getattr(cfg, "KF_STRICT_ALIGN", True))
    allow_index_fallback = bool(getattr(cfg, "KF_ALLOW_INDEX_FALLBACK", True))
    max_allnan_ratio = float(getattr(cfg, "KF_MAX_ALLNAN_RATIO", 0.001))

    # Prefer key-based merge
    join_keys = _infer_join_keys(df, kf_df)
    if join_keys:
        # Make sure KF keys are unique (else merge becomes ambiguous)
        if kf_df.duplicated(join_keys).any():
            dup_n = int(kf_df.duplicated(join_keys).sum())
            print(f"[data][kf] ❌ KF keys are not unique ({join_keys}), dup_rows={dup_n}. Skip merge.")
            return df

        kf_df2 = _rename_collisions(df, kf_df, join_keys)
        feat_cols = [c for c in kf_df2.columns if c not in join_keys]

        # If df has duplicate keys, merge is still defined but probably wrong for our purpose.
        if df.duplicated(join_keys).any():
            print(f"[data][kf] ⚠️ main df has duplicated join keys {join_keys}. "
                  f"Will merge as many-to-one; verify your KF generator.")
            validate = "many_to_one"
        else:
            validate = "one_to_one"

        try:
            merged = df.merge(
                kf_df2[join_keys + feat_cols],
                on=join_keys,
                how="left",
                sort=False,
                validate=validate,
            )
        except Exception as e:
            print(f"[data][kf] ❌ key-merge failed ({join_keys}): {e}. Skip merge.")
            return df

        # Alignment sanity: too many all-NaN rows => almost certainly wrong key mapping
        if feat_cols:
            allnan_ratio = float(merged[feat_cols].isna().all(axis=1).mean())
            if allnan_ratio > max_allnan_ratio:
                msg = (f"[data][kf] ❌ key-merge produced too many missing KF rows: "
                       f"all_nan_ratio={allnan_ratio:.3%} (thr={max_allnan_ratio:.3%}). "
                       f"Keys={join_keys}.")
                if strict:
                    print(msg + " -> refusing merge")
                    return df
                print(msg + " -> keep merge anyway (KF_STRICT_ALIGN=False)")

        print(f"[data] Merged {len(feat_cols)} KF features via keys={join_keys}.")
        return merged

    # Fallback: index-based concat (only if length matches and spot-check passes)
    if allow_index_fallback and (len(kf_df) == len(df)):
        if strict and (not _spotcheck_date_alignment(df, kf_df)):
            return df

        # Avoid duplicating obvious key columns if present
        drop_keys = [c for c in ("row_id", "date_id") if c in kf_df.columns and c in df.columns]
        kf_df3 = kf_df.drop(columns=drop_keys, errors="ignore")
        kf_df3 = _rename_collisions(df, kf_df3, keys=[])
        kf_df3.index = df.index

        merged = pd.concat([df, kf_df3], axis=1)
        print(f"[data] Merged {kf_df3.shape[1]} KF features via index-fallback.")
        return merged

    # Otherwise: refuse to merge (safer than silent misalignment)
    if len(kf_df) != len(df):
        print(f"[data][kf] ⚠️ Length mismatch! main={len(df)}, kf={len(kf_df)}. "
              f"No join keys found, so merge is skipped.")
    else:
        print(f"[data][kf] ⚠️ No join keys found and index fallback disabled. Merge skipped.")
    return df


def load_train(cfg:Any) -> DataBundle:
    """从训练集 CSV 读入并打包（应用 date 过滤、权重、目标等）"""
    path = getattr(cfg, "RAW_DATA_FILE", "train_final_features.csv")
    df = pd.read_csv(path)

    
    date_col = _detect_date_col(df, cfg)
    #丢弃前1055天
    min_id = getattr(cfg, "ANALYSIS_START_DATE_ID", 1055)
    if date_col in df.columns:
        df = df[df[date_col] >= min_id].reset_index(drop=True)
        
    df = _merge_kf_features(df, cfg, mode="dev")

    sol = None
    if ("forward_returns" in df.columns) and ("risk_free_rate" in df.columns):
        sol = df[["forward_returns", "risk_free_rate"]].copy()


    y = _build_y_train(df)
    w = _make_weight_train(df)
    X, cols = _build_X(df,date_col)
    return DataBundle(X, y, df[date_col], w, cols, sol=sol)

def load_holdout(cfg:Any) -> DataBundle:
    path = getattr(cfg, "HOLDOUT_DATA_FILE", "test_final_features.csv")
    df = pd.read_csv(path)
    _assert_holdout_purity(df)
    df = _merge_kf_features(df, cfg, mode="holdout")
    date_col = _detect_date_col(df, cfg)
    X, cols = _build_X(df, date_col)
    w = np.ones(len(df), dtype=np.float32)
    y:Dict[str, np.ndarray] = {}
    return DataBundle(X, y, df[date_col], w, cols)