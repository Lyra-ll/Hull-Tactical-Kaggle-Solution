"""
Orchestrates the run: global filter -> per-fold preprocess -> models -> artifacts.

- 统一指挥流：全局列筛选（MAX_NAN_RATIO）→ 按折预处理 → 训练基线 LGBM → （可选）AE→LGBM Wide&Deep。
- 返回 artifacts（OOF、fold metrics、holdout 预测等）以及“实际使用列”供 main 对齐。

Core:
- fit_evaluate(cfg, train, holdout=None, primary_head="lgbm") -> result dict
  result 包含：
    - feature_cols（全局筛后的列集）
    - models（最后 k 折基线模型）
    - oof/oof_mask/auc_oof/fold_aucs/best_params
    - （可选）holdout_pred_lgbm_wd

TODO:
- Weather 接入：features.weather.fit_on_train()/transform()（折内拟合、仅 transform）
- LGBM 三头（1d/3d/5d）与聚合权重；特征重要性/校准曲线/SHAP（可选）
- Optuna 搜索空间（LGBM/AE），与 N_TRIALS_* 对齐
- KF/RFE/LOFO 候选池接线：全局预筛 + 消融报告
"""



from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from data import DataBundle
import cv as cvmod
import os, json
import shutil
from pathlib import Path
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from encoder import fit_encode
from sklearn.preprocessing import StandardScaler
import pandas as pd
from preprocess import FoldPreprocessor
from pathlib import Path
import numpy as np
import random
import torch

# === [ADD] Weather integration ===
import re
try:
    # Preferred package-style import when running inside project structure
    from features import weather as weather_mod
except Exception:  # pragma: no cover
    # Fallback for environments where `features/` is not a package on sys.path
    import weather as weather_mod


# ===== [ADD] Leak-safe feature filtering =====
_FORBIDDEN_FEATURE_PATTERNS = [
    r'^risk_free_rate$',
    r'^forward_returns$',
    r'^market_forward_excess_returns$',
    r'^market_forward_returns$',
    r'^resp_\d+d$',
    r'^action_\d+d$',
]

def _drop_forbidden_feature_cols(X: pd.DataFrame, *, strict: bool = False, where: str = 'X') -> pd.DataFrame:
    """
    Drop columns that must never be used as model features (leak-prone labels/targets).

    Notes:
    - This does NOT touch the `solution` dataframe used for official scoring (it is separate).
    - `strict=True` raises if any forbidden columns are present; otherwise it will drop and continue.
    """
    if X is None or getattr(X, 'empty', False):
        return X
    cols = list(getattr(X, 'columns', []))
    if not cols:
        return X
    to_drop: List[str] = []
    for c in cols:
        for pat in _FORBIDDEN_FEATURE_PATTERNS:
            if re.match(pat, str(c)):
                to_drop.append(c)
                break
    if not to_drop:
        return X
    msg = f"[leak_guard] forbidden feature cols present in {where}: n={len(to_drop)} | sample={to_drop[:5]}"
    if strict:
        raise RuntimeError(f"[LEAK] {msg}")
    print(msg + " -> dropping")
    return X.drop(columns=to_drop, errors='ignore')

def _parse_weather_patterns(cfg):
    """
    把 config 里的 WEATHER_SPEC / WEATHER_EXCLUDE_REGEX 解析成 weather.fit_on_train 所需的 list[str]
    """
    spec = getattr(cfg, "WEATHER_SPEC", r"regex:.*(ema|sma|mom|rol_mean|rol_std|vol|skew|kurt|rank_|cs_|kf_).*")
    if isinstance(spec, str) and spec.startswith("regex:"):
        include_regex = [spec[len("regex:"):]]
    elif isinstance(spec, (list, tuple)):
        include_regex = list(spec)
    else:
        include_regex = [str(spec)]
    # 缺省黑名单：目标/收益/样本权重/时间索引统统排除，防泄露
    exclude_regex = list(getattr(cfg, "WEATHER_EXCLUDE_REGEX", [
        r"^action_", r"^resp_", r"^dls_target_", r"^sample_weight$",
        r"^date_id$", r"^row_id$", r"^forward_returns$"
    ]))
    return include_regex, exclude_regex

def _infer_engine_from_post_cols(cols):
    """
    从列名里提取engine前缀比如hmm，列名形如weather_hmm_p0
    """
    for c in cols:
        m = re.match(r"^weather_(\w+)_p\d+$", c)
        if m:
            return m.group(1)
    return "hmm"  # 默认

def derive_weather_confidence(W_df:pd.DataFrame) -> pd.DataFrame:
    """
    输入：post形式天气dataframe列
    输出：两列，weather_(eng)_margin, weather_(eng)_entropy
    输出两个天气有关的数据当成特征，也许会有用
    """
    if W_df is None or W_df.empty:
        return None
    engine = _infer_engine_from_post_cols(W_df.columns)
    post_cols = [c for c in W_df.columns if re.match(rf"^weather_{engine}_p\d+$", c)]
    if not post_cols:
        return None

    P = W_df[post_cols].to_numpy(dtype=np.float32, copy=False)
    #margin = top1 - top2
    Ps = np.sort(P, axis=1)[:, ::-1]
    margin = (Ps[:, 0] - Ps[:, 1]).reshape(-1, 1)
    eps = 1e-12
    entropy = (-(P*np.log(P + eps)).sum(axis=1)).reshape(-1, 1)

    out = pd.DataFrame(
        np.hstack([margin, entropy]),
        index = W_df.index,
        columns=[f"weather_{engine}_margin", f"weather_{engine}_entropy"],
    )
    return out

def compute_weather_regime_scale_global(
    X: pd.DataFrame,
    ret_all: np.ndarray,
    cfg: Any,
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    使用训练期的天气 HMM posterior 列 + resp_1d，
    为每个样本构造一个 regime 缩放系数，用于决策层仓位调整。

    返回:
        regime_scale_all: shape=(N,) 的 np.ndarray，或 None
        diag: 诊断信息 dict，或 None
    """
    use_regime = bool(getattr(cfg, "DECISION_USE_WEATHER_REGIME", False))
    if not use_regime:
        return None, None

    engine = str(getattr(cfg, "WEATHER_ENGINE", "hmm"))
    # 只支持 WEATHER_OUTPUT = "post" 的情况
    post_cols = [c for c in X.columns if re.match(rf"^weather_{engine}_p\d+$", c)]
    if not post_cols:
        print("[decision][regime] no weather post columns found in X; disable regime overlay.")
        return None, None

    P = X[post_cols].to_numpy(dtype=np.float32, copy=False)
    if P.ndim != 2 or P.shape[1] < 2:
        print("[decision][regime] weather post has <2 states; disable regime overlay.")
        return None, None

    # 防御性归一化（理论上已经是概率）
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    # 每个样本所属的 HMM state（0..K-1）
    states = P.argmax(axis=1).astype(np.int32)
    n_states = P.shape[1]

    ret_all = np.asarray(ret_all, dtype=float)
    mask_valid = np.isfinite(ret_all)
    if not mask_valid.any():
        print("[decision][regime] no valid returns for regime; disable.")
        return None, None

    # 每个 state 的平均收益 + 样本数
    avg_ret = np.zeros(n_states, dtype=float)
    count = np.zeros(n_states, dtype=int)
    for k in range(n_states):
        m = mask_valid & (states == k)
        count[k] = int(m.sum())
        avg_ret[k] = float(ret_all[m].mean()) if m.any() else 0.0

    min_cnt = int(getattr(cfg, "DECISION_REGIME_MIN_STATE_SAMPLES", 200))
    if (count < min_cnt).any():
        print(f"[decision][regime] some states have <{min_cnt} samples; disable regime overlay.")
        diag = {
            "states": states,
            "avg_ret": avg_ret.tolist(),
            "count": count.tolist(),
            "skipped": True,
        }
        return None, diag

    # 按平均收益从小到大排序
    order = np.argsort(avg_ret)
    ranks = np.empty(n_states, dtype=int)
    ranks[order] = np.arange(n_states)

    q_low, q_high = getattr(cfg, "DECISION_REGIME_QUANTILES", (0.33, 0.67))
    q_low = float(q_low); q_high = float(q_high)
    low_cut = int(np.floor(q_low * n_states))
    high_cut = int(np.ceil(q_high * n_states))

    bull_scale = float(getattr(cfg, "DECISION_REGIME_BULL_SCALE", 1.2))
    neutral_scale = float(getattr(cfg, "DECISION_REGIME_NEUTRAL_SCALE", 1.0))
    bear_scale = float(getattr(cfg, "DECISION_REGIME_BEAR_SCALE", 0.7))

    scale_per_state = np.full(n_states, neutral_scale, dtype=float)
    low_states = np.where(ranks < low_cut)[0]
    high_states = np.where(ranks >= high_cut)[0]
    scale_per_state[low_states] = bear_scale
    scale_per_state[high_states] = bull_scale

    regime_scale_all = scale_per_state[states].astype(np.float32)

    diag = {
        "states": states,
        "avg_ret": avg_ret.tolist(),
        "count": count.tolist(),
        "scale_per_state": scale_per_state.tolist(),
        "low_states": low_states.tolist(),
        "high_states": high_states.tolist(),
        "order": order.tolist(),
        "quantiles": (q_low, q_high),
    }
    return regime_scale_all, diag

def _read_rank_files(paths: list[str], avail_cols:list[str]) -> list[str]:
    """
    从若干排名文件读取候选列名， 按给定顺序优先级合并，去重后与avail cols取交集并保序
    支持.txt(每行一列名) 和 .csv(优先找'feature'列，否则取首列)
    """
    seen, out = set(), []
    avail = set(avail_cols)
    for p in paths:
        if not p:
            continue
        fp = Path(p)
        if not fp.exists():
            continue
        try:
            if fp.suffix.lower() == ".txt":
                with fp.open("r", encoding="utf-8") as f:
                    names = [ln.strip() for ln in f if ln.strip()]
            else:#csv
                df = pd.read_csv(fp)
                col = "feature" if "feature" in df.columns else df.columns[0]
                names = [str(x) for x in df[col].tolist()]
        except Exception:
            continue
        for c in names:
            if (c in avail) and (c not in seen):
                seen.add(c); out.append(c)
    return out

def _corr_prune(df : "pd.DataFrame", cols:list[str], thr:float) ->list[str]:
    """
    对候选列做一个简单的|ρ|≥thr 共线裁剪（在当前 df 上计算）。
    以给定顺序贪心保留，与80列规模匹配计算量可控
    """
    if (thr is None) or (thr <= 0) or (len(cols) <= 1):
        return cols
    keep = []
    for c in cols:
        ok = True
        for k in keep:
            s1, s2 = df[k], df[c]
            rho = s1.corr(s2)
            if pd.notna(rho) and abs(float(rho)) >= thr:
                ok = False
                break;
        if ok:
            keep.append(c)
    return keep

def _resolve_topk_feature_list(cfg, X_df:"pd.DataFrame") -> list[str] | None:
    """
    依据配置解析Top k列池， 返回列名列表或者none
    只针对基表列（此时X df已经预处理做过全局缺失率过滤）
    """
    if not getattr(cfg, "TOPK_ENABLE", False):
        return None
    n = int(getattr(cfg, "TOPK_N", 80))
    files = list(getattr(cfg, "TOPK_FILES", []))

    cands = _read_rank_files(files, list(X_df.columns))
    if not cands:#无可用排名文件
        return None
    
    thr = getattr(cfg, "TOPK_CORR_PRUNE_THR", None)
    if thr is not None:
        cands = _corr_prune(X_df, cands, float(thr))
    
    topk = cands[:n]
    if not topk:
        return None
    
    return topk

def _rank_cols_by_univariate_auc(
    X_df: "pd.DataFrame",
    y: np.ndarray,
    w: np.ndarray | None = None,
    cols: list[str] | None = None,
) -> list[str]:
    """
    折内(训练段)用的快速特征排序：对每列做单变量 AUC（取 max(auc, 1-auc) 视为“有信息量”）。
    - 只使用折内训练数据 (X_df/y/w)；不会读验证段。
    - 处理 NaN：用训练段中位数填充；全空/常数列会回退到 0.5。
    说明：这是“筛选/缩窄候选池”，不是最终建模；交互项仍交给下游模型学习。
    """
    if cols is None:
        cols = list(X_df.columns)
    y = np.asarray(y)
    if w is not None:
        w = np.asarray(w)
        if w.shape[0] != y.shape[0]:
            raise ValueError("[topk] w and y length mismatch in _rank_cols_by_univariate_auc")
    if y.shape[0] != len(X_df):
        raise ValueError("[topk] y and X_df length mismatch in _rank_cols_by_univariate_auc")

    # y 需要至少两类
    y_mask = np.isfinite(y)
    if y_mask.sum() < 10:
        return cols
    y_u = np.unique(y[y_mask])
    if y_u.size < 2:
        return cols

    scores: list[tuple[float, str]] = []
    for c in cols:
        x = X_df[c].to_numpy()
        m = y_mask & np.isfinite(x)
        if m.sum() < 10:
            scores.append((0.5, c))
            continue
        # 常数列直接 0.5
        xu = x[m]
        if np.nanmin(xu) == np.nanmax(xu):
            scores.append((0.5, c))
            continue
        med = np.nanmedian(xu)
        if not np.isfinite(med):
            scores.append((0.5, c))
            continue
        x_f = x.copy()
        x_f[~np.isfinite(x_f)] = med
        try:
            if w is None:
                auc = roc_auc_score(y[m], x_f[m])
            else:
                auc = roc_auc_score(y[m], x_f[m], sample_weight=w[m])
            if not np.isfinite(auc):
                auc = 0.5
        except Exception:
            auc = 0.5
        score = float(max(auc, 1.0 - auc))
        scores.append((score, c))

    scores.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scores]


def _resolve_topk_feature_list_fold(
    cfg,
    X_tr_df: "pd.DataFrame",
    y_tr: np.ndarray,
    w_tr: np.ndarray | None,
) -> list[str] | None:
    """
    折内 Top-K（无泄露）：只用折内训练段计算排序，再做相关性剪枝，再取前 K。
    - 候选池：若 TOPK_FILES 存在，则把文件里出现过的列当作候选；否则候选=训练段所有列。
      （注意：文件里的“顺序”在 fold 模式下不再被信任，只当候选集合。）
    """
    if not getattr(cfg, "TOPK_ENABLE", False):
        return None
    n = int(getattr(cfg, "TOPK_N", 80))
    files = list(getattr(cfg, "TOPK_FILES", []))

    # 候选池：优先用文件限定范围（可显著提速），否则用全列
    cands = _read_rank_files(files, list(X_tr_df.columns))
    if not cands:
        cands = list(X_tr_df.columns)

    ranked = _rank_cols_by_univariate_auc(X_tr_df, y_tr, w_tr, cols=cands)

    thr = getattr(cfg, "TOPK_CORR_PRUNE_THR", None)
    if thr is not None:
        ranked = _corr_prune(X_tr_df, ranked, float(thr))

    topk = ranked[:n]
    if not topk:
        return None
    return topk


#===锁种函数===
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _fold_time_decay_weights(
    date_series,
    base_w:np.ndarray,
    train_idx:np.ndarray,
    val_idx:np.ndarray,
    half_life_days:float,
    floor:float,
    normalize:bool = True,
) -> np.ndarray:
    """
    按折计算时间衰减权重，返回train_idx上的权重
    date_series,
    base_w:np.ndarray,
    train_idx:np.ndarray,
    val_idx:np.ndarray,
    half_life_days:float,
    floor:float,
    normalize:bool = True,
    """
    val_start_date = date_series.iloc[val_idx].min()
    dt = (val_start_date - date_series.iloc[train_idx].values).astype(float)
    w_time = np.maximum(floor, np.power(0.5, dt/float(half_life_days)))
    w = base_w[train_idx] * w_time
    if normalize and w.size > 0:
        w = w / max(w.mean(), 1e-12)
    return w

# =====================================================================
# Decision layer helpers: AUC -> allocation -> Sharpe proxy

def _get_decision_params(cfg: Any) -> dict:
    """从 cfg 中读取决策层相关参数，并补默认值。"""
    return {
        "coef": float(getattr(cfg, "DECISION_COEF", 5.0)),
        "decay": float(getattr(cfg, "DECISION_DECAY", 0.5)),
        "threshold": float(getattr(cfg, "DECISION_THRESHOLD", 0.0)),
        "clip_low": float(getattr(cfg, "DECISION_CLIP_LOW", 0.0)),
        "clip_high": float(getattr(cfg, "DECISION_CLIP_HIGH", 2.0)),
        "vol_window": int(getattr(cfg, "DECISION_VOL_WINDOW", 20)),
        "coverage_floor": float(getattr(cfg, "DECISION_COVERAGE_FLOOR", 0.05)),
    }


def compute_vol_proxy(
    ret_all: np.ndarray,
    window: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    用 resp_1d 估计一个“波动率代理”，用于仓位缩放。

    注意：
    - 用 shift(1) 保证每个时点只用“过去”的收益计算自己的 vol（不看未来）
    - 这是回测评估工具，不进入训练目标
    """
    s = pd.Series(np.asarray(ret_all, dtype=float))
    # 只用过去的收益
    vol = s.shift(1).rolling(window=window, min_periods=max(5, window // 2)).std()
    # 用前向填充 / 全局 std 填空，保证没有 NaN
    vol = vol.bfill()
    if vol.isna().all():
        vol = pd.Series(np.full_like(s, float(s.std() or 1.0)))
    vol = vol.fillna(float(s.std() or 1.0))
    arr = np.maximum(vol.to_numpy(dtype=float), eps)
    return arr

def decision_allocation_from_pred(
    pred: np.ndarray,
    vol: Optional[np.ndarray],
    cfg: Any,
    regime_scale: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """
    将预测概率 -> long-only 仓位区间 [0, 2]。

    - pred: 模型输出的概率（OOF）
    - vol: 预先计算好的波动率 proxy（与 pred 对齐）
    - regime_scale: 每个样本的 Regime 缩放倍数（若为 None 则当作 1.0）
    """
    p = _get_decision_params(cfg)
    pred = np.asarray(pred, dtype=float)

    # 信号：p - 0.5
    signal = pred - 0.5

    # 死区：置信度太低就不交易
    thr = p["threshold"]
    if thr > 0:
        signal = np.where(np.abs(signal) < thr, 0.0, signal)

    # 波动率缩放：vol^decay 放到分母
    if vol is None:
        denom = 1.0
    else:
        v = np.asarray(vol, dtype=float)
        denom = np.power(v, p["decay"]) + 1e-6

    # Regime 缩放：不同状态放大/缩小仓位
    if regime_scale is not None:
        rs = np.asarray(regime_scale, dtype=float)
        if rs.shape != signal.shape:
            raise ValueError(
                f"[decision] regime_scale shape {rs.shape} != signal shape {signal.shape}"
            )
    else:
        rs = 1.0

    alloc = p["coef"] * signal * rs / denom
    alloc = np.clip(alloc, p["clip_low"], p["clip_high"]).astype(np.float32, copy=False)

    coverage = float(np.mean(alloc > 0.01)) if alloc.size else 0.0
    meta = {**p, "coverage": coverage}
    return alloc, meta

def compute_foldwise_weather_regime_scale(
    X: pd.DataFrame,
    ret_all: np.ndarray,
    folds,
    cfg: Any,
) -> tuple[np.ndarray, dict]:
    """
    核心：按折构造“无泄露”的 Regime 缩放向量。

    对每一折：
      - 只用该折训练段 tr 的 resp_1d + weather 状态，估计各 state 的平均收益
      - 按平均收益排序，切成 bear / neutral / bull 三档
      - 只把这个映射应用到该折的验证段 va 样本上

    返回:
        regime_scale_all: shape=(N,) 的缩放系数（初始为 1，只有有定义的 val 才会被覆盖）
        diag: 一些诊断信息（per-fold 平均收益、样本数等）
    """
    enabled = bool(getattr(cfg, "DECISION_USE_WEATHER_REGIME", False))
    N = len(X)
    regime_scale_all = np.ones(N, dtype=np.float32)
    diag: dict = {"enabled": enabled, "per_fold_stats": []}

    if not enabled:
        return regime_scale_all, diag

    engine = str(getattr(cfg, "WEATHER_ENGINE", "hmm"))
    output = str(getattr(cfg, "WEATHER_OUTPUT", "post"))

    # 支持两种输入形式："post" 概率列 或 "state" 整型列
    post_cols = [c for c in X.columns if re.match(rf"^weather_{engine}_p\d+$", c)]
    state_col = None
    if output == "state":
        cand = [c for c in X.columns if c == f"weather_{engine}_state"]
        if cand:
            state_col = cand[0]

    if (state_col is None) and (not post_cols):
        print("[decision][regime] no weather state/post columns found; disable regime overlay.")
        diag["enabled"] = False
        return regime_scale_all, diag

    min_cnt = int(getattr(cfg, "DECISION_REGIME_MIN_STATE_SAMPLES", 200))
    bull_scale = float(getattr(cfg, "DECISION_REGIME_BULL_SCALE", 1.2))
    neutral_scale = float(getattr(cfg, "DECISION_REGIME_NEUTRAL_SCALE", 1.0))
    bear_scale = float(getattr(cfg, "DECISION_REGIME_BEAR_SCALE", 0.7))
    q_low, q_high = getattr(cfg, "DECISION_REGIME_QUANTILES", (0.33, 0.67))
    q_low = float(q_low); q_high = float(q_high)

    ret_all = np.asarray(ret_all, dtype=float)
    mask_ret_valid = np.isfinite(ret_all)

    for fold_idx, (tr, va) in enumerate(folds, start=1):
        tr = np.asarray(tr); va = np.asarray(va)
        if tr.size == 0 or va.size == 0:
            continue

        ret_tr = ret_all[tr]

        # 1) 训练段的 state 序列
        if state_col is not None:
            st_full = X[state_col].to_numpy()
            states_tr = st_full[tr].astype(int)
            states_va = st_full[va].astype(int)
            n_states = int(max(states_tr.max(), states_va.max()) + 1)
        else:
            P_tr = X.iloc[tr][post_cols].to_numpy(dtype=np.float32)
            P_tr = np.clip(P_tr, 1e-12, 1.0)
            P_tr /= P_tr.sum(axis=1, keepdims=True)

            states_tr = P_tr.argmax(axis=1).astype(int)
            n_states = P_tr.shape[1]

            P_va = X.iloc[va][post_cols].to_numpy(dtype=np.float32)
            P_va = np.clip(P_va, 1e-12, 1.0)
            P_va /= P_va.sum(axis=1, keepdims=True)
            states_va = P_va.argmax(axis=1).astype(int)

        # 2) 只在训练段上计算各 state 的平均收益
        avg_ret = np.zeros(n_states, dtype=float)
        count = np.zeros(n_states, dtype=int)
        for k in range(n_states):
            m = (states_tr == k) & mask_ret_valid[tr]
            count[k] = int(m.sum())
            avg_ret[k] = float(ret_tr[m].mean()) if m.any() else 0.0

        if (count < min_cnt).any():
            diag["per_fold_stats"].append({
                "fold": fold_idx,
                "skipped": True,
                "reason": f"min_samples<{min_cnt}",
                "count": count.tolist(),
            })
            continue

        # 3) 按平均收益排序，分档
        order = np.argsort(avg_ret)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n_states)

        low_cut = int(np.floor(q_low * n_states))
        high_cut = int(np.ceil(q_high * n_states))

        scale_per_state = np.full(n_states, neutral_scale, dtype=float)
        low_states = np.where(ranks < low_cut)[0]
        high_states = np.where(ranks >= high_cut)[0]
        scale_per_state[low_states] = bear_scale
        scale_per_state[high_states] = bull_scale

        # 4) 只把映射应用在该折的验证段 va 上（关键：不回头改 tr，更不看未来折的收益）
        if state_col is not None:
            st_full = X[state_col].to_numpy()
            states_va = st_full[va].astype(int)

        scale_va = scale_per_state[states_va]
        regime_scale_all[va] = scale_va.astype(np.float32, copy=False)

        diag["per_fold_stats"].append({
            "fold": fold_idx,
            "skipped": False,
            "avg_ret": avg_ret.tolist(),
            "count": count.tolist(),
            "low_states": low_states.tolist(),
            "high_states": high_states.tolist(),
        })

    return regime_scale_all, diag

def sharpe_proxy(alloc: np.ndarray, ret: np.ndarray) -> float:
    """
    简化版 Sharpe： mean(alloc * ret) / std(alloc * ret)
    不做年化，只看“信号质量”。
    """
    alloc = np.asarray(alloc, dtype=float)
    ret = np.asarray(ret, dtype=float)
    strat_ret = alloc * ret
    mu = float(np.mean(strat_ret))
    sigma = float(np.std(strat_ret) + 1e-9)
    return mu / sigma

def score_official(
    solution: pd.DataFrame,
    position: np.ndarray,
    min_invest: float = 0.0,
    max_invest: float = 2.0,
) -> float:
    """Kaggle 官方同款（或近似同款）离线评分：Volatility-Adjusted Sharpe Ratio。

    重要：
    - solution 必须至少包含列：forward_returns, risk_free_rate
    - position 为策略仓位（0..2），将被 clip 到 [min_invest, max_invest]
    - 该函数只用于离线评估/Optuna 目标；不要在训练中用未来信息反推 position
    """
    if solution is None or len(solution) == 0:
        return 0.0
    if ("forward_returns" not in solution.columns) or ("risk_free_rate" not in solution.columns):
        raise ValueError("solution must contain forward_returns and risk_free_rate")

    fr = solution["forward_returns"].to_numpy(dtype=np.float64, copy=False)
    rf = solution["risk_free_rate"].to_numpy(dtype=np.float64, copy=False)

    p = np.asarray(position, dtype=np.float64).reshape(-1)
    if p.shape[0] != fr.shape[0]:
        raise ValueError(f"position length {p.shape[0]} != solution length {fr.shape[0]}")

    # Strict clip (same as scoring engine expectation)
    p = np.clip(p, min_invest, max_invest)

    # Strategy returns: rf*(1-pos) + pos*forward_returns
    strat_ret = rf * (1.0 - p) + p * fr

    # Excess vs risk free (used for geometric mean)
    strat_excess = strat_ret - rf
    mkt_excess = fr - rf

    # Robust guards
    if strat_ret.size < 2:
        return 0.0

    # Geometric mean of daily returns (same structure as starter)
    one_plus = 1.0 + strat_excess
    one_plus_m = 1.0 + mkt_excess
    if np.any(~np.isfinite(one_plus)) or np.any(~np.isfinite(one_plus_m)):
        return 0.0
    # Avoid invalid geometric mean if <=0 (rare in this competition, but be safe)
    if np.any(one_plus <= 0) or np.any(one_plus_m <= 0):
        return 0.0

    strat_mean = float(np.exp(np.mean(np.log(one_plus))) - 1.0)
    mkt_mean = float(np.exp(np.mean(np.log(one_plus_m))) - 1.0)

    # Std (pandas default ddof=1). Use ddof=1 for consistency.
    strat_std = float(np.std(strat_ret, ddof=1))
    mkt_std = float(np.std(fr, ddof=1))
    if strat_std == 0.0 or mkt_std == 0.0 or (not np.isfinite(strat_std)) or (not np.isfinite(mkt_std)):
        return 0.0

    sharpe = strat_mean / strat_std * np.sqrt(252.0)

    s_vol = strat_std * np.sqrt(252.0) * 100.0
    mkt_vol = mkt_std * np.sqrt(252.0) * 100.0
    if mkt_vol == 0.0 or (not np.isfinite(mkt_vol)):
        return 0.0

    # Penalties (match the starter)
    vol_penalty = 1.0 + max(0.0, s_vol / mkt_vol - 1.2)

    return_gap = max(0.0, (mkt_mean - strat_mean) * 100.0 * 252.0)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    out = sharpe / (vol_penalty * return_penalty)
    if not np.isfinite(out):
        return 0.0
    return float(out)



def fit_evaluate(
    train: DataBundle,
    cfg:Any,
    eval_heads: Tuple[str, ...] = ("head", "lr", "lgbm"),
    primary_head: str = "lgbm",
    holdout: Optional[DataBundle] = None,
) -> Dict[str, Any]:
    """
    训练以及评估骨架
    """
    print(f"[pipeline] eval_heads={eval_heads}, primary_head={primary_head}")
    print(f"[pipeline] train: X={train.X.shape}, y_keys={list(train.y.keys())}, w={train.w.shape}")
    if holdout is not None:
        print(f"[pipeline] holdout: X={holdout.X.shape}, dates={holdout.date.min()}..{holdout.date.max()}")

    # ===== [ADD] Drop leak-prone columns from feature matrices (safe-by-default) =====
    strict_leak = bool(getattr(cfg, 'LEAK_GUARD_STRICT', False))
    train.X = _drop_forbidden_feature_cols(train.X, strict=strict_leak, where='train.X')
    if holdout is not None:
        holdout.X = _drop_forbidden_feature_cols(holdout.X, strict=strict_leak, where='holdout.X')

    #walk forward 折索引
    folds = cvmod.time_series_folds(
        train.date,
        n_splits=getattr(cfg, "N_SPLITS", 5),
        embargo=getattr(cfg, "EMBARGO_SIZE", 30),
        purge=getattr(cfg, "PURGE_SIZE", 0),
        warmup_blocks=getattr(cfg, "WARMUP_BLOCKS", 1)

    )
    use_k = getattr(cfg, "N_LAST_FOLDS_TO_USE_INFERENCE", 3)
    last_fds = cvmod.last_k_folds(folds, k=use_k)
    last_fold_idx = set(range(len(folds) - use_k, len(folds)))
    print(f"[pipeline] CV: total_folds={len(folds)}, last_k={use_k}")
    # ===== [BUNDLE EXPORT] optional: export per-fold models for Kaggle inference =====
    bundle_dir = getattr(cfg, "EXPORT_BUNDLE_DIR", None)
    bundle_enabled = bool(bundle_dir)
    bundle_dir = Path(str(bundle_dir)) if bundle_enabled else None
    bundle_folds = str(getattr(cfg, "EXPORT_BUNDLE_FOLDS", "last_k")).lower()  # "all" or "last_k"
    bundle_copy_code = bool(getattr(cfg, "EXPORT_BUNDLE_COPY_CODE", True))
    bundle_overwrite = bool(getattr(cfg, "EXPORT_BUNDLE_OVERWRITE", True))
    if bundle_enabled:
        if bundle_overwrite and bundle_dir.exists():
            # keep it simple: remove only known files we write, don't nuke whole dir
            bundle_dir.mkdir(parents=True, exist_ok=True)
        else:
            bundle_dir.mkdir(parents=True, exist_ok=True)

        # copy minimal code needed to unpickle objects on Kaggle (preprocess/weather/encoder/config)
        if bundle_copy_code:
            src_dir = Path(__file__).resolve().parent
            for _fname in ["preprocess.py", "weather.py", "encoder.py", "config.py"]:
                _src = src_dir / _fname
                if _src.exists():
                    try:
                        shutil.copy2(_src, bundle_dir / _fname)
                    except Exception as _e:
                        print(f"[bundle][WARN] failed to copy {_fname}: {_e}")

        # meta skeleton
        meta_path = bundle_dir / "bundle_meta.json"
        if not meta_path.exists():
            meta = {
                "created_by": "pipeline.fit_evaluate",
                "export_folds": [],
                "n_splits": int(getattr(cfg, "N_SPLITS", 5)),
                "n_last_folds": int(getattr(cfg, "N_LAST_FOLDS_TO_USE_INFERENCE", 3)),
                "weather_enable": bool(getattr(cfg, "WEATHER_ENABLE", True)),
                "wide_max_weather_cols": int(getattr(cfg, "WIDE_MAX_WEATHER_COLS", 0)),
                "wide_max_ae_z": int(getattr(cfg, "WIDE_MAX_AE_Z", 32)),
            }
            try:
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as _e:
                print(f"[bundle][WARN] failed to init meta: {_e}")

    
    # ========= 插入：全局列掩码 =========
    max_nan_ratio = getattr(cfg, "MAX_NAN_RATIO", None)
    if max_nan_ratio is None:
        max_nan_ratio = getattr(cfg, "MISSING_THRESHOLD", 0.30)
    max_nan_ratio = float(max(0.0, min(1.0, max_nan_ratio)))
    nan_ratio_all = train.X.isna().mean(0)
    global_keep_cols = nan_ratio_all.index[(nan_ratio_all <= max_nan_ratio) & (nan_ratio_all < 1.0)].tolist()
    print(f"[preprocess][global] kept={len(global_keep_cols)} / {train.X.shape[1]} "
        f"| dropped={train.X.shape[1] - len(global_keep_cols)} (>{int(max_nan_ratio*100)}% NaN)")

    train.X = train.X.reindex(columns=global_keep_cols)
    if holdout is not None:
        holdout.X = holdout.X.reindex(columns=global_keep_cols)

    # === [ADD] Top-K 基表列池（只裁基表；天气/AE 另行拼接）===
    topk_mode = str(getattr(cfg, "TOPK_MODE", "global")).lower()
    topk_cols = None
    if getattr(cfg, "TOPK_ENABLE", False) and topk_mode in ("global", "file", "precomputed"):
        topk_cols = _resolve_topk_feature_list(cfg, train.X)
        if topk_cols is not None:
            scope = str(getattr(cfg, "TOPK_APPLY_SCOPE", "both")).lower()
            if scope in ("ae_only", "both", "lgbm_base"):
                train.X = train.X.reindex(columns=topk_cols)
                if holdout is not None:
                    holdout.X = holdout.X.reindex(columns=topk_cols)
            print(f"[topk] enabled: mode=global n={len(topk_cols)} | scope={scope} | sample={topk_cols[:5]}")
    elif getattr(cfg, "TOPK_ENABLE", False) and topk_mode in ("fold", "per_fold", "online"):
        # 折内选择：在 CV 循环里基于训练折计算；这里不提前裁列（避免“用全数据选特征”的CV泄露）
        print(f"[topk] enabled: mode=fold n={int(getattr(cfg,'TOPK_N',80))} | will select inside folds (train-only)")
        
        # ===== [LEAK GUARD] forbid train-only columns =====
        if "risk_free_rate" in train.X.columns:
            raise RuntimeError("[LEAK] risk_free_rate is in train.X after TopK; it must never be used as a feature.")
        if holdout is not None and ("risk_free_rate" in holdout.X.columns):
            raise RuntimeError("[LEAK] risk_free_rate is in holdout.X after TopK; it must never be used as a feature.")
    
    else:
        topk_cols = list(train.X.columns)
        print(f"[topk] disabled or no ranking file found; using {len(topk_cols)} cols")

    try:
        import torch
        device = getattr(cfg, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = getattr(cfg, "DEVICE", "cpu")
    #LGBM主框架
    target_key = "action_1d" if "action_1d" in train.y else list(train.y.keys())[0]
    X = train.X
    y_all = train.y[target_key]

    target_cols = ['action_1d', 'action_3d', 'action_5d']
    valid_targets = [t for t in target_cols if t in train.y]
    if valid_targets:
        print(f"[pipeline] AE Multi-task targets found: {valid_targets}")
        # [防御性编程] 强制转换为 numpy array 再 stack，防止 Pandas Series 报错
        # 形状变为 (N, 3)
        y_multi_list = [np.array(train.y[t]) for t in valid_targets]
        y_all_multi = np.stack(y_multi_list, axis=1)
    else:
        # Fallback: 如果找不到多目标，就回退到只用主目标
        print(f"[pipeline] Warning: Multi-task targets missing, fallback to single target ({target_key}).")
        # [防御性编程] 强制转 numpy 并 reshape，确保是 (N, 1) 矩阵
        y_all_multi = np.array(y_all).reshape(-1, 1)

    # === Optional: real returns for offline Sharpe & regimes ===
    ret_all = None
    if "resp_1d" in train.y:
        try:
            ret_all = np.asarray(train.y["resp_1d"], dtype=float)
        except Exception:
            ret_all = None

    # Pre-allocated per-sample regime scaling (1.0 = neutral)
    use_regime = bool(getattr(cfg, "DECISION_USE_WEATHER_REGIME", False))
    regime_scale_all = np.ones(len(train.X), dtype=np.float32)

    # —— AE→LR 的 OOF 容器与每折指标 ——
    oof_lr = np.full(len(train.X), np.nan, dtype=float)
    fold_aucs_lr: list[float] = []

    # 纯LGBM的OOF 只在“被当验证的样本”上写入预测；其余保持 NaN
    oof = np.full(len(X), np.nan, dtype=float)
    covered = np.zeros(len(X), dtype=bool)

    # —— AE→LGBM 的 OOF 与模型容器 ——
    oof_lgbm_wd = np.full(len(train.X), np.nan, dtype=float)
    fold_aucs_lgbm_wd: list[float] = []
    models_wd: list = []            # 只保留 last_k 折的模型
    preds_ho_wd: list[np.ndarray] = []  # 与 models_wd 同步保存该折对 holdout 的预测
    
    def _train_weight(tr, va):
        if getattr(cfg, "TIME_DECAY_ENABLED", False):
            hl = getattr(cfg, "TIME_DECAY_HALF_LIFE_DAYS", 504)
            fl = getattr(cfg, "TIME_DECAY_FLOOR", 0.2)
            nz = getattr(cfg, "TIME_DECAY_NORMALIZE_PER_FOLD", True)
            w = _fold_time_decay_weights(train.date, train.w, tr, va, hl, fl, normalize=nz)
            print(f"[weights] fold train={len(tr)}, val={len(va)}, "
                f"w_mean={w.mean():.3f}, w_min={w.min():.3f}, w_max={w.max():.3f}")
            return w
        else:
            return train.w[tr]

    #optuna调参
    best_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 48,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        #防止过拟合
        "min_gain_to_split": 0.0001,
        "lambda_l1": 0.05,           # 加上这个！L1 正则
        "lambda_l2": 0.05,
        "n_estimators": 3000,
        "random_state": getattr(cfg, "SEED", 42),
        "n_jobs": getattr(cfg, "N_THREADS", -1),
        "verbose": -1,
    }
    if getattr(cfg, "OPTUNA_ENABLE", False):
        n_trials = int(getattr(cfg, "N_TRIALS_LGBM", 0))
        if n_trials > 0:
            try:
                import optuna
                def objective(trial):
                    # ==========================================
                    # 1. AE 架构 & 损失权重搜索（保留原有能力）
                    # ==========================================
                    ae_hidden = trial.suggest_categorical("ae_hidden", [128, 256, 512])
                    ae_enc_dim = trial.suggest_categorical("ae_enc_dim", [32, 64, 128])
                    ae_dropout = trial.suggest_float("ae_dropout", 0.1, 0.4)
                    recon_val = trial.suggest_float("ae_recon_weight", 0.01, 0.2)

                    # 动态注入到 cfg，fit_encode 会读取这些值
                    cfg.AE_DYN_HIDDEN = ae_hidden
                    cfg.AE_DYN_ENC_DIM = ae_enc_dim
                    cfg.AE_DYN_DROPOUT = ae_dropout
                    cfg.AE_LOSS_WEIGHT_RECON = recon_val
                    cfg.AE_LOSS_WEIGHT_CLS = 1.0  # 仍然保持分类主导

                    # ==========================================
                    # 2. LGBM 超参搜索（只调 LGBM，不再动决策层）
                    # ==========================================
                    lgbm_params = best_params | {
                        "learning_rate": trial.suggest_float("lr", 0.005, 0.05, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 24, 64, step=8),
                        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
                        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
                        # 这里仍然用 min_split_gain 作为 trial 名称
                        "min_split_gain": trial.suggest_categorical("min_split_gain", [0.0, 1e-4, 1e-3]),
                        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
                        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
                        "n_estimators": 1500,  # 调参时略短一点
                    }

                    # ==========================================
                    # 3. 只用 last_k 折，跑 AE + Wide&Deep LGBM，并以 AUC 为目标
                    # ==========================================
                    all_preds: list[np.ndarray] = []
                    all_targets: list[np.ndarray] = []
                    all_weights: list[np.ndarray] = []

                    for tr, va in last_fds:
                        w_tr, w_va = _train_weight(tr, va), train.w[va]
                        X_tr_raw, X_va_raw = X.iloc[tr], X.iloc[va]

                        # --- AE 训练：剔除天气特征，标准化后喂给 AE ---
                        non_weather_cols = [c for c in X_tr_raw.columns if not c.startswith("weather_")]
                        X_tr_ae = X_tr_raw[non_weather_cols]
                        X_va_ae = X_va_raw[non_weather_cols]

                        fp_ae = FoldPreprocessor(strategy="median", scale=True)
                        X_tr_ae_cl = fp_ae.fit_transform(X_tr_ae)
                        X_va_ae_cl = fp_ae.transform(X_va_ae)

                        # 为了调参稳定，这里用单一种子
                        seed_everything(42)
                        Z_tr, Z_va, _, _ = fit_encode(
                            X_tr=X_tr_ae_cl,
                            y_tr=y_all_multi[tr],  # 多任务标签
                            w_tr=w_tr,
                            X_va=X_va_ae_cl,
                            cfg=cfg,
                            device=getattr(cfg, "DEVICE", "cpu"),
                            epochs=20,  # 调参阶段加速
                        )

                        # --- LGBM：wide (基表) + deep (Z) ---
                        fp_lgbm = FoldPreprocessor(strategy="median", scale=False)
                        X_tr_wide = fp_lgbm.fit_transform(X_tr_raw)
                        X_va_wide = fp_lgbm.transform(X_va_raw)

                        X_tr_wd = np.hstack([X_tr_wide, Z_tr])
                        X_va_wd = np.hstack([X_va_wide, Z_va])

                        clf = lgb.LGBMClassifier(**lgbm_params)
                        clf.fit(
                            X_tr_wd, y_all[tr],
                            sample_weight=w_tr,
                            eval_set=[(X_va_wd, y_all[va])],
                            eval_sample_weight=[w_va],
                            eval_metric="auc",
                            callbacks=[lgb.early_stopping(30, verbose=False)],
                        )

                        pred_va = clf.predict_proba(X_va_wd)[:, 1]
                        all_preds.append(pred_va)
                        all_targets.append(y_all[va])
                        all_weights.append(w_va)

                    if not all_preds:
                        return -1.0

                    preds_concat = np.concatenate(all_preds)
                    targets_concat = np.concatenate(all_targets)
                    weights_concat = np.concatenate(all_weights)

                    # === 核心改变：用 AUC 作为 objective，完全不再掺杂 Sharpe / 决策层 ===
                    auc = roc_auc_score(targets_concat, preds_concat, sample_weight=weights_concat)
                    return float(auc)

                study = optuna.create_study(direction="maximize")


                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                best = study.best_params
                best_params.update({
                    "learning_rate": best.get("lr", best_params["learning_rate"]),
                    "num_leaves": best.get("num_leaves", best_params["num_leaves"]),
                    "min_data_in_leaf": best.get("min_data_in_leaf", best_params["min_data_in_leaf"]),
                    "feature_fraction": best.get("feature_fraction", best_params["feature_fraction"]),
                    "bagging_fraction": best.get("bagging_fraction", best_params["bagging_fraction"]),
                    "bagging_freq": best.get("bagging_freq", best_params["bagging_freq"]),
                    "lambda_l1": best.get("lambda_l1", best_params.get("lambda_l1", 0.0)),
                    "lambda_l2": best.get("lambda_l2", best_params.get("lambda_l2", 0.0)),
                    # 注意这里：如果 best 只有 min_split_gain，就映射回 min_gain_to_split
                    "min_gain_to_split": best.get(
                        "min_gain_to_split",
                        best.get("min_split_gain", best_params.get("min_gain_to_split", 0.0)),
                    ),
                    "n_estimators": 2000,
                })

                # === 新增：把最优 AE 超参也固化回 cfg，后面 fit_encode 真正用到它们 ===
                cfg.AE_DYN_HIDDEN = best.get("ae_hidden", getattr(cfg, "AE_DYN_HIDDEN", 128))
                cfg.AE_DYN_ENC_DIM = best.get("ae_enc_dim", getattr(cfg, "AE_DYN_ENC_DIM", 32))
                cfg.AE_DYN_DROPOUT = best.get("ae_dropout", getattr(cfg, "AE_DYN_DROPOUT", 0.1))
                cfg.AE_LOSS_WEIGHT_RECON = best.get("ae_recon_weight", getattr(cfg, "AE_LOSS_WEIGHT_RECON", 0.1))
                cfg.AE_LOSS_WEIGHT_CLS = getattr(cfg, "AE_LOSS_WEIGHT_CLS", 1.0)

                print(f"[optuna] best_params={best_params}")
            except Exception as e:
                print(f"[optuna] skip due to: {e}")
    #按照全部折训练并产出oof
    fold_aucs, models = [], []
                                                                            #1代表从1开始计数而不是0
    for i, (tr, va) in enumerate(folds, 1):
        w_tr, w_va = _train_weight(tr, va), train.w[va]
        # ===== AE 编码（仅用训练折拟合），并跑 LR =====
        X_tr = train.X.iloc[tr]     # ✅ 保持 DataFrame
        X_va = train.X.iloc[va]     # ✅ 保持 DataFrame
        y_tr = y_all[tr]
        y_va = y_all[va]

        y_tr_multi = y_all_multi[tr]

        # 拿到 AE 的压缩表征（z_tr/z_va），以及 AE 监督头在验证集上的概率（p_va_head，先不用）
        # —— 原始切片（Wide 原生 df）——
        X_tr = train.X.iloc[tr]
        X_va = train.X.iloc[va]


        X_ho = holdout.X if holdout is not None else None

        # === [ADD] Fold-TopK (no-leak): select using only fold-train ===
        if getattr(cfg, "TOPK_ENABLE", False) and topk_mode in ("fold", "per_fold", "online"):
            topk_cols_fold = _resolve_topk_feature_list_fold(cfg, X_tr, y_tr, w_tr)
            if topk_cols_fold is not None:
                scope = str(getattr(cfg, "TOPK_APPLY_SCOPE", "both")).lower()
                if scope in ("ae_only", "both", "lgbm_base"):
                    X_tr = X_tr.reindex(columns=topk_cols_fold)
                    X_va = X_va.reindex(columns=topk_cols_fold)
                    if X_ho is not None:
                        X_ho = X_ho.reindex(columns=topk_cols_fold)
                if i == 1:
                    print(f"[topk] fold#{i}: selected={len(topk_cols_fold)} | scope={scope} | sample={topk_cols_fold[:5]}")

        # === [ADD] Weather per-fold fit/transform (wide 分支可用；AE 侧也可拼) ===
        W_tr = W_va = W_ho = None
        W_tr_extra = W_va_extra = W_ho_extra = None
        weather_enabled = bool(getattr(cfg, "WEATHER_ENABLE", True))
        weather_mode = str(getattr(cfg, "WEATHER_MODE", "fit")).lower()
        if i == 1: print(f"[weather] enabled={weather_enabled} mode={weather_mode}")
        if weather_enabled and (weather_mode == "fit"):
            engine = getattr(cfg, "WEATHER_ENGINE", "hmm")
            k = int(getattr(cfg, "WEATHER_K", 4))
            output = getattr(cfg, "WEATHER_OUTPUT", "post")
            std_floor = float(getattr(cfg, "WEATHER_STD_FLOOR", getattr(cfg, "PREPROCESS_STD_FLOOR", 1e-2)))
            include_regex, exclude_regex = _parse_weather_patterns(cfg)

            #仅用训练折拟合，再在本折验证
            w_model = weather_mod.fit_on_train(
                X_tr,
                engine=engine,
                k=k,
                include_regex=include_regex,
                exclude_regex=exclude_regex,
                output=output,
                std_floor=std_floor,
                random_state=int(getattr(cfg, "WEATHER_RANDOM_STATE", 42)),
            )
            W_tr = weather_mod.transform(w_model, X_tr)
            W_va = weather_mod.transform(w_model, X_va)
            if X_ho is not None:
                W_ho = weather_mod.transform(w_model, X_ho)
            # —— 为 post 形态的天气派生 margin/entropy，并拼回 W_* —— 
            W_tr_extra = derive_weather_confidence(W_tr)
            W_va_extra = derive_weather_confidence(W_va)
            W_ho_extra = derive_weather_confidence(W_ho) if W_ho is not None else None

            if W_tr_extra is not None: W_tr = pd.concat([W_tr, W_tr_extra], axis=1)
            if W_va_extra is not None: W_va = pd.concat([W_va, W_va_extra], axis=1)
            if W_ho_extra is not None: W_ho = pd.concat([W_ho, W_ho_extra], axis=1)

                # === [Optional] Fold-wise weather regime scale for decision layer ===
        # 目的：把“天气状态 -> 缩放系数”的映射，严格用本折训练段(tr)估计，
        #      只写入本折验证段(va)到 regime_scale_all，从而保证无泄露。
        if use_regime and (ret_all is not None) and (W_tr is not None) and (W_va is not None):
            try:
                engine_reg = _infer_engine_from_post_cols(W_tr.columns)

                # posterior columns: weather_{engine}_p0..p(K-1)
                post_cols = [c for c in W_tr.columns if re.match(rf"^weather_{engine_reg}_p\d+$", str(c))]
                # optional discrete state column (if present)
                state_col = None
                for cand in (f"weather_{engine_reg}_state", "weather_state", "weather_regime"):
                    if cand in W_tr.columns:
                        state_col = cand
                        break

                if (not post_cols) and (state_col is None):
                    if i == 1:
                        print("[decision][regime] fold#1: no posterior/state columns -> skipped")
                else:
                    # --- 1) 本折训练/验证的状态序列 ---
                    if post_cols:
                        P_tr = W_tr[post_cols].to_numpy(dtype=np.float32, copy=False)
                        P_va = W_va[post_cols].to_numpy(dtype=np.float32, copy=False)
                        # defensive normalize to avoid numerical drift
                        P_tr = np.clip(P_tr, 1e-12, 1.0); P_tr = P_tr / P_tr.sum(axis=1, keepdims=True)
                        P_va = np.clip(P_va, 1e-12, 1.0); P_va = P_va / P_va.sum(axis=1, keepdims=True)
                        states_tr = np.argmax(P_tr, axis=1).astype(np.int32, copy=False)
                        states_va = np.argmax(P_va, axis=1).astype(np.int32, copy=False)
                        n_states = int(P_tr.shape[1])
                    else:
                        states_tr = W_tr[state_col].to_numpy(dtype=np.int32, copy=False)
                        states_va = W_va[state_col].to_numpy(dtype=np.int32, copy=False)
                        max_tr = int(states_tr.max()) if states_tr.size else 0
                        max_va = int(states_va.max()) if states_va.size else 0
                        n_states = int(max(max_tr, max_va) + 1)

                    if n_states < 2:
                        if i == 1:
                            print(f"[decision][regime] fold#1: n_states={n_states} < 2 -> skipped")
                    else:
                        # --- 2) 仅用训练段真实收益估计每个 state 的表现 ---
                        ret_tr = np.asarray(ret_all[tr], dtype=float)
                        mask_ret = np.isfinite(ret_tr)

                        avg_ret = np.full(n_states, np.nan, dtype=float)
                        counts = np.zeros(n_states, dtype=int)
                        for s_idx in range(n_states):
                            mk = mask_ret & (states_tr == s_idx)
                            counts[s_idx] = int(mk.sum())
                            if mk.any():
                                avg_ret[s_idx] = float(ret_tr[mk].mean())

                        min_samples = int(getattr(cfg, "DECISION_REGIME_MIN_STATE_SAMPLES", 32))
                        valid = np.isfinite(avg_ret) & (counts >= min_samples)
                        n_valid = int(valid.sum())

                        if n_valid < 2:
                            if i == 1:
                                print(f"[decision][regime] fold#1: skip (valid_states={n_valid}, min_samples={min_samples}) | "
                                      f"counts={counts.tolist()} | avg_ret={np.round(np.nan_to_num(avg_ret, nan=0.0), 6).tolist()}")
                        else:
                            # --- 3) 只在有效状态上做分位切分，映射回全状态 ---
                            valid_idx = np.flatnonzero(valid)
                            valid_avg = avg_ret[valid_idx]
                            order = np.argsort(valid_avg)
                            ranks_valid = np.empty_like(order)
                            ranks_valid[order] = np.arange(n_valid)

                            rank_full = np.full(n_states, np.nan, dtype=float)
                            rank_full[valid_idx] = ranks_valid.astype(float)

                            q_low = float(getattr(cfg, "DECISION_REGIME_Q_LOW", 0.3))
                            q_high = float(getattr(cfg, "DECISION_REGIME_Q_HIGH", 0.7))
                            q_low = min(max(q_low, 0.0), 0.5)
                            q_high = max(min(q_high, 1.0), 0.5)

                            low_k = int(np.floor(q_low * (n_valid - 1)))
                            high_k = int(np.ceil(q_high * (n_valid - 1)))

                            bear_scale = float(getattr(cfg, "DECISION_REGIME_SCALE_BEAR", 0.7))
                            neutral_scale = float(getattr(cfg, "DECISION_REGIME_SCALE_NEUTRAL", 1.0))
                            bull_scale = float(getattr(cfg, "DECISION_REGIME_SCALE_BULL", 1.3))

                            state_scale = np.full(n_states, neutral_scale, dtype=float)
                            for s_idx in range(n_states):
                                rk = rank_full[s_idx]
                                if not np.isfinite(rk):
                                    continue
                                if rk <= low_k:
                                    state_scale[s_idx] = bear_scale
                                elif rk >= high_k:
                                    state_scale[s_idx] = bull_scale

                            # --- 4) 只写入本折验证段 va 的缩放 ---
                            scale_va = state_scale[states_va]
                            scale_va = np.where(np.isfinite(scale_va), scale_va, 1.0).astype(np.float32, copy=False)
                            regime_scale_all[va] = scale_va

                            if i == 1:
                                print(
                                    f"[decision][regime] fold#1: n_states={n_states}, valid={n_valid}, "
                                    f"bear_ceil={low_k}, bull_floor={high_k}"
                                )
            except Exception as e:
                if i == 1:
                    print(f"[decision][regime] fold#1: disabled due to error: {e}")
#折内预处理
        fp = FoldPreprocessor(
            strategy=getattr(cfg, "PREPROCESS_IMPUTE_STRATEGY", "median"),
            center=getattr(cfg, "PREPROCESS_CENTER", True),
            scale=getattr(cfg, "PREPROCESS_SCALE", True),
            std_floor=getattr(cfg, "PREPROCESS_STD_FLOOR", 1e-2),
            clip_z=getattr(cfg, "PREPROCESS_CLIP_Z", 6.0),
            drop_nan_col_rate=None,
            )
        X_tr_np = fp.fit_transform(X_tr)
        X_va_np = fp.transform(X_va)
        X_ho_np = fp.transform(X_ho) if X_ho is not None else None
        print(f"[preprocess] fold#{i}: tr={X_tr_np.shape}, va={X_va_np.shape}, ho={None if X_ho_np is None else X_ho_np.shape}")
        # ===== [BUNDLE EXPORT] save fold preprocessor / weather model (optional) =====
        do_export_fold = False
        if bundle_enabled:
            do_export_fold = (bundle_folds == "all") or ((i - 1) in last_fold_idx)
            if do_export_fold:
                try:
                    joblib.dump(fp, bundle_dir / f"fold{i:02d}_fp.pkl")
                    print(f"[bundle] saved fp -> fold{i:02d}_fp.pkl")
                except Exception as _e:
                    print(f"[bundle][WARN] failed to save fp for fold{i}: {_e}")
                try:
                    if weather_enabled and (weather_mode == "fit") and ("w_model" in locals()) and (w_model is not None):
                        joblib.dump(w_model, bundle_dir / f"fold{i:02d}_weather.pkl")
                        print(f"[bundle] saved weather -> fold{i:02d}_weather.pkl")
                except Exception as _e:
                    print(f"[bundle][WARN] failed to save weather for fold{i}: {_e}")


        #AE编码
        # Z_ho = None  # 占位，防未定义；未来扩展 fit_encode 返回 Z_ho 时再启用 holdout 分支
        # Z_tr, Z_va, p_va_head, Z_ho = fit_encode(
        #     X_tr=X_tr_np,
        #     y_tr=y_tr,
        #     w_tr=w_tr,
        #     X_va=X_va_np,
        #     X_ho=(X_ho_np if holdout is not None else None),
        #     cfg=cfg,
        #     device=getattr(cfg, "DEVICE", "cpu"),
        # )

        #加入多种子平均功能
        n_ae_seeds = 3  # 训练 3 个 AE
        Z_tr_sum, Z_va_sum, Z_ho_sum = 0, 0, 0
        
        # 1. 记录当前的主种子 (比如 fold=1 时 seed=43)
        base_fold_seed = getattr(cfg, "SEED", 42) + i
        
        for s_idx in range(n_ae_seeds):
            # 2. 临时切换种子: 43, 44, 45...
            current_sub_seed = base_fold_seed + s_idx
            seed_everything(current_sub_seed)
            
            print(f"[ae] Fold {i} | Seed {s_idx+1}/{n_ae_seeds} (val={current_sub_seed})...")
            export_ae = None
            if bundle_enabled:
                do_export_fold = (bundle_folds == "all") or ((i - 1) in last_fold_idx)
                if do_export_fold:
                    export_ae = {"dir": str(bundle_dir), "fold": int(i), "seed": int(s_idx+1), "tag": "ae"}

            
            # 3. 训练 AE
            z_tr_i, z_va_i, _, z_ho_i = fit_encode(
                X_tr=X_tr_np,
                y_tr=y_tr_multi,
                w_tr=w_tr,
                X_va=X_va_np,
                X_ho=(X_ho_np if holdout is not None else None),
                cfg=cfg,
                device=getattr(cfg, "DEVICE", "cpu"),
                export=export_ae,
            )
            
            # 4. 累加结果
            Z_tr_sum += z_tr_i
            Z_va_sum += z_va_i
            if z_ho_i is not None:
                Z_ho_sum = Z_ho_sum + z_ho_i if isinstance(Z_ho_sum, np.ndarray) else z_ho_i

        # 5. 取平均
        Z_tr = Z_tr_sum / n_ae_seeds
        Z_va = Z_va_sum / n_ae_seeds
        Z_ho = (Z_ho_sum / n_ae_seeds) if (holdout is not None) else None
        
        # 6. [关键] 恢复主种子，确保后面的 LGBM 不受影响
        seed_everything(base_fold_seed)
        print(f"[ae] Done. Seed restored to {base_fold_seed} for LGBM.")



# === [优化版] LR 数据准备 ===
        # 1. 使用 fp 处理好的干净 wide 特征 (X_tr_np) + deep 特征 (Z_tr)
        parts_lr_tr = [X_tr_np]
        parts_lr_va = [X_va_np]
        
        # 2. 如果有天气特征，先对齐索引再合并 (修复 Bug 3)
        if W_tr is not None and W_va is not None:
            # 防御性对齐：确保天气特征的行顺序与 X 完全一致
            W_tr_aligned = W_tr.reindex(X_tr.index).fillna(0.0).values
            W_va_aligned = W_va.reindex(X_va.index).fillna(0.0).values
            
            parts_lr_tr.append(W_tr_aligned)
            parts_lr_va.append(W_va_aligned)
            
        # 3. 加入 AE 特征 (Z_tr 已经是 numpy 且顺序一致)
        parts_lr_tr.append(Z_tr)
        parts_lr_va.append(Z_va)
        
        # 4. 拼接
        Xtr_wd = np.hstack(parts_lr_tr)
        Xva_wd = np.hstack(parts_lr_va)
        
        # 5. 标准化 (LR 对量纲敏感，Z_tr 和 W_tr 需要缩放)
        # 虽然 X_tr_np 已经缩放过，但再次 StandardScaler 不会破坏数据，且能统一 Z/W 的量纲
        sc = StandardScaler()
        Xtr_wd = sc.fit_transform(Xtr_wd)
        Xva_wd = sc.transform(Xva_wd) # 验证集只 transform，不泄露
        
        print(f"[ae->lr] input shape: {Xtr_wd.shape} (base+weather+ae)")





        sc = StandardScaler()
        Xtr_wd = sc.fit_transform(Xtr_wd)
        Xva_wd = sc.transform(Xva_wd)#不训练，所以无泄漏

        # 原始wide部分的NaN计数（可换成 Xtr_wd）
        nan_tr = int(np.isnan(Xtr_wd).sum())
        print(f"[ae->lr] impute: train_nan_filled={nan_tr}, all-NaN-cols->0={0}")

        lr_clf = LogisticRegression(
            solver="lbfgs", penalty="l2", C=0.5,
            max_iter=2000, n_jobs=getattr(cfg, "N_THREADS", -1),
        )

        lr_clf.fit(Xtr_wd, y_tr, sample_weight=w_tr)
        pred_lr = lr_clf.predict_proba(Xva_wd)[:, 1]
        auc_lr = roc_auc_score(y_va, pred_lr, sample_weight=w_va)
        fold_aucs_lr.append(float(auc_lr))

        # 写入 AE→LR 的 OOF（只覆盖当前验证索引）
        oof_lr[va] = pred_lr
        print(f"[ae->lr] fold#{i} AUC={auc_lr:.6f} (wd_dim={Xtr_wd.shape[1]})")
        #AE LR部分结束

        # ===== AE→LGBM（用 [X, Z] 训练一个独立的 LGBM）=====
        # 1) 组装带列名的 DataFrame，避免 feature-name 的警告
        base_cols = list(fp.cols_)

        # Xtr_wd_df = pd.concat(
        #     [fp.transform_df(train.X.iloc[tr]).reset_index(drop=True),
        #     pd.DataFrame(Z_tr, columns=z_cols)], axis=1
        # )
        # Xva_wd_df = pd.concat(
        #     [fp.transform_df(train.X.iloc[va]).reset_index(drop=True),
        #     pd.DataFrame(Z_va, columns=z_cols)], axis=1
        # )
        # === [MODIFY] AE→LGBM 的 wide+deep 输入并入天气列（DF 版） ===
        Xtr_base_df = fp.transform_df(train.X.iloc[tr]).reset_index(drop=True)
        Xva_base_df = fp.transform_df(train.X.iloc[va]).reset_index(drop=True)

        parts_tr = [Xtr_base_df]
        parts_va = [Xva_base_df]

        if W_tr is not None and W_va is not None:
            # 对齐索引 + 先做宽侧列上限，避免天气列过多/污染
            W_tr_df = W_tr.reindex(X_tr.index)
            W_va_df = W_va.reindex(X_va.index)
            max_w = int(getattr(cfg, "WIDE_MAX_WEATHER_COLS", 0))
            if max_w > 0:
                w_cols = list(W_tr_df.columns)
                if len(w_cols) > max_w:
                    keep_w_cols = w_cols[:max_w]
                    W_tr_df = W_tr_df[keep_w_cols]
                    W_va_df = W_va_df[keep_w_cols]
            parts_tr.append(W_tr_df.reset_index(drop=True))
            parts_va.append(W_va_df.reset_index(drop=True))
        
        z_cols = [f"AE_z{i}" for i in range(Z_tr.shape[1])]
        parts_tr.append(pd.DataFrame(Z_tr, columns=z_cols, index=None))
        parts_va.append(pd.DataFrame(Z_va, columns=z_cols, index=None))
        
        # === [ADD] 宽侧护栏：限制 AE_z 与天气列的最大数量 ===
        max_z = int(getattr(cfg, "WIDE_MAX_AE_Z", 32))
        if max_z > 0 and len(z_cols) > max_z:
            z_cols = z_cols[:max_z]
            parts_tr[-1] = parts_tr[-1][z_cols]
            parts_va[-1] = parts_va[-1][z_cols]

        Xtr_wd_df = pd.concat(parts_tr, axis=1)
        Xva_wd_df = pd.concat(parts_va, axis=1)
        # ===== [BUNDLE EXPORT] save WD feature columns (optional) =====
        if bundle_enabled and do_export_fold:
            try:
                (bundle_dir / f"fold{i:02d}_wd_cols.json").write_text(
                    json.dumps(list(Xtr_wd_df.columns), indent=2),
                    encoding="utf-8"
                )
                print(f"[bundle] saved wd cols -> fold{i:02d}_wd_cols.json (n={len(Xtr_wd_df.columns)})")
            except Exception as _e:
                print(f"[bundle][WARN] failed to save wd cols for fold{i}: {_e}")

        
        # ===== [LEAK GUARD] final model matrices (wide+deep) =====
        for _df, _name in [(Xtr_wd_df, "Xtr_wd_df"), (Xva_wd_df, "Xva_wd_df")]:
            if "risk_free_rate" in _df.columns:
                raise RuntimeError(f"[LEAK] risk_free_rate found in {_name} columns!")
        
        # === 强校验：确保 Z 列真的拼进来了 ===
        print(f"[ae->lgbm][dbg] base_cols={len(base_cols)}, z_cols={len(z_cols)}, "
            f"Xtr_wd_df.shape={Xtr_wd_df.shape}, Xva_wd_df.shape={Xva_wd_df.shape}")
        extra = [c for c in Xtr_wd_df.columns if c not in set(base_cols)]
        print(f"[ae->lgbm][dbg] extra_cols(sample)={extra[:5]} ... count={len(extra)}")

        # 2) 实例化 AE→LGBM 分类器（参数口径与基线一致；metric="auc"）
        clf_wd = lgb.LGBMClassifier(
            objective="binary", metric="auc",
            learning_rate=getattr(cfg, "LGBM_LR", 0.02),
            num_leaves=48, min_data_in_leaf=20,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
            n_estimators=getattr(cfg, "LGBM_N_ESTIMATORS", 3000),
            random_state=getattr(cfg, "SEED", 42),
            #正则化部分
            min_gain_to_split=0.0001,
            lambda_l1=0.05,           # AE特征更需要正则，稍微大点
            lambda_l2=0.05,
            n_jobs=getattr(cfg, "N_THREADS", -1), verbose=-1,
        )

        # 3) 早停（与基线口径一致；验证权重用 ε 替代 0，避免度量不动）
        eps = getattr(cfg, "EARLYSTOP_EPS", 1e-6)
        w_va_es = np.where(w_va > 0, w_va, eps)
        callbacks = [lgb.early_stopping(
            stopping_rounds=getattr(cfg, "LGBM_ES_ROUNDS", 200),
            verbose=False
        )]

        clf_wd.fit(
            Xtr_wd_df, y_tr,
            sample_weight=w_tr,
            eval_set=[(Xva_wd_df, y_va)],
            eval_sample_weight=[w_va_es],
            eval_metric="auc",
            callbacks=callbacks,
        )

        
        # ===== [BUNDLE EXPORT] save WD LGBM booster (optional) =====
        if bundle_enabled and do_export_fold:
            try:
                booster = None
                if hasattr(clf_wd, "booster_") and clf_wd.booster_ is not None:
                    booster = clf_wd.booster_
                elif hasattr(clf_wd, "_Booster"):
                    booster = clf_wd._Booster
                if booster is not None:
                    booster.save_model(str(bundle_dir / f"fold{i:02d}_wd_lgbm.txt"))
                    print(f"[bundle] saved wd model -> fold{i:02d}_wd_lgbm.txt")
            except Exception as _e:
                print(f"[bundle][WARN] failed to save wd model for fold{i}: {_e}")

            # update meta export list (best-effort)
            try:
                meta_path = bundle_dir / "bundle_meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                meta.setdefault("export_folds", [])
                if int(i) not in meta["export_folds"]:
                    meta["export_folds"].append(int(i))
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception:
                pass

# 4) 本折验证预测与 AUC
        pred_va_wd = clf_wd.predict_proba(Xva_wd_df)[:, 1]
        auc_va_wd = roc_auc_score(y_va, pred_va_wd, sample_weight=w_va)
        # === 诊断：是否反信号？分数方差是否极小？ ===
        auc_inv_wd = roc_auc_score(y_va, 1.0 - pred_va_wd, sample_weight=w_va)  # 等价于 -score
        std_wd = float(np.std(pred_va_wd))
        print(f"[diag] fold#{i} ae->lgbm: auc={auc_va_wd:.6f} | inv={auc_inv_wd:.6f} | std={std_wd:.3e}")
        
        fold_aucs_lgbm_wd.append(float(auc_va_wd))
        oof_lgbm_wd[va] = pred_va_wd
        print(f"[ae->lgbm] fold#{i} AUC={auc_va_wd:.6f} | best_iter={getattr(clf_wd, 'best_iteration_', None)}")

        # 5) 可选：如果“本折 ∈ last_k”，对 holdout 也做一次 AE→LGBM 预测，后面做均值
        # 说明：此处需要 Z_ho（“本折编码器”对 holdout 的编码）。如果你的 fit_encode 目前
        # 只返回 Z_tr, Z_va，可以先把下面这段注释掉；等你把 fit_encode 扩展为也返回 Z_ho，
        # 再把这段解注即可。
        if (holdout is not None) and ((i - 1) in last_fold_idx) and (Z_ho is not None):
            # === [MODIFY] holdout 的 WD 输入并入天气列 ===
            Xho_base_df = fp.transform_df(holdout.X).reset_index(drop=True)
            parts_ho = [Xho_base_df]
            if W_ho is not None:
                W_ho_df = W_ho
                max_w = int(getattr(cfg, "WIDE_MAX_WEATHER_COLS", 0))
                if max_w > 0:
                    w_cols = list(W_ho_df.columns)
                    if len(w_cols) > max_w:
                        W_ho_df = W_ho_df[w_cols[:max_w]]
                parts_ho.append(W_ho_df.reset_index(drop=True))
            parts_ho.append(pd.DataFrame(Z_ho, columns=z_cols, index=None))
            Xho_wd_df = pd.concat(parts_ho, axis=1)
            preds_ho_wd.append(clf_wd.predict_proba(Xho_wd_df)[:, 1])
            models_wd.append(clf_wd)

        clf_base = lgb.LGBMClassifier(**best_params)
        # 验证权重：只供 early-stopping 用；把 0 换成极小 ε，防“度量不动”
        w_va = train.w[va]
        eps = getattr(cfg, "EARLYSTOP_EPS", 1e-6)
        w_va_es = np.where(w_va > 0, w_va, eps)
        callbacks = [lgb.early_stopping(
            stopping_rounds=getattr(cfg, "EARLY_STOPPING_ROUNDS", 200),
            verbose=False
        )]
        # === [ADD] baseline LGBM can also consume per-fold weather features ===
        Xtr_base_df = fp.transform_df(X_tr).reset_index(drop=True)
        Xva_base_df = fp.transform_df(X_va).reset_index(drop=True)
        if weather_enabled and getattr(cfg, "LGBM_USE_WEATHER", True) and (W_tr is not None) and (W_va is not None):
            W_tr_df = W_tr.reset_index(drop=True)
            W_va_df = W_va.reset_index(drop=True)
            max_w = int(getattr(cfg, "WIDE_MAX_WEATHER_COLS", 0))
            if max_w > 0:
                w_cols = list(W_tr_df.columns)
                if len(w_cols) > max_w:
                    W_tr_df = W_tr_df[w_cols[:max_w]]
                    W_va_df = W_va_df[w_cols[:max_w]]
            Xtr_base_df = pd.concat([Xtr_base_df, W_tr_df], axis=1)
            Xva_base_df = pd.concat([Xva_base_df, W_va_df], axis=1)
            if i == 1: print(f"[lgbm][dbg] use_weather=True | dim={Xtr_base_df.shape[1]}")
        else:
            if i == 1: print(f"[lgbm][dbg] use_weather=False | dim={Xtr_base_df.shape[1]}")
        # === leak-guard (baseline) ===
        if "risk_free_rate" in Xtr_base_df.columns:
            raise RuntimeError(f"[LEAK] risk_free_rate found in baseline LGBM columns!")
        clf_base.fit(
            Xtr_base_df, y_tr,
            sample_weight=w_tr,
            eval_set=[(Xva_base_df, y_va)],
            eval_sample_weight=[w_va_es],
            eval_metric="auc",
            callbacks=callbacks,
        )
        pred_base = clf_base.predict_proba(Xva_base_df)[:, 1]
        oof[va] = pred_base
        covered[va] = True
        auc_i = roc_auc_score(y_va, pred_base, sample_weight=w_va)
        fold_aucs.append(float(auc_i)); models.append(clf_base)
        print(f"[lgbm] fold#{i} AUC={auc_i:.6f} | best_iter={clf_base.best_iteration_}")
        pos_rate = y_va.mean()
        pos_w_share = float(train.w[va][y_va == 1].sum() / (train.w[va].sum() + 1e-12))
        print(f"[diag] fold#{i} pos_rate={pos_rate:.3f}, pos_w_share={pos_w_share:.3f}")
        zero_ratio = float((w_va <= 0).mean())
        print(f"[diag] fold#{i} val_w_zero_ratio={zero_ratio:.3f}  (ε={eps:g})")
    # 只在“有验证预测”的样本上评 OOF（避免 warmup/purge/embargo 样本稀释）
    mask_oof = covered  # 或 (~np.isnan(oof))
    cov_ratio = float(mask_oof.mean())
    auc_oof = roc_auc_score(y_all[mask_oof], oof[mask_oof], sample_weight=train.w[mask_oof])
    print(f"[lgbm] OOF AUC ({target_key}) = {auc_oof:.6f} | coverage={cov_ratio:.3f}; per-fold={fold_aucs}")
    mask_lr = ~np.isnan(oof_lr)
    if mask_lr.any():
        auc_oof_lr = roc_auc_score(y_all[mask_lr], oof_lr[mask_lr], sample_weight=train.w[mask_lr])
        print(f"[ae->lr] OOF AUC ({target_key}) = {auc_oof_lr:.6f} | coverage={float(mask_lr.mean()):.3f}; per-fold={fold_aucs_lr}")
    else:
        auc_oof_lr = float("nan")
        print(f"[ae->lr] OOF AUC ({target_key}) = nan | coverage=0.000")
    # —— AE→LGBM 的 OOF 汇总 ——
    mask_wd = ~np.isnan(oof_lgbm_wd)
    if mask_wd.any():
        auc_oof_wd = roc_auc_score(y_all[mask_wd], oof_lgbm_wd[mask_wd], sample_weight=train.w[mask_wd])
        print(f"[ae->lgbm] OOF AUC ({target_key}) = {auc_oof_wd:.6f} | coverage={float(mask_wd.mean()):.3f}; per-fold={fold_aucs_lgbm_wd}")
    else:
        auc_oof_wd = float("nan")
        print(f"[ae->lgbm] OOF AUC ({target_key}) = nan | coverage=0.000")
    
    # ----- Rank-Average 融合 AUC（只看排序，稳健抬一截）-----
    def _percentile_rank(x: np.ndarray) -> np.ndarray:
        r = np.argsort(np.argsort(x))
        return r.astype(np.float32) / max(len(x) - 1, 1)

    cands, names = [], []
    if 'oof' in locals() and oof is not None:                # 基线 LGBM
        cands.append(oof); names.append('lgbm')
    if 'oof_lgbm_wd' in locals() and oof_lgbm_wd is not None:  # AE→LGBM
        cands.append(oof_lgbm_wd); names.append('ae->lgbm')
    if 'oof_lr' in locals() and oof_lr is not None:          # AE→LR
        cands.append(oof_lr); names.append('ae->lr')

    # 新增：保存一份“全长”的融合 OOF，便于后续决策层复用
    oof_blend = np.full(len(X), np.nan, dtype=float)
    auc_blend = None

    if len(cands) >= 2:
        blend_mask = ~np.isnan(cands[0])
        for a in cands[1:]:
            blend_mask &= ~np.isnan(a)
        if blend_mask.any():
            blend_vec = np.mean([_percentile_rank(a[blend_mask]) for a in cands], axis=0)
            oof_blend[blend_mask] = blend_vec
            auc_blend = roc_auc_score(y_all[blend_mask], blend_vec, sample_weight=train.w[blend_mask])
            print(
                f"[blend-rank] OOF AUC ({target_key}) = {auc_blend:.6f} "
                f"| used={'+'.join(names)} | coverage={float(blend_mask.mean()):.3f}"
            )

    # ===== Offline decision-layer Sharpe proxy (sanity check only) =====
    sharpe_oof_proxy: Optional[float] = None
    oof_score_official: Optional[float] = None
    decision_meta: Optional[Dict[str, Any]] = None
    decision_pred_head: Optional[str] = None

    if (ret_all is not None) and ("resp_1d" in train.y):
        # 1) 选择用于决策层的预测头（优先使用 blend）
        if ("oof_blend" in locals()) and (auc_blend is not None) and (not np.all(np.isnan(oof_blend))):
            pred_for_decision = oof_blend
            decision_pred_head = "blend"
        elif ("oof_lgbm_wd" in locals()) and (oof_lgbm_wd is not None) and (not np.all(np.isnan(oof_lgbm_wd))):
            pred_for_decision = oof_lgbm_wd
            decision_pred_head = "ae->lgbm"
        elif ("oof_lr" in locals()) and (oof_lr is not None) and (not np.all(np.isnan(oof_lr))):
            pred_for_decision = oof_lr
            decision_pred_head = "ae->lr"
        elif ("oof" in locals()) and (oof is not None):
            pred_for_decision = oof
            decision_pred_head = "lgbm"
        else:
            pred_for_decision = None

        if pred_for_decision is not None:
            p_dec = _get_decision_params(cfg)
            vol_all = compute_vol_proxy(ret_all, window=p_dec["vol_window"])

            # 只在有 OOF 的位置评估决策层
            mask_dec = ~np.isnan(pred_for_decision)
            if mask_dec.any():
                alloc, decision_meta = decision_allocation_from_pred(
                    pred_for_decision[mask_dec],
                    vol_all[mask_dec],
                    cfg,
                )

                # Regime overlay（已经折内拟合，避免泄露）
                regime_enabled = False
                use_regime = bool(getattr(cfg, "DECISION_USE_WEATHER_REGIME", False))
                if use_regime and ("regime_scale_all" in locals()):
                    try:
                        regime_scale = np.asarray(regime_scale_all[mask_dec], dtype=float)
                        if np.isfinite(regime_scale).any():
                            alloc = alloc * np.where(np.isfinite(regime_scale), regime_scale, 1.0)
                            regime_enabled = True
                    except Exception as e:
                        print(f"[decision][regime] offline overlay disabled due to error: {e}")

                coverage = float(np.mean(alloc > 0.01)) if alloc.size else 0.0
                if decision_meta is None:
                    decision_meta = {}
                decision_meta["coverage"] = coverage
                decision_meta["regime_enabled"] = regime_enabled

                sharpe_oof_proxy = sharpe_proxy(alloc, ret_all[mask_dec])

                # 简单保护：coverage 过低时减去 1.0，提示过于保守
                if coverage < p_dec["coverage_floor"]:
                    sharpe_oof_proxy = float(sharpe_oof_proxy) - 1.0

                # ===== Official scoring (Kaggle-like) on OOF portion =====
                if getattr(train, "sol", None) is not None:
                    try:
                        sol_df = train.sol
                        idx_dec = np.flatnonzero(mask_dec)
                        sol_dec = sol_df.iloc[idx_dec][["forward_returns", "risk_free_rate"]].copy()
                        oof_score_official = score_official(
                            sol_dec,
                            alloc,
                            min_invest=float(getattr(cfg, "MIN_INVESTMENT", 0.0)),
                            max_invest=float(getattr(cfg, "MAX_INVESTMENT", 2.0)),
                        )
                        if decision_meta is None:
                            decision_meta = {}
                        decision_meta["oof_score_official"] = float(oof_score_official)
                        print(
                            f"[score] OOF official score ({decision_pred_head}) = "
                            f"{oof_score_official:.6f} | n={len(sol_dec)}"
                        )
                    except Exception as e:
                        print(f"[score] skipped official score due to error: {e}")
                else:
                    print("[score] skipped: train.sol not available (need forward_returns/risk_free_rate).")

                print(
                    f"[decision] offline Sharpe proxy ({decision_pred_head}) = "
                    f"{sharpe_oof_proxy:.6f} | coverage={coverage:.3f} | regime={regime_enabled}"
                )

        # —— AE→LGBM 的 holdout 均值（若前面在每个 last_k 折存了 preds_ho_wd）——
    pred_ho_wd = None
    if (holdout is not None) and (len(preds_ho_wd) > 0):
        pred_ho_wd = np.stack(preds_ho_wd, axis=1).mean(axis=1)

# ===== [FINAL] last-k models for inference =====
    last_models = models[-use_k:] if (use_k and use_k > 0) else models
    return {
    "status": "ok",
    "primary": primary_head,
    "heads": list(eval_heads),
    "target": target_key,
    "auc_oof": float(auc_oof),
    "fold_aucs": fold_aucs,
    "best_params": best_params,
    "n_folds": len(folds),
    "n_last_models": len(last_models),
    # ↓↓↓ 新增：用于落盘和 train 推理
    "oof_pred": oof,         # np.ndarray，长度 = len(train.X)
    "oof_mask": mask_oof,        # np.ndarray[bool]，True 表示该样本被当作验证、已有 OOF 分数
    "models": last_models,   # 最后 k 折的模型列表（用于 holdout 推理）
    "oof_pred_lr": oof_lr,
    "auc_oof_lr": float(auc_oof_lr),
    "fold_aucs_lr": fold_aucs_lr,
    "oof_pred_lgbm_wd": oof_lgbm_wd,
    "auc_oof_lgbm_wd": float(auc_oof_wd),
    "fold_aucs_lgbm_wd": fold_aucs_lgbm_wd,

    "auc_oof_blend": float(auc_blend) if auc_blend is not None else None,
    "oof_pred_blend": oof_blend,
    "sharpe_oof_proxy": float(sharpe_oof_proxy) if sharpe_oof_proxy is not None else None,
    "oof_score_official": float(oof_score_official) if oof_score_official is not None else None,
    "decision_meta": decision_meta,
    "decision_pred_head": decision_pred_head,
    "holdout_pred_lgbm_wd": pred_ho_wd,
    "feature_cols": list(topk_cols),
}








