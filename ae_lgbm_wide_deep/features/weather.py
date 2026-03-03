



# features/weather.py
from __future__ import annotations
from ast import pattern
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

@dataclass
class WeatherModel:
    engine: str                 # "hmm" / "kmeans" / "gmm"
    k: int
    scaler: StandardScaler      # fit on train (weather base cols only)
    model: object               # hmmlearn model / KMeans / GMM
    base_cols: List[str]        # columns used to fit/transform
    output: str                 # "post"|"onehot"|"state"
    feat_names: List[str]       # output column names (ordered)
    fill_values: np.ndarray     # [ADD] per-column median (learned on train fold) for NaN/Inf fill

def _stdz_fit_transform(X_tr: pd.DataFrame, std_floor:float) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X_tr.values.astype(np.float32, copy=False))
    #std floor
    scale = scaler.scale_.copy()#scale_是标准差
    scale[scale < std_floor] = 1.0
    scaler.scale_ = scale
    return Z, scaler

def _stdz_transform(X:pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X.values.astype(np.float32, copy=False))

def _pick_base_cols(all_cols:List[str], include_regex:List[str], exclude_regex:List[str]) -> List[str]:
    """
    选出不在黑名单上的列，拥有白名单功能。返回list
    """
    def any_match(name, patterns):
        return any(re.search(p, name) for p in patterns) if patterns else False
    cols = []
    for c in all_cols:
        if exclude_regex and any_match(c, exclude_regex):
            continue
        if include_regex:
            if any_match(c, include_regex):
                cols.append(c)
        else:
            cols.append(c)

    return cols
def _post_names(engine:str, k:int) -> List[str]:
    prefix = "hmm" if engine == "hmm" else ("gmm" if engine == "gmm" else "km")
    return [f"weather_{prefix}_p{i}" for i in range(k)]

def _ohe_names(engine: str, k: int) -> List[str]:
    prefix = "hmm" if engine == "hmm" else ("gmm" if engine == "gmm" else "km")
    return [f"weather_{prefix}_s{i}" for i in range(k)]

def _state_name(engine: str) -> str:
    prefix = "hmm" if engine == "hmm" else ("gmm" if engine == "gmm" else "km")
    return f"weather_{prefix}_state"

def fit_on_train(
    X_tr:pd.DataFrame,
    *,
    engine:str,
    k:int,
    include_regex:List[str],
    exclude_regex:List[str],
    output:str,
    std_floor:float = 1e-2,
    random_state = 42,

) -> WeatherModel:
    engine = engine.lower()
    assert engine in {"hmm", "kmeans", "gmm"}, f"Unsupported engine: {engine}"
    assert output in {"post", "onehot", "state"}

    base_cols = _pick_base_cols(list(X_tr.columns), include_regex, exclude_regex)
    if len(base_cols) == 0:
        raise ValueError("[weather] No base columns selected. Check WEATHER_SPEC_REGEX/EXCLUDE_REGEX.")
    Xb = X_tr[base_cols]

    #Z_tr, scaler = _stdz_fit_transform(Xb, std_floor)
    # [ADD] —— 训练折：先清理再填充（中位数），随后再标准化
    V = Xb.values.astype(np.float32, copy=False)
    V[~np.isfinite(V)] = np.nan
    #训练折按列中位数填充
    fill_values = np.nanmedian(V, axis=0)
    V_filled = np.where(np.isnan(V), fill_values, V)

    #标准化
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_tr = scaler.fit_transform(V_filled)

    #std floor
    scale = scaler.scale_.copy()
    scale[scale < std_floor] = 1.0
    scaler.scale_ = scale

    if not np.isfinite(Z_tr).all():
        raise ValueError("[weather] Z_tr still contains NaN/Inf after fill+scale; check include/exclude regex & raw columns.")

    if engine == "kmeans":
        try:
            model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        except TypeError:
            # older sklearn may not accept n_init="auto"
            model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        model.fit(Z_tr)
        if output == "post":
            # approximate posterior by distance-based softmax
            d = model.transform(Z_tr)                   # (n,k) distances
            d = -d
            d = d - d.max(axis=1, keepdims=True)
            post = np.exp(d); post /= post.sum(axis=1, keepdims=True)
            feat_names = _post_names("kmeans", k)
        elif output == "onehot":
            lab = model.predict(Z_tr)
            post = np.eye(k, dtype=np.float32)[lab]
            feat_names = _ohe_names("kmeans", k)
        else:
            lab = model.predict(Z_tr).astype(np.int32)
            post = lab.reshape(-1, 1)
            feat_names = [_state_name("kmeans")]
    elif engine == "gmm":
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)


        gmm.fit(Z_tr)
        if output == "post":
            post = gmm.predict_proba(Z_tr).astype(np.float32)
            feat_names = _post_names("gmm", k)
        elif output == "onehot":
            lab = gmm.predict(Z_tr)
            post = np.eye(k, dtype=np.float32)[lab]
            feat_names = _ohe_names("gmm", k)
        else:
            lab = gmm.predict(Z_tr).astype(np.int32)
            post = lab.reshape(-1, 1)
            feat_names = [_state_name("gmm")]
        model = gmm
    else:#hmm
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as e:
            raise ImportError(
                "[weather] engine='hmm' requires hmmlearn. pip install hmmlearn "
                "or switch WEATHER_ENGINE to 'gmm' (often a good substitute)."
            ) from e
        hmm = GaussianHMM(n_components=k, covariance_type="full", random_state=random_state, n_iter=200)
        hmm.fit(Z_tr)
        if output == "post":
            post = hmm.predict_proba(Z_tr).astype(np.float32)
            feat_names = _post_names("hmm", k)
        elif output == "onehot":
            lab = hmm.predict(Z_tr)
            post = np.eye(k, dtype=np.float32)[lab]
            feat_names = _ohe_names("hmm", k)
        else:
            lab = hmm.predict(Z_tr).astype(np.int32)
            post = lab.reshape(-1, 1)
            feat_names = [_state_name("hmm")]
        model = hmm

    return WeatherModel(
        engine=engine, k=k, scaler=scaler, model=model,
        base_cols=base_cols, output=output, feat_names=feat_names,
        fill_values=fill_values
    )

def transform(model:WeatherModel, X:pd.DataFrame) -> pd.DataFrame:
    # Xb = X.reindex(columns=model.base_cols)
    # Z = _stdz_transform(Xb, model.scaler)
    Xb = X.reindex(columns=model.base_cols, fill_value=np.nan)
    V = Xb.values.astype(np.float32, copy=False)
    V[~np.isfinite(V)] = np.nan
    V_filled = np.where(np.isnan(V), model.fill_values, V)

    Z = model.scaler.transform(V_filled)

    if model.engine == "kmeans":
        d = model.model.transform(Z)
        if model.output == "post":
            d = -d
            d = d - d.max(axis=1, keepdims=True)
            post = np.exp(d); post /= post.sum(axis=1, keepdims=True)
            W = post
        elif model.output == "onehot":
            lab = model.model.predict(Z)
            W = np.eye(model.k, dtype=np.float32)[lab]
        else:
            lab = model.model.predict(Z).astype(np.int32)
            W = lab.reshape(-1, 1)
    elif model.engine == "gmm":
        if model.output == "post":
            W = model.model.predict_proba(Z).astype(np.float32)
        elif model.output == "onehot":
            lab = model.model.predict(Z)
            W = np.eye(model.k, dtype=np.float32)[lab]
        else:
            lab = model.model.predict(Z).astype(np.int32)
            W = lab.reshape(-1, 1)
    else:#hmm
        if model.output == "post":
            W = model.model.predict_proba(Z).astype(np.float32)
        elif model.output == 'onehot':
            lab = model.model.predict(Z)
            W = np.eye(model.k, dtype=np.float32)[lab]
        else:
            lab = model.model.predict(Z).astype(np.int32)
            W = lab.reshape(-1, 1)

    W = np.asarray(W)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    dfw = pd.DataFrame(W, index=X.index, columns=model.feat_names)
    return dfw

        
