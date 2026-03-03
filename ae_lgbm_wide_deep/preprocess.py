"""
Per-fold preprocessing (fit on train, transform on val/holdout).

中文说明：
- 折内预处理器：仅用训练折学习统计量（中心化/标准化等），再对验证与 holdout 进行 transform。
- 默认不在折内按缺失率再次删列（列集已由全局裁定）。

Core:
- FoldPreprocessor(center=True, scale=True, drop_nan_col_rate=None)
  .fit(X_tr, X_va, X_ho=None) -> (Xt, Xv, Xh)
  保证只从训练折学习参数，transform 保持列顺序与维度一致。

TODO:
- RobustScaler / Winsorize 等可插拔策略。
- 数值稳定性告警（如全零方差列、极端值计数报告）。
"""

# preprocess.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def _ensure_numeric_ordered(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    只保留数值列并严格按指定 cols 顺序对齐；缺失列补出来（全 NaN）
    """
    df_num = df.reindex(columns=cols)
    for c in cols:
        if c not in df_num.columns:
            df_num[c] = np.nan
    # 按 cols 顺序确保一致
    return df_num[cols]

@dataclass
class FoldPreprocessor:
    """
    折内预处理器（仅用 train 折统计）：
      - 丢弃缺失率 > drop_nan_col_rate 的列（在训练折上统计）
      - 缺失值填充（median/mean）
      - 标准化（center/scale），std 下限 std_floor
      - 可选按 z-score 裁剪（clip_z）
      - 保证 transform 后没有 NaN，且列顺序与训练一致
      
    用法：
      fp = FoldPreprocessor(...)
      X_tr_np = fp.fit_transform(train_df)         # 只在当前折的训练子集上 fit
      X_va_np = fp.transform(val_df)               # 验证/holdout 仅 transform
    """
    strategy: str = "median"  #median or mean
    center: bool = True
    scale: bool = True
    std_floor: float = 1e-2
    clip_z: Optional[float] = 6.0
    drop_nan_col_rate: float = 0.30
    eps: float = 1e-8

    #拟合后属性
    cols_: Optional[List[str]] = None
    dropped_: Optional[List[str]] = None
    center_vec_: Optional[np.ndarray] = None
    mean_: Optional[np.ndarray] = None
    median_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    low_: Optional[np.ndarray] = None
    high_: Optional[np.ndarray] = None
    fitted_: bool = False

    def _stats_on(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = X.to_numpy(dtype=np.float32)
        all_nan = np.isnan(arr).all(axis=0)          # 整列皆空的列
        median = np.nanmedian(arr, axis=0)
        mean   = np.nanmean(arr, axis=0)
        std    = np.nanstd(arr, axis=0)

        # 对于整列 NaN：强制给出稳定的中心/尺度，避免后续 NaN 传播与警告
        if all_nan.any():
            median[all_nan] = 0.0
            mean[all_nan]   = 0.0
            std[all_nan]    = 1.0

        std = np.maximum(std, self.std_floor)
        return mean.astype(np.float32), median.astype(np.float32), std.astype(np.float32)

    def fit(self, X_train:pd.DataFrame) -> "FoldPreprocessor":
        #1 选出全数值列
        numeric_cols = [c for c in X_train.columns]
        X_num = _ensure_numeric_ordered(X_train, numeric_cols)

        # 2 在训练集上统计缺失率并丢弃（允许关闭）
        arr = X_num.to_numpy(dtype=np.float32)
        if (self.drop_nan_col_rate is None) or (self.drop_nan_col_rate >= 1.0):
            keep_cols = list(X_num.columns)
            drop_cols = []
        else:
            nan_rate = np.isnan(arr).mean(axis=0)
            keep_mask = nan_rate <= self.drop_nan_col_rate
            assert len(numeric_cols) == len(keep_mask), "列和掩码的长度不匹配！"
            keep_cols = [c for c, k in zip(numeric_cols, keep_mask) if k]
            drop_cols = [c for c, k in zip(numeric_cols, keep_mask) if not k]
            if len(keep_cols) == 0:
                raise ValueError("[preprocess] 所有列都被缺失率阈值过滤掉了，请降低 drop_nan_col_rate")
        
        #3 在保留列上面计算统计量
        X_keep = X_num[keep_cols]
        mean, median, std = self._stats_on(X_keep)
        center_vec = median if self.strategy == "median" else mean
        
        #4 预计算裁剪上下界
        if self.clip_z is not None:
            low = center_vec - self.clip_z * std
            high = center_vec + self.clip_z * std
        else:
            low = None
            high = None

        #赋值
        self.cols_ = keep_cols
        self.dropped_ = drop_cols
        self.mean_ = mean
        self.median_ = median
        self.std_ = std
        self.center_vec_ = center_vec
        self.low_ = low
        self.high_ = high
        self.fitted_ = True

        nan_thr_str = f">{self.drop_nan_col_rate:.0%} NaN" if self.drop_nan_col_rate is not None else "off"
        print(
            f"[preprocess] fit: kept={len(keep_cols)}, dropped={len(drop_cols)} "
            f"({nan_thr_str}) | train_nan_ratio_on_kept={float(np.isnan(X_keep.to_numpy()).mean()):.3f}"
        )
        if drop_cols:
            sample = ", ".join(drop_cols[:8])
            print(f"[preprocess] dropped sample: [{sample}]{'...' if len(drop_cols) > 8 else ''}")
        return self

    def _impute_center_scale(self, X:pd.DataFrame) -> np.ndarray:
        """
        数据预处理，使用中位数填充缺失，裁剪，标准化，后返回ndarray
        """
        assert self.fitted_, "Foldpreprocessor not fitted"
        X = _ensure_numeric_ordered(X, self.cols_)
        arr = X.to_numpy(dtype=np.float32)
        arr[~np.isfinite(arr)] = np.nan    # 把 ±Inf 统一按 NaN 处理
        #缺失填充，使用中心值
        mask_nan = np.isnan(arr)
        if mask_nan.any():
            arr[mask_nan] = np.take(self.center_vec_, np.where(mask_nan)[1])

        #可选裁剪
        if (self.low_ is not None) and (self.high_ is not None):
            arr = np.minimum(np.maximum(arr, self.low_), self.high_)

        #标准化
        if self.center:
            arr = arr - self.center_vec_
        if self.scale:
            arr = arr / (self.std_ + self.eps)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr
    
    def transform(self, X:pd.DataFrame) -> np.ndarray:
        return self._impute_center_scale(X)

    def fit_transform(self, X_train:pd.DataFrame) -> np.ndarray:
        return self.fit(X_train)._impute_center_scale(X_train)

    def transform_df(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理后返回带列名的dataframe，用于AE + LGBM的wide 侧拼接
        """
        arr = self.transform(X)
        return pd.DataFrame(arr, columns=self.cols_)