"""
- 实现时间序列前滚（walk-forward）的切分器：
  训练窗口在前、验证窗口在后；val 左侧加 purge，右侧加 embargo。
- 支持 warmup_blocks：前若干块仅参与“学参数”，不计入验证。

Core:
- time_series_folds(n_folds, embargo, purge, warmup_blocks, ...) -> yield (tr_idx, va_idx)

"""


from typing import List, Tuple
import numpy as np
import pandas as pd

def time_series_folds(
    dates: pd.Series,
    n_splits: int = 5,
    embargo: int = 0,
    purge: int = 0,
    warmup_blocks: int = 1,

) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    基于日期进行时间序列切分
    val 使用按日期均分后的连续块
    train 去掉【val_start-embargo, val_end + embargo】区间
    并且在val_start前做purge
    从第warm up block开始验证，之前的仅学习
    dates: pd.Series,
    n_splits: int = 5,
    embargo: int = 0,
    purge: int = 0
    """
    uniq_dates = np.unique(dates.values)
    date_blocks = np.array_split(uniq_dates, n_splits)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    idx_all = np.arange(len(dates))

    for k in range(warmup_blocks, len(date_blocks)):
        block = date_blocks[k]
        #验证集
        val_mask = dates.isin(block).values
        val_idx = np.where(val_mask)[0]
        if val_idx.size == 0:
            continue
        start, end = val_idx.min(), val_idx.max()

        #训练集
        train_mask = idx_all < start
        if purge > 0:
            train_mask &= (idx_all < max(0, start - purge))

        if embargo > 0:
            left = max(0, start - embargo)
            right = min(len(dates), end + embargo + 1)
            emb_zone = (idx_all >= left) & (idx_all < right)
            train_mask &= ~emb_zone

        train_idx = idx_all[train_mask]
        folds.append((train_idx, val_idx))

    return folds

def last_k_folds(folds: List[Tuple[np.ndarray, np.ndarray]], k: int = 3):
    if k <= 0 or k >= len(folds):
        return folds
    return folds[-k:]


