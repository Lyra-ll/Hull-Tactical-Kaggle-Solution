"""
Tiny registry/runner to glue main entry.

中文说明：
- 轻量的运行器：设置随机种子、解析命令、转调 main。
- 不持有业务逻辑，仅封装脚手架。

Core:
- run(argv=None)

TODO:
- 支持 YAML/JSON 配置文件一键导入并覆盖 config（优先级：CLI > file > defaults）
- 统一 logging 格式（时间/级别/进程）
"""
from __future__ import annotations
import os, json
from typing import Any
from typing import Any, Dict, Optional

from data import load_train, load_holdout, DataBundle


def prepare_bundles(cfg:Any, mode: str = "train") -> Dict[str, Optional[DataBundle]]:
    """
    统一的数据装配入口
    加载训练数据集，如果存在holdout data而且模式不是cv only则一起加载holdout data
    返回:{"train": DataBundle, "holdout": Optional[DataBundle]}
    """
    bd_train: DataBundle = load_train(cfg)

    bd_holdout: Optional[DataBundle] = None
    holdout_path = getattr(cfg, "HOLDOUT_DATA_FILE", None)

    if holdout_path and os.path.exists(holdout_path) and mode in ("train", "infer"):
        bd_holdout = load_holdout(cfg)

    try:
        print(f"[registry] train: X={bd_train.X.shape}, y_keys={list(bd_train.y.keys())}, "
              f"date[{bd_train.date.min()}..{bd_train.date.max()}], w={bd_train.w.shape}")
        if bd_holdout is not None:
            print(f"[registry] holdout: X={bd_holdout.X.shape}, "
                  f"date[{bd_holdout.date.min()}..{bd_holdout.date.max()}]")
    except Exception as e:
        print(f"[registry] peek failed: {e}")

    return {"train": bd_train, "holdout": bd_holdout}



def run(mode: str, artifacts_dir: str, cfg: Any) -> None:
    """Minimal orchestrator stub.
    Later it will call: data.load_train -> cv.splits -> features -> models -> eval.
    For now we just snapshot env and data peek so main.py can import us cleanly.
    """
    print(f"[registry] start mode={mode} | artifacts={artifacts_dir}")

    # 1) set global seed if available (optional, non-fatal)
    try:
        import utils  # your existing utils.py at project root
        if hasattr(utils, "set_global_seeds"):
            seed = getattr(cfg, "GLOBAL_SEED", 42)
            utils.set_global_seeds(seed)
            print(f"[registry] seeds set to {seed}")
    except Exception as e:
        print(f"[registry] WARN: seed init skipped: {e}")

    # 2) lightweight data peek (does not load full frame to avoid memory surprises)
    train_path = getattr(cfg, "RAW_DATA_FILE", "train_final_features.csv")
    hold_out_path = getattr(cfg, "HOLDOUT_DATA_FILE", None)

    info = {"path": train_path, "exists": os.path.exists(train_path)}
    if info["exists"]:
        try:
            import pandas as pd
            head = pd.read_csv(train_path, nrows=5)
            info.update({
                "n_cols": head.shape[1],
                "columns": list(map(str, head.columns[:50])),  # cap preview
            })
            print(f"[registry] data peek: cols={info['n_cols']}, sample_rows=5")
        except Exception as e:
            info["read_error"] = str(e)
            print(f"[registry] WARN: cannot peek data: {e}")
    else:
        print(f"[registry] WARN: file not found -> {train_path}")

    if hold_out_path and os.path.exists(hold_out_path):
        info["holdout_file"] = hold_out_path

    with open(os.path.join(artifacts_dir, "registry_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 3) mode routing (stubs)
    if mode in ("train", "validate"):
        print("[registry] pipeline stub: data -> cv -> features -> model -> eval (pending)")
    elif mode == "infer":
        print("[registry] infer stub: to be implemented")

    print("[registry] done")
