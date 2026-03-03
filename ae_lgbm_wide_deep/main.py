"""
CLI entrypoint: validate/train modes and artifact writing.

- validate：只产生 OOF 与折内指标；不写 holdout。
- train：同样走 CV，但额外带回“最后 k 折”模型，对 holdout 推理并落盘。
- 统一写入 config 快照 / fold_metrics / 预测 CSV。

Core:
- parse args -> prepare_bundles(data) -> pipeline.fit_evaluate(...) -> write artifacts
- 严格用 result["feature_cols"] 对齐 holdout 列顺序与集合。

TODO:
- run_dir 结构稳定化（/oof, /holdout, /models, /logs）
- 融合策略：baseline vs AE->LGBM 的加权或 stacking（后续）
"""
#!/usr/bin/env python3
from __future__ import annotations
from inspect import getargs
import argparse, json, os, sys, datetime as dt
from types import ModuleType
import importlib
import argparse
import registry
import pipeline
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

def _load_cfg():
    return importlib.import_module("config")



def _collect_config(cfg:ModuleType) -> dict:
    """
    collect upper case attributes from config.py for snapshotting
    """
    out:dict = {}
    for k in dir(cfg):
        if k.isupper():
            v = getattr(cfg,k)
            try:
                json.dumps(v, default=str)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    return out

def make_artifacts_dir(root: str = "artifacts", tag :str | None = None) -> str:
    """
    生成保存的参数产物目录
    """
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"-{tag}" if tag else ""
    path = os.path.join(root, f"run-{ts}{tag}")
    os.makedirs(path, exist_ok=True)
    return path

def save_config_snapshot(cfg_module: ModuleType, out_dir: str) -> None:
    snap = _collect_config(cfg_module)
    snap["__timestamp__"] = dt.datetime.now().isoformat(timespec="seconds")
    #记录时间运行的时间精确到秒
    # optional: attach git short hash if available
    try:
        import subprocess
        rev = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        snap["__git__"] = rev
    except Exception:
        pass
    with open(os.path.join(out_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2, default=str)

def main() -> None:
    ap = argparse.ArgumentParser("ae_lgbm_wide_deep")
    ap.add_argument("--mode", choices=["train", "validate", "infer"], default="validate")
    ap.add_argument("--tag", default=None, help="optional run tag appended to artifacts dir")
    args = ap.parse_args()


    try:
        import config as cfg
    except Exception as e:
        print(f"[FATAL] cannot import config.py: {e}", file=sys.stderr)
        sys.exit(1)


    out_dir = make_artifacts_dir(tag=args.tag)
    save_config_snapshot(cfg, out_dir)
    print(f"[RUN] mode={args.mode} | artifacts={out_dir}")


# Next step: registry.run(...) will orchestrate the pipeline.
    try:
        import registry, pipeline

        #统一数据装配，已经经过了最初启动时间以及和过滤权重等等
        bundles = registry.prepare_bundles(cfg, mode=args.mode)
        bd_train = bundles["train"]
        bd_holdout = bundles["holdout"]

        #评估的开关部分
        eval_heads = getattr(cfg, "EVAL_HEADS", ("head", "lr", "lgbm"))
        primary = getattr(cfg, "PRIMARY_EVAL_HEAD", "lgbm")

        #进入pipeline管
        result = pipeline.fit_evaluate(
            train = bd_train,
            cfg = cfg,
            eval_heads=eval_heads,
            primary_head=primary,
            holdout=bd_holdout
        )
        # =======================================================
        # === [新增代码] 强行把模型存到硬盘 (用于提交) ===
        # =======================================================
        print("[main] Attempting to save model to disk...")
        # 1. 保存基线 LGBM
        if "models" in result and result["models"]:
            # 我们取最后一个 fold 的模型作为提交模型 (简单有效)
            model_to_save = result["models"][-1]
            
            # 保存两份：一份在 artifacts 目录，一份在根目录(方便上传)
            save_path_root = "model_lgbm.pkl"
            save_path_run = os.path.join(out_dir, "model_lgbm.pkl")
            
            joblib.dump(model_to_save, save_path_root)
            joblib.dump(model_to_save, save_path_run)
            print(f"[SUCCESS] Baseline LGBM saved to: {save_path_root}")
        else:
            print("[WARN] No 'models' found in result to save!")

        # 2. (可选) 如果有 AE->LGBM 模型，也存一下
        if "models_wd" in result and result["models_wd"]:
            wd_model = result["models_wd"][-1]
            joblib.dump(wd_model, "model_ae_lgbm.pkl")
            print(f"[SUCCESS] AE->LGBM saved to: model_ae_lgbm.pkl")
        # =======================================================
        # ====== artifacts 保存 / 推理 ======
        run_dir = Path(out_dir)  # ← 原来写成 artifacts_dir（未定义）
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "validate":
            # —— 只在“有验证”的样本上落 OOF ——
            oof = result.get("oof", result.get("oof_pred"))
            mask = result.get("oof_mask")
            assert oof is not None and mask is not None, "[main] result 里没有 oof / oof_mask"

            oof = np.asarray(oof).reshape(-1)
            mask = np.asarray(mask).astype(bool)
            assert oof.shape[0] == bd_train.X.shape[0] and mask.shape[0] == bd_train.X.shape[0], \
                "[main] oof/mask 与训练集长度不一致"

            oof_df = pd.DataFrame({
                "row_id": bd_train.X.index[mask],
                "pred": oof[mask].astype(float),
            })
            oof_path = run_dir / f"oof_lgbm_{result.get('target', getattr(cfg, 'PRIMARY_TARGET', 'action_1d'))}.csv"
            oof_df.to_csv(oof_path, index=False)

            # --- Save OOF pack for decision-layer tuning (no retraining needed) ---
            # Contains: full timeline aux (resp_1d/forward/rf), OOF mask, and multiple prediction heads.
            try:
                pack = pd.DataFrame({
                    "row_id": bd_train.X.index,
                    "date_id": np.asarray(bd_train.date),
                    "oof_mask": mask.astype(np.int8),
                })

                def _add_pred(colname: str, key: str) -> None:
                    arr = result.get(key, None)
                    if arr is None:
                        return
                    arr = np.asarray(arr).reshape(-1)
                    if arr.shape[0] != pack.shape[0]:
                        return
                    pack[colname] = arr

                _add_pred("pred_lgbm", "oof_pred")
                _add_pred("pred_ae_lr", "oof_pred_lr")
                _add_pred("pred_ae_lgbm", "oof_pred_lgbm_wd")
                _add_pred("pred_blend", "oof_pred_blend")
                _add_pred("regime_scale", "regime_scale_all")

                if isinstance(getattr(bd_train, "y", None), dict) and ("resp_1d" in bd_train.y):
                    pack["resp_1d"] = np.asarray(bd_train.y["resp_1d"], dtype=np.float32)

                if getattr(bd_train, "sol", None) is not None:
                    for c in ("forward_returns", "risk_free_rate"):
                        if c in bd_train.sol.columns:
                            pack[c] = bd_train.sol[c].to_numpy(dtype=np.float32, copy=False)

                pack_path = run_dir / "oof_pack.csv"
                pack.to_csv(pack_path, index=False)
                print(f"[artifacts] OOF pack -> {pack_path.name} (for decision tuning)")
            except Exception as e:
                print(f"[warn] failed to write oof_pack.csv: {e}")


            metrics = {
                "target": result.get("target"),
                "n_folds": result.get("n_folds"),
                # 1. 纯 LGBM 分数
                "oof_auc_lgbm": result.get("auc_oof"), 
                # 2. [新增] AE->LGBM 分数
                "oof_auc_ae_lgbm": result.get("auc_oof_lgbm_wd"),
                # 3. [新增] AE->LR 分数
                "oof_auc_ae_lr": result.get("auc_oof_lr"),
                # 4. [核心新增] 最强 Blend 分数！
                "oof_auc_blend": result.get("auc_oof_blend"),
                "sharpe_oof_proxy": result.get("sharpe_oof_proxy"),
                "oof_score_official": result.get("oof_score_official"),
                "decision_pred_head": result.get("decision_pred_head"),
                "decision_meta": result.get("decision_meta"),
                "oof_auc": result.get("auc_oof"),
                "fold_aucs": result.get("fold_aucs"),
                "oof_coverage": float(mask.mean()),
                "best_params": result.get("best_params"),   # ← 用 get，避免 KeyError
            }
            with open(run_dir / "fold_metrics_lgbm.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            # —— 配置快照 —— 
            cfg_snap = {  # ← 原来写成 config
                k: getattr(cfg, k) for k in dir(cfg)
                if k.isupper() and not k.startswith("__")
            }
            with open(run_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
                json.dump(cfg_snap, f, ensure_ascii=False, indent=2)

            print(f"[artifacts] OOF -> {oof_path.name}; metrics/config 已写入 {run_dir}")

        elif args.mode == "train":
            # 1) 跑一遍CV，同时把最后k折模型带回来（用于holdout推理）
            result = pipeline.fit_evaluate(
                train=bd_train,
                cfg=cfg,
                primary_head="lgbm",
                holdout=bd_holdout,   # ← 关键：train模式要把holdout传进去
            )

            # 2) artifacts 目录
            run_dir = Path(out_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

            # 3) 用“最后k折”模型对 holdout 做均值推理并落盘
            models = result.get("models", [])
            if bd_holdout is None or not models:
                print("[warn] 没有 holdout 或 models，跳过推理")
            else:
                # —— 对齐 holdout 列到训练口径（集合与顺序）——
                feat_cols = result.get("feature_cols", list(bd_train.X.columns))
                miss = [c for c in feat_cols if c not in bd_holdout.X.columns]
                extra = [c for c in bd_holdout.X.columns if c not in feat_cols]
                print(f"[align] holdout missing={len(miss)} extra={len(extra)}"
                    f"{' | missing sample: ' + ','.join(miss[:5]) if miss else ''}"
                    f"{' | extra sample: ' + ','.join(extra[:5]) if extra else ''}")
                X_ho = bd_holdout.X.reindex(columns=feat_cols, fill_value=0.0)
                # （可选）统一 dtype（一些版本更稳）
                X_ho = X_ho.astype(np.float32, copy=False)
                # 均值集成（最后k折）
                preds = np.stack([m.predict_proba(X_ho)[:, 1] for m in models], axis=1).mean(axis=1)
                ho_df = pd.DataFrame({
                    "row_id": X_ho.index,
                    "pred": preds.astype(float),
                })
                ho_path = run_dir / f"pred_holdout_lgbm_{result['target']}.csv"
                ho_df.to_csv(ho_path, index=False)
                print(f"[artifacts] holdout prediction -> {ho_path.name}  (models={len(models)})")

                wd_ho = result.get("holdout_pred_lgbm_wd", None)
                if wd_ho is not None:
                    ho_wd_df = pd.DataFrame({"row_id": X_ho.index, "pred": wd_ho.astype(float)})
                    path_wd = run_dir / f"pred_holdout_ae_lgbm_{result['target']}.csv"
                    ho_wd_df.to_csv(path_wd, index=False)
                    print(f"[artifacts] holdout (AE->LGBM) -> {path_wd.name}")
            # 4) 统一完成日志
            print("[main] done", result)
    except ImportError:
            print("[WARN] registry 或 pipeline 未就绪。稍后接入 prepare_bundles/fit_evaluate。")


if __name__ == "__main__":
    main()







