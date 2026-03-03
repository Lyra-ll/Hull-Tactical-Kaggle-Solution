"""
Supervised AE: fit per fold, then encode (Z) for LGBM.

- 监督式 AE 的折内训练与编码：只在训练折拟合（含标准化），再对 val/holdout 编码。
- 返回：Z_tr/Z_va/Z_ho、可选 head 输出（仅用于评估/消融，不作为 LGBM 特征）。

Core:
- fit_encode(X_tr, y_tr, w_tr, X_va, X_ho=None, cfg) -> dict(Z_tr, Z_va, Z_ho, metrics)
- 保证推断阶段使用“训练折统计量标准化后”的矩阵（避免分布漂移）。

TODO:
- 模型/Scaler 的持久化（每折保存 .pt/.pkl）供复算/断点续训。
- TopK 选择与列名约定：AE_000..AE_K-1，确保稳定复现。
"""



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional, Tuple

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training and self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x

class EncoderNet(nn.Module):
    def __init__(self, in_dim:int, enc_dim:int = 64, hidden:int = 128, dropout:float = 0.1, n_targets: int = 1):
        super().__init__()
        self.noise = GaussianNoise(sigma=0.1)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden // 2, enc_dim),
            nn.BatchNorm1d(enc_dim),
            nn.SiLU(),
        )

        #解码器
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, hidden//2),
            nn.BatchNorm1d(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden // 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, in_dim)
        )

        #预测头
        self.head = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_targets)
        )

    def forward(self, x):
        if self.training:
            x_in = self.noise(x)
        else:
            x_in = x

        z = self.enc(x_in)
        logits = self.head(z)
        recon = self.dec(z)
        return z, logits, recon


def _maybe_export_ae_bundle(
    *,
    export: Optional[dict],
    net: "EncoderNet",
    mu: np.ndarray,
    sigma: np.ndarray,
    in_dim: int,
    n_targets: int,
    hidden: int,
    enc_dim: int,
    dropout: float,
) -> None:
    """
    Export AE weights + normalization stats for later inference (Kaggle bundle).
    This is intentionally 'dumb simple': it just writes files if export is provided.
    """
    if not export:
        return
    try:
        import os
        from pathlib import Path
        import joblib
        import torch

        out_dir = Path(str(export.get("dir", "hull_model_bundle")))
        out_dir.mkdir(parents=True, exist_ok=True)

        fold = export.get("fold", None)
        seed = export.get("seed", None)
        tag = export.get("tag", "ae")
        prefix = export.get("prefix")
        if not prefix:
            if fold is not None and seed is not None:
                prefix = f"fold{int(fold):02d}_seed{int(seed):02d}_{tag}"
            elif fold is not None:
                prefix = f"fold{int(fold):02d}_{tag}"
            else:
                prefix = f"{tag}"

        # weights
        torch.save(net.state_dict(), out_dir / f"{prefix}_state.pth")

        # norm + arch
        meta = {
            "mu": mu.astype("float32"),
            "sigma": sigma.astype("float32"),
            "in_dim": int(in_dim),
            "n_targets": int(n_targets),
            "hidden": int(hidden),
            "enc_dim": int(enc_dim),
            "dropout": float(dropout),
        }
        joblib.dump(meta, out_dir / f"{prefix}_norm.pkl")
        print(f"[bundle] saved AE -> {(out_dir / (prefix + '_state.pth')).as_posix()}")
    except Exception as e:
        print(f"[bundle][WARN] failed to export AE: {e}")


def _to_tensor(x):
    if hasattr(x, "values"): x = x.values
    return torch.from_numpy(np.asarray(x, dtype=np.float32))

def fit_encode(
    X_tr, y_tr, w_tr, X_va, X_ho=None,
    *, cfg: Optional[Any] = None,   # 新增：显式接收 cfg（关键字专用参数）
    epochs: int = 40,
    lr: float = 0.001,
    batch: int = 512,
    device: str = "cpu",
    export: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    AE训练，返回三个东西：训练集潜变量和验证集潜变量以及验证集概率
    
    """

    if y_tr.ndim == 1:
        n_targets = 1
        y_tr = y_tr.reshape(-1, 1)
    else:
        n_targets = y_tr.shape[1]

    # === [核心修改] 从 cfg 读取动态架构参数 (Optuna 注入的) ===
    # 如果 cfg 里没有，就用默认值 (256, 64)
    # 注意：这里的 getattr 会读取我们在 objective 里动态注入的值
    dynamic_hidden = int(getattr(cfg, "AE_DYN_HIDDEN", 256))
    dynamic_enc_dim = int(getattr(cfg, "AE_DYN_ENC_DIM", 64))
    dynamic_dropout = float(getattr(cfg, "AE_DYN_DROPOUT", 0.1))

    # 实例化模型时传入动态参数
    net = EncoderNet(
        in_dim=X_tr.shape[1], 
        n_targets=n_targets,
        hidden=dynamic_hidden,      # <--- 动态深度
        enc_dim=dynamic_enc_dim,    # <--- 动态瓶颈
        dropout=dynamic_dropout     # <--- 动态 Dropout
    ).to(device)
    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    #定义两个loss
    criterion_cls = nn.BCEWithLogitsLoss(reduction="none")
    criterion_recon = nn.MSELoss(reduction="none")

    # === Standardize per-fold (use train stats only) ===
    X_tr_np = np.asarray(X_tr, dtype=np.float32)
    X_va_np = np.asarray(X_va, dtype=np.float32)
    mu = np.nanmean(X_tr_np, axis=0)
    sigma = np.nanstd(X_tr_np, axis=0)
    sigma[sigma < 1e-6] = 1.0  # 防止除0

    X_tr_np = (X_tr_np - mu) / sigma
    X_va_np = (X_va_np - mu) / sigma
    X_tr_np = np.nan_to_num(X_tr_np, nan=0.0)
    X_va_np = np.nan_to_num(X_va_np, nan=0.0)
    if X_ho is not None:
        X_ho_np = (np.asarray(X_ho, dtype=np.float32) - mu) / sigma
        X_ho_np = np.nan_to_num(X_ho_np, nan=0.0)
    else:
        X_ho_np = None

    Xtr = _to_tensor(X_tr_np).to(device)
    y_tensor = _to_tensor(y_tr)
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.view(-1, 1)
    ytr = y_tensor.to(device)
    wtr = _to_tensor(w_tr.reshape(-1, 1)).to(device).view(-1, 1)

    net.train()
    n = Xtr.size(0)

    w_cls = getattr(cfg, "AE_LOSS_WEIGHT_CLS", 0.7)    # 默认 0.7
    w_recon = getattr(cfg, "AE_LOSS_WEIGHT_RECON", 0.3) # 默认 0.3

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            batch_x = Xtr[idx]
            batch_y = ytr[idx]
            batch_w = wtr[idx]

            _, logits, recon = net(batch_x)
            loss_cls = (criterion_cls(logits, batch_y) * batch_w).mean()
            loss_recon = criterion_recon(recon, batch_x).mean()

            loss = (w_cls * loss_cls) + (w_recon * loss_recon)

            opt.zero_grad()
            loss.backward()
            opt.step()

    # —— 评估输出（含可选 Z_ho） ——
    
    # === [BUNDLE EXPORT] save AE weights + norm stats (optional) ===
    _maybe_export_ae_bundle(
        export=export,
        net=net,
        mu=mu,
        sigma=sigma,
        in_dim=X_tr.shape[1],
        n_targets=n_targets,
        hidden=dynamic_hidden,
        enc_dim=dynamic_enc_dim,
        dropout=dynamic_dropout,
    )

    net.eval()
    with torch.no_grad():
        Z_tr, _, _ = net(_to_tensor(X_tr_np).to(device))
        Z_va, logits_va, _ = net(_to_tensor(X_va_np).to(device))
        p_va = torch.sigmoid(logits_va)[:, 0].detach().cpu().numpy()
        Z_tr = Z_tr.cpu().numpy()
        Z_va = Z_va.cpu().numpy()

        Z_ho = None
        if X_ho is not None:
            Z_ho, _, _ = net(_to_tensor(X_ho_np).to(device))
            Z_ho = Z_ho.cpu().numpy()

    return Z_tr, Z_va, p_va, Z_ho

