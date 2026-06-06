"""
BirdCLEF+ 2026 — 5-fold MLP head training with MixUp + soundscape upweighting.

Improvements over train_mlp_local.py:
  - Soundscape 2x dedup (host-confirmed error)
  - GroupKFold(5) instead of single GroupShuffleSplit
  - Embedding-level MixUp (Beta(0.4, 0.4))
  - Soundscape sample upweighting (20x) in BCE loss
  - Saves mlp_fold{0-4}.pt for ensemble inference

Outputs:
  - C:\\birdCLEF\\new\\mlp_fold0.pt ... mlp_fold4.pt
  - C:\\birdCLEF\\new\\fold_results.json
  - C:\\birdCLEF\\new\\fold_curves.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SEED = 42
EMB_DIR = Path(r"C:\birdCLEF\embedding")
EXT_EMB_DIR = Path(r"C:\birdCLEF\embedding_external")
OUT_DIR = Path(r"C:\birdCLEF\new")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_EXTERNAL = EXT_EMB_DIR.exists() and (EXT_EMB_DIR / "external_embeddings.npy").exists()

N_CLASSES = 234
EMB_DIM = 1536
N_FOLDS = 5

EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
PATIENCE = 5
MIXUP_ALPHA = 0.4
SC_WEIGHT = 20.0  # soundscape sample weight multiplier

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] torch={torch.__version__} device={DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[env] gpu={torch.cuda.get_device_name(0)}", flush=True)


class MLPHead(nn.Module):
    def __init__(self, in_dim=EMB_DIM, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(384, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    print("[data] loading embeddings...", flush=True)
    ta_emb = np.load(EMB_DIR / "train_audio_embeddings.npy").astype(np.float32)
    ta_lab = np.load(EMB_DIR / "train_audio_labels.npy").astype(np.float32)
    ss_emb = np.load(EMB_DIR / "soundscape_embeddings.npy").astype(np.float32)
    ss_lab = np.load(EMB_DIR / "soundscape_labels.npy").astype(np.float32)
    meta = pd.read_parquet(EMB_DIR / "train_audio_meta.parquet")

    # Host confirmed exact 2x duplicate in soundscape labels (1478 → 739 unique)
    ss_emb = ss_emb[:len(ss_emb) // 2]
    ss_lab = ss_lab[:len(ss_lab) // 2]

    n_ta = ta_emb.shape[0]
    n_ss = ss_emb.shape[0]

    ta_groups = meta["filename"].astype(str).to_numpy()
    ss_groups = np.array([f"__ss_{i}__" for i in range(n_ss)])

    X = np.concatenate([ta_emb, ss_emb], axis=0)
    Y = np.concatenate([ta_lab, ss_lab], axis=0)
    G = np.concatenate([ta_groups, ss_groups], axis=0)

    # is_soundscape mask for upweighting
    is_sc = np.zeros(len(X), dtype=np.float32)
    is_sc[n_ta:] = 1.0

    print(
        f"[data] X={X.shape} Y={Y.shape} "
        f"train_audio={n_ta} soundscape={n_ss} (deduped)",
        flush=True,
    )

    # Optionally load external embeddings (iNaturalist, Xeno-canto, AnuraSet)
    if USE_EXTERNAL:
        ext_emb = np.load(EXT_EMB_DIR / "external_embeddings.npy").astype(np.float32)
        ext_lab = np.load(EXT_EMB_DIR / "external_labels.npy").astype(np.float32)
        ext_meta = pd.read_parquet(EXT_EMB_DIR / "external_meta.parquet")

        assert ext_emb.shape[1] == EMB_DIM
        assert ext_lab.shape[1] == N_CLASSES

        n_ext = ext_emb.shape[0]
        ext_groups = np.array([f"__ext_{i}__" for i in range(n_ext)])
        ext_is_sc = np.zeros(n_ext, dtype=np.float32)

        X = np.concatenate([X, ext_emb], axis=0)
        Y = np.concatenate([Y, ext_lab], axis=0)
        G = np.concatenate([G, ext_groups], axis=0)
        is_sc = np.concatenate([is_sc, ext_is_sc], axis=0)

        print(
            f"[data] +external: {n_ext} frames, {ext_meta['primary_label'].nunique()} species "
            f"-> total X={X.shape}",
            flush=True,
        )

    # Pseudo-labeled data returned separately (training-only, never in validation)
    ps_emb, ps_lab = None, None
    pseudo_path = EMB_DIR / "pseudo_embeddings.npy"
    if pseudo_path.exists():
        ps_emb = np.load(pseudo_path).astype(np.float32)
        ps_lab = np.load(EMB_DIR / "pseudo_labels.npy").astype(np.float32)
        assert ps_emb.shape[1] == EMB_DIM
        assert ps_lab.shape[1] == N_CLASSES
        print(f"[data] pseudo available: {ps_emb.shape[0]} frames (training-only)", flush=True)

    return X, Y, G, is_sc, ps_emb, ps_lab


def make_sample_weights(is_sc_train):
    """Soundscape rows get SC_WEIGHT, train_audio rows get 1.0."""
    w = np.ones_like(is_sc_train)
    w[is_sc_train > 0] = SC_WEIGHT
    return w


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_tgts = [], []
    for xb, yb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.cpu().numpy())
        all_tgts.append(yb.numpy())
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits, axis=0)))
    tgts = np.concatenate(all_tgts, axis=0)

    per_class_auc = np.full(N_CLASSES, np.nan, dtype=np.float64)
    for c in range(N_CLASSES):
        if 0 < tgts[:, c].sum() < tgts.shape[0]:
            try:
                per_class_auc[c] = roc_auc_score(tgts[:, c], probs[:, c])
            except Exception:
                pass
    macro_auc = float(np.nanmean(per_class_auc))
    return macro_auc


def train_fold(fold_idx, X_tr, Y_tr, W_tr, X_va, Y_va):
    torch.manual_seed(SEED + fold_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold_idx)

    model = MLPHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pin = DEVICE.type == "cuda"
    train_ds = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(Y_tr), torch.from_numpy(W_tr)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_va), torch.from_numpy(Y_va),
        torch.ones(len(X_va), dtype=torch.float32),  # dummy weights for val
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    print(f"\n=== Fold {fold_idx} | params={n_params:,} ===", flush=True)

    history = {"train_loss": [], "val_auc": []}
    best_auc = -1.0
    best_state = None
    patience = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"F{fold_idx} ep{epoch:02d}", leave=False)
        for xb, yb, wb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)

            # MixUp on embeddings
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            idx = torch.randperm(xb.size(0), device=DEVICE)
            xb = lam * xb + (1 - lam) * xb[idx]
            yb = lam * yb + (1 - lam) * yb[idx]
            wb = lam * wb + (1 - lam) * wb[idx]

            opt.zero_grad()
            logits = model(xb)
            # Weighted BCE: per-sample weight applied to mean loss
            loss_per_sample = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            ).mean(dim=1)  # (B,)
            loss = (loss_per_sample * wb).mean()
            loss.backward()
            opt.step()

            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        tr_loss = running / max(n_seen, 1)

        val_auc = evaluate(model, val_loader)
        history["train_loss"].append(tr_loss)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_auc
        flag = "*" if improved else " "
        print(
            f"[F{fold_idx}] ep{epoch:02d} tr_loss={tr_loss:.4f} "
            f"val_auc={val_auc:.4f} {flag}",
            flush=True,
        )

        if improved:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[F{fold_idx}] early stop at epoch {epoch}", flush=True)
                break

    dur = time.time() - t0
    out_path = OUT_DIR / f"mlp_fold{fold_idx}.pt"
    torch.save(best_state, out_path)
    print(f"[F{fold_idx}] best_auc={best_auc:.4f} time={dur:.1f}s -> {out_path}", flush=True)

    return {
        "fold": fold_idx,
        "best_val_auc": float(best_auc),
        "epochs_run": len(history["val_auc"]),
        "train_time_sec": float(dur),
        "history": history,
    }


def main():
    X, Y, G, is_sc, ps_emb, ps_lab = load_data()

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, Y, groups=G)):
        X_tr, Y_tr = X[tr_idx], Y[tr_idx]
        X_va, Y_va = X[va_idx], Y[va_idx]
        is_sc_tr = is_sc[tr_idx]

        # Append pseudo-labeled data to training only (never to validation)
        if ps_emb is not None:
            ps_is_sc = np.ones(len(ps_emb), dtype=np.float32)
            X_tr = np.concatenate([X_tr, ps_emb], axis=0)
            Y_tr = np.concatenate([Y_tr, ps_lab], axis=0)
            is_sc_tr = np.concatenate([is_sc_tr, ps_is_sc], axis=0)

        W_tr = make_sample_weights(is_sc_tr)

        n_sc_tr = int(is_sc_tr.sum())
        n_real_tr = len(tr_idx)
        n_pseudo = len(ps_emb) if ps_emb is not None else 0
        print(
            f"\n[fold {fold_idx}] train={len(X_tr)} (real={n_real_tr} + pseudo={n_pseudo}) "
            f"val={len(va_idx)} sc_in_train={n_sc_tr}",
            flush=True,
        )

        result = train_fold(fold_idx, X_tr, Y_tr, W_tr, X_va, Y_va)
        fold_results.append(result)

    # Summary
    aucs = [r["best_val_auc"] for r in fold_results]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    print(f"\n\n{'='*60}")
    print(f"5-fold OOF:  mean={mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Per-fold:    {['%.4f' % a for a in aucs]}")
    print(f"{'='*60}\n")

    # Save results JSON
    payload = {
        "config": {
            "n_folds": N_FOLDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "patience": PATIENCE, "seed": SEED,
            "mixup_alpha": MIXUP_ALPHA, "sc_weight": SC_WEIGHT,
        },
        "mean_val_auc": mean_auc,
        "std_val_auc": std_auc,
        "folds": fold_results,
    }
    with open(OUT_DIR / "fold_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] {OUT_DIR / 'fold_results.json'}", flush=True)

    # Training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for r in fold_results:
        ep = range(1, len(r["history"]["val_auc"]) + 1)
        axes[0].plot(ep, r["history"]["val_auc"], label=f"fold{r['fold']}", marker="o", markersize=3)
        axes[1].plot(ep, r["history"]["train_loss"], label=f"fold{r['fold']}", marker="o", markersize=3)
    axes[0].set(xlabel="Epoch", ylabel="Val macro AUC", title="5-fold Val AUC")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Train loss", title="5-fold Train Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fold_curves.png", dpi=130)
    plt.close(fig)
    print(f"[save] {OUT_DIR / 'fold_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
