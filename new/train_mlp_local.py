"""
BirdCLEF+ 2026 — Local MLP head training on Perch v2 embeddings.

Ablation over three variants:
  A. Linear probe
  B. MLP head (1536 -> 768 -> 384 -> 234) + BN + Dropout
  C. MLP head + Focal BCE loss

Outputs:
  - C:\\birdCLEF\\new\\ablation_results.json
  - C:\\birdCLEF\\new\\training_curves.png
  - C:\\birdCLEF\\new\\mlp_best.pt
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
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SEED = 42
EMB_DIR = Path(r"C:\birdCLEF\embedding")
OUT_DIR = Path(r"C:\birdCLEF\new")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 234
EMB_DIM = 1536

EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
PATIENCE = 5
VAL_FRAC = 0.15

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] torch={torch.__version__} device={DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[env] gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}", flush=True)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
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

    assert ta_emb.shape == (ta_lab.shape[0], EMB_DIM)
    assert ta_lab.shape[1] == N_CLASSES
    assert len(meta) == ta_emb.shape[0]
    assert ss_emb.shape == (ss_lab.shape[0], EMB_DIM)

    # Groups: use filename for train_audio; one unique group per soundscape segment.
    ta_groups = meta["filename"].astype(str).to_numpy()
    ss_groups = np.array([f"__ss_{i}__" for i in range(ss_lab.shape[0])])

    X = np.concatenate([ta_emb, ss_emb], axis=0)
    Y = np.concatenate([ta_lab, ss_lab], axis=0)
    G = np.concatenate([ta_groups, ss_groups], axis=0)

    print(
        f"[data] X={X.shape} Y={Y.shape} G unique={len(np.unique(G))} "
        f"(train_audio files={len(np.unique(ta_groups))}, soundscape segs={len(ss_groups)})",
        flush=True,
    )
    return X, Y, G


def split_data(X, Y, G):
    assert len(np.unique(G)) > 1, "Need >1 group for GroupShuffleSplit"
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X, Y, groups=G))

    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_va, Y_va = X[va_idx], Y[va_idx]

    val_pos_per_class = Y_va.sum(axis=0)
    n_classes_with_pos = int((val_pos_per_class > 0).sum())

    print(
        f"[split] train={len(tr_idx)} val={len(va_idx)} "
        f"train_pos_frac={Y_tr.mean():.5f} val_pos_frac={Y_va.mean():.5f} "
        f"val_classes_with_pos={n_classes_with_pos}/{N_CLASSES}",
        flush=True,
    )
    return X_tr, Y_tr, X_va, Y_va, n_classes_with_pos


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class LinearProbe(nn.Module):
    def __init__(self, in_dim=EMB_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


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


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return loss.mean()


# -----------------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    all_logits, all_tgts, losses = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(loss.item() * xb.size(0))
        all_logits.append(logits.detach().cpu().numpy())
        all_tgts.append(yb.detach().cpu().numpy())
    n = sum(x.shape[0] for x in all_logits)
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits, axis=0)))
    tgts = np.concatenate(all_tgts, axis=0)

    per_class_auc = np.full(N_CLASSES, np.nan, dtype=np.float64)
    for c in range(N_CLASSES):
        if tgts[:, c].sum() > 0 and tgts[:, c].sum() < tgts.shape[0]:
            try:
                per_class_auc[c] = roc_auc_score(tgts[:, c], probs[:, c])
            except Exception:
                pass
    macro_auc = float(np.nanmean(per_class_auc))
    avg_loss = sum(losses) / max(n, 1)
    return avg_loss, macro_auc, per_class_auc


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_variant(name, model, loss_fn, train_loader, val_loader):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Training {name} | params={n_params:,} ===", flush=True)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc = -1.0
    best_state = None
    best_per_class = None
    patience = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"{name} ep{epoch:02d}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        tr_loss = running / max(n_seen, 1)

        val_loss, val_auc, per_class = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_auc
        flag = "*" if improved else " "
        print(
            f"[{name}] ep{epoch:02d} tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.4f} {flag}",
            flush=True,
        )

        if improved:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_per_class = per_class.copy()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[{name}] early stop at epoch {epoch} (no improve {PATIENCE} ep).", flush=True)
                break

    dur = time.time() - t0
    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # top/bottom per-class AUC
    pc = best_per_class
    order = np.argsort(-np.where(np.isnan(pc), -np.inf, pc))
    top5 = [(int(i), float(pc[i])) for i in order[:5] if not np.isnan(pc[i])]
    valid = ~np.isnan(pc)
    bot_order = np.argsort(np.where(valid, pc, np.inf))
    bottom5 = [(int(i), float(pc[i])) for i in bot_order[:5] if not np.isnan(pc[i])]

    return {
        "name": name,
        "params": int(n_params),
        "best_val_auc": float(best_auc),
        "train_time_sec": float(dur),
        "history": history,
        "top5_classes": top5,
        "bottom5_classes": bottom5,
        "best_state": best_state,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    X, Y, G = load_data()
    X_tr, Y_tr, X_va, Y_va, n_classes_with_pos = split_data(X, Y, G)

    pin = DEVICE.type == "cuda"
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                            num_workers=0, pin_memory=pin)

    bce = nn.BCEWithLogitsLoss()

    variants = []

    # A: Linear probe, BCE
    variants.append(train_variant("A_linear_bce", LinearProbe(), bce, train_loader, val_loader))

    # B: MLP head, BCE
    variants.append(train_variant("B_mlp_bce", MLPHead(), bce, train_loader, val_loader))

    # C: MLP head, Focal
    variants.append(train_variant("C_mlp_focal", MLPHead(), focal_bce, train_loader, val_loader))

    # ---------- save best model ----------
    best_overall = max(variants, key=lambda v: v["best_val_auc"])
    torch.save(best_overall["best_state"], OUT_DIR / "mlp_best.pt")
    print(f"\n[save] best={best_overall['name']} auc={best_overall['best_val_auc']:.4f} "
          f"-> {OUT_DIR / 'mlp_best.pt'}", flush=True)

    # ---------- training curves ----------
    plt.figure(figsize=(8, 5))
    for v in variants:
        plt.plot(range(1, len(v["history"]["val_auc"]) + 1),
                 v["history"]["val_auc"], label=v["name"], marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Val macro AUC (skip-empty)")
    plt.title("BirdCLEF+ 2026 — Perch v2 head ablation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curves.png", dpi=130)
    plt.close()
    print(f"[save] curves -> {OUT_DIR / 'training_curves.png'}", flush=True)

    # ---------- results json ----------
    serializable = []
    for v in variants:
        serializable.append({
            "name": v["name"],
            "params": v["params"],
            "best_val_auc": v["best_val_auc"],
            "train_time_sec": v["train_time_sec"],
            "epochs_run": len(v["history"]["val_auc"]),
            "history": v["history"],
            "top5_classes": v["top5_classes"],
            "bottom5_classes": v["bottom5_classes"],
        })
    result_payload = {
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(DEVICE),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "n_train": int(X_tr.shape[0]),
            "n_val": int(X_va.shape[0]),
            "val_classes_with_pos": int(n_classes_with_pos),
            "n_classes": N_CLASSES,
            "emb_dim": EMB_DIM,
        },
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "patience": PATIENCE, "val_frac": VAL_FRAC, "seed": SEED,
        },
        "best_overall": best_overall["name"],
        "variants": serializable,
    }
    with open(OUT_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    print(f"[save] results -> {OUT_DIR / 'ablation_results.json'}", flush=True)

    # ---------- markdown table ----------
    print("\n\n## Ablation Summary\n", flush=True)
    print("| Variant | Params | Time (s) | Epochs | Val macro AUC |", flush=True)
    print("|---------|--------|----------|--------|---------------|", flush=True)
    for v in variants:
        print(
            f"| {v['name']} | {v['params']:,} | {v['train_time_sec']:.1f} | "
            f"{len(v['history']['val_auc'])} | {v['best_val_auc']:.4f} |",
            flush=True,
        )
    print(f"\nBest overall: **{best_overall['name']}** (val AUC = {best_overall['best_val_auc']:.4f})", flush=True)


if __name__ == "__main__":
    main()
