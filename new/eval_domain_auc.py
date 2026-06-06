"""
Domain-split evaluation: compute val AUC on train_audio portion (ta_auc) and
soundscape portion (sc_auc) separately, for each of the three ablation variants.

This aligns with Route A's reporting style (which separates `val_auc` from `sc_auc`).
Re-trains all three variants deterministically (seed=42) since only Variant B's
weights were persisted. Total runtime ~2-3 minutes on RTX 5070.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
EMB_DIR = Path(r"C:\birdCLEF\embedding")
OUT_DIR = Path(r"C:\birdCLEF\new")
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


def macro_auc_on_subset(probs, tgts, idx):
    """Compute macro AUC (skip classes with no positives) on a row subset."""
    if len(idx) == 0:
        return float("nan"), 0
    sub_probs = probs[idx]
    sub_tgts = tgts[idx]
    aucs = []
    for c in range(N_CLASSES):
        pos = sub_tgts[:, c].sum()
        if 0 < pos < sub_tgts.shape[0]:
            try:
                aucs.append(roc_auc_score(sub_tgts[:, c], sub_probs[:, c]))
            except Exception:
                pass
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(EMB_DIM, N_CLASSES)
    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, 768), nn.BatchNorm1d(768), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(768, 384), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(384, N_CLASSES),
        )
    def forward(self, x):
        return self.net(x)


def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


@torch.no_grad()
def full_eval(model, X_va, Y_va):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va)),
                        batch_size=BATCH_SIZE * 4, shuffle=False, pin_memory=True)
    all_logits, all_tgts = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.cpu().numpy())
        all_tgts.append(yb.numpy())
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits, 0)))
    tgts = np.concatenate(all_tgts, 0)
    return probs, tgts


def train_variant(name, model_cls, loss_fn, train_loader, X_va, Y_va,
                  ta_val_idx, ss_val_idx):
    """Train and also track ta_auc / sc_auc per epoch."""
    model = model_cls().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n_params = sum(p.numel() for p in model.parameters())

    history = {"train_loss": [], "val_auc": [], "ta_auc": [], "sc_auc": [],
               "val_n_classes": [], "ta_n_classes": [], "sc_n_classes": []}
    best_val = -1.0
    best_epoch = 0
    best_probs, best_tgts = None, None
    patience = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        tr_loss = total / n

        probs, tgts = full_eval(model, X_va, Y_va)
        all_idx = np.arange(len(tgts))
        val_auc, val_nc = macro_auc_on_subset(probs, tgts, all_idx)
        ta_auc, ta_nc = macro_auc_on_subset(probs, tgts, ta_val_idx)
        sc_auc, sc_nc = macro_auc_on_subset(probs, tgts, ss_val_idx)

        history["train_loss"].append(tr_loss)
        history["val_auc"].append(val_auc)
        history["ta_auc"].append(ta_auc)
        history["sc_auc"].append(sc_auc)
        history["val_n_classes"].append(val_nc)
        history["ta_n_classes"].append(ta_nc)
        history["sc_n_classes"].append(sc_nc)

        flag = "*" if val_auc > best_val else " "
        print(f"[{name}] ep{epoch:02d} tr_loss={tr_loss:.4f} val_auc={val_auc:.4f} "
              f"ta_auc={ta_auc:.4f} sc_auc={sc_auc:.4f} {flag}", flush=True)

        if val_auc > best_val:
            best_val = val_auc
            best_epoch = epoch
            best_probs = probs.copy()
            best_tgts = tgts.copy()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[{name}] early stop at ep{epoch}", flush=True)
                break

    dur = time.time() - t0
    # Final metrics at best epoch
    best_val_auc, _ = macro_auc_on_subset(best_probs, best_tgts, np.arange(len(best_tgts)))
    best_ta_auc, best_ta_nc = macro_auc_on_subset(best_probs, best_tgts, ta_val_idx)
    best_sc_auc, best_sc_nc = macro_auc_on_subset(best_probs, best_tgts, ss_val_idx)

    return {
        "name": name,
        "params": int(n_params),
        "best_epoch": best_epoch,
        "epochs_run": len(history["val_auc"]),
        "train_time_sec": dur,
        "best_val_auc": best_val_auc,
        "best_ta_auc": best_ta_auc,
        "best_sc_auc": best_sc_auc,
        "best_ta_n_classes": best_ta_nc,
        "best_sc_n_classes": best_sc_nc,
        "history": history,
    }


def main():
    # --- load ---
    ta_emb = np.load(EMB_DIR / "train_audio_embeddings.npy").astype(np.float32)
    ta_lab = np.load(EMB_DIR / "train_audio_labels.npy").astype(np.float32)
    ss_emb = np.load(EMB_DIR / "soundscape_embeddings.npy").astype(np.float32)
    ss_lab = np.load(EMB_DIR / "soundscape_labels.npy").astype(np.float32)
    meta = pd.read_parquet(EMB_DIR / "train_audio_meta.parquet")

    # Host confirmed exact 2x duplicate in soundscape labels (1478 → 739 unique)
    ss_emb = ss_emb[:len(ss_emb) // 2]
    ss_lab = ss_lab[:len(ss_lab) // 2]

    n_ta = ta_emb.shape[0]

    X = np.concatenate([ta_emb, ss_emb], axis=0)
    Y = np.concatenate([ta_lab, ss_lab], axis=0)
    G = np.concatenate([meta["filename"].astype(str).to_numpy(),
                        np.array([f"__ss_{i}__" for i in range(ss_lab.shape[0])])], axis=0)

    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X, Y, groups=G))

    # va_idx points into X. Samples with index < n_ta are train_audio; >= n_ta are soundscape.
    va_ta_mask = va_idx < n_ta
    va_ss_mask = ~va_ta_mask

    # Now, in the subsetted val arrays, positions 0..len(va_idx)-1 correspond to va_idx in order.
    val_positions = np.arange(len(va_idx))
    ta_val_local = val_positions[va_ta_mask]
    ss_val_local = val_positions[va_ss_mask]

    X_tr = X[tr_idx]; Y_tr = Y[tr_idx]
    X_va = X[va_idx]; Y_va = Y[va_idx]

    print(f"[split] train={len(tr_idx)} val={len(va_idx)}  "
          f"val_ta={len(ta_val_local)} val_ss={len(ss_val_local)}", flush=True)
    print(f"[split] train_audio val pos_frac={Y_va[ta_val_local].mean():.5f}", flush=True)
    print(f"[split] soundscape val pos_frac={Y_va[ss_val_local].mean():.5f}", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)

    bce = nn.BCEWithLogitsLoss()

    results = []
    # Need fresh seed for each variant to match original script's behavior
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    results.append(train_variant("A_linear_bce", LinearProbe, bce, train_loader,
                                  X_va, Y_va, ta_val_local, ss_val_local))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    results.append(train_variant("B_mlp_bce", MLPHead, bce, train_loader,
                                  X_va, Y_va, ta_val_local, ss_val_local))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    results.append(train_variant("C_mlp_focal", MLPHead, focal_bce, train_loader,
                                  X_va, Y_va, ta_val_local, ss_val_local))

    payload = {
        "domain_split": {
            "n_train_audio_val": int(len(ta_val_local)),
            "n_soundscape_val": int(len(ss_val_local)),
            "ta_pos_frac": float(Y_va[ta_val_local].mean()),
            "sc_pos_frac": float(Y_va[ss_val_local].mean()),
        },
        "variants": results,
    }
    out_path = OUT_DIR / "domain_auc_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] {out_path}", flush=True)

    # summary table
    print("\n| Variant | Params | Val AUC | TA AUC (train_audio) | SC AUC (soundscape) |")
    print("|---------|--------|---------|----------------------|---------------------|")
    for r in results:
        print(f"| {r['name']} | {r['params']:,} | {r['best_val_auc']:.4f} | "
              f"{r['best_ta_auc']:.4f} ({r['best_ta_n_classes']} cls) | "
              f"{r['best_sc_auc']:.4f} ({r['best_sc_n_classes']} cls) |")


if __name__ == "__main__":
    main()
