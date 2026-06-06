"""
BirdCLEF+ 2026 — ProtoSSM Chain experiment (isolated branch).

Two variants ablating prototype-based classification on Perch v2 embeddings:

  F_proto_nonparam:
      Class-mean prototypes + cosine similarity. Zero learnable parameters.
      Based on Kaggle "ProtoSSM" approach (imaadmahmood, LB 0.925).

  G_proto_learned:
      J=10 learnable prototypes per class, projected to 256-dim,
      non-negative linear aggregation, Asymmetric Loss + orthogonality reg.
      Based on Bird-MAE PPNet (arXiv 2504.12880).

Outputs land in c:\\birdCLEF\\exp_proto_ssm\\ (isolated from main pipeline).
Reads embedding/ data (read-only). Uses same split as main pipeline.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
EMB_DIR = Path(r"C:\birdCLEF\embedding")
OUT_DIR = Path(r"C:\birdCLEF\exp_proto_ssm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 234
EMB_DIM = 1536
PROJ_DIM = 256
J_PER_CLASS = 10

EPOCHS = 40
BATCH_SIZE = 512
LR = 3e-4
PROTO_LR = 0.04
PATIENCE = 8
VAL_FRAC = 0.15

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] torch={torch.__version__} device={DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[env] gpu={torch.cuda.get_device_name(0)}", flush=True)


# ---------------------------------------------------------------------------
# Data loading (same as main pipeline)
# ---------------------------------------------------------------------------
def load_data():
    print("[data] loading embeddings...", flush=True)
    ta_emb = np.load(EMB_DIR / "train_audio_embeddings.npy").astype(np.float32)
    ta_lab = np.load(EMB_DIR / "train_audio_labels.npy").astype(np.float32)
    ss_emb = np.load(EMB_DIR / "soundscape_embeddings.npy").astype(np.float32)
    ss_lab = np.load(EMB_DIR / "soundscape_labels.npy").astype(np.float32)
    meta = pd.read_parquet(EMB_DIR / "train_audio_meta.parquet")

    ss_emb = ss_emb[: len(ss_emb) // 2]
    ss_lab = ss_lab[: len(ss_lab) // 2]

    ta_groups = meta["filename"].astype(str).to_numpy()
    ss_groups = np.array([f"__ss_{i}__" for i in range(ss_lab.shape[0])])

    X = np.concatenate([ta_emb, ss_emb], axis=0)
    Y = np.concatenate([ta_lab, ss_lab], axis=0)
    G = np.concatenate([ta_groups, ss_groups], axis=0)
    print(f"[data] X={X.shape} Y={Y.shape} groups={len(np.unique(G))}", flush=True)
    return X, Y, G


def split_data(X, Y, G):
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X, Y, groups=G))
    return X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def compute_macro_auc(probs, tgts):
    per_class = np.full(N_CLASSES, np.nan)
    for c in range(N_CLASSES):
        if 0 < tgts[:, c].sum() < tgts.shape[0]:
            try:
                per_class[c] = roc_auc_score(tgts[:, c], probs[:, c])
            except Exception:
                pass
    return float(np.nanmean(per_class)), per_class


# ===========================================================================
# Variant F: Non-parametric prototype (class means + cosine sim)
# ===========================================================================
def run_nonparam_proto(X_tr, Y_tr, X_va, Y_va):
    print("\n=== F_proto_nonparam (class-mean prototypes) ===", flush=True)
    t0 = time.time()

    prototypes = np.zeros((N_CLASSES, EMB_DIM), dtype=np.float32)
    for c in range(N_CLASSES):
        mask = Y_tr[:, c] > 0
        if mask.sum() > 0:
            prototypes[c] = X_tr[mask].mean(axis=0)

    norms = np.linalg.norm(prototypes, axis=1, keepdims=True).clip(1e-8)
    prototypes = prototypes / norms

    X_va_norm = X_va / np.linalg.norm(X_va, axis=1, keepdims=True).clip(1e-8)
    scores = X_va_norm @ prototypes.T

    probs = 1.0 / (1.0 + np.exp(-scores * 10.0))

    macro_auc, per_class = compute_macro_auc(probs, Y_va)
    dur = time.time() - t0

    print(f"[F_proto_nonparam] val_auc={macro_auc:.4f} time={dur:.1f}s", flush=True)

    order = np.argsort(-np.where(np.isnan(per_class), -np.inf, per_class))
    top5 = [(int(i), float(per_class[i])) for i in order[:5] if not np.isnan(per_class[i])]
    valid = ~np.isnan(per_class)
    bot_order = np.argsort(np.where(valid, per_class, np.inf))
    bot5 = [(int(i), float(per_class[i])) for i in bot_order[:5] if not np.isnan(per_class[i])]

    return {
        "name": "F_proto_nonparam",
        "params": 0,
        "best_val_auc": macro_auc,
        "train_time_sec": dur,
        "history": {"val_auc": [macro_auc]},
        "top5_classes": top5,
        "bottom5_classes": bot5,
    }


# ===========================================================================
# Variant G: Learned PPNet-style prototypical head
# ===========================================================================
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (from Bird-MAE / ASL paper)."""

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1 - probs

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = xs_pos * targets
            pt_neg = xs_neg * (1 - targets)
            one_sided_gamma_pos = torch.pow(1 - pt_pos, self.gamma_pos)
            one_sided_gamma_neg = torch.pow(1 - pt_neg, self.gamma_neg)
            loss_pos = loss_pos * one_sided_gamma_pos
            loss_neg = loss_neg * one_sided_gamma_neg

        return -(loss_pos + loss_neg).mean()


class PPNetHead(nn.Module):
    """
    Learned prototypical head inspired by Bird-MAE PPNet.

    Architecture:
        input (1536) → Linear(1536, 256) → LayerNorm → ReLU
        → cosine_similarity with J*C prototypes (each 256-dim)
        → reshape to (B, C, J) → non-negative weighted sum → logits (B, C)

    Total params ≈ 1536*256 + 234*10*256 + 234*10 + 234 ≈ 995k
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        proj_dim: int = PROJ_DIM,
        n_classes: int = N_CLASSES,
        j_per_class: int = J_PER_CLASS,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.J = j_per_class

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )

        total_protos = n_classes * j_per_class
        self.prototypes = nn.Parameter(torch.randn(total_protos, proj_dim) * 0.02)

        self.class_weights = nn.Parameter(torch.ones(n_classes, j_per_class))
        self.bias = nn.Parameter(torch.full((n_classes,), -2.0))

    def forward(self, x):
        h = self.proj(x)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = h_norm @ p_norm.T
        sim = sim.view(-1, self.n_classes, self.J)
        w = F.relu(self.class_weights)
        logits = (sim * w).sum(dim=-1) + self.bias
        return logits

    def orthogonality_loss(self):
        """Encourage prototypes within each class to be orthogonal."""
        p_norm = F.normalize(self.prototypes, dim=-1)
        p_grouped = p_norm.view(self.n_classes, self.J, -1)
        gram = torch.bmm(p_grouped, p_grouped.transpose(1, 2))
        eye = torch.eye(self.J, device=gram.device).unsqueeze(0)
        return ((gram - eye) ** 2).mean()


@torch.no_grad()
def evaluate_learned(model, loader):
    model.eval()
    all_logits, all_tgts = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.cpu().numpy())
        all_tgts.append(yb.numpy())
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))
    tgts = np.concatenate(all_tgts)
    macro_auc, per_class = compute_macro_auc(probs, tgts)
    return macro_auc, per_class


def run_learned_proto(X_tr, Y_tr, X_va, Y_va):
    print(f"\n=== G_proto_learned (J={J_PER_CLASS}/class, proj={PROJ_DIM}) ===", flush=True)

    pin = DEVICE.type == "cuda"
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                            num_workers=0, pin_memory=pin)

    model = PPNetHead().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[G] params={n_params:,}", flush=True)

    proto_params = [model.prototypes]
    other_params = [p for n, p in model.named_parameters() if n != "prototypes"]

    opt = torch.optim.AdamW([
        {"params": proto_params, "lr": PROTO_LR},
        {"params": other_params, "lr": LR},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    ortho_weight = 0.01

    history = {"train_loss": [], "val_auc": []}
    best_auc = -1.0
    best_state = None
    best_per_class = None
    patience = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"G ep{epoch:02d}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            cls_loss = loss_fn(logits, yb)
            ortho_loss = model.orthogonality_loss()
            loss = cls_loss + ortho_weight * ortho_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        tr_loss = running / max(n_seen, 1)

        val_auc, per_class = evaluate_learned(model, val_loader)
        history["train_loss"].append(tr_loss)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_auc
        flag = "*" if improved else " "
        print(f"[G] ep{epoch:02d} tr={tr_loss:.4f} auc={val_auc:.4f} {flag}", flush=True)

        if improved:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_per_class = per_class.copy()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[G] early stop at epoch {epoch}", flush=True)
                break

    dur = time.time() - t0

    order = np.argsort(-np.where(np.isnan(best_per_class), -np.inf, best_per_class))
    top5 = [(int(i), float(best_per_class[i])) for i in order[:5] if not np.isnan(best_per_class[i])]
    valid = ~np.isnan(best_per_class)
    bot_order = np.argsort(np.where(valid, best_per_class, np.inf))
    bot5 = [(int(i), float(best_per_class[i])) for i in bot_order[:5] if not np.isnan(best_per_class[i])]

    return {
        "name": "G_proto_learned",
        "params": int(n_params),
        "best_val_auc": float(best_auc),
        "train_time_sec": float(dur),
        "history": history,
        "top5_classes": top5,
        "bottom5_classes": bot5,
        "best_state": best_state,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    X, Y, G = load_data()
    X_tr, Y_tr, X_va, Y_va = split_data(X, Y, G)
    print(f"[split] train={X_tr.shape[0]} val={X_va.shape[0]}", flush=True)

    results = []

    # F: Non-parametric prototype
    results.append(run_nonparam_proto(X_tr, Y_tr, X_va, Y_va))

    # G: Learned PPNet head
    g_result = run_learned_proto(X_tr, Y_tr, X_va, Y_va)
    results.append(g_result)

    # Save best learned model
    if g_result["best_state"] is not None:
        torch.save(g_result["best_state"], OUT_DIR / "ppnet_best.pt")
        print(f"\n[save] ppnet_best.pt -> {OUT_DIR / 'ppnet_best.pt'}", flush=True)

    # Training curves
    plt.figure(figsize=(8, 5))
    for r in results:
        if len(r["history"]["val_auc"]) > 1:
            ep = range(1, len(r["history"]["val_auc"]) + 1)
            plt.plot(ep, r["history"]["val_auc"], label=r["name"], marker="o", markersize=3)
        else:
            plt.axhline(y=r["history"]["val_auc"][0], label=r["name"], linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Val macro AUC")
    plt.title("ProtoSSM Chain — Prototype Head Ablation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "proto_curves.png", dpi=130)
    plt.close()
    print(f"[save] proto_curves.png", flush=True)

    # Results JSON
    serializable = []
    for r in results:
        serializable.append({
            "name": r["name"],
            "params": r["params"],
            "best_val_auc": r["best_val_auc"],
            "train_time_sec": r["train_time_sec"],
            "epochs_run": len(r["history"]["val_auc"]),
            "history": r["history"],
            "top5_classes": r["top5_classes"],
            "bottom5_classes": r["bottom5_classes"],
        })
    payload = {
        "experiment": "EXP-005 ProtoSSM Chain",
        "architecture": {
            "F_proto_nonparam": "class-mean prototypes + cosine sim, scale=10",
            "G_proto_learned": f"PPNet J={J_PER_CLASS}/class, proj {EMB_DIM}->{PROJ_DIM}, "
                               f"non-neg linear, AsymmetricLoss(γ-=4,γ+=1), ortho_reg=0.01",
        },
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "proto_lr": PROTO_LR,
            "patience": PATIENCE, "seed": SEED,
            "j_per_class": J_PER_CLASS, "proj_dim": PROJ_DIM,
        },
        "baseline_comparison": {
            "B_mlp_bce": 0.9646,
            "C_mlp_focal": 0.9633,
        },
        "best_overall": max(serializable, key=lambda v: v["best_val_auc"])["name"],
        "variants": serializable,
    }
    with open(OUT_DIR / "proto_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] proto_results.json", flush=True)

    # Summary
    print("\n\n## ProtoSSM Ablation Summary\n", flush=True)
    print("| Variant | Params | Time (s) | Val macro AUC |", flush=True)
    print("|---------|--------|----------|---------------|", flush=True)
    for r in results:
        print(f"| {r['name']} | {r['params']:,} | {r['train_time_sec']:.1f} | "
              f"{r['best_val_auc']:.4f} |", flush=True)

    best = max(results, key=lambda r: r["best_val_auc"])
    print(f"\nBest: **{best['name']}** (val AUC = {best['best_val_auc']:.4f})", flush=True)
    print(f"\nBaseline reference: B_mlp_bce = 0.9646, C_mlp_focal = 0.9633", flush=True)


if __name__ == "__main__":
    main()
