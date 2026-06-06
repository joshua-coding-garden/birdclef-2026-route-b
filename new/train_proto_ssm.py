"""
BirdCLEF+ 2026 — ProtoSSM Chain on Perch v2 embeddings.

Architecture (per sequence of 5-sec clips from the same source file):

  Perch emb (1536) → Projection (512) + LayerNorm
    → Diagonal SSM (S4D, d_state=64) + residual + LayerNorm
    → Prototypical Head (scaled cosine to 234 learnable prototypes)

Variants:
  D_proto_ssm_bce   — ProtoSSM Chain + BCEWithLogitsLoss
  E_proto_ssm_focal — ProtoSSM Chain + Focal BCE (γ=2.0, α=0.25)

Training: clips grouped by source file into temporal sequences.
Evaluation: per-clip predictions → macro AUC (skip-empty classes).

Outputs:
  - C:\\birdCLEF\\new\\proto_ssm_results.json
  - C:\\birdCLEF\\new\\proto_ssm_curves.png
  - C:\\birdCLEF\\new\\proto_ssm_best.pt
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
EMB_DIR = Path(r"C:\birdCLEF\embedding")
OUT_DIR = Path(r"C:\birdCLEF\new")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 234
EMB_DIM = 1536
SSM_DIM = 512
D_STATE = 64

EPOCHS = 40
BATCH_SIZE = 64          # sequences per batch
LR = 3e-4
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
# Data loading — sequence-level
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

    assert ta_emb.shape == (ta_lab.shape[0], EMB_DIM)
    assert ta_lab.shape[1] == N_CLASSES

    ta_groups = meta["filename"].astype(str).to_numpy()
    ss_groups = np.array([f"__ss_{i}__" for i in range(ss_lab.shape[0])])

    X = np.concatenate([ta_emb, ss_emb], axis=0)
    Y = np.concatenate([ta_lab, ss_lab], axis=0)
    G = np.concatenate([ta_groups, ss_groups], axis=0)

    print(
        f"[data] X={X.shape} Y={Y.shape} "
        f"groups={len(np.unique(G))} "
        f"(ta_files={len(np.unique(ta_groups))}, ss_segs={len(ss_groups)})",
        flush=True,
    )
    return X, Y, G


def split_data(X, Y, G):
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X, Y, groups=G))

    X_tr, Y_tr, G_tr = X[tr_idx], Y[tr_idx], G[tr_idx]
    X_va, Y_va, G_va = X[va_idx], Y[va_idx], G[va_idx]

    val_pos = Y_va.sum(axis=0)
    n_cls_pos = int((val_pos > 0).sum())

    print(
        f"[split] train_clips={len(tr_idx)} val_clips={len(va_idx)} "
        f"val_classes_with_pos={n_cls_pos}/{N_CLASSES}",
        flush=True,
    )
    return X_tr, Y_tr, G_tr, X_va, Y_va, G_va, n_cls_pos


# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """Groups clip embeddings by source file to form temporal sequences."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, groups: np.ndarray):
        group_map: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            group_map[g].append(i)

        self.seqs: list[np.ndarray] = []
        self.labs: list[np.ndarray] = []
        for g in sorted(group_map.keys()):
            idx = sorted(group_map[g])
            self.seqs.append(embeddings[idx])   # (seq_len, 1536)
            self.labs.append(labels[idx])        # (seq_len, 234)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labs[i]


def collate_sequences(batch):
    """Pad sequences to max length in batch; return (seqs, labels, lengths)."""
    seqs, labs = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    bsz = len(seqs)

    s_pad = torch.zeros(bsz, max_len, EMB_DIM)
    l_pad = torch.zeros(bsz, max_len, N_CLASSES)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        n = s.shape[0]
        s_pad[i, :n] = torch.from_numpy(s) if isinstance(s, np.ndarray) else s
        l_pad[i, :n] = torch.from_numpy(l) if isinstance(l, np.ndarray) else l

    return s_pad, l_pad, lengths


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class DiagonalSSM(nn.Module):
    """
    Diagonal State Space Model (S4D-like).

    Discretized recurrence (ZOH):
      x_t = Ā x_{t-1} + B̄ u_t
      y_t = C x_t + D u_t

    where Ā = exp(A · dt), B̄ = (Ā - I) / A · B, and A is diagonal & negative
    (stability guaranteed).
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # HiPPO-style initialization: A_n = -n
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A_init).unsqueeze(0).expand(d_model, -1).clone()
        )
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.01 - 4.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (batch, seq_len, d_model) → y: (batch, seq_len, d_model)"""
        batch, seq_len, _ = u.shape

        dt = F.softplus(self.log_dt)                 # (d_model,)
        A = -torch.exp(self.A_log)                   # (d_model, d_state)
        dtA = A * dt.unsqueeze(-1)                   # (d_model, d_state)
        A_bar = torch.exp(dtA)                       # (d_model, d_state)
        B_bar = (A_bar - 1.0) / A * self.B           # (d_model, d_state)

        x = u.new_zeros(batch, self.d_model, self.d_state)
        ys = []
        for t in range(seq_len):
            ut = u[:, t, :]                          # (batch, d_model)
            x = A_bar * x + B_bar * ut.unsqueeze(-1) # (batch, d_model, d_state)
            y = (self.C * x).sum(-1) + self.D * ut   # (batch, d_model)
            ys.append(y)

        return self.dropout(torch.stack(ys, dim=1))


class PrototypicalHead(nn.Module):
    """Learnable prototypes with scaled cosine similarity → logits."""

    def __init__(self, d_model: int, n_classes: int, init_temp: float = 10.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d_model) → logits: (N, n_classes)"""
        x_n = F.normalize(x, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        return torch.exp(self.log_temp) * (x_n @ p_n.T)


class ProtoSSMChain(nn.Module):
    """
    Full chain: Projection → SSM → Prototypical Head.

    For inference, feed a soundscape's 12 clips as one sequence — the SSM
    gives each clip temporal context from its neighbours.
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        ssm_dim: int = SSM_DIM,
        d_state: int = D_STATE,
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj = nn.Linear(emb_dim, ssm_dim)
        self.norm1 = nn.LayerNorm(ssm_dim)
        self.ssm = DiagonalSSM(ssm_dim, d_state, dropout=0.1)
        self.norm2 = nn.LayerNorm(ssm_dim)
        self.drop = nn.Dropout(dropout)
        self.head = PrototypicalHead(ssm_dim, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        x:       (batch, max_seq_len, emb_dim)
        lengths: (batch,)  — actual clip counts per sequence
        Returns: (total_clips, n_classes) logits, one row per valid clip.
        """
        h = self.norm1(self.proj(x))
        h = h + self.ssm(h)
        h = self.drop(self.norm2(h))

        if lengths is not None:
            clips = [h[i, : lengths[i]] for i in range(h.size(0))]
            h_flat = torch.cat(clips, dim=0)
        else:
            h_flat = h.reshape(-1, h.size(-1))

        return self.head(h_flat)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    all_logits, all_tgts, losses = [], [], []
    for s_pad, l_pad, lengths in loader:
        s_pad = s_pad.to(DEVICE, non_blocking=True)
        l_pad = l_pad.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE, non_blocking=True)

        logits = model(s_pad, lengths)

        tgts_flat = torch.cat(
            [l_pad[i, : lengths[i]] for i in range(l_pad.size(0))], dim=0
        )
        loss = loss_fn(logits, tgts_flat)
        losses.append(loss.item() * logits.size(0))
        all_logits.append(logits.cpu().numpy())
        all_tgts.append(tgts_flat.cpu().numpy())

    total = sum(x.shape[0] for x in all_logits)
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))
    tgts = np.concatenate(all_tgts)

    per_class = np.full(N_CLASSES, np.nan)
    for c in range(N_CLASSES):
        if 0 < tgts[:, c].sum() < tgts.shape[0]:
            try:
                per_class[c] = roc_auc_score(tgts[:, c], probs[:, c])
            except Exception:
                pass
    macro_auc = float(np.nanmean(per_class))
    avg_loss = sum(losses) / max(total, 1)
    return avg_loss, macro_auc, per_class


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_variant(name, model, loss_fn, train_loader, val_loader):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
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
        running, n_clips = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{name} ep{epoch:02d}", leave=False)
        for s_pad, l_pad, lengths in pbar:
            s_pad = s_pad.to(DEVICE, non_blocking=True)
            l_pad = l_pad.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)

            logits = model(s_pad, lengths)
            tgts_flat = torch.cat(
                [l_pad[i, : lengths[i]] for i in range(l_pad.size(0))], dim=0
            )

            loss = loss_fn(logits, tgts_flat)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            nc = logits.size(0)
            running += loss.item() * nc
            n_clips += nc
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        sched.step()
        tr_loss = running / max(n_clips, 1)

        val_loss, val_auc, per_class = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_auc
        flag = "*" if improved else " "
        print(
            f"[{name}] ep{epoch:02d} tr={tr_loss:.4f} val={val_loss:.4f} "
            f"auc={val_auc:.4f} {flag}",
            flush=True,
        )

        if improved:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_per_class = per_class.copy()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(
                    f"[{name}] early stop at epoch {epoch} (no improve {PATIENCE} ep).",
                    flush=True,
                )
                break

    dur = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    pc = best_per_class
    order = np.argsort(-np.where(np.isnan(pc), -np.inf, pc))
    top5 = [(int(i), float(pc[i])) for i in order[:5] if not np.isnan(pc[i])]
    valid = ~np.isnan(pc)
    bot_order = np.argsort(np.where(valid, pc, np.inf))
    bot5 = [(int(i), float(pc[i])) for i in bot_order[:5] if not np.isnan(pc[i])]

    return {
        "name": name,
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
    X_tr, Y_tr, G_tr, X_va, Y_va, G_va, n_cls_pos = split_data(X, Y, G)

    train_ds = SequenceDataset(X_tr, Y_tr, G_tr)
    val_ds = SequenceDataset(X_va, Y_va, G_va)

    print(
        f"[seqs] train={len(train_ds)} val={len(val_ds)} sequences",
        flush=True,
    )

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
        pin_memory=pin,
    )

    bce = nn.BCEWithLogitsLoss()

    variants = []

    # D: ProtoSSM + BCE
    variants.append(
        train_variant(
            "D_proto_ssm_bce",
            ProtoSSMChain(),
            bce,
            train_loader,
            val_loader,
        )
    )

    # E: ProtoSSM + Focal
    variants.append(
        train_variant(
            "E_proto_ssm_focal",
            ProtoSSMChain(),
            focal_bce,
            train_loader,
            val_loader,
        )
    )

    # ---------- save best model ----------
    best = max(variants, key=lambda v: v["best_val_auc"])
    torch.save(best["best_state"], OUT_DIR / "proto_ssm_best.pt")
    print(
        f"\n[save] best={best['name']} auc={best['best_val_auc']:.4f} "
        f"-> {OUT_DIR / 'proto_ssm_best.pt'}",
        flush=True,
    )

    # ---------- training curves ----------
    plt.figure(figsize=(8, 5))
    for v in variants:
        ep = range(1, len(v["history"]["val_auc"]) + 1)
        plt.plot(ep, v["history"]["val_auc"], label=v["name"], marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Val macro AUC (skip-empty)")
    plt.title("BirdCLEF+ 2026 — ProtoSSM Chain ablation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "proto_ssm_curves.png", dpi=130)
    plt.close()
    print(f"[save] curves -> {OUT_DIR / 'proto_ssm_curves.png'}", flush=True)

    # ---------- results JSON ----------
    serializable = []
    for v in variants:
        serializable.append(
            {
                "name": v["name"],
                "params": v["params"],
                "best_val_auc": v["best_val_auc"],
                "train_time_sec": v["train_time_sec"],
                "epochs_run": len(v["history"]["val_auc"]),
                "history": v["history"],
                "top5_classes": v["top5_classes"],
                "bottom5_classes": v["bottom5_classes"],
            }
        )
    payload = {
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(DEVICE),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "architecture": {
            "ssm_dim": SSM_DIM,
            "d_state": D_STATE,
            "emb_dim": EMB_DIM,
            "n_classes": N_CLASSES,
        },
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "patience": PATIENCE,
            "val_frac": VAL_FRAC,
            "seed": SEED,
        },
        "data": {
            "n_train_clips": int(sum(l.shape[0] for l in train_ds.labs)),
            "n_val_clips": int(sum(l.shape[0] for l in val_ds.labs)),
            "n_train_seqs": len(train_ds),
            "n_val_seqs": len(val_ds),
            "val_classes_with_pos": int(n_cls_pos),
        },
        "best_overall": best["name"],
        "variants": serializable,
    }
    with open(OUT_DIR / "proto_ssm_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] results -> {OUT_DIR / 'proto_ssm_results.json'}", flush=True)

    # ---------- summary table ----------
    print("\n\n## ProtoSSM Ablation Summary\n", flush=True)
    print("| Variant | Params | Time (s) | Epochs | Val macro AUC |", flush=True)
    print("|---------|--------|----------|--------|---------------|", flush=True)
    for v in variants:
        print(
            f"| {v['name']} | {v['params']:,} | {v['train_time_sec']:.1f} | "
            f"{len(v['history']['val_auc'])} | {v['best_val_auc']:.4f} |",
            flush=True,
        )
    print(
        f"\nBest: **{best['name']}** (val AUC = {best['best_val_auc']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
