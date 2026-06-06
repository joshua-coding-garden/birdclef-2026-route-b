"""
BirdCLEF+ 2026 — Knowledge distillation: Perch v2 (teacher) → HGNetV2_b0 (student).

Teacher: trained MLP head on Perch embeddings (val AUC 0.9646), soft sigmoid outputs.
Student: HGNetV2_b0 on mel spectrograms, end-to-end from raw audio.
Loss: BCE(student_logits, teacher_soft_labels).

Outputs:
  - C:\\birdCLEF\\new\\hgnet_best.pt
  - C:\\birdCLEF\\new\\distill_results.json
  - C:\\birdCLEF\\new\\distill_curves.png
"""
from __future__ import annotations

import os
os.environ["HF_HOME"] = r"C:\birdCLEF\cache\huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SEED = 42
AUDIO_DIR = Path(r"C:\birdCLEF\birdclef-2026\train_audio")
EMB_DIR = Path(r"C:\birdCLEF\embedding")
OUT_DIR = Path(r"C:\birdCLEF\new")

N_CLASSES = 234
SR = 32000
FRAME_SEC = 5
FRAME_LEN = SR * FRAME_SEC  # 160000

# Mel spectrogram
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 50
F_MAX = 16000
IMG_SIZE = 224

# Training
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 5
VAL_FRAC = 0.15
MIXUP_ALPHA = 0.4
NUM_WORKERS = 4

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] torch={torch.__version__} device={DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[env] gpu={torch.cuda.get_device_name(0)}", flush=True)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class AudioFrameDataset(Dataset):
    """Loads 5-sec audio frames on-the-fly with seek-based reading."""

    def __init__(self, filenames, frame_indices, labels, audio_dir):
        self.filenames = filenames
        self.frame_indices = frame_indices
        self.labels = labels
        self.audio_dir = Path(audio_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        frame_idx = int(self.frame_indices[idx])
        label = self.labels[idx]

        start_sample = frame_idx * FRAME_LEN
        stop_sample = start_sample + FRAME_LEN

        fp = self.audio_dir / fn
        try:
            info = sf.info(str(fp))
            total_samples = info.frames

            if start_sample >= total_samples:
                waveform = np.zeros(FRAME_LEN, dtype=np.float32)
            elif stop_sample > total_samples:
                audio, _ = sf.read(str(fp), start=start_sample, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                waveform = np.zeros(FRAME_LEN, dtype=np.float32)
                waveform[: len(audio)] = audio
            else:
                audio, _ = sf.read(
                    str(fp), start=start_sample, stop=stop_sample, dtype="float32"
                )
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                waveform = audio
        except Exception:
            waveform = np.zeros(FRAME_LEN, dtype=np.float32)

        return torch.from_numpy(waveform), torch.from_numpy(label.astype(np.float32))


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class HGNetStudent(nn.Module):
    """HGNetV2_b0 student with on-GPU mel spectrogram."""

    def __init__(self, n_classes=N_CLASSES, pretrained=True):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
        )
        self.backbone = timm.create_model(
            "hgnetv2_b0.ssld_stage2_ft_in1k",
            pretrained=pretrained,
            in_chans=1,
            num_classes=n_classes,
        )

    def forward(self, waveform):
        # waveform: (B, FRAME_LEN)
        mel = self.mel_spec(waveform)  # (B, N_MELS, T)
        mel = torch.log(mel + 1e-8)
        # per-sample normalization
        mel_flat = mel.reshape(mel.size(0), -1)
        mean = mel_flat.mean(dim=1, keepdim=True).unsqueeze(-1)
        std = mel_flat.std(dim=1, keepdim=True).unsqueeze(-1)
        mel = (mel - mean) / (std + 1e-8)
        mel = mel.unsqueeze(1)  # (B, 1, N_MELS, T)
        mel = F.interpolate(mel, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return self.backbone(mel)


# -----------------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, hard_labels_val):
    model.eval()
    all_logits = []
    for waveform, _ in loader:
        waveform = waveform.to(DEVICE, non_blocking=True)
        logits = model(waveform)
        all_logits.append(logits.cpu().numpy())

    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits, axis=0)))
    tgts = hard_labels_val

    per_class_auc = np.full(N_CLASSES, np.nan, dtype=np.float64)
    for c in range(N_CLASSES):
        if 0 < tgts[:, c].sum() < tgts.shape[0]:
            try:
                per_class_auc[c] = roc_auc_score(tgts[:, c], probs[:, c])
            except Exception:
                pass
    macro_auc = float(np.nanmean(per_class_auc))
    n_valid = int((~np.isnan(per_class_auc)).sum())
    return macro_auc, n_valid


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def main():
    # --- Load metadata and labels ---
    meta = pd.read_parquet(EMB_DIR / "train_audio_meta.parquet")
    hard_labels = np.load(EMB_DIR / "train_audio_labels.npy").astype(np.float32)
    soft_labels = np.load(EMB_DIR / "teacher_soft_labels.npy").astype(np.float32)

    # soft_labels includes soundscape rows at the end; take only train_audio portion
    n_ta = len(meta)
    soft_labels = soft_labels[:n_ta]

    assert len(meta) == hard_labels.shape[0] == soft_labels.shape[0]
    print(f"[data] train_audio frames={n_ta}, classes={N_CLASSES}", flush=True)

    filenames = meta["filename"].astype(str).to_numpy()
    frame_indices = meta["frame_idx"].to_numpy()
    groups = filenames  # same as other scripts

    # --- Split (same GroupShuffleSplit as baseline for comparable validation) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_idx, va_idx = next(gss.split(np.arange(n_ta), hard_labels, groups=groups))

    print(
        f"[split] train={len(tr_idx)} val={len(va_idx)} "
        f"(GroupShuffleSplit, test_size={VAL_FRAC}, seed={SEED})",
        flush=True,
    )

    # --- Datasets ---
    train_ds = AudioFrameDataset(
        filenames[tr_idx], frame_indices[tr_idx], soft_labels[tr_idx], AUDIO_DIR
    )
    val_ds = AudioFrameDataset(
        filenames[va_idx], frame_indices[va_idx], soft_labels[va_idx], AUDIO_DIR
    )
    hard_labels_val = hard_labels[va_idx]

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- Model ---
    model = HGNetStudent(pretrained=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] HGNetV2_b0 student, params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()

    # --- Training loop ---
    history = {"train_loss": [], "val_auc": []}
    best_auc = -1.0
    best_state = None
    patience_count = 0
    t0_total = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        t0_epoch = time.time()

        pbar = tqdm(train_loader, desc=f"ep{epoch:02d}", leave=False)
        for waveform, soft_target in pbar:
            waveform = waveform.to(DEVICE, non_blocking=True)
            soft_target = soft_target.to(DEVICE, non_blocking=True)

            # MixUp on waveforms
            if MIXUP_ALPHA > 0:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                idx = torch.randperm(waveform.size(0), device=DEVICE)
                waveform = lam * waveform + (1 - lam) * waveform[idx]
                soft_target = lam * soft_target + (1 - lam) * soft_target[idx]

            opt.zero_grad()
            logits = model(waveform)
            loss = loss_fn(logits, soft_target)
            loss.backward()
            opt.step()

            running_loss += loss.item() * waveform.size(0)
            n_seen += waveform.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        sched.step()
        tr_loss = running_loss / max(n_seen, 1)
        epoch_time = time.time() - t0_epoch

        # Evaluate on hard labels (for comparable AUC with baseline)
        val_auc, val_nc = evaluate(model, val_loader, hard_labels_val)

        history["train_loss"].append(tr_loss)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_auc
        flag = "*" if improved else " "
        print(
            f"[ep{epoch:02d}] tr_loss={tr_loss:.4f} val_auc={val_auc:.4f} "
            f"({val_nc} cls) time={epoch_time:.0f}s {flag}",
            flush=True,
        )

        if improved:
            best_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"[early stop] no improve for {PATIENCE} epochs", flush=True)
                break

    total_time = time.time() - t0_total
    print(
        f"\n[done] best_val_auc={best_auc:.4f} total_time={total_time:.0f}s",
        flush=True,
    )

    # --- Save ---
    torch.save(best_state, OUT_DIR / "hgnet_best.pt")
    print(f"[save] {OUT_DIR / 'hgnet_best.pt'}", flush=True)

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ep_range = range(1, len(history["val_auc"]) + 1)
    ax1.plot(ep_range, history["val_auc"], marker="o", markersize=3)
    ax1.set(xlabel="Epoch", ylabel="Val macro AUC", title="HGNetV2 Distillation — Val AUC")
    ax1.grid(alpha=0.3)
    ax2.plot(ep_range, history["train_loss"], marker="o", markersize=3, color="orange")
    ax2.set(xlabel="Epoch", ylabel="Train loss (BCE vs soft labels)", title="Train Loss")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distill_curves.png", dpi=130)
    plt.close(fig)
    print(f"[save] {OUT_DIR / 'distill_curves.png'}", flush=True)

    # Results JSON
    payload = {
        "experiment": "EXP-003: Perch v2 -> HGNetV2 distillation",
        "env": {
            "torch": torch.__version__,
            "timm": timm.__version__,
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        },
        "model": {
            "backbone": "hgnetv2_b0.ssld_stage2_ft_in1k",
            "in_chans": 1,
            "params": n_params,
        },
        "mel_spec": {
            "sr": SR,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "f_min": F_MIN,
            "f_max": F_MAX,
            "img_size": IMG_SIZE,
        },
        "training": {
            "teacher": "mlp_best.pt (B_mlp_bce, val AUC 0.9646)",
            "loss": "BCE(student_logits, teacher_soft_labels)",
            "optimizer": "AdamW",
            "lr": LR,
            "epochs_max": EPOCHS,
            "epochs_run": len(history["val_auc"]),
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "mixup_alpha": MIXUP_ALPHA,
            "val_frac": VAL_FRAC,
            "seed": SEED,
        },
        "data": {
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "source": "train_audio only (no soundscapes)",
        },
        "results": {
            "best_val_auc": float(best_auc),
            "total_time_sec": float(total_time),
        },
        "history": history,
    }
    with open(OUT_DIR / "distill_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] {OUT_DIR / 'distill_results.json'}", flush=True)

    # Comparison with baseline
    print("\n## Comparison with MLP baseline")
    print("| Model | Val AUC | Params | Notes |")
    print("|-------|---------|--------|-------|")
    print(f"| MLP (B_mlp_bce) | 0.9646 | 1,573,098 | Perch embedding input |")
    print(f"| HGNetV2_b0 distilled | {best_auc:.4f} | {n_params:,} | Raw audio input |")


if __name__ == "__main__":
    main()
