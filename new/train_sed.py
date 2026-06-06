"""
BirdCLEF+ 2026 — SED (Sound Event Detection) training on mel spectrograms.

Architecture: EfficientNet-B0 backbone + AttBlockV2 attention pooling.
Loss: 0.5 * clip-level BCE + 0.5 * max-frame BCE (standard BirdCLEF SED recipe).
References:
  - PANNs (Kong et al.): AttBlock original definition
  - BirdCLEF 2023 2nd place (LIHANG-HONG): AttBlockV2 + EfficientNet SED
  - BirdCLEF 2025 2nd place (VSydorskyy): BCEFocal2WayLoss

Outputs:
  - C:\\birdCLEF\\new\\sed_results.json
  - C:\\birdCLEF\\new\\sed_curves.png
  - C:\\birdCLEF\\new\\sed_best.pt
"""
from __future__ import annotations

import os
os.environ["HF_HOME"] = r"C:\birdCLEF\cache\huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import ast
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
DATA_ROOT = Path(r"C:\birdCLEF\birdclef-2026")
OUT_DIR = Path(r"C:\birdCLEF\new")

N_CLASSES = 234
SR = 32000
DURATION_SEC = 5
FRAME_LEN = SR * DURATION_SEC  # 160000

# Mel spectrogram
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 20
F_MAX = 16000

# Training
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 7
VAL_FRAC = 0.15
NUM_WORKERS = 4

# Augmentation
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
SPEC_FREQ_MASK = 10
SPEC_TIME_MASK = 20

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] torch={torch.__version__} device={DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[env] gpu={torch.cuda.get_device_name(0)}", flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BirdAudioDataset(Dataset):
    """Loads raw audio, crops/pads to DURATION_SEC, returns waveform + label."""

    def __init__(self, file_paths, labels, groups, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.groups = groups
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        label = self.labels[idx]

        try:
            audio, sr = sf.read(str(fp), dtype="float32")
        except Exception:
            audio = np.zeros(FRAME_LEN, dtype="float32")
            sr = SR

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (should be 32kHz already)
        if sr != SR:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio).unsqueeze(0), sr, SR
            ).squeeze(0).numpy()

        n = len(audio)
        if n < FRAME_LEN:
            # Pad short audio
            audio = np.pad(audio, (0, FRAME_LEN - n))
        elif n > FRAME_LEN and self.is_train:
            # Random crop during training
            start = np.random.randint(0, n - FRAME_LEN)
            audio = audio[start:start + FRAME_LEN]
        else:
            # Center crop for val / short files
            if n > FRAME_LEN:
                start = (n - FRAME_LEN) // 2
                audio = audio[start:start + FRAME_LEN]

        return torch.from_numpy(audio), torch.from_numpy(label)


def build_file_list():
    """Build list of (filepath, multi-hot label, group) from train.csv."""
    train_df = pd.read_csv(DATA_ROOT / "train.csv")
    taxonomy_df = pd.read_csv(DATA_ROOT / "taxonomy.csv")
    species_list = taxonomy_df["primary_label"].astype(str).tolist()
    sp2idx = {sp: i for i, sp in enumerate(species_list)}

    file_paths = []
    labels = []
    groups = []

    for _, row in train_df.iterrows():
        fp = DATA_ROOT / "train_audio" / row["filename"]
        if not fp.exists():
            continue

        label = np.zeros(N_CLASSES, dtype=np.float32)
        pl = str(row["primary_label"])
        if pl in sp2idx:
            label[sp2idx[pl]] = 1.0

        # Secondary labels
        sec = row.get("secondary_labels", "[]")
        if isinstance(sec, str) and sec != "[]":
            try:
                sec_list = ast.literal_eval(sec)
                for s in sec_list:
                    s = str(s)
                    if s in sp2idx:
                        label[sp2idx[s]] = 1.0
            except Exception:
                pass

        file_paths.append(fp)
        labels.append(label)
        groups.append(row["filename"].split("/")[0])

    # Also add soundscape labeled segments
    ss_label_path = DATA_ROOT / "train_soundscapes_labels.csv"
    if ss_label_path.exists():
        ss_df = pd.read_csv(ss_label_path)
        ss_df["start_sec"] = pd.to_timedelta(ss_df["start"]).dt.total_seconds()
        ss_df["end_sec"] = pd.to_timedelta(ss_df["end"]).dt.total_seconds()

        # Dedup: host confirmed 2x duplicate
        ss_df = ss_df.drop_duplicates(
            subset=["filename", "start", "end", "primary_label"], keep="first"
        )

        ss_audio_dir = DATA_ROOT / "train_soundscapes"
        for _, row in ss_df.iterrows():
            fp = ss_audio_dir / row["filename"]
            if not fp.exists():
                continue

            label = np.zeros(N_CLASSES, dtype=np.float32)
            lbl_str = str(row["primary_label"])
            for lbl in lbl_str.split(";"):
                lbl = lbl.strip()
                if lbl in sp2idx:
                    label[sp2idx[lbl]] = 1.0

            file_paths.append(fp)
            labels.append(label)
            groups.append(f"__ss_{row['filename']}_{row['start']}__")

    labels = np.array(labels, dtype=np.float32)
    groups = np.array(groups)

    print(
        f"[data] files={len(file_paths)} "
        f"(train_audio={len(train_df)}, soundscape={len(file_paths)-len(train_df)}) "
        f"unique_groups={len(np.unique(groups))}",
        flush=True,
    )
    return file_paths, labels, groups


class SoundscapeSegmentDataset(Dataset):
    """Loads a specific 5-sec segment from a soundscape file."""

    def __init__(self, file_path, start_sec, end_sec, label, group):
        self.file_path = file_path
        self.start_sample = int(start_sec * SR)
        self.end_sample = int(end_sec * SR)
        self.label = label
        self.group = group

    def get_audio(self):
        audio, sr = sf.read(str(self.file_path), dtype="float32",
                            start=self.start_sample,
                            stop=self.end_sample)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < FRAME_LEN:
            audio = np.pad(audio, (0, FRAME_LEN - len(audio)))
        return audio[:FRAME_LEN]


class BirdAudioDatasetV2(Dataset):
    """
    Better dataset that handles both train_audio (random crop from full file)
    and soundscape segments (read specific 5-sec window).
    """

    def __init__(self, entries, is_train=True):
        """entries: list of dict with keys: filepath, label, group, start_sample, end_sample"""
        self.entries = entries
        self.is_train = is_train

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        fp = e["filepath"]
        label = e["label"]

        try:
            if e.get("is_soundscape"):
                audio, sr = sf.read(
                    str(fp), dtype="float32",
                    start=e["start_sample"], stop=e["end_sample"]
                )
            else:
                audio, sr = sf.read(str(fp), dtype="float32")
        except Exception:
            audio = np.zeros(FRAME_LEN, dtype="float32")
            sr = SR

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        n = len(audio)
        if n < FRAME_LEN:
            audio = np.pad(audio, (0, FRAME_LEN - n))
        elif n > FRAME_LEN and self.is_train:
            start = np.random.randint(0, n - FRAME_LEN)
            audio = audio[start:start + FRAME_LEN]
        elif n > FRAME_LEN:
            start = (n - FRAME_LEN) // 2
            audio = audio[start:start + FRAME_LEN]

        return torch.from_numpy(audio.copy()), torch.from_numpy(label)


def build_entries():
    """Build list of dataset entries (dict) for train_audio + soundscape."""
    train_df = pd.read_csv(DATA_ROOT / "train.csv")
    taxonomy_df = pd.read_csv(DATA_ROOT / "taxonomy.csv")
    species_list = taxonomy_df["primary_label"].astype(str).tolist()
    sp2idx = {sp: i for i, sp in enumerate(species_list)}

    entries = []

    for _, row in train_df.iterrows():
        fp = DATA_ROOT / "train_audio" / row["filename"]
        if not fp.exists():
            continue

        label = np.zeros(N_CLASSES, dtype=np.float32)
        pl = str(row["primary_label"])
        if pl in sp2idx:
            label[sp2idx[pl]] = 1.0

        sec = row.get("secondary_labels", "[]")
        if isinstance(sec, str) and sec != "[]":
            try:
                sec_list = ast.literal_eval(sec)
                for s in sec_list:
                    s = str(s)
                    if s in sp2idx:
                        label[sp2idx[s]] = 1.0
            except Exception:
                pass

        entries.append({
            "filepath": fp,
            "label": label,
            "group": str(row["filename"]).split("/")[0],
            "is_soundscape": False,
        })

    # Soundscape segments
    ss_label_path = DATA_ROOT / "train_soundscapes_labels.csv"
    if ss_label_path.exists():
        ss_df = pd.read_csv(ss_label_path)
        ss_df["start_sec"] = pd.to_timedelta(ss_df["start"]).dt.total_seconds()
        ss_df["end_sec"] = pd.to_timedelta(ss_df["end"]).dt.total_seconds()
        ss_df = ss_df.drop_duplicates(
            subset=["filename", "start", "end", "primary_label"], keep="first"
        )

        ss_audio_dir = DATA_ROOT / "train_soundscapes"
        for _, row in ss_df.iterrows():
            fp = ss_audio_dir / row["filename"]
            if not fp.exists():
                continue

            label = np.zeros(N_CLASSES, dtype=np.float32)
            for lbl in str(row["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in sp2idx:
                    label[sp2idx[lbl]] = 1.0

            entries.append({
                "filepath": fp,
                "label": label,
                "group": f"__ss_{row['filename']}_{row['start_sec']}__",
                "is_soundscape": True,
                "start_sample": int(row["start_sec"] * SR),
                "end_sample": int(row["end_sec"] * SR),
            })

    print(
        f"[data] total entries={len(entries)} "
        f"(train_audio={len(train_df)}, soundscape={len(entries)-len(train_df)})",
        flush=True,
    )
    return entries


# ---------------------------------------------------------------------------
# Mel spectrogram transform (on GPU)
# ---------------------------------------------------------------------------
class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform):
        # waveform: (B, T)
        spec = self.mel(waveform)     # (B, n_mels, time_frames)
        spec = self.db(spec)          # dB scale
        spec = (spec + 80) / 80       # normalize to [0, 1] range
        return spec


# ---------------------------------------------------------------------------
# AttBlockV2 (from BirdCLEF 2023 2nd place, via PANNs)
# ---------------------------------------------------------------------------
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class AttBlockV2(nn.Module):
    def __init__(self, in_features, out_features, activation="sigmoid"):
        super().__init__()
        self.activation = activation
        self.att = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (B, C, T)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        clipwise_output = torch.sum(norm_att * cla, dim=2)
        return clipwise_output, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# SED Model: EfficientNet-B0 + AttBlockV2
# ---------------------------------------------------------------------------
class SEDModel(nn.Module):
    def __init__(
        self,
        backbone_name="tf_efficientnet_b0_ns",
        n_classes=N_CLASSES,
        n_mels=N_MELS,
        pretrained=True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.mel_transform = MelSpecTransform()

        # Spectrogram batch norm
        self.bn0 = nn.BatchNorm2d(n_mels)
        init_bn(self.bn0)

        # Backbone (strip final pooling + classifier)
        base_model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,
            drop_path_rate=0.2,
            drop_rate=0.5,
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        in_features = base_model.classifier.in_features  # 1280 for B0

        # SED head
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        init_layer(self.fc1)
        self.att_block = AttBlockV2(in_features, n_classes, activation="sigmoid")

        # SpecAugment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(SPEC_FREQ_MASK)
        self.time_mask = torchaudio.transforms.TimeMasking(SPEC_TIME_MASK)

    def forward(self, waveform, apply_augment=False):
        # waveform: (B, T) raw audio
        # 1. Mel spectrogram
        spec = self.mel_transform(waveform)  # (B, n_mels, time_frames)

        # 2. SpecAugment (training only)
        if apply_augment and self.training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        # 3. Add channel dim -> (B, 1, n_mels, T)
        x = spec.unsqueeze(1)

        # 4. BN on mel axis: (B, 1, n_mels, T) -> transpose -> BN -> transpose
        x = x.transpose(1, 2)  # (B, n_mels, 1, T)
        x = self.bn0(x)
        x = x.transpose(1, 2)  # (B, 1, n_mels, T)

        # 5. CNN backbone
        x = self.encoder(x)    # (B, C, F', T')

        # 6. Collapse frequency axis
        x = torch.mean(x, dim=2)  # (B, C, T')

        # 7. Channel smoothing (max + avg pool over time with k=3)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)     # (B, T', C)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)     # (B, C, T')
        x = F.dropout(x, p=0.5, training=self.training)

        # 8. Attention pooling
        # Get raw logits from cla branch (before sigmoid)
        norm_att = torch.softmax(torch.tanh(self.att_block.att(x)), dim=-1)
        segmentwise_logit = self.att_block.cla(x)  # (B, n_classes, T') raw logits

        clipwise_logit = torch.sum(norm_att * segmentwise_logit, dim=2)  # (B, n_classes)
        framewise_logit_max = segmentwise_logit.max(dim=2)[0]  # (B, n_classes)
        clipwise_output = torch.sigmoid(clipwise_logit)

        return {
            "clipwise_logit": clipwise_logit,
            "framewise_logit_max": framewise_logit_max,
            "clipwise_output": clipwise_output,
        }


# ---------------------------------------------------------------------------
# Loss: 0.5 * clip BCE + 0.5 * max-frame BCE
# ---------------------------------------------------------------------------
class BCEDualLoss(nn.Module):
    def __init__(self, clip_weight=0.5, frame_weight=0.5):
        super().__init__()
        self.clip_weight = clip_weight
        self.frame_weight = frame_weight

    def forward(self, output, target):
        clip_loss = F.binary_cross_entropy_with_logits(
            output["clipwise_logit"], target
        )
        frame_loss = F.binary_cross_entropy_with_logits(
            output["framewise_logit_max"], target
        )
        return self.clip_weight * clip_loss + self.frame_weight * frame_loss


# ---------------------------------------------------------------------------
# MixUp on waveforms
# ---------------------------------------------------------------------------
def mixup(waveform, labels, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(waveform.size(0), device=waveform.device)
    waveform = lam * waveform + (1 - lam) * waveform[idx]
    labels = lam * labels + (1 - lam) * labels[idx]
    return waveform, labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_tgts = [], []
    for waveform, yb in loader:
        waveform = waveform.to(DEVICE, non_blocking=True)
        out = model(waveform, apply_augment=False)
        probs = out["clipwise_output"].cpu().numpy()
        all_probs.append(probs)
        all_tgts.append(yb.numpy())

    probs = np.concatenate(all_probs, axis=0)
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, epoch):
    model.train()
    running_loss = 0.0
    n_seen = 0
    pbar = tqdm(loader, desc=f"ep{epoch:02d}", leave=False)
    for waveform, yb in pbar:
        waveform = waveform.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        # MixUp
        if np.random.rand() < MIXUP_PROB:
            waveform, yb = mixup(waveform, yb)

        optimizer.zero_grad()
        out = model(waveform, apply_augment=True)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveform.size(0)
        n_seen += waveform.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n_seen, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[1/5] Building dataset entries...", flush=True)
    entries = build_entries()

    groups = np.array([e["group"] for e in entries])
    labels = np.array([e["label"] for e in entries])

    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_idx, va_idx = next(gss.split(np.arange(len(entries)), labels, groups=groups))

    train_entries = [entries[i] for i in tr_idx]
    val_entries = [entries[i] for i in va_idx]

    # Count stats
    n_train_sc = sum(1 for e in train_entries if e.get("is_soundscape"))
    n_val_sc = sum(1 for e in val_entries if e.get("is_soundscape"))
    print(
        f"[split] train={len(train_entries)} (sc={n_train_sc}) "
        f"val={len(val_entries)} (sc={n_val_sc})",
        flush=True,
    )

    train_ds = BirdAudioDatasetV2(train_entries, is_train=True)
    val_ds = BirdAudioDatasetV2(val_entries, is_train=False)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=True,
    )

    print("[2/5] Building SED model...", flush=True)
    model = SEDModel(pretrained=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       params={n_params:,}", flush=True)

    loss_fn = BCEDualLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("[3/5] Training...", flush=True)
    history = {"train_loss": [], "val_auc": []}
    best_auc = -1.0
    best_state = None
    patience_count = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch)
        scheduler.step()

        val_auc = evaluate(model, val_loader)
        history["train_loss"].append(tr_loss)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_auc
        flag = "*" if improved else " "
        print(
            f"[SED] ep{epoch:02d} tr_loss={tr_loss:.4f} val_auc={val_auc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} {flag}",
            flush=True,
        )

        if improved:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"[SED] early stop at epoch {epoch}", flush=True)
                break

    total_time = time.time() - t0
    best_epoch = history["val_auc"].index(best_auc) + 1
    print(
        f"\n[4/5] Done. best_auc={best_auc:.4f} at epoch {best_epoch}, "
        f"total_time={total_time:.1f}s",
        flush=True,
    )

    # Save best model (state_dict only — no mel transform weights needed,
    # they are parameter-free transforms in torchaudio)
    if best_state is not None:
        save_keys = {k: v for k, v in best_state.items()
                     if not k.startswith("mel_transform.")}
        torch.save(save_keys, OUT_DIR / "sed_best.pt")
        print(f"[save] weights -> {OUT_DIR / 'sed_best.pt'}", flush=True)

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ep_range = range(1, len(history["val_auc"]) + 1)
    ax1.plot(ep_range, history["val_auc"], marker="o", markersize=3, label="val_auc")
    ax1.axhline(y=best_auc, color="r", linestyle="--", alpha=0.5, label=f"best={best_auc:.4f}")
    ax1.set(xlabel="Epoch", ylabel="Val macro AUC", title="SED Val AUC")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ep_range, history["train_loss"], marker="o", markersize=3, color="orange")
    ax2.set(xlabel="Epoch", ylabel="Train loss", title="SED Train Loss")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sed_curves.png", dpi=130)
    plt.close(fig)
    print(f"[save] curves -> {OUT_DIR / 'sed_curves.png'}", flush=True)

    # Results JSON
    result = {
        "model": "SEDModel (EfficientNet-B0 + AttBlockV2)",
        "params": n_params,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "epochs_run": len(history["val_auc"]),
        "total_time_sec": total_time,
        "config": {
            "backbone": "tf_efficientnet_b0_ns",
            "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
            "f_min": F_MIN, "f_max": F_MAX, "sr": SR,
            "duration_sec": DURATION_SEC,
            "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
            "patience": PATIENCE, "seed": SEED,
            "mixup_prob": MIXUP_PROB, "mixup_alpha": MIXUP_ALPHA,
            "loss": "0.5*clip_bce + 0.5*maxframe_bce",
        },
        "history": history,
    }
    with open(OUT_DIR / "sed_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[save] results -> {OUT_DIR / 'sed_results.json'}", flush=True)

    print(f"\n[5/5] Summary: SED val_auc={best_auc:.4f} | "
          f"MLP baseline val_auc=0.9646 (single split) / 0.9552 (5-fold mean)",
          flush=True)


if __name__ == "__main__":
    main()
