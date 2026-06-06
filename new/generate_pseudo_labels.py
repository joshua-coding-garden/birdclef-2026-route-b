"""
用 5-fold ensemble 生成 pseudo-labels。

輸入：
  - embedding/sc_all_embeddings.npy  (從 Kaggle 下載的全部 soundscape embedding)
  - embedding/sc_all_meta.parquet
  - new/mlp_fold{0-4}.pt

輸出：
  - embedding/pseudo_embeddings.npy   (未標記部分的 embedding)
  - embedding/pseudo_labels.npy       (soft pseudo-labels)
  - embedding/pseudo_meta.parquet     (metadata)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EMB_DIR = Path(r"C:\birdCLEF\embedding")
NEW_DIR = Path(r"C:\birdCLEF\new")
DATA_DIR = Path(r"C:\birdCLEF\birdclef-2026")

N_CLASSES = 234
EMB_DIM = 1536
N_FOLDS = 5
BATCH = 2048


class MLPHead(nn.Module):
    def __init__(self, in_dim=EMB_DIM, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 768), nn.BatchNorm1d(768), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(768, 384), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(384, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def main():
    # 1. 載入全部 soundscape embeddings（從 Kaggle 下載的）
    print("[1/4] loading soundscape embeddings...")
    sc_emb = np.load(EMB_DIR / "sc_all_embeddings.npy").astype(np.float32)
    sc_meta = pd.read_parquet(EMB_DIR / "sc_all_meta.parquet")
    print(f"       total: {sc_emb.shape[0]} frames, {sc_meta['filename'].nunique()} files")

    # 2. 找出哪些是已標記的（排除它們，只對未標記的生成 pseudo-label）
    print("[2/4] identifying unlabeled frames...")
    labels_csv = pd.read_csv(DATA_DIR / "train_soundscapes_labels.csv")
    labeled_files = set(labels_csv["filename"].unique())
    print(f"       labeled files: {len(labeled_files)}")

    unlabeled_mask = ~sc_meta["filename"].isin(labeled_files)
    pseudo_emb = sc_emb[unlabeled_mask.values]
    pseudo_meta = sc_meta[unlabeled_mask].reset_index(drop=True)
    print(f"       unlabeled: {len(pseudo_meta)} frames, {pseudo_meta['filename'].nunique()} files")

    # 3. 載入 5-fold ensemble，生成 soft pseudo-labels
    print("[3/4] generating pseudo-labels with 5-fold ensemble...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = []
    for i in range(N_FOLDS):
        h = MLPHead()
        h.load_state_dict(torch.load(NEW_DIR / f"mlp_fold{i}.pt", map_location="cpu"))
        h.eval()
        h.to(device)
        heads.append(h)
    print(f"       {N_FOLDS} heads loaded on {device}")

    all_probs = []
    n_batches = (len(pseudo_emb) + BATCH - 1) // BATCH
    with torch.no_grad():
        for i, start in enumerate(range(0, len(pseudo_emb), BATCH)):
            end = min(start + BATCH, len(pseudo_emb))
            x = torch.from_numpy(pseudo_emb[start:end]).to(device)
            acc = torch.zeros(x.size(0), N_CLASSES, device=device)
            for h in heads:
                acc += torch.sigmoid(h(x))
            probs = acc / N_FOLDS
            all_probs.append(probs.cpu().numpy())
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f"       batch {i+1}/{n_batches}", flush=True)
    pseudo_labels = np.concatenate(all_probs, axis=0).astype(np.float32)
    print(f"       pseudo_labels shape: {pseudo_labels.shape}")
    print(f"       mean prob: {pseudo_labels.mean():.4f}, max: {pseudo_labels.max():.4f}")

    # 4. 儲存
    print("[4/4] saving...")
    np.save(EMB_DIR / "pseudo_embeddings.npy", pseudo_emb)
    np.save(EMB_DIR / "pseudo_labels.npy", pseudo_labels)
    pseudo_meta.to_parquet(EMB_DIR / "pseudo_meta.parquet", index=False)

    print(f"       {EMB_DIR / 'pseudo_embeddings.npy'} ({pseudo_emb.nbytes / 1e6:.1f} MB)")
    print(f"       {EMB_DIR / 'pseudo_labels.npy'} ({pseudo_labels.nbytes / 1e6:.1f} MB)")
    print(f"       {EMB_DIR / 'pseudo_meta.parquet'}")

    # 統計
    top_species = pseudo_labels.mean(axis=0).argsort()[::-1][:10]
    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    species_list = taxonomy["primary_label"].tolist()
    print("\n       Top 10 predicted species in pseudo-labels:")
    for rank, idx in enumerate(top_species):
        print(f"         {rank+1}. {species_list[idx]}: mean_prob={pseudo_labels[:, idx].mean():.4f}")

    print("\n[OK] pseudo-labels generated.")


if __name__ == "__main__":
    main()
