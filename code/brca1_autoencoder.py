#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering of BRCA1 Mutation Reads Using an Autoencoder for Haplotype Analysis
------------------------------------------------------------------------------

Pipeline
- Load gnomAD BRCA1 CSV (hard-coded path).
- Feature engineering: keep numeric columns + restrained one-hot for frequent categories.
- Standardize features.
- Train a symmetric autoencoder (PyTorch). Fallback to PCA if PyTorch is missing.
- Reconstruction error as anomaly score.
- Latent-space clustering (DBSCAN + KMeans).
- 2D t-SNE embedding for qualitative inspection.
- Export a single CSV with metrics, labels, latent coords, and t-SNE coords.
- Save simple matplotlib figures (no specific colors/styles set).

Note
This file is the "ready-to-run" variant. Paths are hard-coded for the current session.
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)

# --------------------
# Hard-coded paths
# --------------------
DATA_PATH = "/mnt/data/gnomAD_v4.1.0_ENSG00000012048_2025_07_31_15_05_59.csv"
OUT_DIR = "/mnt/data/brca1_autoencoder_results"

CAT_KEYS = [
    "ref", "alt", "allele", "consequence", "impact",
    "effect", "annotation", "biotype", "class", "variant_type"
]

def detect_categoricals(df: pd.DataFrame):
    cand = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in CAT_KEYS):
            cand.append(c)
    return cand

def safe_one_hot(df: pd.DataFrame, cat_cols, top_k=20):
    used = []
    parts = [df]
    for c in cat_cols:
        if c not in df.columns:
            continue
        s = df[c].astype("string").fillna("NA")
        top_vals = s.value_counts().nlargest(top_k).index.tolist()
        s = s.where(s.isin(top_vals), other="OTHER")
        dummies = pd.get_dummies(s, prefix=c, dtype=np.uint8)
        parts.append(dummies)
        used.append(c)
    base = df.drop(columns=[c for c in used if c in df.columns], errors="ignore")
    out = pd.concat([base] + parts[1:], axis=1)
    return out, used

def build_features(df: pd.DataFrame, top_k_cat=20):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = detect_categoricals(df)
    keep = list(set(numeric_cols + cat_cols))
    feat = df[keep].copy() if keep else df.select_dtypes(include=[np.number]).copy()

    feat, used_cat = safe_one_hot(feat, cat_cols, top_k=top_k_cat)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(axis=1, how="all")

    for c in feat.columns:
        if feat[c].dtype.kind in "iufc":
            if feat[c].isna().any():
                feat[c] = feat[c].fillna(feat[c].median())
        else:
            feat[c] = feat[c].fillna(0)

    if feat.shape[1] == 0:
        raise ValueError("No usable features after preprocessing.")
    return feat

def train_autoencoder(X, latent_dim=None, epochs=120, batch_size=64, lr=1e-3):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    torch.manual_seed(RNG_SEED)

    n_features = X.shape[1]
    if latent_dim is None:
        latent_dim = max(2, min(16, n_features // 4))

    class AE(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            hidden1 = max(32, min(256, input_dim // 2))
            hidden2 = max(latent_dim * 2, 16)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out, z

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(n_features, latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    n = X_tensor.shape[0]
    steps = math.ceil(n / batch_size)

    losses = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        batch_losses = []
        for i in range(steps):
            idx = perm[i*batch_size:(i+1)*batch_size]
            xb = X_tensor[idx]
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))

    model.eval()
    with torch.no_grad():
        recon_all, Z_t = model(X_tensor)
        Z = Z_t.cpu().numpy()
        X_recon = recon_all.cpu().numpy()
    recon_error = ((X - X_recon) ** 2).mean(axis=1)
    return Z, recon_error, losses, latent_dim

def embed_tsne(Z):
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, Z.shape[0] // 50))
    tsne = TSNE(n_components=2, init="pca", random_state=RNG_SEED,
                learning_rate="auto", perplexity=perplexity)
    return tsne.fit_transform(Z)

def cluster_latent(Z):
    from sklearn.cluster import DBSCAN, KMeans
    db = DBSCAN(eps=0.8, min_samples=5).fit(Z)
    db_labels = db.labels_
    k = min(6, max(2, Z.shape[0] // 200))
    km = KMeans(n_clusters=k, random_state=RNG_SEED, n_init="auto").fit(Z)
    km_labels = km.labels_
    return db_labels, km_labels

def plot_training_loss(losses, out_png):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_recon_hist(errors, out_png):
    plt.figure()
    plt.hist(errors, bins=50)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Distribution")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_tsne_clusters(Z2, labels, title, out_png):
    plt.figure()
    for lab in np.unique(labels):
        mask = labels == lab
        plt.scatter(Z2[mask, 0], Z2[mask, 1], s=10, label=f"{lab}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.legend(markerscale=2, loc="best")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_tsne_outliers(Z2, outlier_mask, out_png):
    plt.figure()
    m_in = ~outlier_mask
    plt.scatter(Z2[m_in, 0], Z2[m_in, 1], s=10, label="Inliers")
    plt.scatter(Z2[outlier_mask, 0], Z2[outlier_mask, 1], s=10, label="Top 5% error")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Latent Space (t-SNE) — Reconstruction Error Outliers")
    plt.legend(markerscale=2, loc="best")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    df = pd.read_csv(DATA_PATH)
    df_orig = df.copy()

    # Features
    feats = build_features(df, top_k_cat=20)

    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values.astype(np.float32))

    # Autoencoder with PCA fallback
    try:
        Z, recon_error, losses, latent_dim = train_autoencoder(X, latent_dim=None, epochs=120, batch_size=64, lr=1e-3)
        backend = "PyTorch Autoencoder"
    except Exception as e:
        from sklearn.decomposition import PCA
        latent_dim = max(2, min(16, X.shape[1] // 4))
        pca = PCA(n_components=latent_dim, random_state=RNG_SEED)
        Z = pca.fit_transform(X)
        X_recon = pca.inverse_transform(Z)
        recon_error = ((X - X_recon) ** 2).mean(axis=1)
        losses = None
        backend = "PCA fallback"

    # Outliers
    q95 = np.quantile(recon_error, 0.95)
    outlier_mask = recon_error >= q95

    # t-SNE + clustering
    Z2 = embed_tsne(Z)
    db_labels, km_labels = cluster_latent(Z)

    # Assemble result table
    res = df_orig.copy()
    res["reconstruction_error"] = recon_error
    res["ae_outlier_p95"] = outlier_mask.astype(int)
    res["dbscan_label"] = db_labels
    res["kmeans_label"] = km_labels
    for i in range(Z.shape[1]):
        res[f"latent_{i+1}"] = Z[:, i]
    res["tsne_x"] = Z2[:, 0]
    res["tsne_y"] = Z2[:, 1]

    out_csv = os.path.join(OUT_DIR, "brca1_autoencoder_results.csv")
    res.to_csv(out_csv, index=False)

    # Plots
    if losses is not None:
        plot_training_loss(losses, os.path.join(OUT_DIR, "training_loss.png"))
    plot_recon_hist(recon_error, os.path.join(OUT_DIR, "reconstruction_error_hist.png"))
    plot_tsne_clusters(Z2, km_labels, "Latent (t-SNE) — KMeans Clusters", os.path.join(OUT_DIR, "tsne_kmeans.png"))
    plot_tsne_outliers(Z2, outlier_mask, os.path.join(OUT_DIR, "tsne_outliers.png"))

    print(f"[OK] Backend: {backend}")
    print(f"[OK] Rows: {df.shape[0]} | Features after encoding: {feats.shape[1]} | Latent dim: {latent_dim}")
    print(f"[OK] Results CSV: {out_csv}")
    print(f"[OK] Figures in: {OUT_DIR}")

if __name__ == "__main__":
    main()
