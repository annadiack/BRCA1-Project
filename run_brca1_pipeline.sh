#!/bin/bash
set -euo pipefail

echo "=== BRCA1 Autoencoder Pipeline (one-click) ==="

PROJECT_ROOT="$HOME/Desktop/BRCA1_Project"
DATA_DIR="$PROJECT_ROOT/data"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
RESULTS_DIR="$PROJECT_ROOT/results"
VENV_DIR="$PROJECT_ROOT/.venv"

mkdir -p "$DATA_DIR" "$SCRIPTS_DIR" "$RESULTS_DIR"

echo "[1/5] Creating virtual environment…"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[2/5] Installing Python dependencies (OpenBLAS wheels, no MKL)…"
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "[3/5] Writing autoencoder script to $SCRIPTS_DIR/brca1_autoencoder.py …"
cat > "$SCRIPTS_DIR/brca1_autoencoder.py" << 'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering of BRCA1 Mutation Reads Using an Autoencoder for Haplotype Analysis
------------------------------------------------------------------------------

Desktop-ready:
- Project root: ~/Desktop/BRCA1_Project (by default)
- Input CSV:   BRCA1_Project/data/<your gnomAD BRCA1 CSV>
- Outputs:     BRCA1_Project/results/

Behavior:
- Loads the first CSV in data/ if the canonical gnomAD filename is not present.
- Numeric + restrained one-hot features, standardization.
- PyTorch autoencoder with PCA fallback.
- Reconstruction error, DBSCAN & KMeans on latent space.
- t-SNE visualization. Exports wide results CSV + PNGs.
"""

import os, sys, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)

def project_root():
    # Default root: ~/Desktop/BRCA1_Project
    return os.path.expanduser(os.getenv("BRCA1_PROJECT_ROOT", "~/Desktop/BRCA1_Project"))

def resolve_paths():
    root = project_root()
    data_dir = os.path.join(root, "data")
    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    exact = os.path.join(data_dir, "gnomAD_v4.1.0_ENSG00000012048_2025_07_31_15_05_59.csv")
    if os.path.isfile(exact):
        csv_path = exact
    else:
        candidates = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not candidates:
            raise FileNotFoundError(f"No CSV found in {data_dir}. Place your gnomAD BRCA1 CSV there.")
        csv_path = candidates[0]

    return csv_path, results_dir

CAT_KEYS = [
    "ref", "alt", "allele", "consequence", "impact",
    "effect", "annotation", "biotype", "class", "variant_type"
]

def detect_categoricals(df):
    out = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in CAT_KEYS):
            out.append(c)
    return out

def safe_one_hot(df, cat_cols, top_k=20):
    used, parts = [], [df]
    for c in cat_cols:
        if c not in df.columns: 
            continue
        s = df[c].astype("string").fillna("NA")
        top_vals = s.value_counts().nlargest(top_k).index.tolist()
        s = s.where(s.isin(top_vals), other="OTHER")
        dummies = pd.get_dummies(s, prefix=c, dtype=np.uint8)
        parts.append(dummies); used.append(c)
    base = df.drop(columns=[c for c in used if c in df.columns], errors="ignore")
    return pd.concat([base] + parts[1:], axis=1)

def build_features(df, top_k_cat=20):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cats = detect_categoricals(df)
    keep = list(set(num + cats))
    feat = df[keep].copy() if keep else df.select_dtypes(include=[np.number]).copy()
    feat = safe_one_hot(feat, cats, top_k=top_k_cat)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(axis=1, how="all")
    for c in feat.columns:
        if feat[c].dtype.kind in "iufc":
            if feat[c].isna().any(): feat[c] = feat[c].fillna(feat[c].median())
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
        def __init__(self, d, z):
            super().__init__()
            h1 = max(32, min(256, d // 2))
            h2 = max(z*2, 16)
            self.enc = nn.Sequential(nn.Linear(d, h1), nn.ReLU(),
                                     nn.Linear(h1, h2), nn.ReLU(),
                                     nn.Linear(h2, z))
            self.dec = nn.Sequential(nn.Linear(z, h2), nn.ReLU(),
                                     nn.Linear(h2, h1), nn.ReLU(),
                                     nn.Linear(h1, d))
        def forward(self, x):
            z = self.enc(x); xr = self.dec(z); return xr, z

    device = "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"
    model = AE(X.shape[1], latent_dim).__class__(X.shape[1], latent_dim)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = __import__("torch").tensor(X, dtype=__import__("torch").float32).to(device)
    n = X_t.shape[0]; steps = (n + batch_size - 1) // batch_size

    losses = []
    model.train()
    for _ in range(epochs):
        perm = __import__("torch").randperm(n, device=device)
        bl = []
        for i in range(steps):
            idx = perm[i*batch_size:(i+1)*batch_size]
            xb = X_t[idx]
            opt.zero_grad()
            xr, _ = model(xb)
            loss = loss_fn(xr, xb)
            loss.backward(); opt.step()
            bl.append(loss.item())
        losses.append(float(np.mean(bl)))

    model.eval()
    with __import__("torch").no_grad():
        xr, z = model(X_t)
        Z = z.cpu().numpy(); Xr = xr.cpu().numpy()
    err = ((X - Xr) ** 2).mean(axis=1)
    return Z, err, losses

def embed_tsne(Z):
    from sklearn.manifold import TSNE
    per = min(30, max(5, max(1, Z.shape[0] // 50)))
    return TSNE(n_components=2, init="pca", random_state=RNG_SEED,
                learning_rate="auto", perplexity=per).fit_transform(Z)

def cluster_latent(Z):
    from sklearn.cluster import DBSCAN, KMeans
    db = DBSCAN(eps=0.8, min_samples=5).fit(Z)
    k = min(6, max(2, max(2, Z.shape[0] // 200)))
    km = KMeans(n_clusters=k, random_state=RNG_SEED, n_init="auto").fit(Z)
    return db.labels_, km.labels_

def plot_training_loss(losses, out_png):
    plt.figure(); plt.plot(losses)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Autoencoder Training Loss")
    plt.savefig(out_png, bbox_inches="tight"); plt.close()

def plot_recon_hist(errors, out_png):
    plt.figure(); plt.hist(errors, bins=50)
    plt.xlabel("Reconstruction Error"); plt.ylabel("Count"); plt.title("Reconstruction Error Distribution")
    plt.savefig(out_png, bbox_inches="tight"); plt.close()

def plot_tsne_clusters(Z2, labels, title, out_png):
    plt.figure()
    for lab in np.unique(labels):
        m = labels == lab
        plt.scatter(Z2[m,0], Z2[m,1], s=10, label=f"{lab}")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.title(title)
    plt.legend(markerscale=2, loc="best"); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def plot_tsne_outliers(Z2, mask, out_png):
    plt.figure()
    plt.scatter(Z2[~mask,0], Z2[~mask,1], s=10, label="Inliers")
    plt.scatter(Z2[mask,0], Z2[mask,1], s=10, label="Top 5% error")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.title("Latent (t‑SNE) — Reconstruction Error Outliers")
    plt.legend(markerscale=2, loc="best"); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def main():
    root = project_root()
    csv_path, out_dir = resolve_paths()
    print(f"[i] Project root: {root}")
    print(f"[i] CSV: {csv_path}")
    print(f"[i] Results: {out_dir}")

    df = pd.read_csv(csv_path)
    feats = build_features(df, top_k_cat=20)

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(feats.values.astype(np.float32))

    backend = "AE"
    try:
        Z, err, losses = train_autoencoder(X, latent_dim=None, epochs=120, batch_size=64, lr=1e-3)
    except Exception as e:
        from sklearn.decomposition import PCA
        z_dim = max(2, min(16, X.shape[1] // 4))
        pca = PCA(n_components=z_dim, random_state=RNG_SEED)
        Z = pca.fit_transform(X)
        Xr = pca.inverse_transform(Z)
        err = ((X - Xr) ** 2).mean(axis=1)
        losses = None
        backend = f"PCA fallback ({e})"

    q95 = float(np.quantile(err, 0.95))
    out_mask = err >= q95

    Z2 = embed_tsne(Z)
    db, km = cluster_latent(Z)

    res = df.copy()
    res["reconstruction_error"] = err
    res["ae_outlier_p95"] = out_mask.astype(int)
    res["dbscan_label"] = db
    res["kmeans_label"] = km
    for i in range(Z.shape[1]):
        res[f"latent_{i+1}"] = Z[:, i]
    res["tsne_x"] = Z2[:,0]; res["tsne_y"] = Z2[:,1]

    out_csv = os.path.join(out_dir, "brca1_autoencoder_results.csv")
    res.to_csv(out_csv, index=False)

    if losses is not None:
        plot_training_loss(losses, os.path.join(out_dir, "training_loss.png"))
    plot_recon_hist(err, os.path.join(out_dir, "reconstruction_error_hist.png"))
    plot_tsne_clusters(Z2, km, "Latent (t‑SNE) — KMeans Clusters", os.path.join(out_dir, "tsne_kmeans.png"))
    plot_tsne_outliers(Z2, out_mask, os.path.join(out_dir, "tsne_outliers.png"))

    print(f"[OK] Backend: {backend}")
    print(f"[OK] Rows: {df.shape[0]} | Features: {feats.shape[1]} | Latent dims: {res.filter(like='latent_').shape[1]}")
    print(f"[OK] Results CSV: {out_csv}")
    print(f"[OK] Figures: {out_dir}")

if __name__ == "__main__":
    main()
PY
chmod +x "$SCRIPTS_DIR/brca1_autoencoder.py"

echo "[4/5] Checking for input CSV in $DATA_DIR …"
CSV_CANON="$DATA_DIR/gnomAD_v4.1.0_ENSG00000012048_2025_07_31_15_05_59.csv"
if [ ! -f "$CSV_CANON" ]; then
  # If canonical file is missing, ensure there is at least *some* CSV present
  shopt -s nullglob
  CSV_LIST=("$DATA_DIR"/*.csv)
  if [ ${#CSV_LIST[@]} -eq 0 ]; then
    echo ">>> Put your gnomAD BRCA1 CSV into: $DATA_DIR"
    echo ">>> Then re-run: bash ~/Desktop/run_brca1_pipeline.sh"
    exit 1
  fi
fi

echo "[5/5] Running the pipeline…"
python "$SCRIPTS_DIR/brca1_autoencoder.py"

echo "=== Done. Results are in: $RESULTS_DIR ==="
