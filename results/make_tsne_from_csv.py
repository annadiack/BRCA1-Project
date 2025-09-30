# make_tsne_from_csv.py
import argparse, os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def detect_feature_columns(df):
    # Prefer latent columns (e.g., z1, z2, ..., latent_*)
    latent_like = [c for c in df.columns
                   if c.lower().startswith(("z", "latent"))
                   and pd.api.types.is_numeric_dtype(df[c])]
    if len(latent_like) >= 2:
        return latent_like
    # Fallback: all numeric columns minus obvious metadata
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    drop = ("id","rs","chrom","chr","pos","position","label","class","cluster","sample")
    return [c for c in numeric_cols if not any(k in c.lower() for k in drop)]

def detect_label_column(df):
    for c in ("cluster","label","group","class"):
        if c in df.columns:
            return c
    return None

def tsne_png(X, labels, perplexity, out_path, title_suffix=""):
    tsne = TSNE(n_components=2, perplexity=perplexity,
                learning_rate="auto", init="pca", random_state=42)
    emb = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(emb[idx, 0], emb[idx, 1], s=10, label=str(lab))
        plt.legend(title="Label", fontsize=9, markerscale=1.5)
    else:
        plt.scatter(emb[:, 0], emb[:, 1], s=10)
    plt.title(f"t-SNE (perplexity={perplexity}){title_suffix}")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to your CSV (e.g., brca1_autoencoder_results.csv)")
    ap.add_argument("--rows", type=int, default=20000, help="Optional row cap for speed/memory")
    ap.add_argument("--outdir", default=".", help="Output directory for PNGs")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, nrows=args.rows, low_memory=False)

    feat = detect_feature_columns(df)
    if len(feat) < 2:
        raise SystemExit(f"Not enough numeric feature columns found. Got {len(feat)}. "
                         f"Expected latent columns like z1,z2,... or latent_*.")

    labcol = detect_label_column(df)
    labels = df[labcol].values if labcol is not None else None

    X = df[feat].values
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if labels is not None:
        labels = labels[mask]

    os.makedirs(args.outdir, exist_ok=True)
    for p in [5, 30, 50]:
        out = os.path.join(args.outdir, f"tsne_perplexity_{p}.png")
        tsne_png(X, labels, p, out, title_suffix=f"  •  {len(feat)} features, {X.shape[0]} samples")
        print("Saved", out)

if __name__ == "__main__":
    main()
