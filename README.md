# BRCA1 Project: Autoencoder-Based Clustering of Mutation Reads

This repository implements an **autoencoder-based pipeline** to analyze and cluster **BRCA1 mutation reads** for haplotype analysis.  
The project combines **deep learning** (autoencoders), **clustering**, and **population genetics data** (e.g., gnomAD, MyHeritage raw DNA) to explore the structure and distribution of BRCA1 variants.

---

## 📌 Overview
- **Goal:** Detect patterns and outliers in BRCA1 mutations using dimensionality reduction and clustering.  
- **Methods:**  
  - Feedforward autoencoder with latent space representation  
  - Reconstruction error analysis  
  - t-SNE & k-means clustering for visualization  
- **Applications:** Supports genetic research, haplotype analysis, and variant interpretation.

---

## ⚙️ Installation

Clone this repository:
```bash
git clone https://github.com/annadiack/BRCA1-Project.git
cd BRCA1-Project
