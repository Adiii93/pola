# 1. Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # <-- Lengkap!
import umap.umap_ as umap

# 2. Load dataset
iris = load_iris()
X = iris.data    # <-- Lengkap!
y = iris.target
labels = iris.target_names   # <-- Lengkap!

# 3. PCA
pca = PCA(n_components=2)   # <-- Lengkap!
X_pca = pca.fit_transform(X)   # <-- Lengkap!

plt.figure(figsize=(6, 4))
for i in range(3):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label=labels[i])
plt.title("PCA - Iris Dataset")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.show()

# 4. t-SNE
tsne = TSNE(n_components=2, random_state=42)  # <-- Lengkap!
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(6, 4))
for i in range(3):
    plt.scatter(X_tsne[y==i, 0], X_tsne[y==i, 1], label=labels[i])  # <-- Lengkap!
plt.title("t-SNE - Iris Dataset")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.grid(True)
plt.show()

# 5. UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)  # <-- Lengkap!
X_umap = umap_model.fit_transform(X)  # <-- Lengkap!

plt.figure(figsize=(6, 4))
for i in range(3):
    plt.scatter(X_umap[y==i, 0], X_umap[y==i, 1], label=labels[i])  # <-- Lengkap!
plt.title("UMAP - Iris Dataset")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.grid(True)
plt.show()
