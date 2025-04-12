import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Load data Iris
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

# Tampilkan 5 baris pertama
print(pd.DataFrame(X, columns=iris.feature_names).head())

#PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualisasi
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label=labels[i])
plt.title("PCA - Iris Dataset")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.show()

#TSNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(6, 4))
for i in range(3):
    plt.scatter(X_tsne[y==i, 0], X_tsne[y==i, 1], label=labels[i])
plt.title("t-SNE - Iris Dataset")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.grid(True)
plt.show()

#UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

plt.figure(figsize=(6, 4))
for i in range(3):
    plt.scatter(X_umap[y==i, 0], X_umap[y==i, 1], label=labels[i])
plt.title("UMAP - Iris Dataset")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.grid(True)
plt.show()
