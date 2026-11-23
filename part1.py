import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import fit_pca, fix_signs, load_data, save_table


def part1(z_train, names):
    print("\nP1. PCA from first principles")
    cov, values, vectors = fit_pca(z_train)
    for j, value in enumerate(values, start=1):
        print(f"lambda {j} = {value:.4f}")

    identity = np.eye(len(values))
    print(f"Orthogonality residual = {np.linalg.norm(vectors.T @ vectors - identity, 'fro'):.4e}")
    print(f"Trace = {np.trace(cov):.4f}")
    print(f"Trace minus sum of eigenvalues = {np.trace(cov) - values.sum():.4e}")

    reference = PCA(n_components=13, svd_solver="full").fit(z_train)
    reference_vectors = fix_signs(reference.components_.T)
    print(f"Max eigenvalue difference = {np.max(np.abs(values - reference.explained_variance_)):.4e}")
    print(f"Max eigenvector difference = {np.max(np.abs(vectors - reference_vectors)):.4e}")

    loadings = pd.DataFrame({"Feature": names, "PC1": vectors[:, 0], "PC2": vectors[:, 1]})
    save_table(loadings, "p1_loadings.csv")
    for j in range(2):
        top = np.argsort(-np.abs(vectors[:, j]))[:3]
        print(f"PC{j + 1} largest absolute loadings: {', '.join(names[i] for i in top)}")
    return cov, values, vectors


if __name__ == "__main__":
    variants, _, _, names = load_data()
    z_train, _ = variants["V3"]
    part1(z_train, names)
